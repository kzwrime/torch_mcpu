#include "../native/MCPUFallback.h"
#include "Common.h"
#include "RawPlan.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ops/searchsorted.h>
#include <ATen/ops/searchsorted_ops.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>

#if TORCH_MCPU_ENABLE_CPU_FALLBACK
namespace at::native::mcpu {
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);
} // namespace at::native::mcpu
#endif

namespace at::mcpu {
namespace {

void boxed_cpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
#if TORCH_MCPU_ENABLE_CPU_FALLBACK
  at::native::mcpu::cpu_fallback(op, stack);
#else
  TORCH_CHECK(false, "CPU fallback is disabled for ", op.schema().name());
#endif
}

bool is_standard_mcpu_tensor(const at::Tensor& tensor) {
  return ops::is_mcpu_tensor(tensor) &&
      tensor.layout() == c10::Layout::Strided && !tensor.is_quantized() &&
      !tensor.is_neg() && !tensor.is_conj();
}

bool has_non_expanding_strides(const at::Tensor& out) {
  if (out.is_contiguous()) {
    return true;
  }
  const auto sizes = out.sizes();
  const auto strides = out.strides();
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] > 1 && strides[i] == 0) {
      return false;
    }
  }
  return true;
}

bool is_cpu_scalar_tensor(const at::Tensor& tensor) {
  return tensor.defined() && tensor.is_cpu() && tensor.dim() == 0;
}

bool is_whitelisted_searchsorted_dtype(at::ScalarType dtype) {
  return ops::is_raw_dtype_supported(dtype) && dtype != at::ScalarType::Bool;
}

bool is_whitelisted_numeric_scalar(const at::Scalar& scalar) {
  return !scalar.isSymbolic() &&
      (scalar.type() == at::ScalarType::Long ||
       scalar.type() == at::ScalarType::Double);
}

bool is_whitelisted_sorter(
    const at::Tensor& sorted_sequence,
    const std::optional<at::Tensor>& sorter) {
  if (!sorter.has_value() || !sorter->defined()) {
    return true;
  }
  return is_standard_mcpu_tensor(*sorter) &&
      sorter->device() == sorted_sequence.device() &&
      sorter->scalar_type() == at::ScalarType::Long &&
      sorter->sizes().equals(sorted_sequence.sizes());
}

bool has_no_sorter_alias(
    const at::Tensor& out,
    const std::optional<at::Tensor>& sorter) {
  return !sorter.has_value() || !sorter->defined() || !out.is_alias_of(*sorter);
}

enum class SearchSortedSide : uint8_t {
  None,
  Left,
  Right,
};

std::optional<SearchSortedSide> try_parse_whitelisted_side(
    bool right,
    const std::optional<c10::string_view>& side) {
  if (!side.has_value()) {
    return SearchSortedSide::None;
  }
  if (*side == "right") {
    return SearchSortedSide::Right;
  }
  if (*side == "left" && !right) {
    return SearchSortedSide::Left;
  }
  return std::nullopt;
}

std::optional<c10::string_view> side_to_string_view(SearchSortedSide side) {
  switch (side) {
    case SearchSortedSide::Left:
      return c10::string_view("left");
    case SearchSortedSide::Right:
      return c10::string_view("right");
    case SearchSortedSide::None:
    default:
      return std::nullopt;
  }
}

bool has_valid_searchsorted_tensor_shapes(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self) {
  if (sorted_sequence.dim() < 1) {
    return false;
  }
  if (sorted_sequence.dim() == 1) {
    return true;
  }
  if (self.dim() != sorted_sequence.dim()) {
    return false;
  }
  const int64_t prefix_dims = sorted_sequence.dim() - 1;
  return sorted_sequence.sizes()
      .slice(0, prefix_dims)
      .equals(self.sizes().slice(0, prefix_dims));
}

std::optional<SearchSortedSide> try_validate_searchsorted_common_whitelist(
    const at::Tensor& sorted_sequence,
    bool out_int32,
    bool right,
    const std::optional<c10::string_view>& side,
    const std::optional<at::Tensor>& sorter) {
  if (!is_standard_mcpu_tensor(sorted_sequence) || sorted_sequence.dim() < 1 ||
      !is_whitelisted_searchsorted_dtype(sorted_sequence.scalar_type()) ||
      !is_whitelisted_sorter(sorted_sequence, sorter)) {
    return std::nullopt;
  }
  if (out_int32 &&
      sorted_sequence.size(-1) >= std::numeric_limits<int32_t>::max()) {
    return std::nullopt;
  }
  return try_parse_whitelisted_side(right, side);
}

struct SearchSortedTensorArgs {
  ops::TensorViewSpec sorted_sequence;
  ops::TensorViewSpec self;
  std::optional<ops::TensorViewSpec> sorter;
  ops::TensorViewSpec out;
  bool out_int32 = false;
  bool right = false;
  SearchSortedSide side = SearchSortedSide::None;
};

struct SearchSortedScalarArgs {
  ops::TensorViewSpec sorted_sequence;
  at::Scalar self;
  std::optional<ops::TensorViewSpec> sorter;
  ops::TensorViewSpec out;
  bool out_int32 = false;
  bool right = false;
  SearchSortedSide side = SearchSortedSide::None;
};

void launch_searchsorted_tensor_kernel(
    const char* record_name,
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    SearchSortedSide side,
    const std::optional<at::Tensor>& sorter,
    at::Tensor& out) {
  if (out.numel() == 0) {
    return;
  }
  std::optional<ops::TensorViewSpec> sorter_spec;
  if (sorter.has_value() && sorter->defined()) {
    sorter_spec = ops::make_cpu_view_spec(*sorter);
  }
  SearchSortedTensorArgs args{
      ops::make_cpu_view_spec(sorted_sequence),
      ops::make_cpu_view_spec(self),
      std::move(sorter_spec),
      ops::make_cpu_view_spec(out),
      out_int32,
      right,
      side};
  at::mcpu::launch_timed_kernel_on_stream(
      at::mcpu::detail::get_kernel_launch_stream(out),
      record_name,
      [args = std::move(args),
       record_name](at::mcpu::kernel_timing::Event* timing_event) mutable {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(record_name, timing_event);
        KernelPointerMemoryGuard guard(
            {args.sorted_sequence.data,
             args.self.data,
             args.sorter.has_value() ? args.sorter->data : nullptr,
             args.out.data});
        auto cpu_seq = ops::cpu_view_from_spec(args.sorted_sequence);
        auto cpu_self = ops::cpu_view_from_spec(args.self);
        auto cpu_out = ops::cpu_view_from_spec(args.out);
        std::optional<at::Tensor> cpu_sorter;
        if (args.sorter.has_value()) {
          cpu_sorter = ops::cpu_view_from_spec(*args.sorter);
        }
        at::searchsorted_out(
            cpu_out,
            cpu_seq,
            cpu_self,
            args.out_int32,
            args.right,
            side_to_string_view(args.side),
            cpu_sorter);
      });
}

void launch_searchsorted_scalar_kernel(
    const char* record_name,
    const at::Tensor& sorted_sequence,
    const at::Scalar& self,
    bool out_int32,
    bool right,
    SearchSortedSide side,
    const std::optional<at::Tensor>& sorter,
    at::Tensor& out) {
  std::optional<ops::TensorViewSpec> sorter_spec;
  if (sorter.has_value() && sorter->defined()) {
    sorter_spec = ops::make_cpu_view_spec(*sorter);
  }
  SearchSortedScalarArgs args{
      ops::make_cpu_view_spec(sorted_sequence),
      self,
      std::move(sorter_spec),
      ops::make_cpu_view_spec(out),
      out_int32,
      right,
      side};
  at::mcpu::launch_timed_kernel_on_stream(
      at::mcpu::detail::get_kernel_launch_stream(out),
      record_name,
      [args = std::move(args),
       record_name](at::mcpu::kernel_timing::Event* timing_event) mutable {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(record_name, timing_event);
        KernelPointerMemoryGuard guard(
            {args.sorted_sequence.data,
             args.sorter.has_value() ? args.sorter->data : nullptr,
             args.out.data});
        auto cpu_seq = ops::cpu_view_from_spec(args.sorted_sequence);
        auto cpu_out = ops::cpu_view_from_spec(args.out);
        std::optional<at::Tensor> cpu_sorter;
        if (args.sorter.has_value()) {
          cpu_sorter = ops::cpu_view_from_spec(*args.sorter);
        }
        at::searchsorted_out(
            cpu_out,
            cpu_seq,
            args.self,
            args.out_int32,
            args.right,
            side_to_string_view(args.side),
            cpu_sorter);
      });
}

at::Tensor& searchsorted_Tensor_out(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    std::optional<c10::string_view> side,
    const std::optional<at::Tensor>& sorter,
    at::Tensor& out) {
  const auto expected_dtype =
      out_int32 ? at::ScalarType::Int : at::ScalarType::Long;
  const auto parsed_side = try_validate_searchsorted_common_whitelist(
      sorted_sequence, out_int32, right, side, sorter);

  // Enforce MCPU .out size contracts before CPU fallback so MCPUFallback
  // cannot silently resize `out`.
  if (parsed_side.has_value() && is_standard_mcpu_tensor(out) &&
      out.device() == sorted_sequence.device() &&
      out.scalar_type() == expected_dtype &&
      ((is_standard_mcpu_tensor(self) &&
        self.device() == sorted_sequence.device()) ||
       is_cpu_scalar_tensor(self)) &&
      is_whitelisted_searchsorted_dtype(self.scalar_type()) &&
      has_valid_searchsorted_tensor_shapes(sorted_sequence, self)) {
    ops::check_out_sizes("aten::searchsorted.Tensor_out", out, self.sizes());
    if (is_standard_mcpu_tensor(self) && has_non_expanding_strides(out) &&
        !out.is_alias_of(sorted_sequence) && !out.is_alias_of(self) &&
        has_no_sorter_alias(out, sorter)) {
      launch_searchsorted_tensor_kernel(
          "mcpu::aten::searchsorted.Tensor_out",
          sorted_sequence,
          self,
          out_int32,
          right,
          *parsed_side,
          sorter,
          out);
      return out;
    }
  }

  return at::native::mcpu::custom::
      call_fallback_fn<&boxed_cpu_fallback, at::_ops::searchsorted_Tensor_out>::
          call(sorted_sequence, self, out_int32, right, side, sorter, out);
}

at::Tensor searchsorted_Tensor(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    std::optional<c10::string_view> side,
    const std::optional<at::Tensor>& sorter) {
  const auto parsed_side = try_validate_searchsorted_common_whitelist(
      sorted_sequence, out_int32, right, side, sorter);
  if (parsed_side.has_value() && is_standard_mcpu_tensor(self) &&
      self.device() == sorted_sequence.device() &&
      is_whitelisted_searchsorted_dtype(self.scalar_type()) &&
      has_valid_searchsorted_tensor_shapes(sorted_sequence, self)) {
    const auto out_dtype =
        out_int32 ? at::ScalarType::Int : at::ScalarType::Long;
    auto out =
        at::empty(self.sizes(), sorted_sequence.options().dtype(out_dtype));
    launch_searchsorted_tensor_kernel(
        "mcpu::aten::searchsorted.Tensor",
        sorted_sequence,
        self,
        out_int32,
        right,
        *parsed_side,
        sorter,
        out);
    return out;
  }
  return at::native::mcpu::custom::
      call_fallback_fn<&boxed_cpu_fallback, at::_ops::searchsorted_Tensor>::
          call(sorted_sequence, self, out_int32, right, side, sorter);
}

at::Tensor& searchsorted_Scalar_out(
    const at::Tensor& sorted_sequence,
    const at::Scalar& self,
    bool out_int32,
    bool right,
    std::optional<c10::string_view> side,
    const std::optional<at::Tensor>& sorter,
    at::Tensor& out) {
  const auto expected_dtype =
      out_int32 ? at::ScalarType::Int : at::ScalarType::Long;
  const auto parsed_side = try_validate_searchsorted_common_whitelist(
      sorted_sequence, out_int32, right, side, sorter);
  if (parsed_side.has_value() && sorted_sequence.dim() == 1 &&
      is_whitelisted_numeric_scalar(self) && is_standard_mcpu_tensor(out) &&
      out.device() == sorted_sequence.device() &&
      out.scalar_type() == expected_dtype) {
    ops::check_out_sizes(
        "aten::searchsorted.Scalar_out", out, c10::IntArrayRef{});
    if (has_non_expanding_strides(out) && !out.is_alias_of(sorted_sequence) &&
        has_no_sorter_alias(out, sorter)) {
      launch_searchsorted_scalar_kernel(
          "mcpu::aten::searchsorted.Scalar_out",
          sorted_sequence,
          self,
          out_int32,
          right,
          *parsed_side,
          sorter,
          out);
      return out;
    }
  }
  return at::native::mcpu::custom::
      call_fallback_fn<&boxed_cpu_fallback, at::_ops::searchsorted_Scalar_out>::
          call(sorted_sequence, self, out_int32, right, side, sorter, out);
}

at::Tensor searchsorted_Scalar(
    const at::Tensor& sorted_sequence,
    const at::Scalar& self,
    bool out_int32,
    bool right,
    std::optional<c10::string_view> side,
    const std::optional<at::Tensor>& sorter) {
  const auto parsed_side = try_validate_searchsorted_common_whitelist(
      sorted_sequence, out_int32, right, side, sorter);
  if (parsed_side.has_value() && sorted_sequence.dim() == 1 &&
      is_whitelisted_numeric_scalar(self)) {
    const auto out_dtype =
        out_int32 ? at::ScalarType::Int : at::ScalarType::Long;
    auto out = at::empty({}, sorted_sequence.options().dtype(out_dtype));
    launch_searchsorted_scalar_kernel(
        "mcpu::aten::searchsorted.Scalar",
        sorted_sequence,
        self,
        out_int32,
        right,
        *parsed_side,
        sorter,
        out);
    return out;
  }
  return at::native::mcpu::custom::
      call_fallback_fn<&boxed_cpu_fallback, at::_ops::searchsorted_Scalar>::
          call(sorted_sequence, self, out_int32, right, side, sorter);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("searchsorted.Tensor", TORCH_FN(searchsorted_Tensor));
  m.impl("searchsorted.Tensor_out", TORCH_FN(searchsorted_Tensor_out));
  m.impl("searchsorted.Scalar", TORCH_FN(searchsorted_Scalar));
  m.impl("searchsorted.Scalar_out", TORCH_FN(searchsorted_Scalar_out));
}

} // namespace at::mcpu
