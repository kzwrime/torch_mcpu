#include "../native/MCPUFallback.h"
#include "Common.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/AccumulateType.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/arange_ops.h>
#include <c10/core/DefaultDtype.h>
#include <c10/util/overflows.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
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

bool is_whitelisted_numeric_scalar(const at::Scalar& scalar) {
  return !scalar.isSymbolic() &&
      (scalar.type() == at::ScalarType::Long ||
       scalar.type() == at::ScalarType::Double);
}

bool is_whitelisted_arange_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Byte:
    case at::ScalarType::Char:
    case at::ScalarType::Short:
    case at::ScalarType::Int:
    case at::ScalarType::Long:
    case at::ScalarType::Half:
    case at::ScalarType::BFloat16:
    case at::ScalarType::Float:
    case at::ScalarType::Double:
      return true;
    default:
      return false;
  }
}

std::optional<at::ScalarType> try_resolve_whitelisted_arange_dtype(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    std::optional<at::ScalarType> dtype) {
  if (!is_whitelisted_numeric_scalar(start) ||
      !is_whitelisted_numeric_scalar(end) ||
      !is_whitelisted_numeric_scalar(step)) {
    return std::nullopt;
  }
  const auto resolved = dtype.has_value()
      ? *dtype
      : ((start.type() == at::ScalarType::Long &&
          end.type() == at::ScalarType::Long &&
          step.type() == at::ScalarType::Long)
             ? at::ScalarType::Long
             : c10::get_default_dtype_as_scalartype());
  if (!is_whitelisted_arange_dtype(resolved)) {
    return std::nullopt;
  }
  return resolved;
}

double extract_whitelisted_scalar_double(const at::Scalar& scalar) {
  return scalar.type() == at::ScalarType::Long
      ? static_cast<double>(scalar.to<int64_t>())
      : scalar.to<double>();
}

struct ArangeWhitelistPlan {
  int64_t size = 0;
  bool use_contiguous_integral_raw = false;
  uint64_t start_bits = 0;
  uint64_t step_bits = 0;
};

template <typename scalar_t>
std::optional<ArangeWhitelistPlan> try_build_integral_arange_plan(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step) {
  // Reject floating-point scalars on integral dtypes so fractional range/step
  // corner cases follow CPU ATen semantics cleanly via fallback.
  if (start.type() != at::ScalarType::Long ||
      end.type() != at::ScalarType::Long ||
      step.type() != at::ScalarType::Long) {
    return std::nullopt;
  }
  const int64_t xstart = start.to<int64_t>();
  const int64_t xend = end.to<int64_t>();
  const int64_t xstep = step.to<int64_t>();
  if (xstep == 0 || (xstep > 0 && xstart > xend) ||
      (xstep < 0 && xstart < xend)) {
    return std::nullopt;
  }
  if (c10::overflows<scalar_t>(xstart, /*strict_unsigned=*/true) ||
      c10::overflows<scalar_t>(xend, /*strict_unsigned=*/true) ||
      c10::overflows<scalar_t>(xstep, /*strict_unsigned=*/true)) {
    return std::nullopt;
  }

  const int64_t sgn = (xstep > 0) - (xstep < 0);
  const __int128 numer = static_cast<__int128>(xend) -
      static_cast<__int128>(xstart) + static_cast<__int128>(xstep) -
      static_cast<__int128>(sgn);
  if (numer < static_cast<__int128>(std::numeric_limits<int64_t>::min()) ||
      numer > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
    return std::nullopt;
  }

  const __int128 delta =
      static_cast<__int128>(xend) - static_cast<__int128>(xstart);
  int64_t count = 0;
  if (delta != 0) {
    const __int128 abs_delta = delta >= 0 ? delta : -delta;
    const __int128 abs_step = xstep >= 0 ? static_cast<__int128>(xstep)
                                         : -static_cast<__int128>(xstep);
    const __int128 count128 = (abs_delta + abs_step - 1) / abs_step;
    if (count128 > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
      return std::nullopt;
    }
    count = static_cast<int64_t>(count128);
    if constexpr (!std::is_same_v<scalar_t, int64_t>) {
      const double double_count = std::ceil(
          (static_cast<double>(xend) - static_cast<double>(xstart)) /
          static_cast<double>(xstep));
      if (double_count != static_cast<double>(count)) {
        return std::nullopt;
      }
    }
    const __int128 last128 = static_cast<__int128>(xstart) +
        static_cast<__int128>(count - 1) * static_cast<__int128>(xstep);
    if (last128 < static_cast<__int128>(std::numeric_limits<scalar_t>::min()) ||
        last128 > static_cast<__int128>(std::numeric_limits<scalar_t>::max())) {
      return std::nullopt;
    }
  }

  ArangeWhitelistPlan plan;
  plan.size = count;
  plan.use_contiguous_integral_raw = true;
  plan.start_bits = static_cast<uint64_t>(xstart);
  plan.step_bits = static_cast<uint64_t>(xstep);
  return plan;
}

template <typename scalar_t>
std::optional<ArangeWhitelistPlan> try_build_floating_arange_plan(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step) {
  const double dstart = extract_whitelisted_scalar_double(start);
  const double dend = extract_whitelisted_scalar_double(end);
  const double dstep = extract_whitelisted_scalar_double(step);
  if (!std::isfinite(dstart) || !std::isfinite(dend) || !std::isfinite(dstep) ||
      dstep == 0.0 || (dstep > 0.0 && dstart > dend) ||
      (dstep < 0.0 && dstart < dend)) {
    return std::nullopt;
  }

  using accscalar_t = at::acc_type<scalar_t, false>;
  if (c10::overflows<accscalar_t>(dstart) ||
      c10::overflows<accscalar_t>(dend) || c10::overflows<accscalar_t>(dstep) ||
      static_cast<accscalar_t>(dstep) == accscalar_t(0)) {
    return std::nullopt;
  }

  const double dsize = std::ceil((dend - dstart) / dstep);
  if (!std::isfinite(dsize) || dsize < 0.0 || dsize >= std::ldexp(1.0, 62)) {
    return std::nullopt;
  }

  ArangeWhitelistPlan plan;
  plan.size = static_cast<int64_t>(dsize);
  plan.use_contiguous_integral_raw = false;
  return plan;
}

std::optional<ArangeWhitelistPlan> try_build_arange_whitelist_plan(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::ScalarType scalar_type) {
  if (!is_whitelisted_numeric_scalar(start) ||
      !is_whitelisted_numeric_scalar(end) ||
      !is_whitelisted_numeric_scalar(step) ||
      !is_whitelisted_arange_dtype(scalar_type)) {
    return std::nullopt;
  }

  switch (scalar_type) {
    case at::ScalarType::Byte:
      return try_build_integral_arange_plan<uint8_t>(start, end, step);
    case at::ScalarType::Char:
      return try_build_integral_arange_plan<int8_t>(start, end, step);
    case at::ScalarType::Short:
      return try_build_integral_arange_plan<int16_t>(start, end, step);
    case at::ScalarType::Int:
      return try_build_integral_arange_plan<int32_t>(start, end, step);
    case at::ScalarType::Long:
      return try_build_integral_arange_plan<int64_t>(start, end, step);
    case at::ScalarType::Half:
      return try_build_floating_arange_plan<c10::Half>(start, end, step);
    case at::ScalarType::BFloat16:
      return try_build_floating_arange_plan<c10::BFloat16>(start, end, step);
    case at::ScalarType::Float:
      return try_build_floating_arange_plan<float>(start, end, step);
    case at::ScalarType::Double:
      return try_build_floating_arange_plan<double>(start, end, step);
    default:
      return std::nullopt;
  }
}

void launch_raw_integral_arange_kernel(
    const char* record_name,
    uint64_t start_bits,
    uint64_t step_bits,
    int64_t size,
    at::Tensor& out) {
  void* out_data = out.data_ptr();
  const at::ScalarType dtype = out.scalar_type();
  at::mcpu::launch_timed_kernel_on_stream(
      at::mcpu::detail::get_kernel_launch_stream(out),
      record_name,
      [out_data, dtype, start_bits, step_bits, size, record_name](
          at::mcpu::kernel_timing::Event* timing_event) {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(record_name, timing_event);
        KernelPointerMemoryGuard guard({out_data});
        AT_DISPATCH_INTEGRAL_TYPES(dtype, "mcpu_arange_integral_raw", [&] {
          auto* out_ptr = static_cast<scalar_t*>(out_data);
          for (int64_t i = 0; i < size; ++i) {
            out_ptr[i] = static_cast<scalar_t>(
                start_bits + static_cast<uint64_t>(i) * step_bits);
          }
        });
      });
}

void launch_arange_kernel(
    const char* record_name,
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& out) {
  void* out_data = out.data_ptr();
  const int64_t size = out.size(0);
  const int64_t stride = out.stride(0);
  const at::ScalarType dtype = out.scalar_type();
  at::mcpu::launch_timed_kernel_on_stream(
      at::mcpu::detail::get_kernel_launch_stream(out),
      record_name,
      [out_data, size, stride, dtype, start, end, step, record_name](
          at::mcpu::kernel_timing::Event* timing_event) {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(record_name, timing_event);
        c10::InferenceMode inference_guard(false);
        KernelPointerMemoryGuard guard({out_data});
        auto cpu_out = at::from_blob(
            out_data,
            {size},
            {stride},
            at::TensorOptions().dtype(dtype).device(c10::DeviceType::CPU));
        at::arange_out(cpu_out, start, end, step);
      });
}

void dispatch_whitelisted_arange_kernel(
    const char* record_name,
    const ArangeWhitelistPlan& plan,
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& out) {
  if (plan.size == 0) {
    return;
  }
  if (plan.use_contiguous_integral_raw && out.is_contiguous()) {
    launch_raw_integral_arange_kernel(
        record_name, plan.start_bits, plan.step_bits, plan.size, out);
    return;
  }
  launch_arange_kernel(record_name, start, end, step, out);
}

template <typename FallbackOp, typename... FallbackArgs>
at::Tensor arange_impl(
    const char* record_name,
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    FallbackArgs&&... fallback_args) {
  ops::check_factory_options(layout, device, pin_memory);
  const auto resolved_dtype =
      try_resolve_whitelisted_arange_dtype(start, end, step, dtype);
  if (resolved_dtype.has_value()) {
    const auto plan =
        try_build_arange_whitelist_plan(start, end, step, *resolved_dtype);
    if (plan.has_value()) {
      auto options = ops::build_mcpu_options(*resolved_dtype, layout, device);
      auto out = at::empty({plan->size}, options);
      dispatch_whitelisted_arange_kernel(
          record_name, *plan, start, end, step, out);
      return out;
    }
  }
  return at::native::mcpu::custom::
      call_fallback_fn<&boxed_cpu_fallback, FallbackOp>::call(
          std::forward<FallbackArgs>(fallback_args)...);
}

template <typename FallbackOp, typename... FallbackArgs>
at::Tensor& arange_out_impl(
    const char* op_name,
    const char* record_name,
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& out,
    FallbackArgs&&... fallback_args) {
  if (is_standard_mcpu_tensor(out)) {
    const auto plan =
        try_build_arange_whitelist_plan(start, end, step, out.scalar_type());
    if (plan.has_value()) {
      const std::array<int64_t, 1> expected_sizes{plan->size};
      ops::check_out_sizes(op_name, out, expected_sizes);
      TORCH_CHECK(
          plan->size <= 1 || out.stride(0) != 0,
          op_name,
          ": unsupported output tensor with internal memory overlap");
      dispatch_whitelisted_arange_kernel(
          record_name, *plan, start, end, step, out);
      return out;
    }
  }

  return at::native::mcpu::custom::
      call_fallback_fn<&boxed_cpu_fallback, FallbackOp>::call(
          std::forward<FallbackArgs>(fallback_args)...);
}

at::Tensor arange(
    const at::Scalar& end,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  const at::Scalar start(0);
  const at::Scalar step(1);
  return arange_impl<at::_ops::arange>(
      "mcpu::aten::arange",
      start,
      end,
      step,
      dtype,
      layout,
      device,
      pin_memory,
      end,
      dtype,
      layout,
      device,
      pin_memory);
}

at::Tensor arange_start(
    const at::Scalar& start,
    const at::Scalar& end,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  const at::Scalar step(1);
  return arange_impl<at::_ops::arange_start>(
      "mcpu::aten::arange.start",
      start,
      end,
      step,
      dtype,
      layout,
      device,
      pin_memory,
      start,
      end,
      dtype,
      layout,
      device,
      pin_memory);
}

at::Tensor arange_start_step(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  return arange_impl<at::_ops::arange_start_step>(
      "mcpu::aten::arange.start_step",
      start,
      end,
      step,
      dtype,
      layout,
      device,
      pin_memory,
      start,
      end,
      step,
      dtype,
      layout,
      device,
      pin_memory);
}

at::Tensor& arange_out(const at::Scalar& end, at::Tensor& out) {
  const at::Scalar start(0);
  const at::Scalar step(1);
  return arange_out_impl<at::_ops::arange_out>(
      "aten::arange.out",
      "mcpu::aten::arange.out",
      start,
      end,
      step,
      out,
      end,
      out);
}

at::Tensor& arange_start_out(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& out) {
  return arange_out_impl<at::_ops::arange_start_out>(
      "aten::arange.start_out",
      "mcpu::aten::arange.start_out",
      start,
      end,
      step,
      out,
      start,
      end,
      step,
      out);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("arange", TORCH_FN(arange));
  m.impl("arange.start", TORCH_FN(arange_start));
  m.impl("arange.start_step", TORCH_FN(arange_start_step));
  m.impl("arange.out", TORCH_FN(arange_out));
  m.impl("arange.start_out", TORCH_FN(arange_start_out));
}

} // namespace at::mcpu
