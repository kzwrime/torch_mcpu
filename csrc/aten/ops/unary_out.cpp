#include "../native/MCPUFallback.h"
#include "Common.h"
#include "RawPlan.h"

#if TORCH_MCPU_ENABLE_CPU_FALLBACK
namespace at::native::mcpu {
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);
} // namespace at::native::mcpu
#endif

#include <ATen/ExpandUtils.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/bitwise_not.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/clamp_max.h>
#include <ATen/ops/clamp_max_ops.h>
#include <ATen/ops/clamp_min.h>
#include <ATen/ops/clamp_min_ops.h>
#include <ATen/ops/clamp_ops.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/count_nonzero.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/masked_fill.h>
#include <ATen/ops/masked_fill_cpu_dispatch.h>
#include <ATen/ops/masked_select.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/reciprocal.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/zero.h>
#include <c10/core/DefaultDtype.h>
#include <c10/util/overflows.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <vector>

namespace at::mcpu {
namespace {

enum class RawUnaryOp {
  Sigmoid,
  Silu,
  Cos,
  Sin,
  Reciprocal,
  Neg,
};

struct UnaryKernelArgs {
  ops::TensorViewSpec self;
  ops::TensorViewSpec out;
};

struct MaskedSelectCountArgs {
  ops::TensorViewSpec mask;
  c10::SmallVector<int64_t, 4> broadcast_sizes;
  int64_t* count;
};

struct MaskedSelectWriteArgs {
  ops::TensorViewSpec self;
  ops::TensorViewSpec mask;
  ops::TensorViewSpec out;
};

template <typename scalar_t>
inline bool is_nonzero_value(const scalar_t& value) {
  if constexpr (c10::is_complex<scalar_t>::value) {
    return value.real() != 0 || value.imag() != 0;
  } else {
    return value != scalar_t(0);
  }
}

template <RawUnaryOp op, typename scalar_t>
inline scalar_t raw_unary_value(scalar_t value) {
  const float x = static_cast<float>(value);
  if constexpr (op == RawUnaryOp::Sigmoid) {
    return static_cast<scalar_t>(1.0f / (1.0f + std::exp(-x)));
  } else if constexpr (op == RawUnaryOp::Silu) {
    return static_cast<scalar_t>(x / (1.0f + std::exp(-x)));
  } else if constexpr (op == RawUnaryOp::Cos) {
    return static_cast<scalar_t>(std::cos(x));
  } else if constexpr (op == RawUnaryOp::Sin) {
    return static_cast<scalar_t>(std::sin(x));
  } else if constexpr (op == RawUnaryOp::Reciprocal) {
    return static_cast<scalar_t>(1.0f / x);
  } else {
    return static_cast<scalar_t>(-value);
  }
}

template <RawUnaryOp op, typename scalar_t>
void raw_unary_kernel(
    const scalar_t* self,
    scalar_t* out,
    const ops::RawTensorPairPlan& plan) {
  ops::for_each_raw_tensor_pair_row(
      plan,
      [self, out](int64_t self_offset, int64_t out_offset, int64_t inner_size) {
        const auto* self_row = self + self_offset;
        auto* out_row = out + out_offset;
        for (int64_t i = 0; i < inner_size; ++i) {
          out_row[i] = raw_unary_value<op>(self_row[i]);
        }
      });
}

template <typename scalar_t>
void raw_bitwise_not_kernel(
    const scalar_t* self,
    scalar_t* out,
    const ops::RawTensorPairPlan& plan) {
  ops::for_each_raw_tensor_pair_row(
      plan,
      [self, out](int64_t self_offset, int64_t out_offset, int64_t inner_size) {
        const auto* self_row = self + self_offset;
        auto* out_row = out + out_offset;
        for (int64_t i = 0; i < inner_size; ++i) {
          out_row[i] = ~self_row[i];
        }
      });
}

template <>
void raw_bitwise_not_kernel<bool>(
    const bool* self,
    bool* out,
    const ops::RawTensorPairPlan& plan) {
  ops::for_each_raw_tensor_pair_row(
      plan,
      [self, out](int64_t self_offset, int64_t out_offset, int64_t inner_size) {
        const auto* self_row = self + self_offset;
        auto* out_row = out + out_offset;
        for (int64_t i = 0; i < inner_size; ++i) {
          out_row[i] = !self_row[i];
        }
      });
}

template <typename scalar_t>
void raw_ne_scalar_kernel(
    const scalar_t* self,
    scalar_t other,
    bool* out,
    const ops::RawTensorPairPlan& plan) {
  ops::for_each_raw_tensor_pair_row(
      plan,
      [self, other, out](
          int64_t self_offset, int64_t out_offset, int64_t inner_size) {
        const auto* self_row = self + self_offset;
        auto* out_row = out + out_offset;
        for (int64_t i = 0; i < inner_size; ++i) {
          out_row[i] = self_row[i] != other;
        }
      });
}

template <typename scalar_t>
void raw_zero_kernel(scalar_t* self, const ops::RawTensorPlan& plan) {
  ops::for_each_raw_tensor_row(
      plan, [self](int64_t offset, int64_t inner_size) {
        auto* row = self + offset;
        for (int64_t i = 0; i < inner_size; ++i) {
          row[i] = scalar_t(0);
        }
      });
}

template <typename scalar_t>
void raw_clamp_contiguous_kernel(
    const scalar_t* self,
    scalar_t* out,
    int64_t numel,
    bool has_min,
    scalar_t min_value,
    bool has_max,
    scalar_t max_value) {
  for (int64_t i = 0; i < numel; ++i) {
    auto value = self[i];
    if (has_min && value < min_value) {
      value = min_value;
    }
    if (has_max && value > max_value) {
      value = max_value;
    }
    out[i] = value;
  }
}

template <typename scalar_t>
void raw_clamp_kernel(
    const scalar_t* self,
    scalar_t* out,
    bool has_min,
    scalar_t min_value,
    bool has_max,
    scalar_t max_value,
    const ops::RawTensorPairPlan& plan) {
  ops::for_each_raw_tensor_pair_row(
      plan,
      [self, out, has_min, min_value, has_max, max_value](
          int64_t self_offset, int64_t out_offset, int64_t inner_size) {
        const auto* self_row = self + self_offset;
        auto* out_row = out + out_offset;
        for (int64_t i = 0; i < inner_size; ++i) {
          auto value = self_row[i];
          if (has_min && value < min_value) {
            value = min_value;
          }
          if (has_max && value > max_value) {
            value = max_value;
          }
          out_row[i] = value;
        }
      });
}

template <typename scalar_t>
void raw_masked_fill_scalar_kernel(
    scalar_t* self,
    const bool* mask,
    scalar_t value,
    const ops::RawTensorPairPlan& plan) {
  ops::for_each_raw_tensor_pair_row(
      plan,
      [self, mask, value](
          int64_t mask_offset, int64_t self_offset, int64_t inner_size) {
        const auto* mask_row = mask + mask_offset;
        auto* self_row = self + self_offset;
        for (int64_t i = 0; i < inner_size; ++i) {
          if (mask_row[i]) {
            self_row[i] = value;
          }
        }
      });
}

at::Tensor empty_unary_mcpu(
    const at::Tensor& self,
    at::ScalarType result_dtype) {
  return at::empty_like(
      self, self.options().dtype(result_dtype), at::MemoryFormat::Preserve);
}

template <RawUnaryOp op>
bool raw_unary_out(
    const char* record_name,
    const at::Tensor& self,
    at::Tensor& out) {
  if (self.scalar_type() != out.scalar_type()) {
    return false;
  }
  const auto dtype = out.scalar_type();
  if (dtype != at::ScalarType::Float && dtype != at::ScalarType::Half &&
      dtype != at::ScalarType::BFloat16) {
    return false;
  }

  auto plan = ops::make_same_shape_raw_tensor_pair_plan(self, out);
  if (!plan.has_value()) {
    return false;
  }

  if (self.numel() == 0) {
    return true;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      out.scalar_type(),
      "mcpu_raw_unary_out",
      [&] {
        auto plan_ptr =
            std::make_shared<ops::RawTensorPairPlan>(*std::move(plan));
        const auto* self_ptr = self.const_data_ptr<scalar_t>();
        auto* out_ptr = out.mutable_data_ptr<scalar_t>();
        MCPU_LAUNCH_TIMED_KERNEL(
            record_name, ([ self_ptr, out_ptr, plan_ptr, record_name ]), {
              KernelPointerMemoryGuard guard({self_ptr, out_ptr});
              raw_unary_kernel<op>(self_ptr, out_ptr, *plan_ptr);
            });
      });
  return true;
}

bool raw_zero_(at::Tensor& self) {
  if (!ops::is_raw_dtype_supported(self.scalar_type())) {
    return false;
  }

  auto plan = ops::make_raw_tensor_plan(self);
  if (!plan.has_value()) {
    return false;
  }

  if (self.numel() == 0) {
    return true;
  }

  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "mcpu_raw_zero_",
      [&] {
        auto* self_ptr = self.mutable_data_ptr<scalar_t>();
        auto plan_ptr = std::make_shared<ops::RawTensorPlan>(*plan);
        MCPU_LAUNCH_TIMED_KERNEL(
            "mcpu::aten::zero_.raw", ([ self_ptr, plan_ptr ]), {
              KernelPointerMemoryGuard guard({self_ptr});
              raw_zero_kernel(self_ptr, *plan_ptr);
            });
      });
  return true;
}

void boxed_cpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
#if TORCH_MCPU_ENABLE_CPU_FALLBACK
  at::native::mcpu::cpu_fallback(op, stack);
#else
  TORCH_CHECK(false, "CPU fallback is disabled for ", op.schema().name());
#endif
}

bool is_standard_clamp_mcpu_tensor(const at::Tensor& tensor) {
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

bool is_whitelisted_clamp_nonbool_dtype(at::ScalarType dtype) {
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

bool is_whitelisted_clamp_numeric_scalar(const at::Scalar& scalar) {
  return !scalar.isSymbolic() &&
      (scalar.type() == at::ScalarType::Long ||
       scalar.type() == at::ScalarType::Double);
}

template <typename scalar_t>
bool can_convert_clamp_scalar_typed(
    const at::Scalar& scalar,
    bool require_finite) {
  if (!is_whitelisted_clamp_numeric_scalar(scalar)) {
    return false;
  }
  if (scalar.type() == at::ScalarType::Long) {
    return !c10::overflows<scalar_t>(
        scalar.to<int64_t>(), /*strict_unsigned=*/true);
  }
  const double value = scalar.to<double>();
  if (require_finite && !std::isfinite(value)) {
    return false;
  }
  return !c10::overflows<scalar_t>(value, /*strict_unsigned=*/true);
}

bool can_convert_clamp_scalar(
    at::ScalarType dtype,
    const at::Scalar& scalar,
    bool require_finite) {
  switch (dtype) {
    case at::ScalarType::Byte:
      return can_convert_clamp_scalar_typed<uint8_t>(scalar, require_finite);
    case at::ScalarType::Char:
      return can_convert_clamp_scalar_typed<int8_t>(scalar, require_finite);
    case at::ScalarType::Short:
      return can_convert_clamp_scalar_typed<int16_t>(scalar, require_finite);
    case at::ScalarType::Int:
      return can_convert_clamp_scalar_typed<int32_t>(scalar, require_finite);
    case at::ScalarType::Long:
      return can_convert_clamp_scalar_typed<int64_t>(scalar, require_finite);
    case at::ScalarType::Half:
      return can_convert_clamp_scalar_typed<c10::Half>(scalar, require_finite);
    case at::ScalarType::BFloat16:
      return can_convert_clamp_scalar_typed<c10::BFloat16>(
          scalar, require_finite);
    case at::ScalarType::Float:
      return can_convert_clamp_scalar_typed<float>(scalar, require_finite);
    case at::ScalarType::Double:
      return can_convert_clamp_scalar_typed<double>(scalar, require_finite);
    default:
      return false;
  }
}

at::ScalarType resolve_whitelisted_clamp_scalar_dtype(
    at::ScalarType self_dtype,
    const std::optional<at::Scalar>& min,
    const std::optional<at::Scalar>& max) {
  if (c10::isFloatingType(self_dtype)) {
    return self_dtype;
  }
  if ((min.has_value() && min->type() == at::ScalarType::Double) ||
      (max.has_value() && max->type() == at::ScalarType::Double)) {
    return c10::get_default_dtype_as_scalartype();
  }
  return self_dtype;
}

bool raw_clamp_out(
    const at::Tensor& self,
    const std::optional<at::Scalar>& min,
    const std::optional<at::Scalar>& max,
    at::Tensor& out,
    const char* record_name = "mcpu::aten::clamp.out.raw") {
  if ((!min.has_value() && !max.has_value()) ||
      !is_standard_clamp_mcpu_tensor(self) ||
      !is_standard_clamp_mcpu_tensor(out) || self.device() != out.device() ||
      !is_whitelisted_clamp_nonbool_dtype(self.scalar_type()) ||
      self.scalar_type() != out.scalar_type() ||
      (min.has_value() && !is_whitelisted_clamp_numeric_scalar(*min)) ||
      (max.has_value() && !is_whitelisted_clamp_numeric_scalar(*max)) ||
      resolve_whitelisted_clamp_scalar_dtype(self.scalar_type(), min, max) !=
          out.scalar_type() ||
      (min.has_value() &&
       !can_convert_clamp_scalar(
           out.scalar_type(), *min, /*require_finite=*/true)) ||
      (max.has_value() &&
       !can_convert_clamp_scalar(
           out.scalar_type(), *max, /*require_finite=*/true))) {
    return false;
  }

  if (self.sizes().equals(out.sizes()) && self.is_contiguous() &&
      out.is_contiguous() && ops::has_safe_raw_output_overlap(out, self)) {
    const int64_t numel = self.numel();
    if (numel == 0) {
      return true;
    }
    const auto launch_stream = at::mcpu::detail::get_kernel_launch_stream(out);
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        out.scalar_type(),
        "mcpu_raw_clamp_contiguous_out",
        [&] {
          const auto* self_ptr = self.const_data_ptr<scalar_t>();
          auto* out_ptr = out.mutable_data_ptr<scalar_t>();
          const bool has_min = min.has_value();
          const bool has_max = max.has_value();
          const auto min_value = has_min ? min->to<scalar_t>() : scalar_t(0);
          const auto max_value = has_max ? max->to<scalar_t>() : scalar_t(0);
          at::mcpu::launch_timed_kernel_on_stream(
              launch_stream,
              record_name,
              [self_ptr,
               out_ptr,
               numel,
               has_min,
               min_value,
               has_max,
               max_value,
               record_name](at::mcpu::kernel_timing::Event* timing_event) {
                MCPU_KERNEL_TIMING_SCOPE_EVENT(record_name, timing_event);
                KernelPointerMemoryGuard guard({self_ptr, out_ptr});
                raw_clamp_contiguous_kernel(
                    self_ptr,
                    out_ptr,
                    numel,
                    has_min,
                    min_value,
                    has_max,
                    max_value);
              });
        });
    return true;
  }

  auto plan = ops::make_same_shape_raw_tensor_pair_plan(self, out);
  if (!plan.has_value()) {
    return false;
  }

  if (self.numel() == 0) {
    return true;
  }

  const auto launch_stream = at::mcpu::detail::get_kernel_launch_stream(out);
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      out.scalar_type(),
      "mcpu_raw_clamp_out",
      [&] {
        const auto* self_ptr = self.const_data_ptr<scalar_t>();
        auto* out_ptr = out.mutable_data_ptr<scalar_t>();
        const bool has_min = min.has_value();
        const bool has_max = max.has_value();
        const auto min_value = has_min ? min->to<scalar_t>() : scalar_t(0);
        const auto max_value = has_max ? max->to<scalar_t>() : scalar_t(0);
        at::mcpu::launch_timed_kernel_on_stream(
            launch_stream,
            record_name,
            [self_ptr,
             out_ptr,
             has_min,
             min_value,
             has_max,
             max_value,
             plan = std::move(*plan),
             record_name](at::mcpu::kernel_timing::Event* timing_event) {
              MCPU_KERNEL_TIMING_SCOPE_EVENT(record_name, timing_event);
              KernelPointerMemoryGuard guard({self_ptr, out_ptr});
              raw_clamp_kernel(
                  self_ptr,
                  out_ptr,
                  has_min,
                  min_value,
                  has_max,
                  max_value,
                  plan);
            });
      });
  return true;
}

bool raw_ne_scalar_out(
    const at::Tensor& self,
    const at::Scalar& other,
    at::Tensor& out) {
  if (out.scalar_type() != at::ScalarType::Bool ||
      !ops::is_raw_dtype_supported(self.scalar_type())) {
    return false;
  }

  auto plan = ops::make_same_shape_raw_tensor_pair_plan(self, out);
  if (!plan.has_value()) {
    return false;
  }

  if (self.numel() == 0) {
    return true;
  }

  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "mcpu_raw_ne_scalar_out",
      [&] {
        const auto* self_ptr = self.const_data_ptr<scalar_t>();
        auto* out_ptr = out.mutable_data_ptr<bool>();
        const auto other_value = other.to<scalar_t>();
        auto plan_ptr = std::make_shared<ops::RawTensorPairPlan>(*plan);
        MCPU_LAUNCH_TIMED_KERNEL(
            "mcpu::aten::ne.Scalar.raw",
            ([ self_ptr, other_value, out_ptr, plan_ptr ]),
            {
              KernelPointerMemoryGuard guard({self_ptr, out_ptr});
              raw_ne_scalar_kernel(self_ptr, other_value, out_ptr, *plan_ptr);
            });
      });
  return true;
}

bool raw_masked_fill__scalar(
    at::Tensor& self,
    const at::Tensor& mask,
    const at::Scalar& value) {
  if (mask.scalar_type() != at::ScalarType::Bool ||
      !ops::is_raw_dtype_supported(self.scalar_type()) ||
      self.is_alias_of(mask)) {
    return false;
  }

  auto plan = ops::make_same_shape_raw_tensor_pair_plan(mask, self);
  if (!plan.has_value()) {
    return false;
  }

  if (self.numel() == 0) {
    return true;
  }

  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "mcpu_raw_masked_fill__scalar",
      [&] {
        auto* self_ptr = self.mutable_data_ptr<scalar_t>();
        const auto* mask_ptr = mask.const_data_ptr<bool>();
        const auto fill_value = value.to<scalar_t>();
        auto plan_ptr = std::make_shared<ops::RawTensorPairPlan>(*plan);
        MCPU_LAUNCH_TIMED_KERNEL(
            "mcpu::aten::masked_fill_.Scalar.raw",
            ([ self_ptr, mask_ptr, fill_value, plan_ptr ]),
            {
              KernelPointerMemoryGuard guard({self_ptr, mask_ptr});
              raw_masked_fill_scalar_kernel(
                  self_ptr, mask_ptr, fill_value, *plan_ptr);
            });
      });
  return true;
}

at::Tensor& sigmoid_out(const at::Tensor& self, at::Tensor& out) {
  ops::check_out_sizes("aten::sigmoid.out", out, self.sizes());

  if (raw_unary_out<RawUnaryOp::Sigmoid>(
          "mcpu::aten::sigmoid.raw", self, out)) {
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::sigmoid", ([ self, out ]), {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::sigmoid_out(cpu_out, cpu_self);
  });
  return out;
}

at::Tensor& abs_out(const at::Tensor& self, at::Tensor& out) {
  ops::check_out_sizes("aten::abs.out", out, self.sizes());
  TORCH_CHECK(
      self.scalar_type() != at::ScalarType::Bool,
      "aten::abs is not implemented for bool tensors");

  auto args = std::make_unique<UnaryKernelArgs>(UnaryKernelArgs{
      ops::make_cpu_view_spec(self), ops::make_cpu_view_spec(out)});
  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::abs.out", ([args = std::move(args)]), {
    KernelPointerMemoryGuard guard({args->self.data, args->out.data});
    auto cpu_self = ops::cpu_view_from_spec(args->self);
    auto cpu_out = ops::cpu_view_from_spec(args->out);
    at::abs_out(cpu_out, cpu_self);
  });
  return out;
}

at::Tensor abs(const at::Tensor& self) {
  auto out = at::empty_like(
      self,
      self.options().dtype(c10::toRealValueType(self.scalar_type())),
      at::MemoryFormat::Preserve);
  abs_out(self, out);
  return out;
}

at::Tensor sigmoid(const at::Tensor& self) {
  auto result_dtype =
      c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)
      ? at::kFloat
      : self.scalar_type();
  auto out = empty_unary_mcpu(self, result_dtype);
  sigmoid_out(self, out);
  return out;
}

at::Tensor& sigmoid_(at::Tensor& self) {
  if (raw_unary_out<RawUnaryOp::Sigmoid>(
          "mcpu::aten::sigmoid_.raw", self, self)) {
    return self;
  }

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::sigmoid_", ([self]), {
    KernelMemoryGuard guard(self);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    at::sigmoid_(cpu_self);
  });
  return self;
}

at::Tensor& silu_out(const at::Tensor& self, at::Tensor& out) {
  ops::check_out_sizes("aten::silu.out", out, self.sizes());

  if (raw_unary_out<RawUnaryOp::Silu>("mcpu::aten::silu.raw", self, out)) {
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::silu.out", ([ self, out ]), {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::silu_out(cpu_out, cpu_self);
  });
  return out;
}

at::Tensor silu(const at::Tensor& self) {
  auto out = empty_unary_mcpu(self, self.scalar_type());
  silu_out(self, out);
  return out;
}

at::Tensor& silu_(at::Tensor& self) {
  if (raw_unary_out<RawUnaryOp::Silu>("mcpu::aten::silu_.raw", self, self)) {
    return self;
  }

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::silu_", ([self]), {
    KernelMemoryGuard guard(self);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    at::silu_(cpu_self);
  });
  return self;
}

at::Tensor& cos_out(const at::Tensor& self, at::Tensor& out) {
  ops::check_out_sizes("aten::cos.out", out, self.sizes());

  if (raw_unary_out<RawUnaryOp::Cos>("mcpu::aten::cos.raw", self, out)) {
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::cos.out", ([ self, out ]), {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::cos_out(cpu_out, cpu_self);
  });
  return out;
}

at::Tensor& sin_out(const at::Tensor& self, at::Tensor& out) {
  ops::check_out_sizes("aten::sin.out", out, self.sizes());

  if (raw_unary_out<RawUnaryOp::Sin>("mcpu::aten::sin.raw", self, out)) {
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::sin.out", ([ self, out ]), {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::sin_out(cpu_out, cpu_self);
  });
  return out;
}

at::Tensor& reciprocal_out(const at::Tensor& self, at::Tensor& out) {
  ops::check_out_sizes("aten::reciprocal.out", out, self.sizes());

  if (raw_unary_out<RawUnaryOp::Reciprocal>(
          "mcpu::aten::reciprocal.raw", self, out)) {
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::reciprocal.out", ([ self, out ]), {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::reciprocal_out(cpu_out, cpu_self);
  });
  return out;
}

at::Tensor& rsqrt_out(const at::Tensor& self, at::Tensor& out) {
  ops::check_out_sizes("aten::rsqrt.out", out, self.sizes());

  auto args = std::make_unique<UnaryKernelArgs>(UnaryKernelArgs{
      ops::make_cpu_view_spec(self), ops::make_cpu_view_spec(out)});
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::rsqrt.out", ([args = std::move(args)]), {
        KernelPointerMemoryGuard guard({args->self.data, args->out.data});
        auto cpu_self = ops::cpu_view_from_spec(args->self);
        auto cpu_out = ops::cpu_view_from_spec(args->out);
        at::rsqrt_out(cpu_out, cpu_self);
      });
  return out;
}

at::Tensor rsqrt(const at::Tensor& self) {
  const auto result_dtype =
      c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)
      ? c10::get_default_dtype_as_scalartype()
      : self.scalar_type();
  auto out = at::empty_like(
      self, self.options().dtype(result_dtype), at::MemoryFormat::Preserve);
  rsqrt_out(self, out);
  return out;
}

at::Tensor& neg_out(const at::Tensor& self, at::Tensor& out) {
  ops::check_out_sizes("aten::neg.out", out, self.sizes());

  if (raw_unary_out<RawUnaryOp::Neg>("mcpu::aten::neg.raw", self, out)) {
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::neg.out", ([ self, out ]), {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::neg_out(cpu_out, cpu_self);
  });
  return out;
}

at::Tensor& bitwise_not_out(const at::Tensor& self, at::Tensor& out) {
  ops::check_out_sizes("aten::bitwise_not.out", out, self.sizes());

  auto plan = ops::make_same_shape_raw_tensor_pair_plan(self, out);
  if (plan.has_value() && self.scalar_type() == out.scalar_type() &&
      c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    if (self.numel() == 0) {
      return out;
    }

    AT_DISPATCH_INTEGRAL_TYPES_AND(
        at::ScalarType::Bool,
        self.scalar_type(),
        "mcpu_raw_bitwise_not_out",
        [&] {
          auto plan_ptr =
              std::make_shared<ops::RawTensorPairPlan>(*std::move(plan));
          const auto* self_ptr = self.const_data_ptr<scalar_t>();
          auto* out_ptr = out.mutable_data_ptr<scalar_t>();
          MCPU_LAUNCH_TIMED_KERNEL(
              "mcpu::aten::bitwise_not.out.raw",
              ([ self_ptr, out_ptr, plan_ptr ]),
              {
                KernelPointerMemoryGuard guard({self_ptr, out_ptr});
                raw_bitwise_not_kernel(self_ptr, out_ptr, *plan_ptr);
              });
        });
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::bitwise_not.out", ([ self, out ]), {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::bitwise_not_out(cpu_out, cpu_self);
  });
  return out;
}

at::Tensor bitwise_not(const at::Tensor& self) {
  auto out = empty_unary_mcpu(self, self.scalar_type());
  bitwise_not_out(self, out);
  return out;
}

at::Tensor& bitwise_not_(at::Tensor& self) {
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    bitwise_not_out(static_cast<const at::Tensor&>(self), self);
    return self;
  }

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::bitwise_not_", ([self]), {
    KernelMemoryGuard guard(self);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    cpu_self.bitwise_not_();
  });
  return self;
}

enum class ClampKind {
  Clamp,
  ClampMin,
  ClampMax,
};

bool is_cpu_scalar_tensor(const at::Tensor& tensor) {
  return tensor.defined() && tensor.is_cpu() && tensor.dim() == 0;
}

struct ClampTensorKernelArgs {
  ops::TensorViewSpec self;
  std::optional<ops::TensorViewSpec> min;
  std::optional<ops::TensorViewSpec> max;
  ops::TensorViewSpec out;
};

template <ClampKind Kind, bool IsInplace = false>
at::Tensor& clamp_scalar_out_impl(
    const char* op_name,
    const char* record_name,
    const char* raw_record_name,
    const at::Tensor& self,
    const std::optional<at::Scalar>& min,
    const std::optional<at::Scalar>& max,
    at::Tensor& out) {
  TORCH_CHECK(
      min.has_value() || max.has_value(),
      "At least one of 'min' or 'max' must not be None");

  if (is_standard_clamp_mcpu_tensor(self) &&
      is_standard_clamp_mcpu_tensor(out) && self.device() == out.device() &&
      is_whitelisted_clamp_nonbool_dtype(self.scalar_type()) &&
      is_whitelisted_clamp_nonbool_dtype(out.scalar_type()) &&
      (!min.has_value() || is_whitelisted_clamp_numeric_scalar(*min)) &&
      (!max.has_value() || is_whitelisted_clamp_numeric_scalar(*max))) {
    const auto expected_dtype =
        resolve_whitelisted_clamp_scalar_dtype(self.scalar_type(), min, max);
    if (is_whitelisted_clamp_nonbool_dtype(expected_dtype)) {
      TORCH_CHECK(
          expected_dtype == out.scalar_type() ||
              at::canCast(expected_dtype, out.scalar_type()),
          op_name,
          ": result type ",
          expected_dtype,
          " can't be cast to the desired output type ",
          out.scalar_type());
      ops::check_out_sizes(op_name, out, self.sizes());

      if (out.scalar_type() == expected_dtype &&
          raw_clamp_out(self, min, max, out, raw_record_name)) {
        return out;
      }

      if ((!min.has_value() ||
           can_convert_clamp_scalar(
               expected_dtype, *min, /*require_finite=*/false)) &&
          (!max.has_value() ||
           can_convert_clamp_scalar(
               expected_dtype, *max, /*require_finite=*/false)) &&
          has_non_expanding_strides(out) &&
          ops::has_safe_raw_output_overlap(out, self)) {
        if (self.numel() == 0) {
          return out;
        }
        UnaryKernelArgs args{
            ops::make_cpu_view_spec(self), ops::make_cpu_view_spec(out)};
        at::mcpu::launch_timed_kernel_on_stream(
            at::mcpu::detail::get_kernel_launch_stream(out),
            record_name,
            [args = std::move(args), min, max, record_name](
                at::mcpu::kernel_timing::Event* timing_event) mutable {
              MCPU_KERNEL_TIMING_SCOPE_EVENT(record_name, timing_event);
              KernelPointerMemoryGuard guard({args.self.data, args.out.data});
              auto cpu_self = ops::cpu_view_from_spec(args.self);
              auto cpu_out = ops::cpu_view_from_spec(args.out);
              if constexpr (Kind == ClampKind::Clamp) {
                at::clamp_out(cpu_out, cpu_self, min, max);
              } else if constexpr (Kind == ClampKind::ClampMin) {
                at::clamp_min_out(cpu_out, cpu_self, *min);
              } else {
                at::clamp_max_out(cpu_out, cpu_self, *max);
              }
            });
        return out;
      }
    }
  }

  if constexpr (Kind == ClampKind::Clamp) {
    return at::native::mcpu::custom::
        call_fallback_fn<&boxed_cpu_fallback, at::_ops::clamp_out>::call(
            self, min, max, out);
  } else if constexpr (Kind == ClampKind::ClampMin && IsInplace) {
    return at::native::mcpu::custom::
        call_fallback_fn<&boxed_cpu_fallback, at::_ops::clamp_min_>::call(
            out, *min);
  } else if constexpr (Kind == ClampKind::ClampMin) {
    return at::native::mcpu::custom::
        call_fallback_fn<&boxed_cpu_fallback, at::_ops::clamp_min_out>::call(
            self, *min, out);
  } else {
    return at::native::mcpu::custom::
        call_fallback_fn<&boxed_cpu_fallback, at::_ops::clamp_max_out>::call(
            self, *max, out);
  }
}

template <ClampKind Kind, bool IsInplace = false>
at::Tensor& clamp_tensor_out_impl(
    const char* op_name,
    const char* record_name,
    const at::Tensor& self,
    const std::optional<at::Tensor>& min,
    const std::optional<at::Tensor>& max,
    at::Tensor& out) {
  const bool has_min = min.has_value() && min->defined();
  const bool has_max = max.has_value() && max->defined();
  TORCH_CHECK(
      has_min || has_max, "At least one of 'min' or 'max' must not be None");

  if (is_standard_clamp_mcpu_tensor(self) &&
      is_standard_clamp_mcpu_tensor(out) && self.device() == out.device() &&
      (!has_min ||
       (is_standard_clamp_mcpu_tensor(*min) &&
        min->device() == out.device())) &&
      (!has_max ||
       (is_standard_clamp_mcpu_tensor(*max) &&
        max->device() == out.device())) &&
      is_whitelisted_clamp_nonbool_dtype(self.scalar_type()) &&
      is_whitelisted_clamp_nonbool_dtype(out.scalar_type()) &&
      (!has_min || is_whitelisted_clamp_nonbool_dtype(min->scalar_type())) &&
      (!has_max || is_whitelisted_clamp_nonbool_dtype(max->scalar_type()))) {
    at::ScalarType common_dtype = self.scalar_type();
    if ((has_min && min->scalar_type() != common_dtype) ||
        (has_max && max->scalar_type() != common_dtype)) {
      at::native::ResultTypeState state = {};
      state = at::native::update_result_type_state(self, state);
      if (has_min) {
        state = at::native::update_result_type_state(*min, state);
      }
      if (has_max) {
        state = at::native::update_result_type_state(*max, state);
      }
      common_dtype = at::native::result_type(state);
    }
    if (is_whitelisted_clamp_nonbool_dtype(common_dtype)) {
      TORCH_CHECK(
          common_dtype == out.scalar_type() ||
              at::canCast(common_dtype, out.scalar_type()),
          op_name,
          ": result type ",
          common_dtype,
          " can't be cast to the desired output type ",
          out.scalar_type());
      const bool same_min_shape = !has_min || min->sizes().equals(self.sizes());
      const bool same_max_shape = !has_max || max->sizes().equals(self.sizes());
      if (same_min_shape && same_max_shape) {
        ops::check_out_sizes(op_name, out, self.sizes());
      } else {
        c10::DimVector expected_sizes(self.sizes().begin(), self.sizes().end());
        if (has_min && !min->sizes().equals(expected_sizes)) {
          expected_sizes =
              at::infer_size_dimvector(expected_sizes, min->sizes());
        }
        if (has_max && !max->sizes().equals(expected_sizes)) {
          expected_sizes =
              at::infer_size_dimvector(expected_sizes, max->sizes());
        }
        ops::check_out_sizes(op_name, out, expected_sizes);
      }

      if (has_non_expanding_strides(out) &&
          ops::has_safe_raw_output_overlap(out, self) &&
          (!has_min || ops::has_safe_raw_output_overlap(out, *min)) &&
          (!has_max || ops::has_safe_raw_output_overlap(out, *max))) {
        if (out.numel() == 0) {
          return out;
        }
        std::optional<ops::TensorViewSpec> min_spec;
        if (has_min) {
          min_spec = ops::make_cpu_view_spec(*min);
        }
        std::optional<ops::TensorViewSpec> max_spec;
        if (has_max) {
          max_spec = ops::make_cpu_view_spec(*max);
        }
        ClampTensorKernelArgs args{
            ops::make_cpu_view_spec(self),
            std::move(min_spec),
            std::move(max_spec),
            ops::make_cpu_view_spec(out)};
        at::mcpu::launch_timed_kernel_on_stream(
            at::mcpu::detail::get_kernel_launch_stream(out),
            record_name,
            [args = std::move(args), record_name](
                at::mcpu::kernel_timing::Event* timing_event) mutable {
              MCPU_KERNEL_TIMING_SCOPE_EVENT(record_name, timing_event);
              KernelPointerMemoryGuard guard(
                  {args.self.data,
                   args.min.has_value() ? args.min->data : nullptr,
                   args.max.has_value() ? args.max->data : nullptr,
                   args.out.data});
              auto cpu_self = ops::cpu_view_from_spec(args.self);
              std::optional<at::Tensor> cpu_min;
              if (args.min.has_value()) {
                cpu_min = ops::cpu_view_from_spec(*args.min);
              }
              std::optional<at::Tensor> cpu_max;
              if (args.max.has_value()) {
                cpu_max = ops::cpu_view_from_spec(*args.max);
              }
              auto cpu_out = ops::cpu_view_from_spec(args.out);
              if constexpr (Kind == ClampKind::Clamp) {
                at::clamp_out(cpu_out, cpu_self, cpu_min, cpu_max);
              } else if constexpr (Kind == ClampKind::ClampMin) {
                at::clamp_min_out(cpu_out, cpu_self, *cpu_min);
              } else {
                at::clamp_max_out(cpu_out, cpu_self, *cpu_max);
              }
            });
        return out;
      }
    }
  }

  const bool has_cpu_scalar = is_cpu_scalar_tensor(self) ||
      (has_min && is_cpu_scalar_tensor(*min)) ||
      (has_max && is_cpu_scalar_tensor(*max));
  if (has_cpu_scalar && is_standard_clamp_mcpu_tensor(out) &&
      ((is_standard_clamp_mcpu_tensor(self) && self.device() == out.device()) ||
       is_cpu_scalar_tensor(self)) &&
      (!has_min ||
       (is_standard_clamp_mcpu_tensor(*min) && min->device() == out.device()) ||
       is_cpu_scalar_tensor(*min)) &&
      (!has_max ||
       (is_standard_clamp_mcpu_tensor(*max) && max->device() == out.device()) ||
       is_cpu_scalar_tensor(*max)) &&
      is_whitelisted_clamp_nonbool_dtype(self.scalar_type()) &&
      is_whitelisted_clamp_nonbool_dtype(out.scalar_type()) &&
      (!has_min || is_whitelisted_clamp_nonbool_dtype(min->scalar_type())) &&
      (!has_max || is_whitelisted_clamp_nonbool_dtype(max->scalar_type()))) {
    c10::DimVector expected_sizes(self.sizes().begin(), self.sizes().end());
    if (has_min) {
      expected_sizes = at::infer_size_dimvector(expected_sizes, min->sizes());
    }
    if (has_max) {
      expected_sizes = at::infer_size_dimvector(expected_sizes, max->sizes());
    }
    ops::check_out_sizes(op_name, out, expected_sizes);
  }

  if constexpr (Kind == ClampKind::Clamp) {
    return at::native::mcpu::custom::
        call_fallback_fn<&boxed_cpu_fallback, at::_ops::clamp_Tensor_out>::call(
            self, min, max, out);
  } else if constexpr (Kind == ClampKind::ClampMin && IsInplace) {
    return at::native::mcpu::custom::call_fallback_fn<
        &boxed_cpu_fallback,
        at::_ops::clamp_min__Tensor>::call(out, *min);
  } else if constexpr (Kind == ClampKind::ClampMin) {
    return at::native::mcpu::custom::call_fallback_fn<
        &boxed_cpu_fallback,
        at::_ops::clamp_min_Tensor_out>::call(self, *min, out);
  } else {
    return at::native::mcpu::custom::call_fallback_fn<
        &boxed_cpu_fallback,
        at::_ops::clamp_max_Tensor_out>::call(self, *max, out);
  }
}

at::Tensor& clamp_out(
    const at::Tensor& self,
    const std::optional<at::Scalar>& min,
    const std::optional<at::Scalar>& max,
    at::Tensor& out) {
  return clamp_scalar_out_impl<ClampKind::Clamp>(
      "aten::clamp.out",
      "mcpu::aten::clamp.out",
      "mcpu::aten::clamp.out.raw",
      self,
      min,
      max,
      out);
}

at::Tensor& clamp_Tensor_out(
    const at::Tensor& self,
    const std::optional<at::Tensor>& min,
    const std::optional<at::Tensor>& max,
    at::Tensor& out) {
  return clamp_tensor_out_impl<ClampKind::Clamp>(
      "aten::clamp.Tensor_out",
      "mcpu::aten::clamp.Tensor_out",
      self,
      min,
      max,
      out);
}

at::Tensor& clamp_min_out(
    const at::Tensor& self,
    const at::Scalar& min,
    at::Tensor& out) {
  return clamp_scalar_out_impl<ClampKind::ClampMin>(
      "aten::clamp_min.out",
      "mcpu::aten::clamp_min.out",
      "mcpu::aten::clamp_min.out.raw",
      self,
      min,
      std::nullopt,
      out);
}

at::Tensor& clamp_min_Tensor_out(
    const at::Tensor& self,
    const at::Tensor& min,
    at::Tensor& out) {
  return clamp_tensor_out_impl<ClampKind::ClampMin>(
      "aten::clamp_min.Tensor_out",
      "mcpu::aten::clamp_min.Tensor_out",
      self,
      min,
      std::nullopt,
      out);
}

at::Tensor clamp_min(const at::Tensor& self, const at::Scalar& min) {
  if (is_standard_clamp_mcpu_tensor(self) &&
      is_whitelisted_clamp_nonbool_dtype(self.scalar_type()) &&
      is_whitelisted_clamp_numeric_scalar(min)) {
    const auto expected_dtype = resolve_whitelisted_clamp_scalar_dtype(
        self.scalar_type(), min, std::nullopt);
    if (is_whitelisted_clamp_nonbool_dtype(expected_dtype) &&
        can_convert_clamp_scalar(
            expected_dtype, min, /*require_finite=*/false)) {
      auto out = at::empty_like(
          self,
          self.options().dtype(expected_dtype),
          at::MemoryFormat::Preserve);
      clamp_min_out(self, min, out);
      return out;
    }
  }
  return at::native::mcpu::custom::
      call_fallback_fn<&boxed_cpu_fallback, at::_ops::clamp_min>::call(
          self, min);
}

at::Tensor& clamp_min_(at::Tensor& self, const at::Scalar& min) {
  return clamp_scalar_out_impl<ClampKind::ClampMin, /*IsInplace=*/true>(
      "aten::clamp_min_",
      "mcpu::aten::clamp_min.out",
      "mcpu::aten::clamp_min.out.raw",
      self,
      min,
      std::nullopt,
      self);
}

at::Tensor clamp_min_Tensor(const at::Tensor& self, const at::Tensor& min) {
  if (is_standard_clamp_mcpu_tensor(self) &&
      is_standard_clamp_mcpu_tensor(min) && self.device() == min.device() &&
      is_whitelisted_clamp_nonbool_dtype(self.scalar_type()) &&
      is_whitelisted_clamp_nonbool_dtype(min.scalar_type())) {
    const auto common_dtype = (self.scalar_type() == min.scalar_type())
        ? self.scalar_type()
        : at::result_type(self, min);
    if (is_whitelisted_clamp_nonbool_dtype(common_dtype)) {
      if (self.sizes().equals(min.sizes())) {
        auto out = at::empty_like(
            self,
            self.options().dtype(common_dtype),
            at::MemoryFormat::Preserve);
        clamp_min_Tensor_out(self, min, out);
        return out;
      }
      const auto out_sizes =
          at::infer_size_dimvector(self.sizes(), min.sizes());
      auto out = self.sizes().equals(out_sizes)
          ? at::empty_like(
                self,
                self.options().dtype(common_dtype),
                at::MemoryFormat::Preserve)
          : (min.sizes().equals(out_sizes)
                 ? at::empty_like(
                       min,
                       min.options().dtype(common_dtype),
                       at::MemoryFormat::Preserve)
                 : at::empty(out_sizes, self.options().dtype(common_dtype)));
      clamp_min_Tensor_out(self, min, out);
      return out;
    }
  }
  return at::native::mcpu::custom::
      call_fallback_fn<&boxed_cpu_fallback, at::_ops::clamp_min_Tensor>::call(
          self, min);
}

at::Tensor& clamp_min__Tensor(at::Tensor& self, const at::Tensor& min) {
  return clamp_tensor_out_impl<ClampKind::ClampMin, /*IsInplace=*/true>(
      "aten::clamp_min_.Tensor",
      "mcpu::aten::clamp_min.Tensor_out",
      self,
      min,
      std::nullopt,
      self);
}

at::Tensor& clamp_max_out(
    const at::Tensor& self,
    const at::Scalar& max,
    at::Tensor& out) {
  return clamp_scalar_out_impl<ClampKind::ClampMax>(
      "aten::clamp_max.out",
      "mcpu::aten::clamp_max.out",
      "mcpu::aten::clamp_max.out.raw",
      self,
      std::nullopt,
      max,
      out);
}

at::Tensor& clamp_max_Tensor_out(
    const at::Tensor& self,
    const at::Tensor& max,
    at::Tensor& out) {
  return clamp_tensor_out_impl<ClampKind::ClampMax>(
      "aten::clamp_max.Tensor_out",
      "mcpu::aten::clamp_max.Tensor_out",
      self,
      std::nullopt,
      max,
      out);
}

at::Tensor& zero_(at::Tensor& self) {
  if (raw_zero_(self)) {
    return self;
  }

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::zero_", ([self]), {
    KernelMemoryGuard guard(self);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    at::zero_(cpu_self);
  });
  return self;
}

at::Tensor& masked_fill__Scalar(
    at::Tensor& self,
    const at::Tensor& mask,
    const at::Scalar& value) {
  if (raw_masked_fill__scalar(self, mask, value)) {
    return self;
  }

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::masked_fill_.Scalar", ([ self, mask, value ]), {
        KernelMemoryGuard guard(self, mask);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_mask = ops::get_cpu_tensor_view_if_needed(mask);
        at::cpu::masked_fill_(cpu_self, cpu_mask, value);
      });
  return self;
}

int64_t masked_select_count(
    const at::Tensor& mask,
    at::IntArrayRef broadcast_sizes) {
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Bool,
      "aten::masked_select expected a bool mask");

  auto count_tensor = at::empty(
      {}, at::TensorOptions().device(c10::DeviceType::CPU).dtype(at::kLong));
  auto args = std::make_unique<MaskedSelectCountArgs>(MaskedSelectCountArgs{
      ops::make_cpu_view_spec(mask),
      c10::SmallVector<int64_t, 4>(
          broadcast_sizes.begin(), broadcast_sizes.end()),
      count_tensor.mutable_data_ptr<int64_t>()});
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::masked_select.count", ([args = std::move(args)]), {
        KernelPointerMemoryGuard guard({args->mask.data});
        auto cpu_mask = ops::cpu_view_from_spec(args->mask);
        auto expanded_mask = cpu_mask.expand(args->broadcast_sizes);
        *args->count = at::count_nonzero(expanded_mask).item<int64_t>();
      });
  at::native::mcpu::synchronize_if_mcpu(mask);
  return count_tensor.item<int64_t>();
}

at::Tensor& masked_select_write(
    const at::Tensor& self,
    const at::Tensor& mask,
    at::Tensor& out) {
  auto args = std::make_unique<MaskedSelectWriteArgs>(MaskedSelectWriteArgs{
      ops::make_cpu_view_spec(self),
      ops::make_cpu_view_spec(mask),
      ops::make_cpu_view_spec(out)});
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::masked_select", ([args = std::move(args)]), {
        KernelPointerMemoryGuard guard(
            {args->self.data, args->mask.data, args->out.data});
        auto cpu_self = ops::cpu_view_from_spec(args->self);
        auto cpu_mask = ops::cpu_view_from_spec(args->mask);
        auto cpu_out = ops::cpu_view_from_spec(args->out);
        at::masked_select_out(cpu_out, cpu_self, cpu_mask);
      });
  return out;
}

at::Tensor masked_select(const at::Tensor& self, const at::Tensor& mask) {
  const auto broadcast_sizes = at::infer_size(self.sizes(), mask.sizes());
  const auto count = masked_select_count(mask, broadcast_sizes);
  auto out = at::empty({count}, self.options());
  return masked_select_write(self, mask, out);
}

at::Tensor& masked_select_out(
    const at::Tensor& self,
    const at::Tensor& mask,
    at::Tensor& out) {
  const auto broadcast_sizes = at::infer_size(self.sizes(), mask.sizes());
  const auto count = masked_select_count(mask, broadcast_sizes);
  const std::array<int64_t, 1> expected_sizes{count};
  ops::check_out_sizes("aten::masked_select.out", out, expected_sizes);
  TORCH_CHECK(
      out.scalar_type() == self.scalar_type(),
      "aten::masked_select.out expected out dtype ",
      self.scalar_type(),
      ", but got ",
      out.scalar_type());
  return masked_select_write(self, mask, out);
}

at::Tensor& ne_Scalar_out(
    const at::Tensor& self,
    const at::Scalar& other,
    at::Tensor& out) {
  ops::check_out_sizes("aten::ne.Scalar_out", out, self.sizes());

  if (raw_ne_scalar_out(self, other, out)) {
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::ne.Scalar_out", ([ self, out, other ]), {
        KernelMemoryGuard guard(self, out);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::ne_out(cpu_out, cpu_self, other);
      });
  return out;
}

at::Tensor nonzero(const at::Tensor& self) {
  const auto ndim = self.dim();
  const auto numel = self.numel();
  const auto sizes = self.sizes().vec();
  const auto strides = self.strides().vec();
  auto count = at::empty(
      {}, self.options().device(c10::DeviceType::CPU).dtype(at::kLong));
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::nonzero.count",
      ([ self, count, ndim, numel, sizes, strides ]),
      {
        KernelMemoryGuard guard(self);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        int64_t count_value = 0;
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            at::ScalarType::Bool,
            cpu_self.scalar_type(),
            "mcpu_nonzero_count",
            [&] {
              const auto* data = cpu_self.const_data_ptr<scalar_t>();
              std::vector<int64_t> index(ndim, 0);
              for (int64_t linear = 0; linear < numel; ++linear) {
                int64_t offset = 0;
                for (int64_t d = 0; d < ndim; ++d) {
                  offset += index[d] * strides[d];
                }
                if (is_nonzero_value(data[offset])) {
                  ++count_value;
                }
                for (int64_t d = ndim - 1; d >= 0; --d) {
                  if (++index[d] < sizes[d]) {
                    break;
                  }
                  index[d] = 0;
                }
              }
            });
        *count.mutable_data_ptr<int64_t>() = count_value;
      });
  at::native::mcpu::synchronize_if_mcpu(self);
  const auto count_value = count.item<int64_t>();
  auto out = at::empty(
      {count_value, ndim},
      self.options().device(c10::DeviceType::PrivateUse1).dtype(at::kLong));

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::nonzero.write",
      ([ self, out, ndim, numel, sizes, strides ]),
      {
        KernelMemoryGuard guard(self, out);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self).clone();
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        const auto ndim = cpu_self.dim();
        const auto numel = cpu_self.numel();
        const auto sizes = cpu_self.sizes().vec();
        const auto strides = cpu_self.strides().vec();

        int64_t row = 0;
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            at::ScalarType::Bool,
            cpu_self.scalar_type(),
            "mcpu_nonzero_write",
            [&] {
              const auto* data = cpu_self.const_data_ptr<scalar_t>();
              auto* out_ptr = cpu_out.mutable_data_ptr<int64_t>();
              std::vector<int64_t> index(ndim, 0);
              for (int64_t linear = 0; linear < numel; ++linear) {
                int64_t offset = 0;
                for (int64_t d = 0; d < ndim; ++d) {
                  offset += index[d] * strides[d];
                }
                if (is_nonzero_value(data[offset])) {
                  for (int64_t d = 0; d < ndim; ++d) {
                    out_ptr[row * ndim + d] = index[d];
                  }
                  ++row;
                }
                for (int64_t d = ndim - 1; d >= 0; --d) {
                  if (++index[d] < sizes[d]) {
                    break;
                  }
                  index[d] = 0;
                }
              }
            });
      });
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("abs", &abs);
  m.impl("abs.out", &abs_out);
  m.impl("bitwise_not", &bitwise_not);
  m.impl("bitwise_not.out", &bitwise_not_out);
  m.impl("bitwise_not_", &bitwise_not_);
  m.impl("clamp.out", &clamp_out);
  m.impl("clamp.Tensor_out", &clamp_Tensor_out);
  m.impl("clamp_min", &clamp_min);
  m.impl("clamp_min_", &clamp_min_);
  m.impl("clamp_min.out", &clamp_min_out);
  m.impl("clamp_min.Tensor", &clamp_min_Tensor);
  m.impl("clamp_min_.Tensor", &clamp_min__Tensor);
  m.impl("clamp_min.Tensor_out", &clamp_min_Tensor_out);
  m.impl("clamp_max.out", &clamp_max_out);
  m.impl("clamp_max.Tensor_out", &clamp_max_Tensor_out);
  m.impl("cos.out", &cos_out);
  m.impl("masked_fill_.Scalar", &masked_fill__Scalar);
  m.impl("masked_select", &masked_select);
  m.impl("masked_select.out", &masked_select_out);
  m.impl("ne.Scalar_out", &ne_Scalar_out);
  m.impl("neg.out", &neg_out);
  m.impl("nonzero", &nonzero);
  m.impl("sigmoid", &sigmoid);
  m.impl("sigmoid.out", &sigmoid_out);
  m.impl("sigmoid_", &sigmoid_);
  m.impl("sin.out", &sin_out);
  m.impl("silu", &silu);
  m.impl("silu.out", &silu_out);
  m.impl("silu_", &silu_);
  m.impl("reciprocal.out", &reciprocal_out);
  m.impl("rsqrt", &rsqrt);
  m.impl("rsqrt.out", &rsqrt_out);
  m.impl("zero_", &zero_);
}

} // namespace at::mcpu
