#include "Common.h"
#include "RawPlan.h"

#include <ATen/ExpandUtils.h>
#include <ATen/ops/bitwise_not.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/masked_fill.h>
#include <ATen/ops/masked_fill_cpu_dispatch.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/reciprocal.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/zero.h>
#include <torch/library.h>

#include <cmath>
#include <memory>

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
            record_name,
            ([ self_ptr, out_ptr, plan_ptr, record_name ]),
            {
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
            "mcpu::aten::zero_.raw",
            ([ self_ptr, plan_ptr ]),
            {
              KernelPointerMemoryGuard guard({self_ptr});
              raw_zero_kernel(self_ptr, *plan_ptr);
            });
      });
  return true;
}

bool raw_clamp_out(
    const at::Tensor& self,
    const std::optional<at::Scalar>& min,
    const std::optional<at::Scalar>& max,
    at::Tensor& out) {
  if ((!min.has_value() && !max.has_value()) ||
      self.scalar_type() != out.scalar_type() ||
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
      out.scalar_type(),
      "mcpu_raw_clamp_out",
      [&] {
        const auto* self_ptr = self.const_data_ptr<scalar_t>();
        auto* out_ptr = out.mutable_data_ptr<scalar_t>();
        const bool has_min = min.has_value();
        const bool has_max = max.has_value();
        const auto min_value =
            has_min ? min->to<scalar_t>() : scalar_t(0);
        const auto max_value =
            has_max ? max->to<scalar_t>() : scalar_t(0);
        auto plan_ptr = std::make_shared<ops::RawTensorPairPlan>(*plan);
        MCPU_LAUNCH_TIMED_KERNEL(
            "mcpu::aten::clamp.out.raw",
            ([ self_ptr,
               out_ptr,
               has_min,
               min_value,
               has_max,
               max_value,
               plan_ptr ]),
            {
              KernelPointerMemoryGuard guard({self_ptr, out_ptr});
              raw_clamp_kernel(
                  self_ptr,
                  out_ptr,
                  has_min,
                  min_value,
                  has_max,
                  max_value,
                  *plan_ptr);
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

at::Tensor& clamp_out(
    const at::Tensor& self,
    const std::optional<at::Scalar>& min,
    const std::optional<at::Scalar>& max,
    at::Tensor& out) {
  ops::check_out_sizes("aten::clamp.out", out, self.sizes());

  if (raw_clamp_out(self, min, max, out)) {
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::clamp.out", ([ self, out, min, max ]), {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::clamp_out(cpu_out, cpu_self, min, max);
  });
  return out;
}

at::Tensor& clamp_Tensor_out(
    const at::Tensor& self,
    const std::optional<at::Tensor>& min,
    const std::optional<at::Tensor>& max,
    at::Tensor& out) {
  auto expected_sizes = self.sizes().vec();
  if (min.has_value() && min->defined()) {
    expected_sizes = at::infer_size(expected_sizes, min->sizes());
  }
  if (max.has_value() && max->defined()) {
    expected_sizes = at::infer_size(expected_sizes, max->sizes());
  }
  ops::check_out_sizes("aten::clamp.Tensor_out", out, expected_sizes);

  const at::Tensor min_guard =
      min.has_value() && min->defined() ? *min : at::Tensor();
  const at::Tensor max_guard =
      max.has_value() && max->defined() ? *max : at::Tensor();
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::clamp.Tensor_out",
      ([ self, min_guard, max_guard, min, max, out ]),
      {
        KernelMemoryGuard guard(self, min_guard, max_guard, out);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_min = min.has_value() && min->defined()
            ? std::make_optional(ops::get_cpu_tensor_view_if_needed(*min))
            : std::nullopt;
        auto cpu_max = max.has_value() && max->defined()
            ? std::make_optional(ops::get_cpu_tensor_view_if_needed(*max))
            : std::nullopt;
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::clamp_out(cpu_out, cpu_self, cpu_min, cpu_max);
      });
  return out;
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
  m.impl("bitwise_not", &bitwise_not);
  m.impl("bitwise_not.out", &bitwise_not_out);
  m.impl("bitwise_not_", &bitwise_not_);
  m.impl("clamp.out", &clamp_out);
  m.impl("clamp.Tensor_out", &clamp_Tensor_out);
  m.impl("cos.out", &cos_out);
  m.impl("masked_fill_.Scalar", &masked_fill__Scalar);
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
  m.impl("zero_", &zero_);
}

} // namespace at::mcpu
