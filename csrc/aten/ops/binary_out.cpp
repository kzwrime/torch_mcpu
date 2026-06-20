#include "Common.h"
#include "RawPlan.h"

#include <ATen/ExpandUtils.h>
#include <ATen/ops/add.h>
#include <ATen/ops/div.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/remainder.h>
#include <ATen/ops/sub.h>
#include <torch/library.h>

#include <memory>
#include <optional>

namespace at::mcpu {
namespace {

enum class RawBinaryOp {
  Add,
  Sub,
  Mul,
};

template <RawBinaryOp op, typename scalar_t>
void raw_binary_kernel(
    const scalar_t* self,
    const scalar_t* other,
    scalar_t* out,
    const ops::RawTensorTripletPlan& plan) {
  ops::for_each_raw_tensor_triplet_row(
      plan,
      [self, other, out](
          int64_t self_offset,
          int64_t other_offset,
          int64_t out_offset,
          int64_t inner_size) {
        const auto* self_row = self + self_offset;
        const auto* other_row = other + other_offset;
        auto* out_row = out + out_offset;
        for (int64_t i = 0; i < inner_size; ++i) {
          if constexpr (op == RawBinaryOp::Add) {
            out_row[i] = self_row[i] + other_row[i];
          } else if constexpr (op == RawBinaryOp::Sub) {
            out_row[i] = self_row[i] - other_row[i];
          } else {
            out_row[i] = self_row[i] * other_row[i];
          }
        }
      });
}

template <typename scalar_t>
void raw_div_kernel(
    const scalar_t* self,
    const scalar_t* other,
    scalar_t* out,
    const ops::RawTensorTripletPlan& plan) {
  ops::for_each_raw_tensor_triplet_row(
      plan,
      [self, other, out](
          int64_t self_offset,
          int64_t other_offset,
          int64_t out_offset,
          int64_t inner_size) {
        const auto* self_row = self + self_offset;
        const auto* other_row = other + other_offset;
        auto* out_row = out + out_offset;
        for (int64_t i = 0; i < inner_size; ++i) {
          out_row[i] = self_row[i] / other_row[i];
        }
      });
}

template <typename scalar_t>
void raw_gt_scalar_kernel(
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
          out_row[i] = self_row[i] > other;
        }
      });
}

template <typename scalar_t>
void raw_gt_tensor_kernel(
    const scalar_t* self,
    const scalar_t* other,
    bool* out,
    const ops::RawTensorTripletPlan& plan) {
  ops::for_each_raw_tensor_triplet_row(
      plan,
      [self, other, out](
          int64_t self_offset,
          int64_t other_offset,
          int64_t out_offset,
          int64_t inner_size) {
        const auto* self_row = self + self_offset;
        const auto* other_row = other + other_offset;
        auto* out_row = out + out_offset;
        for (int64_t i = 0; i < inner_size; ++i) {
          out_row[i] = self_row[i] > other_row[i];
        }
      });
}

template <RawBinaryOp op, typename scalar_t>
void raw_binary_scalar_kernel(
    const scalar_t* self,
    scalar_t other,
    scalar_t* out,
    const ops::RawTensorPairPlan& plan) {
  ops::for_each_raw_tensor_pair_row(
      plan,
      [self, other, out](
          int64_t self_offset, int64_t out_offset, int64_t inner_size) {
        const auto* self_row = self + self_offset;
        auto* out_row = out + out_offset;
        for (int64_t i = 0; i < inner_size; ++i) {
          if constexpr (op == RawBinaryOp::Add) {
            out_row[i] = self_row[i] + other;
          } else {
            out_row[i] = self_row[i] - other;
          }
        }
      });
}

std::optional<ops::RawTensorTripletPlan> make_raw_binary_plan(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    const at::Tensor& out) {
  if (!alpha.equal(1) || self.scalar_type() != other.scalar_type() ||
      self.scalar_type() != out.scalar_type() ||
      self.scalar_type() == at::ScalarType::Bool ||
      c10::isComplexType(self.scalar_type())) {
    return std::nullopt;
  }

  return ops::make_same_shape_raw_tensor_triplet_plan(self, other, out);
}

template <RawBinaryOp op>
bool raw_binary_out(
    const char* record_name,
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& out) {
  auto plan = make_raw_binary_plan(self, other, alpha, out);
  if (!plan.has_value()) {
    return false;
  }

  const auto numel = out.numel();
  if (numel == 0) {
    return true;
  }

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      out.scalar_type(),
      "mcpu_raw_binary",
      [&] {
        const auto* self_ptr = self.const_data_ptr<scalar_t>();
        const auto* other_ptr = other.const_data_ptr<scalar_t>();
        auto* out_ptr = out.mutable_data_ptr<scalar_t>();
        auto plan_ptr = std::make_shared<ops::RawTensorTripletPlan>(*plan);
        MCPU_LAUNCH_TIMED_KERNEL(
            record_name,
            ([ self_ptr, other_ptr, out_ptr, plan_ptr, record_name ]),
            {
              KernelPointerMemoryGuard guard({self_ptr, other_ptr, out_ptr});
              raw_binary_kernel<op>(self_ptr, other_ptr, out_ptr, *plan_ptr);
            });
      });
  return true;
}

template <RawBinaryOp op>
bool raw_binary_scalar_out(
    const char* record_name,
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha,
    at::Tensor& out) {
  if (!alpha.equal(1) || self.scalar_type() != out.scalar_type() ||
      self.scalar_type() == at::ScalarType::Bool ||
      c10::isComplexType(self.scalar_type())) {
    return false;
  }

  auto plan = ops::make_same_shape_raw_tensor_pair_plan(self, out);
  if (!plan.has_value()) {
    return false;
  }

  if (self.numel() == 0) {
    return true;
  }

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      out.scalar_type(),
      "mcpu_raw_binary_scalar",
      [&] {
        const auto* self_ptr = self.const_data_ptr<scalar_t>();
        auto* out_ptr = out.mutable_data_ptr<scalar_t>();
        const auto other_value = other.to<scalar_t>();
        auto plan_ptr = std::make_shared<ops::RawTensorPairPlan>(*plan);
        MCPU_LAUNCH_TIMED_KERNEL(
            record_name,
            ([ self_ptr, other_value, out_ptr, plan_ptr, record_name ]),
            {
              KernelPointerMemoryGuard guard({self_ptr, out_ptr});
              raw_binary_scalar_kernel<op>(
                  self_ptr, other_value, out_ptr, *plan_ptr);
            });
      });
  return true;
}

bool raw_div_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  auto plan = make_raw_binary_plan(self, other, at::Scalar(1), out);
  if (!plan.has_value() || !c10::isFloatingType(out.scalar_type())) {
    return false;
  }

  const auto numel = out.numel();
  if (numel == 0) {
    return true;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      out.scalar_type(),
      "mcpu_raw_div",
      [&] {
        const auto* self_ptr = self.const_data_ptr<scalar_t>();
        const auto* other_ptr = other.const_data_ptr<scalar_t>();
        auto* out_ptr = out.mutable_data_ptr<scalar_t>();
        auto plan_ptr = std::make_shared<ops::RawTensorTripletPlan>(*plan);
        MCPU_LAUNCH_TIMED_KERNEL(
            "mcpu::aten::div.raw",
            ([ self_ptr, other_ptr, out_ptr, plan_ptr ]),
            {
              KernelPointerMemoryGuard guard({self_ptr, other_ptr, out_ptr});
              raw_div_kernel(self_ptr, other_ptr, out_ptr, *plan_ptr);
            });
      });
  return true;
}

bool raw_gt_tensor_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  if (self.scalar_type() != other.scalar_type() ||
      out.scalar_type() != at::ScalarType::Bool ||
      !ops::is_raw_dtype_supported(self.scalar_type())) {
    return false;
  }

  auto plan = ops::make_same_shape_raw_tensor_triplet_plan(self, other, out);
  if (!plan.has_value()) {
    return false;
  }

  if (out.numel() == 0) {
    return true;
  }

  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "mcpu_raw_gt_tensor_out",
      [&] {
        const auto* self_ptr = self.const_data_ptr<scalar_t>();
        const auto* other_ptr = other.const_data_ptr<scalar_t>();
        auto* out_ptr = out.mutable_data_ptr<bool>();
        auto plan_ptr = std::make_shared<ops::RawTensorTripletPlan>(*plan);
        MCPU_LAUNCH_TIMED_KERNEL(
            "mcpu::aten::gt.Tensor.raw",
            ([ self_ptr, other_ptr, out_ptr, plan_ptr ]),
            {
              KernelPointerMemoryGuard guard({self_ptr, other_ptr, out_ptr});
              raw_gt_tensor_kernel(self_ptr, other_ptr, out_ptr, *plan_ptr);
            });
      });
  return true;
}

at::Tensor empty_binary_mcpu(const at::Tensor& self, const at::Tensor& other) {
  auto out_sizes = at::infer_size(self.sizes(), other.sizes());
  return at::empty(
      out_sizes, self.options().dtype(at::result_type(self, other)));
}

at::Tensor empty_like_mcpu_result(
    const at::Tensor& self,
    const at::Scalar& other) {
  return at::empty(
      self.sizes(), self.options().dtype(at::result_type(self, other)));
}

at::Tensor empty_comparison_mcpu(
    const at::Tensor& self,
    at::IntArrayRef out_sizes) {
  return at::empty(out_sizes, self.options().dtype(at::kBool));
}

at::Tensor& add_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& out) {
  auto expected_sizes = at::infer_size(self.sizes(), other.sizes());
  TORCH_CHECK(
      out.sizes().equals(expected_sizes),
      "aten::add.out: expected out.sizes() == ",
      expected_sizes,
      ", but got ",
      out.sizes());

  if (raw_binary_out<RawBinaryOp::Add>(
          "mcpu::aten::add.raw", self, other, alpha, out)) {
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::add", ([ alpha, self, other, out ]), {
    KernelMemoryGuard guard(self, other, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_other = ops::get_cpu_tensor_view_if_needed(other);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::add_out(cpu_out, cpu_self, cpu_other, alpha);
  });
  return out;
}

at::Tensor add_Tensor(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  auto out = empty_binary_mcpu(self, other);
  add_out(self, other, alpha, out);
  return out;
}

at::Tensor& add_Tensor_(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  add_out(self, other, alpha, self);
  return self;
}

at::Tensor add_Scalar(
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  auto out = empty_like_mcpu_result(self, other);
  if (raw_binary_scalar_out<RawBinaryOp::Add>(
          "mcpu::aten::add.Scalar.raw", self, other, alpha, out)) {
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::add.Scalar", ([ self, other, out, alpha ]), {
        KernelMemoryGuard guard(self, out);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::add_out(cpu_out, cpu_self, other, alpha);
      });
  return out;
}

at::Tensor& add_Scalar_(
    at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  if (raw_binary_scalar_out<RawBinaryOp::Add>(
          "mcpu::aten::add_.Scalar.raw", self, other, alpha, self)) {
    return self;
  }

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::add_.Scalar", ([ self, other, alpha ]), {
        KernelMemoryGuard guard(self);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        at::add_out(cpu_self, cpu_self, other, alpha);
      });
  return self;
}

at::Tensor& div_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  auto expected_sizes = at::infer_size(self.sizes(), other.sizes());
  ops::check_out_sizes("aten::div.out", out, expected_sizes);

  if (raw_div_out(self, other, out)) {
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::div.out", ([ self, other, out ]), {
    KernelMemoryGuard guard(self, other, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_other = ops::get_cpu_tensor_view_if_needed(other);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::div_out(cpu_out, cpu_self, cpu_other);
  });
  return out;
}

at::Tensor& gt_Tensor_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  auto expected_sizes = at::infer_size(self.sizes(), other.sizes());
  ops::check_out_sizes("aten::gt.Tensor_out", out, expected_sizes);

  if (raw_gt_tensor_out(self, other, out)) {
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::gt.Tensor_out", ([ self, other, out ]), {
        KernelMemoryGuard guard(self, other, out);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_other = ops::get_cpu_tensor_view_if_needed(other);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::gt_out(cpu_out, cpu_self, cpu_other);
      });
  return out;
}

at::Tensor gt_Tensor(const at::Tensor& self, const at::Tensor& other) {
  auto out_sizes = at::infer_size(self.sizes(), other.sizes());
  auto out = empty_comparison_mcpu(self, out_sizes);
  gt_Tensor_out(self, other, out);
  return out;
}

at::Tensor& gt_Scalar_out(
    const at::Tensor& self,
    const at::Scalar& other,
    at::Tensor& out) {
  ops::check_out_sizes("aten::gt.Scalar_out", out, self.sizes());

  auto plan = ops::make_same_shape_raw_tensor_pair_plan(self, out);
  if (plan.has_value() && out.scalar_type() == at::ScalarType::Bool &&
      ops::is_raw_dtype_supported(self.scalar_type())) {
    if (self.numel() == 0) {
      return out;
    }

    AT_DISPATCH_ALL_TYPES_AND3(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        at::ScalarType::Bool,
        self.scalar_type(),
        "mcpu_raw_gt_scalar_out",
        [&] {
          auto plan_ptr =
              std::make_shared<ops::RawTensorPairPlan>(*std::move(plan));
          const auto* self_ptr = self.const_data_ptr<scalar_t>();
          auto* out_ptr = out.mutable_data_ptr<bool>();
          const auto other_value = other.to<scalar_t>();
          MCPU_LAUNCH_TIMED_KERNEL(
              "mcpu::aten::gt.Scalar.raw",
              ([ self_ptr, other_value, out_ptr, plan_ptr ]),
              {
                KernelPointerMemoryGuard guard({self_ptr, out_ptr});
                raw_gt_scalar_kernel(self_ptr, other_value, out_ptr, *plan_ptr);
              });
        });
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::gt.Scalar_out", ([ self, other, out ]), {
        KernelMemoryGuard guard(self, out);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::gt_out(cpu_out, cpu_self, other);
      });
  return out;
}

at::Tensor gt_Scalar(const at::Tensor& self, const at::Scalar& other) {
  auto out = empty_comparison_mcpu(self, self.sizes());
  gt_Scalar_out(self, other, out);
  return out;
}

at::Tensor& mul_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  auto expected_sizes = at::infer_size(self.sizes(), other.sizes());
  ops::check_out_sizes("aten::mul.out", out, expected_sizes);

  if (raw_binary_out<RawBinaryOp::Mul>(
          "mcpu::aten::mul.raw", self, other, at::Scalar(1), out)) {
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::mul.out", ([ self, other, out ]), {
    KernelMemoryGuard guard(self, other, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_other = ops::get_cpu_tensor_view_if_needed(other);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::mul_out(cpu_out, cpu_self, cpu_other);
  });
  return out;
}

at::Tensor& remainder_Tensor_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  auto expected_sizes = at::infer_size(self.sizes(), other.sizes());
  ops::check_out_sizes("aten::remainder.Tensor_out", out, expected_sizes);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::remainder.Tensor_out", ([ self, other, out ]), {
        KernelMemoryGuard guard(self, other, out);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_other = ops::get_cpu_tensor_view_if_needed(other);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::remainder_out(cpu_out, cpu_self, cpu_other);
      });
  return out;
}

at::Tensor sub_Tensor(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  auto out = empty_binary_mcpu(self, other);
  if (raw_binary_out<RawBinaryOp::Sub>(
          "mcpu::aten::sub.raw", self, other, alpha, out)) {
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::sub.Tensor", ([ self, other, out, alpha ]), {
        KernelMemoryGuard guard(self, other, out);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_other = ops::get_cpu_tensor_view_if_needed(other);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::sub_out(cpu_out, cpu_self, cpu_other, alpha);
      });
  return out;
}

at::Tensor& sub_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& out) {
  auto expected_sizes = at::infer_size(self.sizes(), other.sizes());
  TORCH_CHECK(
      out.sizes().equals(expected_sizes),
      "aten::sub.out: expected out.sizes() == ",
      expected_sizes,
      ", but got ",
      out.sizes());

  if (raw_binary_out<RawBinaryOp::Sub>(
          "mcpu::aten::sub.raw", self, other, alpha, out)) {
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::sub.out", ([ self, other, out, alpha ]), {
        KernelMemoryGuard guard(self, other, out);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_other = ops::get_cpu_tensor_view_if_needed(other);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::sub_out(cpu_out, cpu_self, cpu_other, alpha);
      });
  return out;
}

at::Tensor& sub_Tensor_(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  sub_out(self, other, alpha, self);
  return self;
}

at::Tensor sub_Scalar(
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  auto out = empty_like_mcpu_result(self, other);
  if (raw_binary_scalar_out<RawBinaryOp::Sub>(
          "mcpu::aten::sub.Scalar.raw", self, other, alpha, out)) {
    return out;
  }

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::sub.Scalar", ([ self, out, other, alpha ]), {
        KernelMemoryGuard guard(self, out);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::sub_out(cpu_out, cpu_self, other, alpha);
      });
  return out;
}

at::Tensor& sub_Scalar_(
    at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  if (raw_binary_scalar_out<RawBinaryOp::Sub>(
          "mcpu::aten::sub_.Scalar.raw", self, other, alpha, self)) {
    return self;
  }

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::sub_.Scalar", ([ self, other, alpha ]), {
        KernelMemoryGuard guard(self);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        at::sub_out(cpu_self, cpu_self, other, alpha);
      });
  return self;
}

at::Tensor& pow_Scalar_out(
    const at::Scalar& self,
    const at::Tensor& exponent,
    at::Tensor& out) {
  ops::check_out_sizes("aten::pow.Scalar_out", out, exponent.sizes());

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::pow.Scalar_out", ([ self, exponent, out ]), {
        KernelMemoryGuard guard(exponent, out);
        auto cpu_exponent = ops::get_cpu_tensor_view_if_needed(exponent);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::pow_out(cpu_out, self, cpu_exponent);
      });
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", &add_Tensor);
  m.impl("add.out", &add_out);
  m.impl("add_.Tensor", &add_Tensor_);
  m.impl("add.Scalar", &add_Scalar);
  m.impl("add_.Scalar", &add_Scalar_);
  m.impl("div.out", &div_out);
  m.impl("gt.Tensor", &gt_Tensor);
  m.impl("gt.Tensor_out", &gt_Tensor_out);
  m.impl("gt.Scalar", &gt_Scalar);
  m.impl("gt.Scalar_out", &gt_Scalar_out);
  m.impl("mul.out", &mul_out);
  m.impl("sub.Tensor", &sub_Tensor);
  m.impl("sub.out", &sub_out);
  m.impl("sub_.Tensor", &sub_Tensor_);
  m.impl("sub.Scalar", &sub_Scalar);
  m.impl("sub_.Scalar", &sub_Scalar_);
  m.impl("pow.Scalar_out", &pow_Scalar_out);
  m.impl("remainder.Tensor_out", &remainder_Tensor_out);
}

} // namespace at::mcpu
