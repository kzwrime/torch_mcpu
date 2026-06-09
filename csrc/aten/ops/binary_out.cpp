#include "Common.h"
#include "runtime/McpuKernelTiming.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ExpandUtils.h>
#include <ATen/ops/add.h>
#include <ATen/ops/div.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/remainder.h>
#include <ATen/ops/sub.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

void raw_add_kernel(
    const float* self,
    const float* other,
    float* out,
    int64_t rows,
    int64_t cols,
    int64_t self_s0,
    int64_t self_s1,
    int64_t other_s0,
    int64_t other_s1,
    int64_t out_s0,
    int64_t out_s1,
    bool other_is_1d,
    float alpha) {
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      const int64_t other_offset =
          other_is_1d ? j * other_s0 : i * other_s0 + j * other_s1;
      out[i * out_s0 + j * out_s1] =
          self[i * self_s0 + j * self_s1] + alpha * other[other_offset];
    }
  }
}

bool is_float_mcpu_or_cpu(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::kFloat &&
      (tensor.device().type() == c10::DeviceType::PrivateUse1 ||
       tensor.device().type() == c10::DeviceType::CPU);
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

  const bool other_is_1d = other.dim() == 1;
  const bool raw_shape_ok = self.dim() == 2 && out.dim() == 2 &&
      ((other_is_1d && other.size(0) == self.size(1)) ||
       (other.dim() == 2 && other.sizes().equals(self.sizes())));
  if (raw_shape_ok && is_float_mcpu_or_cpu(self) &&
      is_float_mcpu_or_cpu(other) && is_float_mcpu_or_cpu(out)) {
    const float* self_ptr = self.const_data_ptr<float>();
    const float* other_ptr = other.const_data_ptr<float>();
    float* out_ptr = out.mutable_data_ptr<float>();
    const int64_t rows = self.size(0);
    const int64_t cols = self.size(1);
    const int64_t self_s0 = self.stride(0);
    const int64_t self_s1 = self.stride(1);
    const int64_t other_s0 = other.stride(0);
    const int64_t other_s1 = other_is_1d ? 0 : other.stride(1);
    const int64_t out_s0 = out.stride(0);
    const int64_t out_s1 = out.stride(1);
    const float alpha_value = alpha.to<float>();
    launch_timed_kernel(
        "aten::add",
        [self_ptr,
         other_ptr,
         out_ptr,
         rows,
         cols,
         self_s0,
         self_s1,
         other_s0,
         other_s1,
         out_s0,
         out_s1,
         other_is_1d,
         alpha_value](at::mcpu::kernel_timing::Event* timing_event) {
          MCPU_KERNEL_TIMING_SCOPE_EVENT("mcpu::aten::add", timing_event);
          KernelPointerMemoryGuard guard({self_ptr, other_ptr, out_ptr});
          raw_add_kernel(
              self_ptr,
              other_ptr,
              out_ptr,
              rows,
              cols,
              self_s0,
              self_s1,
              other_s0,
              other_s1,
              out_s0,
              out_s1,
              other_is_1d,
              alpha_value);
        });
    return out;
  }

  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_other = ops::get_cpu_tensor_view_if_needed(other);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  launch_kernel(
      out, [alpha, cpu_self, cpu_other, cpu_out]() mutable {
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
  launch_kernel(out, [self, other, out, alpha]() mutable {
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
  launch_kernel(self, [self, other, alpha]() mutable {
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

  launch_kernel(out, [self, other, out]() mutable {
    KernelMemoryGuard guard(self, other, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_other = ops::get_cpu_tensor_view_if_needed(other);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::div_out(cpu_out, cpu_self, cpu_other);
  });
  return out;
}

at::Tensor& mul_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  auto expected_sizes = at::infer_size(self.sizes(), other.sizes());
  ops::check_out_sizes("aten::mul.out", out, expected_sizes);

  launch_kernel(out, [self, other, out]() mutable {
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

  launch_kernel(out, [self, other, out]() mutable {
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
  launch_kernel(out, [self, other, out, alpha]() mutable {
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

  launch_kernel(out, [self, other, out, alpha]() mutable {
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
  launch_kernel(out, [self, out, other, alpha]() mutable {
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
  launch_kernel(self, [self, other, alpha]() mutable {
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

  launch_kernel(out, [self, exponent, out]() mutable {
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
