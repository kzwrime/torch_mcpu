#include "Common.h"
#include "runtime/McpuKernelTiming.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ops/addmm.h>
#include <ATen/ops/mm.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

void raw_mm_kernel(
    const float* self,
    const float* mat2,
    float* out,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t self_s0,
    int64_t self_s1,
    int64_t mat2_s0,
    int64_t mat2_s1,
    int64_t out_s0,
    int64_t out_s1) {
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int64_t p = 0; p < k; ++p) {
        acc += self[i * self_s0 + p * self_s1] *
            mat2[p * mat2_s0 + j * mat2_s1];
      }
      out[i * out_s0 + j * out_s1] = acc;
    }
  }
}

void raw_addmm_kernel(
    const float* self,
    const float* mat1,
    const float* mat2,
    float* out,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t self_s0,
    int64_t self_s1,
    int64_t mat1_s0,
    int64_t mat1_s1,
    int64_t mat2_s0,
    int64_t mat2_s1,
    int64_t out_s0,
    int64_t out_s1,
    bool self_is_1d,
    float beta,
    float alpha) {
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int64_t p = 0; p < k; ++p) {
        acc += mat1[i * mat1_s0 + p * mat1_s1] *
            mat2[p * mat2_s0 + j * mat2_s1];
      }
      const int64_t self_offset =
          self_is_1d ? j * self_s0 : i * self_s0 + j * self_s1;
      out[i * out_s0 + j * out_s1] =
          beta * self[self_offset] + alpha * acc;
    }
  }
}

bool is_float_mcpu_or_cpu(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::kFloat &&
      (tensor.device().type() == c10::DeviceType::PrivateUse1 ||
       tensor.device().type() == c10::DeviceType::CPU);
}

at::Tensor& mm_out(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  TORCH_CHECK(self.dim() == 2, "mm: self must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mm: mat2 must be a matrix");
  TORCH_CHECK(
      self.size(1) == mat2.size(0), "mm: self.size(1) must match mat2.size(0)");
  ops::check_out_sizes("aten::mm.out", out, {self.size(0), mat2.size(1)});

  if (is_float_mcpu_or_cpu(self) && is_float_mcpu_or_cpu(mat2) &&
      is_float_mcpu_or_cpu(out)) {
    const float* self_ptr = self.const_data_ptr<float>();
    const float* mat2_ptr = mat2.const_data_ptr<float>();
    float* out_ptr = out.mutable_data_ptr<float>();
    const int64_t m = self.size(0);
    const int64_t n = mat2.size(1);
    const int64_t k = self.size(1);
    const int64_t self_s0 = self.stride(0);
    const int64_t self_s1 = self.stride(1);
    const int64_t mat2_s0 = mat2.stride(0);
    const int64_t mat2_s1 = mat2.stride(1);
    const int64_t out_s0 = out.stride(0);
    const int64_t out_s1 = out.stride(1);
    launch_timed_kernel(
        "aten::mm",
        [self_ptr,
         mat2_ptr,
         out_ptr,
         m,
         n,
         k,
         self_s0,
         self_s1,
         mat2_s0,
         mat2_s1,
         out_s0,
         out_s1](at::mcpu::kernel_timing::Event* timing_event) {
          MCPU_KERNEL_TIMING_SCOPE_EVENT("mcpu::aten::mm", timing_event);
          KernelPointerMemoryGuard guard({self_ptr, mat2_ptr, out_ptr});
          raw_mm_kernel(
              self_ptr,
              mat2_ptr,
              out_ptr,
              m,
              n,
              k,
              self_s0,
              self_s1,
              mat2_s0,
              mat2_s1,
              out_s0,
              out_s1);
        });
    return out;
  }

  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_mat2 = ops::get_cpu_tensor_view_if_needed(mat2);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  launch_kernel(out, [cpu_self, cpu_mat2, cpu_out]() mutable {
    at::mm_out(cpu_out, cpu_self, cpu_mat2);
  });
  return out;
}

at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
  auto out = at::empty(
      {self.size(0), mat2.size(1)},
      self.options().dtype(at::result_type(self, mat2)));
  mm_out(self, mat2, out);
  return out;
}

at::Tensor& addmm_out(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  TORCH_CHECK(mat1.dim() == 2, "addmm: mat1 must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "addmm: mat2 must be a matrix");
  TORCH_CHECK(
      mat1.size(1) == mat2.size(0),
      "addmm: mat1.size(1) must match mat2.size(0)");
  ops::check_out_sizes(
      "aten::addmm.out", out, {mat1.size(0), mat2.size(1)});

  const bool self_is_1d = self.dim() == 1;
  const bool self_shape_ok = self_is_1d
      ? self.size(0) == mat2.size(1)
      : self.dim() == 2 && self.size(0) == mat1.size(0) &&
          self.size(1) == mat2.size(1);
  if (self_shape_ok && is_float_mcpu_or_cpu(self) &&
      is_float_mcpu_or_cpu(mat1) && is_float_mcpu_or_cpu(mat2) &&
      is_float_mcpu_or_cpu(out)) {
    const float* self_ptr = self.const_data_ptr<float>();
    const float* mat1_ptr = mat1.const_data_ptr<float>();
    const float* mat2_ptr = mat2.const_data_ptr<float>();
    float* out_ptr = out.mutable_data_ptr<float>();
    const int64_t m = mat1.size(0);
    const int64_t n = mat2.size(1);
    const int64_t k = mat1.size(1);
    const int64_t self_s0 = self.stride(0);
    const int64_t self_s1 = self_is_1d ? 0 : self.stride(1);
    const int64_t mat1_s0 = mat1.stride(0);
    const int64_t mat1_s1 = mat1.stride(1);
    const int64_t mat2_s0 = mat2.stride(0);
    const int64_t mat2_s1 = mat2.stride(1);
    const int64_t out_s0 = out.stride(0);
    const int64_t out_s1 = out.stride(1);
    const float beta_value = beta.to<float>();
    const float alpha_value = alpha.to<float>();
    launch_timed_kernel(
        "aten::addmm",
        [self_ptr,
         mat1_ptr,
         mat2_ptr,
         out_ptr,
         m,
         n,
         k,
         self_s0,
         self_s1,
         mat1_s0,
         mat1_s1,
         mat2_s0,
         mat2_s1,
         out_s0,
         out_s1,
         self_is_1d,
         beta_value,
         alpha_value](at::mcpu::kernel_timing::Event* timing_event) {
          MCPU_KERNEL_TIMING_SCOPE_EVENT("mcpu::aten::addmm", timing_event);
          KernelPointerMemoryGuard guard(
              {self_ptr, mat1_ptr, mat2_ptr, out_ptr});
          raw_addmm_kernel(
              self_ptr,
              mat1_ptr,
              mat2_ptr,
              out_ptr,
              m,
              n,
              k,
              self_s0,
              self_s1,
              mat1_s0,
              mat1_s1,
              mat2_s0,
              mat2_s1,
              out_s0,
              out_s1,
              self_is_1d,
              beta_value,
              alpha_value);
        });
    return out;
  }

  auto cpu_self = ops::get_cpu_tensor_view_if_needed(self);
  auto cpu_mat1 = ops::get_cpu_view_from_mcpu_tensor(mat1);
  auto cpu_mat2 = ops::get_cpu_tensor_view_if_needed(mat2);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  launch_kernel(
      out,
      [beta,
       alpha,
       cpu_self,
       cpu_mat1,
       cpu_mat2,
       cpu_out]() mutable {
        at::addmm_out(cpu_out, cpu_self, cpu_mat1, cpu_mat2, beta, alpha);
      });
  return out;
}

at::Tensor addmm(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  auto out = at::empty(
      {mat1.size(0), mat2.size(1)},
      mat1.options().dtype(at::result_type(mat1, mat2)));
  addmm_out(self, mat1, mat2, beta, alpha, out);
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("mm", &mm);
  m.impl("mm.out", &mm_out);
  m.impl("addmm", &addmm);
  m.impl("addmm.out", &addmm_out);
}

} // namespace at::mcpu
