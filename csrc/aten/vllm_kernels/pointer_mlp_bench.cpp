// SPDX-License-Identifier: Apache-2.0
//
// Pointer-only kernels for stream scheduling overhead experiments.
//
// The Python-facing ops accept tensors to select the stream, but the submitted
// OpenReg tasks intentionally store only raw pointers, dimensions, and strides.
// These ops are benchmark-only: callers must keep every tensor alive and avoid
// reusing its storage until the stream is synchronized.

#include "common.h"
#include "runtime/McpuKernelLaunch.h"
#include "runtime/OpenRegException.h"
#include "runtime/OpenRegStream.h"

#include <include/openreg.h>

#include <chrono>
#include <cmath>

namespace {

void pointer_mm_kernel(
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

void pointer_add_kernel(
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
    bool other_is_1d) {
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      const int64_t other_offset =
          other_is_1d ? j * other_s0 : i * other_s0 + j * other_s1;
      out[i * out_s0 + j * out_s1] =
          self[i * self_s0 + j * self_s1] + other[other_offset];
    }
  }
}

void pointer_sigmoid_kernel(const float* self, float* out, int64_t numel) {
  for (int64_t i = 0; i < numel; ++i) {
    out[i] = 1.0f / (1.0f + std::exp(-self[i]));
  }
}

int64_t pointer_now_ns() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             clock::now().time_since_epoch())
      .count();
}

void pointer_write_time_kernel(int64_t* out) {
  *out = pointer_now_ns();
}

void check_float_tensor(const at::Tensor& tensor, const char* name) {
  if (!vllm_mcpu::CHECK_INPUTS) {
    return;
  }
  TORCH_CHECK(
      tensor.scalar_type() == at::kFloat,
      name,
      " must be float32, got ",
      tensor.scalar_type());
  TORCH_CHECK(
      tensor.device().type() == c10::DeviceType::PrivateUse1,
      name,
      " must be an mcpu tensor");
}

void pointer_mm_out_impl(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  VLLM_MCPU_CHECK_DIM(self, 2, "self");
  VLLM_MCPU_CHECK_DIM(mat2, 2, "mat2");
  VLLM_MCPU_CHECK_DIM(out, 2, "out");
  check_float_tensor(self, "self");
  check_float_tensor(mat2, "mat2");
  check_float_tensor(out, "out");
  VLLM_MCPU_CHECK(self.size(1) == mat2.size(0), "mm shape mismatch");
  VLLM_MCPU_CHECK(out.size(0) == self.size(0), "out rows mismatch");
  VLLM_MCPU_CHECK(out.size(1) == mat2.size(1), "out cols mismatch");

  const float* self_ptr = self.const_data_ptr<float>();
  const float* mat2_ptr = mat2.const_data_ptr<float>();
  float* out_ptr = out.mutable_data_ptr<float>();
  auto stream = c10::mcpu::getCurrentMcpuStream();
  MCPU_CHECK(orLaunchKernel(
      stream,
      pointer_mm_kernel,
      self_ptr,
      mat2_ptr,
      out_ptr,
      self.size(0),
      mat2.size(1),
      self.size(1),
      self.stride(0),
      self.stride(1),
      mat2.stride(0),
      mat2.stride(1),
      out.stride(0),
      out.stride(1)));
}

void pointer_add_out_impl(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  VLLM_MCPU_CHECK_DIM(self, 2, "self");
  VLLM_MCPU_CHECK_DIM(out, 2, "out");
  check_float_tensor(self, "self");
  check_float_tensor(other, "other");
  check_float_tensor(out, "out");
  VLLM_MCPU_CHECK(out.sizes().equals(self.sizes()), "out shape mismatch");

  const bool other_is_1d = other.dim() == 1;
  if (other_is_1d) {
    VLLM_MCPU_CHECK(other.size(0) == self.size(1), "bias shape mismatch");
  } else {
    VLLM_MCPU_CHECK_DIM(other, 2, "other");
    VLLM_MCPU_CHECK(other.sizes().equals(self.sizes()), "other shape mismatch");
  }

  const float* self_ptr = self.const_data_ptr<float>();
  const float* other_ptr = other.const_data_ptr<float>();
  float* out_ptr = out.mutable_data_ptr<float>();
  auto stream = c10::mcpu::getCurrentMcpuStream();
  MCPU_CHECK(orLaunchKernel(
      stream,
      pointer_add_kernel,
      self_ptr,
      other_ptr,
      out_ptr,
      self.size(0),
      self.size(1),
      self.stride(0),
      self.stride(1),
      other.stride(0),
      other_is_1d ? 0 : other.stride(1),
      out.stride(0),
      out.stride(1),
      other_is_1d));
}

void pointer_sigmoid_out_impl(const at::Tensor& self, at::Tensor& out) {
  check_float_tensor(self, "self");
  check_float_tensor(out, "out");
  VLLM_MCPU_CHECK(self.sizes().equals(out.sizes()), "out shape mismatch");
  VLLM_MCPU_CHECK(self.is_contiguous(), "self must be contiguous");
  VLLM_MCPU_CHECK(out.is_contiguous(), "out must be contiguous");

  auto stream = c10::mcpu::getCurrentMcpuStream();
  MCPU_CHECK(orLaunchKernel(
      stream,
      pointer_sigmoid_kernel,
      self.const_data_ptr<float>(),
      out.mutable_data_ptr<float>(),
      self.numel()));
}

void pointer_write_time_impl(at::Tensor& out) {
  if (vllm_mcpu::CHECK_INPUTS) {
    TORCH_CHECK(out.device().type() == c10::DeviceType::PrivateUse1);
    TORCH_CHECK(out.scalar_type() == at::kLong);
    TORCH_CHECK(out.numel() >= 1);
    TORCH_CHECK(out.is_contiguous());
  }
  auto stream = c10::mcpu::getCurrentMcpuStream();
  MCPU_CHECK(orLaunchKernel(
      stream, pointer_write_time_kernel, out.mutable_data_ptr<int64_t>()));
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def("pointer_mm_out(Tensor self, Tensor mat2, Tensor(a!) out) -> ()");
  m.def("pointer_add_out(Tensor self, Tensor other, Tensor(a!) out) -> ()");
  m.def("pointer_sigmoid_out(Tensor self, Tensor(a!) out) -> ()");
  m.def("pointer_write_time(Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("pointer_mm_out", &pointer_mm_out_impl);
  m.impl("pointer_add_out", &pointer_add_out_impl);
  m.impl("pointer_sigmoid_out", &pointer_sigmoid_out_impl);
  m.impl("pointer_write_time", &pointer_write_time_impl);
}
