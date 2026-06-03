// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <torch/library.h>

#include <runtime/McpuKernelLaunch.h>

namespace {

template <typename scalar_t>
void axpby_kernel_typed(
    scalar_t* out_ptr,
    const scalar_t* x_ptr,
    int64_t numel,
    double alpha,
    double beta) {
  for (int64_t i = 0; i < numel; ++i) {
    out_ptr[i] =
        static_cast<scalar_t>(static_cast<double>(x_ptr[i]) * alpha + beta);
  }
}

void external_axpby_impl(
    at::Tensor& out,
    const at::Tensor& x,
    double alpha,
    double beta) {
  TORCH_CHECK(
      out.device().type() == c10::DeviceType::PrivateUse1,
      "out must be an mcpu tensor");
  TORCH_CHECK(
      x.device().type() == c10::DeviceType::PrivateUse1,
      "x must be an mcpu tensor");
  TORCH_CHECK(
      out.scalar_type() == x.scalar_type(),
      "out and x must have the same dtype");
  TORCH_CHECK(out.numel() == x.numel(), "out and x must have the same numel");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

  const int64_t numel = x.numel();

  switch (x.scalar_type()) {
    case at::kFloat: {
      float* out_ptr = out.data_ptr<float>();
      const float* x_ptr = x.data_ptr<float>();
      at::mcpu::launch_kernel(
          out, [out, x, out_ptr, x_ptr, numel, alpha, beta]() mutable {
            at::mcpu::KernelMemoryGuard guard(out, x);
            axpby_kernel_typed<float>(out_ptr, x_ptr, numel, alpha, beta);
          });
      break;
    }
    case at::kBFloat16: {
      at::BFloat16* out_ptr = out.data_ptr<at::BFloat16>();
      const at::BFloat16* x_ptr = x.data_ptr<at::BFloat16>();
      at::mcpu::launch_kernel(
          out, [out, x, out_ptr, x_ptr, numel, alpha, beta]() mutable {
            at::mcpu::KernelMemoryGuard guard(out, x);
            axpby_kernel_typed<at::BFloat16>(
                out_ptr, x_ptr, numel, alpha, beta);
          });
      break;
    }
    default:
      TORCH_CHECK(false, "external_axpby only supports float32 and bfloat16");
  }
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu_external, m) {
  m.def(
      "axpby("
      "Tensor(a!) out, "
      "Tensor x, "
      "float alpha, "
      "float beta"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu_external, PrivateUse1, m) {
  m.impl("axpby", &external_axpby_impl);
}
