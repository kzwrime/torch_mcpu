#include "Common.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ops/cumsum.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

at::ScalarType cumsum_result_dtype(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype) {
  if (dtype.has_value()) {
    return *dtype;
  }
  auto self_dtype = self.scalar_type();
  if (c10::isIntegralType(self_dtype, /*includeBool=*/true)) {
    return at::kLong;
  }
  return self_dtype;
}

at::Tensor cumsum(
    const at::Tensor& self,
    int64_t dim,
    std::optional<at::ScalarType> dtype) {
  auto result_dtype = cumsum_result_dtype(self, dtype);
  auto out = at::empty(self.sizes(), self.options().dtype(result_dtype));
  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::cumsum", ([ self, out, dim, dtype ]), {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::cumsum_out(cpu_out, cpu_self, dim, dtype);
  });
  return out;
}

at::Tensor& cumsum_out(
    const at::Tensor& self,
    int64_t dim,
    std::optional<at::ScalarType> dtype,
    at::Tensor& out) {
  ops::check_out_sizes("aten::cumsum.out", out, self.sizes());

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::cumsum.out", ([ self, out, dim, dtype ]), {
        KernelMemoryGuard guard(self, out);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::cumsum_out(cpu_out, cpu_self, dim, dtype);
      });
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("cumsum", &cumsum);
  m.impl("cumsum.out", &cumsum_out);
}

} // namespace at::mcpu
