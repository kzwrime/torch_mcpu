#include "Common.h"

#include <ATen/ops/cumsum.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

at::Tensor cumsum(
    const at::Tensor& self,
    int64_t dim,
    std::optional<at::ScalarType> dtype) {
  auto result_dtype = dtype.value_or(self.scalar_type());
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::empty(
      {0}, self.options().dtype(result_dtype).device(c10::DeviceType::Meta));
  at::cumsum_out(meta_out, meta_self, dim, dtype);

  auto out =
      ops::empty_mcpu_from_meta(meta_out, self.options().dtype(result_dtype));
  at::native::mcpu::MemoryGuard guard(self, out);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::cumsum_out(cpu_out, cpu_self, dim, dtype);
  return out;
}

at::Tensor& cumsum_out(
    const at::Tensor& self,
    int64_t dim,
    std::optional<at::ScalarType> dtype,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::cumsum_out(meta_out, meta_self, dim, dtype);
  ops::check_out_sizes("aten::cumsum.out", out, meta_out);

  at::native::mcpu::MemoryGuard guard(self, out);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::cumsum_out(cpu_out, cpu_self, dim, dtype);
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("cumsum", &cumsum);
  m.impl("cumsum.out", &cumsum_out);
}

} // namespace at::mcpu
