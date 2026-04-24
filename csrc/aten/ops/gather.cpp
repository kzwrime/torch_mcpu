#include "Common.h"

#include <ATen/ops/gather.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

at::Tensor& gather_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_index = ops::to_meta_tensor(index);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::gather_out(meta_out, meta_self, dim, meta_index, sparse_grad);
  ops::check_out_sizes("aten::gather.out", out, meta_out);

  at::native::mcpu::MemoryGuard guard(self, index, out);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_index = ops::get_cpu_tensor_view_if_needed(index);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::gather_out(cpu_out, cpu_self, dim, cpu_index, sparse_grad);
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("gather.out", &gather_out);
}

} // namespace at::mcpu
