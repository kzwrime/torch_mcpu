#include "Common.h"

#include <ATen/ops/index.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

at::Tensor index_Tensor(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_indices = ops::to_meta_indices(indices);
  auto meta_out = at::empty({0}, self.options().device(c10::DeviceType::Meta));
  at::index_out(meta_out, meta_self, meta_indices);

  auto out = ops::empty_mcpu_from_meta(meta_out, self.options());
  at::native::mcpu::MemoryGuard guard(self, out, c10::IValue(indices));
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  auto cpu_indices = ops::to_cpu_indices(indices);
  at::index_out(cpu_out, cpu_self, cpu_indices);
  return out;
}

at::Tensor& index_Tensor_out(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_indices = ops::to_meta_indices(indices);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::index_out(meta_out, meta_self, meta_indices);
  ops::check_out_sizes("aten::index.Tensor_out", out, meta_out);

  at::native::mcpu::MemoryGuard guard(self, out, c10::IValue(indices));
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  auto cpu_indices = ops::to_cpu_indices(indices);
  at::index_out(cpu_out, cpu_self, cpu_indices);
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("index.Tensor", &index_Tensor);
  m.impl("index.Tensor_out", &index_Tensor_out);
}

} // namespace at::mcpu
