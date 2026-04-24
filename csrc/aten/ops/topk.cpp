#include "Common.h"

#include <ATen/ops/topk.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

::std::tuple<at::Tensor&, at::Tensor&> topk_values(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    at::Tensor& values,
    at::Tensor& indices) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_values =
      at::empty({0}, values.options().device(c10::DeviceType::Meta));
  auto meta_indices =
      at::empty({0}, indices.options().device(c10::DeviceType::Meta));
  at::topk_out(meta_values, meta_indices, meta_self, k, dim, largest, sorted);
  ops::check_out_sizes("aten::topk.values", "values", values, meta_values);
  ops::check_out_sizes("aten::topk.values", "indices", indices, meta_indices);

  at::native::mcpu::MemoryGuard guard(self, values, indices);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_values = ops::get_cpu_view_from_mcpu_tensor(values);
  auto cpu_indices = ops::get_cpu_view_from_mcpu_tensor(indices);
  at::topk_out(cpu_values, cpu_indices, cpu_self, k, dim, largest, sorted);
  return std::forward_as_tuple(values, indices);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("topk.values", &topk_values);
}

} // namespace at::mcpu
