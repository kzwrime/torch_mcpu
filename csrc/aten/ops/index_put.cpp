#include "Common.h"

#include <ATen/ops/_index_put_impl.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

at::Tensor& _index_put_impl_(
    at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate,
    bool unsafe) {
  at::native::mcpu::MemoryGuard guard(self, values, c10::IValue(indices));
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_values = ops::get_cpu_tensor_view_if_needed(values);
  auto cpu_indices = ops::to_cpu_indices(indices);
  at::_index_put_impl_(cpu_self, cpu_indices, cpu_values, accumulate, unsafe);
  return self;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_index_put_impl_", &_index_put_impl_);
}

} // namespace at::mcpu
