#include "Common.h"

#include <ATen/ops/index_copy.h>
#include <ATen/ops/index_select.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

std::vector<int64_t> index_select_sizes(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index) {
  const auto wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
  auto sizes = self.sizes().vec();
  if (sizes.empty()) {
    return sizes;
  }
  sizes[wrapped_dim] = index.numel();
  return sizes;
}

at::Tensor index_select(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index) {
  auto sizes = index_select_sizes(self, dim, index);
  auto out = at::empty(sizes, self.options());

  at::native::mcpu::MemoryGuard guard(self, index, out);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_index = ops::get_cpu_tensor_view_if_needed(index);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::index_select_out(cpu_out, cpu_self, dim, cpu_index);
  return out;
}

at::Tensor& index_select_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    at::Tensor& out) {
  auto sizes = index_select_sizes(self, dim, index);
  ops::check_out_sizes("aten::index_select.out", out, sizes);

  at::native::mcpu::MemoryGuard guard(self, index, out);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_index = ops::get_cpu_tensor_view_if_needed(index);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::index_select_out(cpu_out, cpu_self, dim, cpu_index);
  return out;
}

at::Tensor& index_copy_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source) {
  at::native::mcpu::MemoryGuard guard(self, index, source);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_index = ops::get_cpu_tensor_view_if_needed(index);
  auto cpu_source = ops::get_cpu_tensor_view_if_needed(source);
  at::_ops::index_copy_::call(cpu_self, dim, cpu_index, cpu_source);
  return self;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("index_select", &index_select);
  m.impl("index_select.out", &index_select_out);
  m.impl("index_copy_", &index_copy_);
}

} // namespace at::mcpu
