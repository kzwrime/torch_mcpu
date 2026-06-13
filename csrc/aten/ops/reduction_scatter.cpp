#include "Common.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ops/max.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/scatter_add.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

::std::tuple<at::Tensor&, at::Tensor&> max_dim_max(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& values,
    at::Tensor& indices) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_values =
      at::empty({0}, values.options().device(c10::DeviceType::Meta));
  auto meta_indices =
      at::empty({0}, indices.options().device(c10::DeviceType::Meta));
  at::max_out(meta_values, meta_indices, meta_self, dim, keepdim);
  ops::check_out_sizes("aten::max.dim_max", "max", values, meta_values);
  ops::check_out_sizes(
      "aten::max.dim_max", "max_values", indices, meta_indices);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::max.dim_max", ([ self, values, indices, dim, keepdim ]), {
        KernelMemoryGuard guard(self, values, indices);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_values = ops::get_cpu_view_from_mcpu_tensor(values);
        auto cpu_indices = ops::get_cpu_view_from_mcpu_tensor(indices);
        at::max_out(cpu_values, cpu_indices, cpu_self, dim, keepdim);
      });
  return std::forward_as_tuple(values, indices);
}

at::Tensor& mean_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype,
    at::Tensor& out) {
  auto expected_sizes = ops::reduction_sizes(self.sizes(), dim, keepdim);
  ops::check_out_sizes("aten::mean.out", out, expected_sizes);

  std::optional<std::vector<int64_t>> dim_vec;
  if (dim.has_value()) {
    dim_vec = std::vector<int64_t>(dim->begin(), dim->end());
  }

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::mean.out", ([ self, out, dim_vec, keepdim, dtype ]), {
        KernelMemoryGuard guard(self, out);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        auto cpu_dim = dim_vec.has_value() ? at::OptionalIntArrayRef(*dim_vec)
                                           : at::OptionalIntArrayRef();
        at::mean_out(cpu_out, cpu_self, cpu_dim, keepdim, dtype);
      });
  return out;
}

at::Tensor& scatter_add_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_index = ops::to_meta_tensor(index);
  auto meta_src = ops::to_meta_tensor(src);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::scatter_add_out(meta_out, meta_self, dim, meta_index, meta_src);
  ops::check_out_sizes("aten::scatter_add.out", out, meta_out);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::scatter_add.out", ([ self, index, src, out, dim ]), {
        KernelMemoryGuard guard(self, index, src, out);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_index = ops::get_cpu_tensor_view_if_needed(index);
        auto cpu_src = ops::get_cpu_tensor_view_if_needed(src);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::scatter_add_out(cpu_out, cpu_self, dim, cpu_index, cpu_src);
      });
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("max.dim_max", &max_dim_max);
  m.impl("mean.out", &mean_out);
  m.impl("scatter_add.out", &scatter_add_out);
}

} // namespace at::mcpu
