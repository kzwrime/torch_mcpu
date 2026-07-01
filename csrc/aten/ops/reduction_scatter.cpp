#include "Common.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ops/amax.h>
#include <ATen/ops/amin.h>
#include <ATen/ops/max.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/min.h>
#include <ATen/ops/scatter_add.h>
#include <ATen/ops/sum.h>
#include <torch/library.h>

#include <memory>

namespace at::mcpu {
namespace {

using ops::cpu_view_from_spec;
using ops::make_cpu_view_spec;
using ops::TensorViewSpec;

struct ScatterAddArgs {
  TensorViewSpec self;
  TensorViewSpec index;
  TensorViewSpec src;
  TensorViewSpec out;
  int64_t dim = 0;
};

::std::tuple<at::Tensor&, at::Tensor&> max_dim_max(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& values,
    at::Tensor& indices);

::std::tuple<at::Tensor&, at::Tensor&> min_dim_min(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& values,
    at::Tensor& indices);

at::Tensor& amax_out(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    at::Tensor& out);

at::Tensor& amin_out(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    at::Tensor& out);

std::optional<std::vector<int64_t>> copy_optional_dim(
    at::OptionalIntArrayRef dim) {
  if (!dim.has_value()) {
    return std::nullopt;
  }
  return std::vector<int64_t>(dim->begin(), dim->end());
}

at::Tensor max_unary(const at::Tensor& self) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::max(meta_self);
  auto out = ops::empty_mcpu_from_meta(meta_out, meta_out.options());
  auto self_spec = make_cpu_view_spec(self);
  auto out_spec = make_cpu_view_spec(out);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::max",
      ([ self_spec = std::move(self_spec), out_spec = std::move(out_spec) ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data, out_spec.data});
        auto cpu_self = cpu_view_from_spec(self_spec);
        auto cpu_out = cpu_view_from_spec(out_spec);
        at::max_out(cpu_out, cpu_self);
      });
  return out;
}

at::Tensor& max_unary_out(const at::Tensor& self, at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::max_out(meta_out, meta_self);
  ops::check_out_sizes("aten::max.unary_out", out, meta_out);

  auto self_spec = make_cpu_view_spec(self);
  auto out_spec = make_cpu_view_spec(out);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::max.unary_out",
      ([ self_spec = std::move(self_spec), out_spec = std::move(out_spec) ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data, out_spec.data});
        auto cpu_self = cpu_view_from_spec(self_spec);
        auto cpu_out = cpu_view_from_spec(out_spec);
        at::max_out(cpu_out, cpu_self);
      });
  return out;
}

::std::tuple<at::Tensor, at::Tensor> max_dim(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto meta_self = ops::to_meta_tensor(self);
  auto [meta_values, meta_indices] = at::max(meta_self, dim, keepdim);
  auto values = ops::empty_mcpu_from_meta(meta_values, meta_values.options());
  auto indices = ops::empty_mcpu_from_meta(meta_indices, meta_indices.options());
  max_dim_max(self, dim, keepdim, values, indices);
  return std::make_tuple(values, indices);
}

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

  auto self_spec = make_cpu_view_spec(self);
  auto values_spec = make_cpu_view_spec(values);
  auto indices_spec = make_cpu_view_spec(indices);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::max.dim_max",
      ([
        self_spec = std::move(self_spec),
        values_spec = std::move(values_spec),
        indices_spec = std::move(indices_spec),
        dim,
        keepdim
      ]),
      {
        KernelPointerMemoryGuard guard(
            {self_spec.data, values_spec.data, indices_spec.data});
        auto cpu_self = cpu_view_from_spec(self_spec);
        auto cpu_values = cpu_view_from_spec(values_spec);
        auto cpu_indices = cpu_view_from_spec(indices_spec);
        at::max_out(cpu_values, cpu_indices, cpu_self, dim, keepdim);
      });
  return std::forward_as_tuple(values, indices);
}

at::Tensor min_unary(const at::Tensor& self) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::min(meta_self);
  auto out = ops::empty_mcpu_from_meta(meta_out, meta_out.options());
  auto self_spec = make_cpu_view_spec(self);
  auto out_spec = make_cpu_view_spec(out);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::min",
      ([ self_spec = std::move(self_spec), out_spec = std::move(out_spec) ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data, out_spec.data});
        auto cpu_self = cpu_view_from_spec(self_spec);
        auto cpu_out = cpu_view_from_spec(out_spec);
        at::min_out(cpu_out, cpu_self);
      });
  return out;
}

at::Tensor& min_unary_out(const at::Tensor& self, at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::min_out(meta_out, meta_self);
  ops::check_out_sizes("aten::min.unary_out", out, meta_out);

  auto self_spec = make_cpu_view_spec(self);
  auto out_spec = make_cpu_view_spec(out);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::min.unary_out",
      ([ self_spec = std::move(self_spec), out_spec = std::move(out_spec) ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data, out_spec.data});
        auto cpu_self = cpu_view_from_spec(self_spec);
        auto cpu_out = cpu_view_from_spec(out_spec);
        at::min_out(cpu_out, cpu_self);
      });
  return out;
}

::std::tuple<at::Tensor, at::Tensor> min_dim(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto meta_self = ops::to_meta_tensor(self);
  auto [meta_values, meta_indices] = at::min(meta_self, dim, keepdim);
  auto values = ops::empty_mcpu_from_meta(meta_values, meta_values.options());
  auto indices = ops::empty_mcpu_from_meta(meta_indices, meta_indices.options());
  min_dim_min(self, dim, keepdim, values, indices);
  return std::make_tuple(values, indices);
}

::std::tuple<at::Tensor&, at::Tensor&> min_dim_min(
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
  at::min_out(meta_values, meta_indices, meta_self, dim, keepdim);
  ops::check_out_sizes("aten::min.dim_min", "min", values, meta_values);
  ops::check_out_sizes(
      "aten::min.dim_min", "min_indices", indices, meta_indices);

  auto self_spec = make_cpu_view_spec(self);
  auto values_spec = make_cpu_view_spec(values);
  auto indices_spec = make_cpu_view_spec(indices);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::min.dim_min",
      ([
        self_spec = std::move(self_spec),
        values_spec = std::move(values_spec),
        indices_spec = std::move(indices_spec),
        dim,
        keepdim
      ]),
      {
        KernelPointerMemoryGuard guard(
            {self_spec.data, values_spec.data, indices_spec.data});
        auto cpu_self = cpu_view_from_spec(self_spec);
        auto cpu_values = cpu_view_from_spec(values_spec);
        auto cpu_indices = cpu_view_from_spec(indices_spec);
        at::min_out(cpu_values, cpu_indices, cpu_self, dim, keepdim);
      });
  return std::forward_as_tuple(values, indices);
}

at::Tensor& mean_out_impl(
    const char* op_name,
    const char* kernel_name,
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::mean_out(meta_out, meta_self, dim, keepdim, dtype);
  ops::check_out_sizes(op_name, out, meta_out);

  auto dim_vec = copy_optional_dim(dim);
  auto self_spec = make_cpu_view_spec(self);
  auto out_spec = make_cpu_view_spec(out);

  at::mcpu::launch_timed_kernel(
      kernel_name,
      [
        self_spec = std::move(self_spec),
        out_spec = std::move(out_spec),
        dim_vec = std::move(dim_vec),
        keepdim,
        dtype,
        kernel_name
      ](::at::mcpu::kernel_timing::Event* timing_event) mutable {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(kernel_name, timing_event);
        KernelPointerMemoryGuard guard({self_spec.data, out_spec.data});
        auto cpu_self = cpu_view_from_spec(self_spec);
        auto cpu_out = cpu_view_from_spec(out_spec);
        auto cpu_dim = dim_vec.has_value() ? at::OptionalIntArrayRef(*dim_vec)
                                           : at::OptionalIntArrayRef();
        at::mean_out(cpu_out, cpu_self, cpu_dim, keepdim, dtype);
      });
  return out;
}

at::Tensor mean(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::mean(meta_self, dtype);
  auto out = ops::empty_mcpu_from_meta(meta_out, meta_out.options());
  mean_out_impl(
      "aten::mean.dtype_out",
      "mcpu::aten::mean",
      self,
      at::OptionalIntArrayRef(),
      false,
      dtype,
      out);
  return out;
}

at::Tensor& mean_dtype_out(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype,
    at::Tensor& out) {
  return mean_out_impl(
      "aten::mean.dtype_out",
      "mcpu::aten::mean.dtype_out",
      self,
      at::OptionalIntArrayRef(),
      false,
      dtype,
      out);
}

at::Tensor mean_dim(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::mean(meta_self, dim, keepdim, dtype);
  auto out = ops::empty_mcpu_from_meta(meta_out, meta_out.options());
  mean_out_impl(
      "aten::mean.out", "mcpu::aten::mean.dim", self, dim, keepdim, dtype, out);
  return out;
}

at::Tensor& mean_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype,
    at::Tensor& out) {
  return mean_out_impl(
      "aten::mean.out", "mcpu::aten::mean.out", self, dim, keepdim, dtype, out);
}

at::Tensor amax(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::amax(meta_self, dim, keepdim);
  auto out = ops::empty_mcpu_from_meta(meta_out, meta_out.options());
  amax_out(self, dim, keepdim, out);
  return out;
}

at::Tensor& amax_out(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::amax_out(meta_out, meta_self, dim, keepdim);
  ops::check_out_sizes("aten::amax.out", out, meta_out);

  auto dim_vec = std::vector<int64_t>(dim.begin(), dim.end());
  auto self_spec = make_cpu_view_spec(self);
  auto out_spec = make_cpu_view_spec(out);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::amax.out",
      ([
        self_spec = std::move(self_spec),
        out_spec = std::move(out_spec),
        dim_vec = std::move(dim_vec),
        keepdim
      ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data, out_spec.data});
        auto cpu_self = cpu_view_from_spec(self_spec);
        auto cpu_out = cpu_view_from_spec(out_spec);
        at::amax_out(cpu_out, cpu_self, dim_vec, keepdim);
      });
  return out;
}

at::Tensor amin(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::amin(meta_self, dim, keepdim);
  auto out = ops::empty_mcpu_from_meta(meta_out, meta_out.options());
  amin_out(self, dim, keepdim, out);
  return out;
}

at::Tensor& amin_out(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::amin_out(meta_out, meta_self, dim, keepdim);
  ops::check_out_sizes("aten::amin.out", out, meta_out);

  auto dim_vec = std::vector<int64_t>(dim.begin(), dim.end());
  auto self_spec = make_cpu_view_spec(self);
  auto out_spec = make_cpu_view_spec(out);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::amin.out",
      ([
        self_spec = std::move(self_spec),
        out_spec = std::move(out_spec),
        dim_vec = std::move(dim_vec),
        keepdim
      ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data, out_spec.data});
        auto cpu_self = cpu_view_from_spec(self_spec);
        auto cpu_out = cpu_view_from_spec(out_spec);
        at::amin_out(cpu_out, cpu_self, dim_vec, keepdim);
      });
  return out;
}

at::Tensor& sum_out_impl(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::sum_out(meta_out, meta_self, dtype);
  ops::check_out_sizes("aten::sum.out", out, meta_out);

  auto self_spec = make_cpu_view_spec(self);
  auto out_spec = make_cpu_view_spec(out);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::sum.out",
      ([ self_spec = std::move(self_spec), out_spec = std::move(out_spec), dtype ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data, out_spec.data});
        auto cpu_self = cpu_view_from_spec(self_spec);
        auto cpu_out = cpu_view_from_spec(out_spec);
        at::sum_out(cpu_out, cpu_self, dtype);
      });
  return out;
}

at::Tensor sum(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::sum(meta_self, dtype);
  auto out = ops::empty_mcpu_from_meta(meta_out, meta_out.options());
  sum_out_impl(self, dtype, out);
  return out;
}

at::Tensor& sum_out(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype,
    at::Tensor& out) {
  return sum_out_impl(self, dtype, out);
}

at::Tensor& sum_IntList_out_impl(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::sum_out(meta_out, meta_self, dim, keepdim, dtype);
  ops::check_out_sizes("aten::sum.IntList_out", out, meta_out);

  auto dim_vec = copy_optional_dim(dim);
  auto self_spec = make_cpu_view_spec(self);
  auto out_spec = make_cpu_view_spec(out);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::sum.IntList_out",
      ([
        self_spec = std::move(self_spec),
        out_spec = std::move(out_spec),
        dim_vec = std::move(dim_vec),
        keepdim,
        dtype
      ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data, out_spec.data});
        auto cpu_self = cpu_view_from_spec(self_spec);
        auto cpu_out = cpu_view_from_spec(out_spec);
        auto cpu_dim = dim_vec.has_value() ? at::OptionalIntArrayRef(*dim_vec)
                                           : at::OptionalIntArrayRef();
        at::sum_out(cpu_out, cpu_self, cpu_dim, keepdim, dtype);
      });
  return out;
}

at::Tensor sum_dim_IntList(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::sum(meta_self, dim, keepdim, dtype);
  auto out = ops::empty_mcpu_from_meta(meta_out, meta_out.options());
  sum_IntList_out_impl(self, dim, keepdim, dtype, out);
  return out;
}

at::Tensor& sum_IntList_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype,
    at::Tensor& out) {
  return sum_IntList_out_impl(self, dim, keepdim, dtype, out);
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

  auto args = std::make_unique<ScatterAddArgs>(ScatterAddArgs{
      make_cpu_view_spec(self),
      make_cpu_view_spec(index),
      make_cpu_view_spec(src),
      make_cpu_view_spec(out),
      dim});

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::scatter_add.out",
      ([ args = std::move(args) ]),
      {
        KernelPointerMemoryGuard guard(
            {args->self.data,
             args->index.data,
             args->src.data,
             args->out.data});
        auto cpu_self = cpu_view_from_spec(args->self);
        auto cpu_index = cpu_view_from_spec(args->index);
        auto cpu_src = cpu_view_from_spec(args->src);
        auto cpu_out = cpu_view_from_spec(args->out);
        at::scatter_add_out(cpu_out, cpu_self, args->dim, cpu_index, cpu_src);
      });
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("amax", &amax);
  m.impl("amax.out", &amax_out);
  m.impl("amin", &amin);
  m.impl("amin.out", &amin_out);
  m.impl("max", &max_unary);
  m.impl("max.dim", &max_dim);
  m.impl("max.dim_max", &max_dim_max);
  m.impl("max.unary_out", &max_unary_out);
  m.impl("mean", &mean);
  m.impl("mean.dim", &mean_dim);
  m.impl("mean.dtype_out", &mean_dtype_out);
  m.impl("mean.out", &mean_out);
  m.impl("min", &min_unary);
  m.impl("min.dim", &min_dim);
  m.impl("min.dim_min", &min_dim_min);
  m.impl("min.unary_out", &min_unary_out);
  m.impl("scatter_add.out", &scatter_add_out);
  m.impl("sum", &sum);
  m.impl("sum.dim_IntList", &sum_dim_IntList);
  m.impl("sum.IntList_out", &sum_IntList_out);
  m.impl("sum.out", &sum_out);
}

} // namespace at::mcpu
