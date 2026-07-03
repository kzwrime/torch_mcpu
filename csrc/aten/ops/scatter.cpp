#include "Common.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ops/scatter.h>
#include <torch/library.h>

#include <memory>

namespace at::mcpu {
namespace {

using ops::cpu_view_from_spec;
using ops::make_cpu_view_spec;
using ops::TensorViewSpec;

struct ScatterSrcOutArgs {
  TensorViewSpec self;
  TensorViewSpec index;
  TensorViewSpec src;
  TensorViewSpec out;
  int64_t dim = 0;
};

at::Tensor& scatter_src_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_index = ops::to_meta_tensor(index);
  auto meta_src = ops::to_meta_tensor(src);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::scatter_out(meta_out, meta_self, dim, meta_index, meta_src);
  ops::check_out_sizes("aten::scatter.src_out", out, meta_out);

  auto args = std::make_unique<ScatterSrcOutArgs>(ScatterSrcOutArgs{
      make_cpu_view_spec(self),
      make_cpu_view_spec(index),
      make_cpu_view_spec(src),
      make_cpu_view_spec(out),
      dim});

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::scatter.src_out", ([args = std::move(args)]), {
        KernelPointerMemoryGuard guard(
            {args->self.data,
             args->index.data,
             args->src.data,
             args->out.data});
        auto cpu_self = cpu_view_from_spec(args->self);
        auto cpu_index = cpu_view_from_spec(args->index);
        auto cpu_src = cpu_view_from_spec(args->src);
        auto cpu_out = cpu_view_from_spec(args->out);
        at::scatter_out(cpu_out, cpu_self, args->dim, cpu_index, cpu_src);
      });
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("scatter.src_out", &scatter_src_out);
}

} // namespace at::mcpu
