#include "Common.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ops/index_fill_cpu_dispatch.h>
#include <torch/library.h>

#include <memory>

namespace at::mcpu {
namespace {

using ops::cpu_view_from_spec;
using ops::make_cpu_view_spec;
using ops::TensorViewSpec;

struct IndexFillScalarArgs {
  TensorViewSpec self;
  TensorViewSpec index;
  int64_t dim = 0;
  at::Scalar value;
};

at::Tensor& index_fill__int_Scalar(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Scalar& value) {
  auto args = std::make_unique<IndexFillScalarArgs>(IndexFillScalarArgs{
      make_cpu_view_spec(self), make_cpu_view_spec(index), dim, value});

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::index_fill_.int_Scalar", ([args = std::move(args)]), {
        KernelPointerMemoryGuard guard({args->self.data, args->index.data});
        auto cpu_self = cpu_view_from_spec(args->self);
        auto cpu_index = cpu_view_from_spec(args->index);
        at::cpu::index_fill_(cpu_self, args->dim, cpu_index, args->value);
      });
  return self;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("index_fill_.int_Scalar", &index_fill__int_Scalar);
}

} // namespace at::mcpu
