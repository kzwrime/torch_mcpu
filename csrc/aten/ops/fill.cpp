#include "Common.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ops/fill.h>
#include <torch/library.h>

#include <utility>

namespace at::mcpu {
namespace {

at::Tensor& fill__scalar(at::Tensor& self, const at::Scalar& value) {
  auto self_spec = ops::make_cpu_view_spec(self);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::fill_.Scalar",
      ([ self_spec = std::move(self_spec), value ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data});
        auto cpu_self = ops::cpu_view_from_spec(self_spec);
        at::fill_(cpu_self, value);
      });
  return self;
}

at::Tensor& fill__tensor(at::Tensor& self, const at::Tensor& value) {
  auto self_spec = ops::make_cpu_view_spec(self);
  auto value_spec = ops::make_cpu_view_spec(value);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::fill_.Tensor",
      ([
        self_spec = std::move(self_spec),
        value_spec = std::move(value_spec)
      ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data, value_spec.data});
        auto cpu_self = ops::cpu_view_from_spec(self_spec);
        auto cpu_value = ops::cpu_view_from_spec(value_spec);
        at::fill_(cpu_self, cpu_value);
      });
  return self;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("fill_.Scalar", &fill__scalar);
  m.impl("fill_.Tensor", &fill__tensor);
}

} // namespace at::mcpu
