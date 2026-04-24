#include "Common.h"

#include <ATen/ops/fill.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

at::Tensor& fill__scalar(at::Tensor& self, const at::Scalar& value) {
  at::native::mcpu::MemoryGuard guard(self);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  at::fill_(cpu_self, value);
  return self;
}

at::Tensor& fill__tensor(at::Tensor& self, const at::Tensor& value) {
  at::native::mcpu::MemoryGuard guard(self, value);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_value = ops::get_cpu_tensor_view_if_needed(value);
  at::fill_(cpu_self, cpu_value);
  return self;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("fill_.Scalar", &fill__scalar);
  m.impl("fill_.Tensor", &fill__tensor);
}

} // namespace at::mcpu
