#include "Common.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ops/_softmax.h>
#include <ATen/ops/argmax.h>
#include <ATen/ops/exponential.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

at::Tensor& _softmax_out(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float,
    at::Tensor& out) {
  ops::check_out_sizes("aten::_softmax.out", out, self.sizes());

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::_softmax.out", ([ self, out, dim, half_to_float ]), {
        KernelMemoryGuard guard(self, out);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::_softmax_out(cpu_out, cpu_self, dim, half_to_float);
      });
  return out;
}

at::Tensor& exponential_(
    at::Tensor& self,
    double lambd,
    std::optional<at::Generator> generator) {
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::exponential_", ([ self, lambd, generator ]), {
        KernelMemoryGuard guard(self);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        at::_ops::exponential_::call(cpu_self, lambd, generator);
      });
  return self;
}

at::Tensor& argmax_out(
    const at::Tensor& self,
    std::optional<int64_t> dim,
    bool keepdim,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::argmax_out(meta_out, meta_self, dim, keepdim);
  ops::check_out_sizes("aten::argmax.out", out, meta_out);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::argmax.out", ([ self, out, dim, keepdim ]), {
        KernelMemoryGuard guard(self, out);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::argmax_out(cpu_out, cpu_self, dim, keepdim);
      });
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_softmax.out", &_softmax_out);
  m.impl("argmax.out", &argmax_out);
  m.impl("exponential_", &exponential_);
}

} // namespace at::mcpu
