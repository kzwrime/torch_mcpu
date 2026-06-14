#include "Common.h"

#include <ATen/ops/empty_like.h>
#include <ATen/ops/uniform.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

at::Tensor& uniform_(
    at::Tensor& self,
    double from,
    double to,
    std::optional<at::Generator> generator) {
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::uniform_", ([ self, from, to, generator ]), {
        KernelMemoryGuard guard(self);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        at::_ops::uniform_::call(cpu_self, from, to, generator);
      });
  return self;
}

at::Tensor& uniform_out(
    const at::Tensor& self,
    double from,
    double to,
    std::optional<at::Generator> generator,
    at::Tensor& out) {
  ops::check_out_sizes("aten::uniform.out", out, self.sizes());

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::uniform.out", ([ out, from, to, generator ]), {
        KernelMemoryGuard guard(out);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::_ops::uniform_::call(cpu_out, from, to, generator);
      });
  return out;
}

at::Tensor uniform(
    const at::Tensor& self,
    double from,
    double to,
    std::optional<at::Generator> generator) {
  auto out = at::empty_like(self, self.options(), at::MemoryFormat::Preserve);
  uniform_out(self, from, to, generator, out);
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("uniform", &uniform);
  m.impl("uniform.out", &uniform_out);
  m.impl("uniform_", &uniform_);
}

} // namespace at::mcpu
