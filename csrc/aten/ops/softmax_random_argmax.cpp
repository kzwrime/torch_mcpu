#include "Common.h"
#include "runtime/McpuKernelLaunch.h"
#include "runtime/OpenRegGenerator.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/DistributionTemplates.h>
#include <ATen/ops/_softmax.h>
#include <ATen/ops/argmax.h>
#include <ATen/ops/exponential.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

at::CPUGeneratorImpl* get_mcpu_exponential_generator(
    const at::Tensor& self,
    const std::optional<at::Generator>& generator) {
  if (generator.has_value() && generator->defined()) {
    if (generator->device().type() == c10::DeviceType::CPU) {
      return generator->get<at::CPUGeneratorImpl>();
    }
    TORCH_CHECK(
        generator->device().type() == c10::DeviceType::PrivateUse1,
        "Expected an mcpu generator for mcpu exponential_, but got ",
        generator->device());
    return generator->get<c10::mcpu::McpuGeneratorImpl>();
  }

  const auto device_index = self.device().index();
  const at::Generator& default_generator =
      c10::mcpu::getDefaultMcpuGenerator(device_index);
  return default_generator.get<c10::mcpu::McpuGeneratorImpl>();
}

void exponential_mcpu_impl(
    const at::Tensor& self,
    double lambd,
    const std::optional<at::Generator>& generator) {
  TORCH_CHECK(
      lambd > 0.0,
      "exponential_ expects lambda > 0.0, but found lambda=",
      lambd);
  if (self.numel() == 0) {
    return;
  }

  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto* gen = get_mcpu_exponential_generator(self, generator);
  auto iter = at::TensorIterator::borrowing_nullary_op(cpu_self);
  at::native::templates::cpu::exponential_kernel(iter, lambd, gen);
}

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
        exponential_mcpu_impl(self, lambd, generator);
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
