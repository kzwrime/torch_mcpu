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

#include <vector>

namespace at::mcpu {
namespace {

using ops::cpu_view_from_spec;
using ops::make_cpu_view_spec;
using ops::TensorViewSpec;

at::CPUGeneratorImpl* get_mcpu_exponential_generator(
    c10::DeviceIndex device_index,
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

  const at::Generator& default_generator =
      c10::mcpu::getDefaultMcpuGenerator(device_index);
  return default_generator.get<c10::mcpu::McpuGeneratorImpl>();
}

void exponential_mcpu_impl(
    const at::Tensor& cpu_self,
    double lambd,
    at::CPUGeneratorImpl* gen) {
  TORCH_CHECK(
      lambd > 0.0,
      "exponential_ expects lambda > 0.0, but found lambda=",
      lambd);
  if (cpu_self.numel() == 0) {
    return;
  }

  auto iter = at::TensorIterator::borrowing_nullary_op(cpu_self);
  at::native::templates::cpu::exponential_kernel(iter, lambd, gen);
}

at::Tensor& _softmax_out(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float,
    at::Tensor& out) {
  ops::check_out_sizes("aten::_softmax.out", out, self.sizes());

  auto self_spec = make_cpu_view_spec(self);
  auto out_spec = make_cpu_view_spec(out);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::_softmax.out",
      ([
        self_spec = std::move(self_spec),
        out_spec = std::move(out_spec),
        dim,
        half_to_float
      ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data, out_spec.data});
        auto cpu_self = cpu_view_from_spec(self_spec);
        auto cpu_out = cpu_view_from_spec(out_spec);
        at::_softmax_out(cpu_out, cpu_self, dim, half_to_float);
      });
  return out;
}

at::Tensor& exponential_(
    at::Tensor& self,
    double lambd,
    std::optional<at::Generator> generator) {
  auto self_spec = make_cpu_view_spec(self);
  auto device_index = self.device().index();

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::exponential_",
      ([
        self_spec = std::move(self_spec),
        lambd,
        generator = std::move(generator),
        device_index
      ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data});
        auto cpu_self = cpu_view_from_spec(self_spec);
        auto* gen = get_mcpu_exponential_generator(device_index, generator);
        exponential_mcpu_impl(cpu_self, lambd, gen);
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

  auto self_spec = make_cpu_view_spec(self);
  auto out_spec = make_cpu_view_spec(out);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::argmax.out",
      ([
        self_spec = std::move(self_spec),
        out_spec = std::move(out_spec),
        dim,
        keepdim
      ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data, out_spec.data});
        auto cpu_self = cpu_view_from_spec(self_spec);
        auto cpu_out = cpu_view_from_spec(out_spec);
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
