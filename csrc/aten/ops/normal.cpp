#include "Common.h"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/normal.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

at::Tensor& normal_(
    at::Tensor& self,
    double mean,
    double std,
    std::optional<at::Generator> generator) {
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::normal_", ([ self, mean, std, generator ]), {
        KernelMemoryGuard guard(self);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        at::_ops::normal_::call(cpu_self, mean, std, generator);
      });
  return self;
}

at::Tensor& normal_out(
    const at::Tensor& self,
    double mean,
    double std,
    std::optional<at::Generator> generator,
    at::Tensor& out) {
  ops::check_out_sizes("aten::normal.out", out, self.sizes());

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::normal.out", ([ self, out, mean, std, generator ]), {
        KernelMemoryGuard guard(self, out);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::_ops::normal_::call(cpu_out, mean, std, generator);
      });
  return out;
}

at::Tensor normal_functional(
    const at::Tensor& self,
    double mean,
    double std,
    std::optional<at::Generator> generator) {
  auto out = at::empty_like(self, self.options(), at::MemoryFormat::Preserve);
  normal_out(self, mean, std, generator, out);
  return out;
}

at::Tensor& normal_Tensor_float_out(
    const at::Tensor& mean,
    double std,
    std::optional<at::Generator> generator,
    at::Tensor& out) {
  ops::check_out_sizes("aten::normal.Tensor_float_out", out, mean.sizes());

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::normal.Tensor_float_out", ([ mean, out, std, generator ]), {
        KernelMemoryGuard guard(mean, out);
        auto cpu_mean = ops::get_cpu_tensor_view_if_needed(mean);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::normal_out(cpu_out, cpu_mean, std, generator);
      });
  return out;
}

at::Tensor normal_Tensor_float(
    const at::Tensor& mean,
    double std,
    std::optional<at::Generator> generator) {
  auto out = at::empty_like(mean, mean.options(), at::MemoryFormat::Preserve);
  normal_Tensor_float_out(mean, std, generator, out);
  return out;
}

at::Tensor& normal_float_Tensor_out(
    double mean,
    const at::Tensor& std,
    std::optional<at::Generator> generator,
    at::Tensor& out) {
  ops::check_out_sizes("aten::normal.float_Tensor_out", out, std.sizes());

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::normal.float_Tensor_out", ([ mean, std, out, generator ]), {
        KernelMemoryGuard guard(std, out);
        auto cpu_std = ops::get_cpu_tensor_view_if_needed(std);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::normal_out(cpu_out, mean, cpu_std, generator);
      });
  return out;
}

at::Tensor normal_float_Tensor(
    double mean,
    const at::Tensor& std,
    std::optional<at::Generator> generator) {
  auto out = at::empty_like(std, std.options(), at::MemoryFormat::Preserve);
  normal_float_Tensor_out(mean, std, generator, out);
  return out;
}

at::Tensor& normal_Tensor_Tensor_out(
    const at::Tensor& mean,
    const at::Tensor& std,
    std::optional<at::Generator> generator,
    at::Tensor& out) {
  auto expected_sizes = at::infer_size(mean.sizes(), std.sizes());
  ops::check_out_sizes("aten::normal.Tensor_Tensor_out", out, expected_sizes);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::normal.Tensor_Tensor_out", ([ mean, std, out, generator ]), {
        KernelMemoryGuard guard(mean, std, out);
        auto cpu_mean = ops::get_cpu_tensor_view_if_needed(mean);
        auto cpu_std = ops::get_cpu_tensor_view_if_needed(std);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::normal_out(cpu_out, cpu_mean, cpu_std, generator);
      });
  return out;
}

at::Tensor normal_Tensor_Tensor(
    const at::Tensor& mean,
    const at::Tensor& std,
    std::optional<at::Generator> generator) {
  auto expected_sizes = at::infer_size(mean.sizes(), std.sizes());
  auto out = at::empty(
      expected_sizes, mean.options().device(c10::DeviceType::PrivateUse1));
  normal_Tensor_Tensor_out(mean, std, generator, out);
  return out;
}

at::Tensor& normal_float_float_out(
    double mean,
    double std,
    c10::SymIntArrayRef size,
    std::optional<at::Generator> generator,
    at::Tensor& out) {
  auto expected_sizes = C10_AS_INTARRAYREF_SLOW(size);
  ops::check_out_sizes("aten::normal.float_float_out", out, expected_sizes);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::normal.float_float_out", ([ out, mean, std, generator ]), {
        KernelMemoryGuard guard(out);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::_ops::normal_::call(cpu_out, mean, std, generator);
      });
  return out;
}

at::Tensor normal_float_float(
    double mean,
    double std,
    c10::SymIntArrayRef size,
    std::optional<at::Generator> generator,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  ops::check_factory_options(layout, device, pin_memory);
  auto options = ops::build_mcpu_options(dtype, layout, device);
  auto out = at::empty(C10_AS_INTARRAYREF_SLOW(size), options);
  normal_float_float_out(mean, std, size, generator, out);
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("normal_", &normal_);
  m.impl("normal_functional", &normal_functional);
  m.impl("normal.out", &normal_out);
  m.impl("normal.Tensor_float", &normal_Tensor_float);
  m.impl("normal.Tensor_float_out", &normal_Tensor_float_out);
  m.impl("normal.float_Tensor", &normal_float_Tensor);
  m.impl("normal.float_Tensor_out", &normal_float_Tensor_out);
  m.impl("normal.Tensor_Tensor", &normal_Tensor_Tensor);
  m.impl("normal.Tensor_Tensor_out", &normal_Tensor_Tensor_out);
  m.impl("normal.float_float", &normal_float_float);
  m.impl("normal.float_float_out", &normal_float_float_out);
}

} // namespace at::mcpu
