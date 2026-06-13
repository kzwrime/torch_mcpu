#include "Common.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ops/arange.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

at::Tensor empty_mcpu_like_cpu_result(
    const at::Tensor& cpu_result,
    const at::TensorOptions& options) {
  return at::empty_strided(
      cpu_result.sizes(),
      cpu_result.strides(),
      options.device(c10::DeviceType::PrivateUse1));
}

at::Tensor arange(
    const at::Scalar& end,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  ops::check_factory_options(layout, device, pin_memory);
  auto options = ops::build_mcpu_options(dtype, layout, device);
  auto cpu_result = at::arange(end, options.device(c10::DeviceType::CPU));
  auto out = empty_mcpu_like_cpu_result(cpu_result, options);
  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::arange", ([ out, cpu_result ]), {
    KernelMemoryGuard guard(out);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    cpu_out.copy_(cpu_result);
  });
  return out;
}

at::Tensor arange_start(
    const at::Scalar& start,
    const at::Scalar& end,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  ops::check_factory_options(layout, device, pin_memory);
  auto options = ops::build_mcpu_options(dtype, layout, device);
  auto cpu_result =
      at::arange(start, end, 1, options.device(c10::DeviceType::CPU));
  auto out = empty_mcpu_like_cpu_result(cpu_result, options);
  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::arange.start", ([ out, cpu_result ]), {
    KernelMemoryGuard guard(out);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    cpu_out.copy_(cpu_result);
  });
  return out;
}

at::Tensor arange_start_step(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  ops::check_factory_options(layout, device, pin_memory);
  auto options = ops::build_mcpu_options(dtype, layout, device);
  auto cpu_result =
      at::arange(start, end, step, options.device(c10::DeviceType::CPU));
  auto out = empty_mcpu_like_cpu_result(cpu_result, options);
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::arange.start_step", ([ out, cpu_result ]), {
        KernelMemoryGuard guard(out);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        cpu_out.copy_(cpu_result);
      });
  return out;
}

at::Tensor& arange_out(const at::Scalar& end, at::Tensor& out) {
  auto cpu_result = at::arange(end, out.options().device(c10::DeviceType::CPU));
  ops::check_out_sizes("aten::arange.out", out, cpu_result.sizes());

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::arange.out", ([ out, cpu_result ]), {
    KernelMemoryGuard guard(out);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    cpu_out.copy_(cpu_result);
  });
  return out;
}

at::Tensor& arange_start_out(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& out) {
  auto cpu_result =
      at::arange(start, end, step, out.options().device(c10::DeviceType::CPU));
  ops::check_out_sizes("aten::arange.start_out", out, cpu_result.sizes());

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::arange.start_out", ([ out, cpu_result ]), {
        KernelMemoryGuard guard(out);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        cpu_out.copy_(cpu_result);
      });
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("arange", &arange);
  m.impl("arange.start", &arange_start);
  m.impl("arange.start_step", &arange_start_step);
  m.impl("arange.out", &arange_out);
  m.impl("arange.start_out", &arange_start_out);
}

} // namespace at::mcpu
