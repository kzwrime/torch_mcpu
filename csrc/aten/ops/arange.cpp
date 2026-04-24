#include "Common.h"

#include <ATen/ops/arange.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

at::Tensor arange(
    const at::Scalar& end,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  ops::check_factory_options(layout, device, pin_memory);
  auto options = ops::build_mcpu_options(dtype, layout, device);
  auto meta_out = at::empty({0}, options.device(c10::DeviceType::Meta));
  at::arange_out(meta_out, end);

  auto out = ops::empty_mcpu_from_meta(meta_out, options);
  at::native::mcpu::MemoryGuard guard(out);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::arange_out(cpu_out, end);
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
  auto meta_out = at::empty({0}, options.device(c10::DeviceType::Meta));
  at::arange_out(meta_out, start, end, 1);

  auto out = ops::empty_mcpu_from_meta(meta_out, options);
  at::native::mcpu::MemoryGuard guard(out);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::arange_out(cpu_out, start, end, 1);
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
  auto meta_out = at::empty({0}, options.device(c10::DeviceType::Meta));
  at::arange_out(meta_out, start, end, step);

  auto out = ops::empty_mcpu_from_meta(meta_out, options);
  at::native::mcpu::MemoryGuard guard(out);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::arange_out(cpu_out, start, end, step);
  return out;
}

at::Tensor& arange_out(const at::Scalar& end, at::Tensor& out) {
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::arange_out(meta_out, end);
  ops::check_out_sizes("aten::arange.out", out, meta_out);

  at::native::mcpu::MemoryGuard guard(out);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::arange_out(cpu_out, end);
  return out;
}

at::Tensor& arange_start_out(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& out) {
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::arange_out(meta_out, start, end, step);
  ops::check_out_sizes("aten::arange.start_out", out, meta_out);

  at::native::mcpu::MemoryGuard guard(out);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::arange_out(cpu_out, start, end, step);
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
