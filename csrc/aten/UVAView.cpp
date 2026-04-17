#include <ATen/ATen.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

at::Tensor get_mcpu_view_from_cpu_tensor_impl(const at::Tensor& cpu_tensor) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  if (cpu_tensor.numel() == 0) {
    return at::empty(
        cpu_tensor.sizes(),
        cpu_tensor.options().device(c10::DeviceType::PrivateUse1));
  }

  // Ensure the backing memory is pinned (allocated via the mcpu host
  // allocator). If the input is not pinned, take a contiguous snapshot first.
  at::Tensor pinned = cpu_tensor.is_pinned()
      ? cpu_tensor
      : cpu_tensor.contiguous().pin_memory();

  return at::from_blob(
      pinned.data_ptr(),
      pinned.sizes(),
      pinned.strides(),
      /*deleter=*/[base = pinned](void*) {},
      pinned.options().device(c10::DeviceType::PrivateUse1));
}

at::Tensor get_cpu_view_from_mcpu_tensor_impl(const at::Tensor& mcpu_tensor) {
  TORCH_CHECK(
      mcpu_tensor.device().type() == c10::DeviceType::PrivateUse1,
      "Input tensor must be on mcpu");

  if (mcpu_tensor.numel() == 0) {
    return at::empty(
        mcpu_tensor.sizes(),
        mcpu_tensor.options().device(c10::DeviceType::CPU));
  }

  return at::from_blob(
      mcpu_tensor.data_ptr(),
      mcpu_tensor.sizes(),
      mcpu_tensor.strides(),
      /*deleter=*/[base = mcpu_tensor](void*) {},
      mcpu_tensor.options().device(c10::DeviceType::CPU));
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def("get_mcpu_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor");
  m.def("get_cpu_view_from_mcpu_tensor(Tensor mcpu_tensor) -> Tensor");
}

TORCH_LIBRARY_IMPL(mcpu, CPU, m) {
  m.impl("get_mcpu_view_from_cpu_tensor", &get_mcpu_view_from_cpu_tensor_impl);
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("get_cpu_view_from_mcpu_tensor", &get_cpu_view_from_mcpu_tensor_impl);
}

} // namespace at::mcpu
