#include <ATen/ATen.h>
#include <torch/library.h>

#include <aten/McpuTensorView.hpp>

#include <include/openreg.h>

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
  return at::mcpu::get_cpu_view_from_mcpu_tensor(mcpu_tensor);
}

void unprotect_mcpu_tensor_memory_impl(const at::Tensor& mcpu_tensor) {
  TORCH_CHECK(
      mcpu_tensor.device().type() == c10::DeviceType::PrivateUse1,
      "Input tensor must be on mcpu");

  if (!mcpu_tensor.has_storage() || mcpu_tensor.numel() == 0) {
    return;
  }

  orPointerAttributes attr;
  auto attr_status = orPointerGetAttributes(&attr, mcpu_tensor.data_ptr());
  if (attr_status != orSuccess || attr.type != orMemoryTypeDevice) {
    return;
  }

  auto unprotect_status = orMemoryUnprotect(attr.pointer);
  TORCH_CHECK(
      unprotect_status == orSuccess, "Failed to unprotect mcpu tensor memory");
}

at::Tensor get_unprotected_cpu_view_from_mcpu_tensor_impl(
    const at::Tensor& mcpu_tensor) {
  unprotect_mcpu_tensor_memory_impl(mcpu_tensor);
  return get_cpu_view_from_mcpu_tensor_impl(mcpu_tensor);
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def("get_mcpu_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor");
  m.def("get_cpu_view_from_mcpu_tensor(Tensor mcpu_tensor) -> Tensor");
  m.def(
      "get_unprotected_cpu_view_from_mcpu_tensor(Tensor mcpu_tensor) -> Tensor");
  m.def("unprotect_mcpu_tensor_memory(Tensor mcpu_tensor) -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, CPU, m) {
  m.impl("get_mcpu_view_from_cpu_tensor", &get_mcpu_view_from_cpu_tensor_impl);
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("get_cpu_view_from_mcpu_tensor", &get_cpu_view_from_mcpu_tensor_impl);
  m.impl(
      "get_unprotected_cpu_view_from_mcpu_tensor",
      &get_unprotected_cpu_view_from_mcpu_tensor_impl);
  m.impl("unprotect_mcpu_tensor_memory", &unprotect_mcpu_tensor_memory_impl);
}

} // namespace at::mcpu
