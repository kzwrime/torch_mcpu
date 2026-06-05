#pragma once

#include <ATen/ATen.h>

namespace at::mcpu {

inline bool is_mcpu_tensor(const at::Tensor& tensor) {
  return tensor.defined() &&
      tensor.device().type() == c10::DeviceType::PrivateUse1;
}

inline at::Tensor get_cpu_view_from_mcpu_tensor(const at::Tensor& mcpu_tensor) {
  TORCH_CHECK(
      mcpu_tensor.device().type() == c10::DeviceType::PrivateUse1,
      "Input tensor must be on mcpu");

  c10::InferenceMode inference_guard(false);
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

inline at::Tensor get_cpu_tensor_view_if_needed(const at::Tensor& tensor) {
  return is_mcpu_tensor(tensor) ? get_cpu_view_from_mcpu_tensor(tensor)
                                : tensor;
}

} // namespace at::mcpu
