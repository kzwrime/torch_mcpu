#pragma once

#include "../native/Common.h"

#include <ATen/ATen.h>

namespace at::mcpu::ops {

inline at::Tensor get_cpu_view_from_mcpu_tensor(const at::Tensor& mcpu_tensor) {
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

inline bool is_mcpu_tensor(const at::Tensor& tensor) {
  return tensor.defined() &&
      tensor.device().type() == c10::DeviceType::PrivateUse1;
}

inline at::Tensor get_cpu_tensor_view_if_needed(const at::Tensor& tensor) {
  return is_mcpu_tensor(tensor) ? get_cpu_view_from_mcpu_tensor(tensor)
                                : tensor;
}

inline at::Tensor to_meta_tensor(const at::Tensor& tensor) {
  return at::empty_strided(
      tensor.sizes(),
      tensor.strides(),
      tensor.options().device(c10::DeviceType::Meta));
}

inline std::vector<at::Tensor> to_cpu_tensors_if_needed(
    at::TensorList tensors) {
  std::vector<at::Tensor> cpu_tensors;
  cpu_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    cpu_tensors.push_back(get_cpu_tensor_view_if_needed(tensor));
  }
  return cpu_tensors;
}

inline std::vector<at::Tensor> to_meta_tensors(at::TensorList tensors) {
  std::vector<at::Tensor> meta_tensors;
  meta_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    meta_tensors.push_back(to_meta_tensor(tensor));
  }
  return meta_tensors;
}

inline at::Tensor empty_mcpu_from_meta(
    const at::Tensor& meta_tensor,
    const at::TensorOptions& options) {
  return at::empty_strided(
      meta_tensor.sizes(),
      meta_tensor.strides(),
      options.device(c10::DeviceType::PrivateUse1));
}

inline void check_factory_options(
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  TORCH_CHECK(
      c10::layout_or_default(layout) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory),
      "Pin memory can only be on CPU");
  TORCH_CHECK(
      device.has_value() && device->type() == c10::DeviceType::PrivateUse1,
      "Expected an mcpu device");
}

inline at::TensorOptions build_mcpu_options(
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device) {
  return at::TensorOptions()
      .dtype(c10::dtype_or_default(dtype))
      .layout(c10::layout_or_default(layout))
      .device(*device);
}

inline void check_out_sizes(
    const char* op_name,
    const at::Tensor& out,
    const at::Tensor& meta_out) {
  TORCH_CHECK(
      out.sizes().equals(meta_out.sizes()),
      op_name,
      ": expected out.sizes() == ",
      meta_out.sizes(),
      ", but got ",
      out.sizes());
}

inline void check_out_sizes(
    const char* op_name,
    const char* out_name,
    const at::Tensor& out,
    const at::Tensor& meta_out) {
  TORCH_CHECK(
      out.sizes().equals(meta_out.sizes()),
      op_name,
      ": expected ",
      out_name,
      ".sizes() == ",
      meta_out.sizes(),
      ", but got ",
      out.sizes());
}

inline c10::List<std::optional<at::Tensor>> to_cpu_indices(
    const c10::List<std::optional<at::Tensor>>& indices) {
  c10::List<std::optional<at::Tensor>> cpu_indices;
  cpu_indices.reserve(indices.size());
  for (const auto& index : indices) {
    const std::optional<at::Tensor> optional_index = index;
    if (optional_index.has_value() && optional_index->defined()) {
      cpu_indices.push_back(get_cpu_tensor_view_if_needed(*optional_index));
    } else {
      cpu_indices.push_back(std::nullopt);
    }
  }
  return cpu_indices;
}

inline c10::List<std::optional<at::Tensor>> to_meta_indices(
    const c10::List<std::optional<at::Tensor>>& indices) {
  c10::List<std::optional<at::Tensor>> meta_indices;
  meta_indices.reserve(indices.size());
  for (const auto& index : indices) {
    const std::optional<at::Tensor> optional_index = index;
    if (optional_index.has_value() && optional_index->defined()) {
      meta_indices.push_back(to_meta_tensor(*optional_index));
    } else {
      meta_indices.push_back(std::nullopt);
    }
  }
  return meta_indices;
}

} // namespace at::mcpu::ops
