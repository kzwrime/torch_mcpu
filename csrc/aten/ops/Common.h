#pragma once

#include "../native/Common.h"
#include "runtime/McpuKernelLaunch.h"

#include <aten/McpuTensorView.hpp>

#include <ATen/ATen.h>
#include <ATen/WrapDimUtils.h>
#include <c10/util/SmallVector.h>

#include <vector>

namespace at::mcpu::ops {

using at::mcpu::get_cpu_tensor_view_if_needed;
using at::mcpu::get_cpu_view_from_mcpu_tensor;
using at::mcpu::is_mcpu_tensor;

struct TensorViewSpec {
  void* data = nullptr;
  c10::SmallVector<int64_t, 1> sizes;
  c10::SmallVector<int64_t, 1> strides;
  at::TensorOptions options;
};

using TensorPointerList = c10::SmallVector<const void*, 8>;

inline int64_t numel_from_sizes(c10::ArrayRef<int64_t> sizes) {
  int64_t numel = 1;
  for (const auto size : sizes) {
    numel *= size;
  }
  return numel;
}

inline TensorViewSpec make_cpu_view_spec(const at::Tensor& tensor) {
  TORCH_CHECK(
      is_mcpu_tensor(tensor),
      "make_cpu_view_spec expects an mcpu tensor, but got ",
      tensor.device());
  return TensorViewSpec{
      tensor.numel() == 0 ? nullptr : tensor.data_ptr(),
      c10::SmallVector<int64_t, 1>(
          tensor.sizes().begin(), tensor.sizes().end()),
      c10::SmallVector<int64_t, 1>(
          tensor.strides().begin(), tensor.strides().end()),
      tensor.options().device(c10::DeviceType::CPU)};
}

inline at::Tensor cpu_view_from_spec(const TensorViewSpec& spec) {
  c10::InferenceMode inference_guard(false);
  if (numel_from_sizes(spec.sizes) == 0) {
    return at::empty_strided(spec.sizes, spec.strides, spec.options);
  }
  return at::from_blob(spec.data, spec.sizes, spec.strides, spec.options);
}

inline std::vector<at::Tensor> cpu_views_from_specs(
    c10::ArrayRef<TensorViewSpec> specs) {
  std::vector<at::Tensor> tensors;
  tensors.reserve(specs.size());
  for (const auto& spec : specs) {
    tensors.push_back(cpu_view_from_spec(spec));
  }
  return tensors;
}

inline TensorPointerList pointer_list(
    const TensorViewSpec& first,
    c10::ArrayRef<TensorViewSpec> rest) {
  TensorPointerList ptrs;
  ptrs.reserve(rest.size() + 1);
  if (first.data != nullptr) {
    ptrs.push_back(first.data);
  }
  for (const auto& spec : rest) {
    if (spec.data != nullptr) {
      ptrs.push_back(spec.data);
    }
  }
  return ptrs;
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
    const at::Tensor& out,
    at::IntArrayRef expected_sizes) {
  TORCH_CHECK(
      out.sizes().equals(expected_sizes),
      op_name,
      ": expected out.sizes() == ",
      expected_sizes,
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

inline std::vector<int64_t> reduction_sizes(
    at::IntArrayRef input_sizes,
    at::OptionalIntArrayRef dim,
    bool keepdim) {
  const auto ndim = static_cast<int64_t>(input_sizes.size());
  std::vector<bool> reduce_dims(ndim, false);

  if (dim.has_value() && !dim.value().empty()) {
    for (const auto raw_dim : dim.value()) {
      auto wrapped_dim = at::maybe_wrap_dim(raw_dim, ndim);
      TORCH_CHECK(
          !reduce_dims[wrapped_dim],
          "dim ",
          raw_dim,
          " appears multiple times in the list of dims");
      reduce_dims[wrapped_dim] = true;
    }
  } else {
    std::fill(reduce_dims.begin(), reduce_dims.end(), true);
  }

  std::vector<int64_t> result;
  result.reserve(input_sizes.size());
  for (int64_t i = 0; i < ndim; ++i) {
    if (reduce_dims[i]) {
      if (keepdim) {
        result.push_back(1);
      }
    } else {
      result.push_back(input_sizes[i]);
    }
  }
  return result;
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
