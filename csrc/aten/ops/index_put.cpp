#include "Common.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ops/_index_put_impl.h>
#include <c10/util/SmallVector.h>
#include <torch/library.h>

#include <memory>
#include <optional>
#include <utility>

namespace at::mcpu {
namespace {

struct TensorViewSpec {
  void* data = nullptr;
  c10::SmallVector<int64_t, 1> sizes;
  c10::SmallVector<int64_t, 1> strides;
  at::TensorOptions options;
};

struct OptionalTensorViewSpec {
  bool has_value = false;
  TensorViewSpec spec;
};

struct IndexPutArgs {
  TensorViewSpec self;
  TensorViewSpec values;
  c10::SmallVector<OptionalTensorViewSpec, 8> indices;
  bool accumulate = false;
  bool unsafe = false;
};

using TensorPointerList = c10::SmallVector<const void*, 8>;

int64_t numel_from_sizes(c10::ArrayRef<int64_t> sizes) {
  int64_t numel = 1;
  for (const auto size : sizes) {
    numel *= size;
  }
  return numel;
}

TensorViewSpec make_cpu_view_spec(const at::Tensor& tensor) {
  return TensorViewSpec{
      tensor.numel() == 0 ? nullptr : tensor.data_ptr(),
      c10::SmallVector<int64_t, 1>(
          tensor.sizes().begin(), tensor.sizes().end()),
      c10::SmallVector<int64_t, 1>(
          tensor.strides().begin(), tensor.strides().end()),
      tensor.options().device(c10::DeviceType::CPU)};
}

at::Tensor cpu_view_from_spec(const TensorViewSpec& spec) {
  c10::InferenceMode inference_guard(false);
  if (numel_from_sizes(spec.sizes) == 0) {
    return at::empty_strided(spec.sizes, spec.strides, spec.options);
  }
  return at::from_blob(spec.data, spec.sizes, spec.strides, spec.options);
}

OptionalTensorViewSpec make_optional_cpu_view_spec(
    const std::optional<at::Tensor>& tensor) {
  if (tensor.has_value() && tensor->defined()) {
    return OptionalTensorViewSpec{true, make_cpu_view_spec(*tensor)};
  }
  return OptionalTensorViewSpec{};
}

c10::List<std::optional<at::Tensor>> cpu_indices_from_specs(
    c10::ArrayRef<OptionalTensorViewSpec> index_specs) {
  c10::List<std::optional<at::Tensor>> cpu_indices;
  cpu_indices.reserve(index_specs.size());
  for (const auto& index_spec : index_specs) {
    if (index_spec.has_value) {
      cpu_indices.push_back(cpu_view_from_spec(index_spec.spec));
    } else {
      cpu_indices.push_back(std::nullopt);
    }
  }
  return cpu_indices;
}

TensorPointerList pointer_list(
    const TensorViewSpec& self_spec,
    const TensorViewSpec& values_spec,
    c10::ArrayRef<OptionalTensorViewSpec> index_specs) {
  TensorPointerList ptrs;
  ptrs.reserve(index_specs.size() + 2);
  if (self_spec.data != nullptr) {
    ptrs.push_back(self_spec.data);
  }
  if (values_spec.data != nullptr) {
    ptrs.push_back(values_spec.data);
  }
  for (const auto& index_spec : index_specs) {
    if (index_spec.has_value && index_spec.spec.data != nullptr) {
      ptrs.push_back(index_spec.spec.data);
    }
  }
  return ptrs;
}

at::Tensor& _index_put_impl_(
    at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate,
    bool unsafe) {
  TORCH_CHECK_INDEX(
      indices.size() <= static_cast<size_t>(self.dim()),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      indices.size(),
      ")");

  c10::SmallVector<OptionalTensorViewSpec, 8> index_specs;
  index_specs.reserve(indices.size());
  for (const auto& index : indices) {
    index_specs.push_back(make_optional_cpu_view_spec(index));
  }

  auto args = std::make_unique<IndexPutArgs>(IndexPutArgs{
      make_cpu_view_spec(self),
      make_cpu_view_spec(values),
      std::move(index_specs),
      accumulate,
      unsafe});

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::_index_put_impl_", ([args = std::move(args)]), {
        auto ptrs = pointer_list(args->self, args->values, args->indices);
        KernelPointerMemoryGuard guard(ptrs);
        auto cpu_self = cpu_view_from_spec(args->self);
        auto cpu_values = cpu_view_from_spec(args->values);
        auto cpu_indices = cpu_indices_from_specs(args->indices);
        at::_index_put_impl_(
            cpu_self, cpu_indices, cpu_values, args->accumulate, args->unsafe);
      });
  return self;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_index_put_impl_", &_index_put_impl_);
}

} // namespace at::mcpu
