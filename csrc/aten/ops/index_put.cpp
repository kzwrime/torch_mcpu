#include "Common.h"
#include "RawPlan.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ops/_index_put_impl.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

struct IndexPutDim0Plan {
  int64_t index_count = 0;
  int64_t inner_size = 1;
  int64_t self_size0 = 0;
  int64_t self_outer_stride = 1;
  int64_t values_outer_stride = 1;
  int64_t index_stride = 1;
  bool values_is_scalar = false;
  bool values_is_single_row = false;
  bool index_is_bool_mask = false;
};

bool has_contiguous_inner_dims(const at::Tensor& tensor, int64_t start_dim) {
  int64_t expected_stride = 1;
  for (int64_t dim = tensor.dim() - 1; dim >= start_dim; --dim) {
    if (tensor.size(dim) > 1 && tensor.stride(dim) != expected_stride) {
      return false;
    }
    expected_stride *= tensor.size(dim);
  }
  return true;
}

bool has_non_overlapping_compressed_rows(
    const at::Tensor& tensor,
    int64_t inner_size) {
  return tensor.size(0) <= 1 || inner_size == 0 ||
      tensor.stride(0) >= inner_size;
}

bool is_dim0_raw_index_dtype(at::ScalarType dtype) {
  return dtype == at::ScalarType::Long || dtype == at::ScalarType::Int ||
      dtype == at::ScalarType::Bool;
}

std::optional<IndexPutDim0Plan> make_index_put_dim0_plan(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate,
    bool unsafe) {
  (void)unsafe;
  if (accumulate || indices.size() > static_cast<size_t>(self.dim()) ||
      indices.empty() || !indices[0].has_value() ||
      !ops::is_mcpu_tensor(self) || !ops::is_mcpu_tensor(values) ||
      !ops::is_mcpu_tensor(*indices[0]) ||
      !ops::is_raw_dtype_supported(self.scalar_type()) ||
      self.scalar_type() != values.scalar_type() ||
      !is_dim0_raw_index_dtype(indices[0]->scalar_type()) || self.dim() == 0 ||
      (values.dim() != 0 && values.dim() != self.dim() &&
       values.dim() != self.dim() - 1) ||
      indices[0]->dim() != 1 || self.layout() != c10::Layout::Strided ||
      values.layout() != c10::Layout::Strided ||
      indices[0]->layout() != c10::Layout::Strided ||
      indices[0]->stride(0) < 0 || self.is_alias_of(values) ||
      self.is_alias_of(*indices[0])) {
    return std::nullopt;
  }
  for (size_t i = 1; i < indices.size(); ++i) {
    if (indices[i].has_value()) {
      return std::nullopt;
    }
  }
  const bool index_is_bool_mask =
      indices[0]->scalar_type() == at::ScalarType::Bool;
  const bool values_is_scalar = values.dim() == 0;
  if (index_is_bool_mask) {
    if (indices[0]->numel() != self.size(0) || !values_is_scalar) {
      return std::nullopt;
    }
  }
  const bool values_is_single_row = !values_is_scalar &&
      values.dim() == self.dim() - 1 && indices[0]->numel() == 1 &&
      !index_is_bool_mask;
  if (!values_is_scalar && !values_is_single_row &&
      values.size(0) != indices[0]->numel()) {
    return std::nullopt;
  }
  if (values_is_single_row) {
    for (int64_t dim = 1; dim < self.dim(); ++dim) {
      if (values.size(dim - 1) != self.size(dim)) {
        return std::nullopt;
      }
    }
  } else if (!values_is_scalar) {
    for (int64_t dim = 1; dim < self.dim(); ++dim) {
      if (values.size(dim) != self.size(dim)) {
        return std::nullopt;
      }
    }
  }
  if (!has_contiguous_inner_dims(self, 1) ||
      (values_is_single_row && !has_contiguous_inner_dims(values, 0)) ||
      (!values_is_scalar && !values_is_single_row &&
       !has_contiguous_inner_dims(values, 1))) {
    return std::nullopt;
  }

  const auto inner_size = c10::multiply_integers(self.sizes().slice(1));
  if (!has_non_overlapping_compressed_rows(self, inner_size) ||
      (!values_is_scalar && !values_is_single_row &&
       !has_non_overlapping_compressed_rows(values, inner_size))) {
    return std::nullopt;
  }

  IndexPutDim0Plan plan;
  plan.index_count = indices[0]->numel();
  plan.inner_size = inner_size;
  plan.self_size0 = self.size(0);
  plan.self_outer_stride = self.stride(0);
  plan.values_outer_stride = values_is_scalar ? 0 : values.stride(0);
  plan.index_stride = indices[0]->stride(0);
  plan.values_is_scalar = values_is_scalar;
  plan.values_is_single_row = values_is_single_row;
  plan.index_is_bool_mask = index_is_bool_mask;
  return plan;
}

template <typename scalar_t>
void index_put_dim0_kernel(
    scalar_t* self_ptr,
    const scalar_t* values_ptr,
    const bool* index_ptr,
    const IndexPutDim0Plan& plan) {
  for (int64_t row = 0; row < plan.index_count; ++row) {
    if (!index_ptr[row * plan.index_stride]) {
      continue;
    }
    auto* self_row_ptr = self_ptr + row * plan.self_outer_stride;
    for (int64_t inner = 0; inner < plan.inner_size; ++inner) {
      self_row_ptr[inner] = values_ptr[0];
    }
  }
}

template <typename scalar_t, typename index_t>
void index_put_dim0_kernel(
    scalar_t* self_ptr,
    const scalar_t* values_ptr,
    const index_t* index_ptr,
    int64_t* invalid_index_ptr,
    const IndexPutDim0Plan& plan) {
  for (int64_t outer = 0; outer < plan.index_count; ++outer) {
    const int64_t original_index =
        static_cast<int64_t>(index_ptr[outer * plan.index_stride]);
    int64_t self_row = original_index;
    if (self_row < 0) {
      self_row += plan.self_size0;
    }
    if (self_row < 0 || self_row >= plan.self_size0) {
      invalid_index_ptr[0] = 1;
      invalid_index_ptr[1] = original_index;
      return;
    }
  }

  for (int64_t outer = 0; outer < plan.index_count; ++outer) {
    int64_t self_row =
        static_cast<int64_t>(index_ptr[outer * plan.index_stride]);
    if (self_row < 0) {
      self_row += plan.self_size0;
    }
    auto* self_row_ptr = self_ptr + self_row * plan.self_outer_stride;
    const int64_t values_outer = plan.values_is_single_row ? 0 : outer;
    const auto* values_row_ptr =
        values_ptr + values_outer * plan.values_outer_stride;
    for (int64_t inner = 0; inner < plan.inner_size; ++inner) {
      self_row_ptr[inner] = values_row_ptr[plan.values_is_scalar ? 0 : inner];
    }
  }
}

void launch_index_put_dim0_bool(
    at::Tensor& self,
    const at::Tensor& index,
    const at::Tensor& values,
    IndexPutDim0Plan plan) {
  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "mcpu_index_put_dim0",
      [&] {
        auto* self_ptr = self.mutable_data_ptr<scalar_t>();
        const auto* values_ptr = values.const_data_ptr<scalar_t>();
        const auto* index_ptr = index.const_data_ptr<bool>();
        MCPU_LAUNCH_TIMED_KERNEL(
            "mcpu::aten::_index_put_impl_.raw",
            ([ self_ptr, values_ptr, index_ptr, plan ]),
            {
              KernelPointerMemoryGuard guard({self_ptr, values_ptr, index_ptr});
              index_put_dim0_kernel(self_ptr, values_ptr, index_ptr, plan);
            });
      });
}

template <typename index_t>
void launch_index_put_dim0_checked(
    at::Tensor& self,
    const at::Tensor& index,
    const at::Tensor& values,
    IndexPutDim0Plan plan) {
  auto invalid_index =
      at::empty({2}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto* invalid_index_ptr = invalid_index.mutable_data_ptr<int64_t>();
  invalid_index_ptr[0] = 0;
  invalid_index_ptr[1] = 0;
  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "mcpu_index_put_dim0",
      [&] {
        auto* self_ptr = self.mutable_data_ptr<scalar_t>();
        const auto* values_ptr = values.const_data_ptr<scalar_t>();
        const auto* index_ptr = index.const_data_ptr<index_t>();
        MCPU_LAUNCH_TIMED_KERNEL(
            "mcpu::aten::_index_put_impl_.raw",
            ([ self_ptr, values_ptr, index_ptr, invalid_index_ptr, plan ]),
            {
              KernelPointerMemoryGuard guard(
                  {self_ptr, values_ptr, index_ptr, invalid_index_ptr});
              index_put_dim0_kernel(
                  self_ptr, values_ptr, index_ptr, invalid_index_ptr, plan);
            });
      });
  at::native::mcpu::synchronize_if_mcpu(index);
  TORCH_CHECK(
      invalid_index_ptr[0] == 0,
      "index ",
      invalid_index_ptr[1],
      " is out of bounds for dimension 0 with size ",
      plan.self_size0);
}

bool index_put_dim0(
    at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate,
    bool unsafe) {
  auto plan =
      make_index_put_dim0_plan(self, indices, values, accumulate, unsafe);
  if (!plan.has_value()) {
    return false;
  }
  if (plan->index_count == 0 || plan->inner_size == 0) {
    return true;
  }

  const auto& index = *indices[0];
  if (plan->index_is_bool_mask) {
    launch_index_put_dim0_bool(self, index, values, *plan);
  } else if (index.scalar_type() == at::ScalarType::Int) {
    launch_index_put_dim0_checked<int32_t>(self, index, values, *plan);
  } else {
    launch_index_put_dim0_checked<int64_t>(self, index, values, *plan);
  }
  return true;
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

  if (index_put_dim0(self, indices, values, accumulate, unsafe)) {
    return self;
  }

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::_index_put_impl_",
      ([ self, indices, values, accumulate, unsafe ]),
      {
        KernelMemoryGuard guard(self, values, c10::IValue(indices));
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_values = ops::get_cpu_tensor_view_if_needed(values);
        auto cpu_indices = ops::to_cpu_indices(indices);
        at::_index_put_impl_(
            cpu_self, cpu_indices, cpu_values, accumulate, unsafe);
      });
  return self;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_index_put_impl_", &_index_put_impl_);
}

} // namespace at::mcpu
