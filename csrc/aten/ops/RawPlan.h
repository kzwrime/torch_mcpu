#pragma once

#include "Common.h"

#include <c10/util/SmallVector.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

namespace at::mcpu::ops {

struct RawTensorPlan {
  int64_t inner_size = 1;
  int64_t outer_count = 1;
  c10::SmallVector<int64_t, 8> outer_sizes;
  c10::SmallVector<int64_t, 8> outer_strides;
};

struct RawTensorPairPlan {
  int64_t inner_size = 1;
  int64_t outer_count = 1;
  c10::SmallVector<int64_t, 8> outer_sizes;
  c10::SmallVector<int64_t, 8> first_outer_strides;
  c10::SmallVector<int64_t, 8> second_outer_strides;
};

struct RawTensorTripletPlan {
  int64_t inner_size = 1;
  int64_t outer_count = 1;
  c10::SmallVector<int64_t, 8> outer_sizes;
  c10::SmallVector<int64_t, 8> first_outer_strides;
  c10::SmallVector<int64_t, 8> second_outer_strides;
  c10::SmallVector<int64_t, 8> third_outer_strides;
};

inline bool is_last_dim_contiguous(const at::Tensor& tensor) {
  const auto dim = tensor.dim();
  return dim == 0 || tensor.size(dim - 1) <= 1 || tensor.stride(dim - 1) == 1;
}

inline bool has_non_overlapping_positive_strides(const at::Tensor& tensor) {
  if (tensor.numel() == 0) {
    return true;
  }

  c10::SmallVector<std::pair<int64_t, int64_t>, 8> dims;
  const auto ndim = tensor.dim();
  for (int64_t dim = 0; dim < ndim; ++dim) {
    const auto size = tensor.size(dim);
    if (size <= 1) {
      continue;
    }
    const auto stride = tensor.stride(dim);
    if (stride <= 0) {
      return false;
    }
    dims.emplace_back(stride, size);
  }

  std::sort(dims.begin(), dims.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first < rhs.first;
  });

  int64_t covered_span = 1;
  for (const auto& [stride, size] : dims) {
    if (stride < covered_span) {
      return false;
    }
    const auto extent = size - 1;
    if (extent > 0 &&
        stride >
            (std::numeric_limits<int64_t>::max() - covered_span) / extent) {
      return false;
    }
    covered_span += extent * stride;
  }
  return true;
}

inline bool has_raw_plan_layout(const at::Tensor& tensor) {
  return is_mcpu_tensor(tensor) && tensor.layout() == c10::Layout::Strided &&
      is_last_dim_contiguous(tensor) &&
      has_non_overlapping_positive_strides(tensor);
}

inline bool is_raw_dtype_supported(at::ScalarType dtype) {
  return dtype == at::ScalarType::Byte || dtype == at::ScalarType::Char ||
      dtype == at::ScalarType::Double || dtype == at::ScalarType::Float ||
      dtype == at::ScalarType::Int || dtype == at::ScalarType::Long ||
      dtype == at::ScalarType::Short || dtype == at::ScalarType::Half ||
      dtype == at::ScalarType::BFloat16 || dtype == at::ScalarType::Bool;
}

inline std::optional<RawTensorPlan> make_raw_tensor_plan(
    const at::Tensor& tensor) {
  if (!has_raw_plan_layout(tensor)) {
    return std::nullopt;
  }

  RawTensorPlan plan;
  const auto ndim = tensor.dim();
  if (ndim == 0) {
    return plan;
  }

  plan.inner_size = tensor.size(ndim - 1);
  for (int64_t dim = 0; dim < ndim - 1; ++dim) {
    plan.outer_count *= tensor.size(dim);
    plan.outer_sizes.push_back(tensor.size(dim));
    plan.outer_strides.push_back(tensor.stride(dim));
  }
  return plan;
}

inline bool has_safe_raw_output_overlap(
    const at::Tensor& out,
    const at::Tensor& input) {
  if (!out.is_alias_of(input)) {
    return true;
  }
  return out.const_data_ptr() == input.const_data_ptr() &&
      out.sizes().equals(input.sizes()) &&
      out.strides().equals(input.strides());
}

inline std::optional<RawTensorPairPlan> make_same_shape_raw_tensor_pair_plan(
    const at::Tensor& first,
    const at::Tensor& second) {
  if (!first.sizes().equals(second.sizes()) || !has_raw_plan_layout(first) ||
      !has_raw_plan_layout(second) ||
      !has_safe_raw_output_overlap(second, first)) {
    return std::nullopt;
  }

  RawTensorPairPlan plan;
  const auto ndim = first.dim();
  if (ndim == 0) {
    return plan;
  }

  plan.inner_size = first.size(ndim - 1);
  for (int64_t dim = 0; dim < ndim - 1; ++dim) {
    plan.outer_count *= first.size(dim);
    plan.outer_sizes.push_back(first.size(dim));
    plan.first_outer_strides.push_back(first.stride(dim));
    plan.second_outer_strides.push_back(second.stride(dim));
  }
  return plan;
}

inline std::optional<RawTensorTripletPlan>
make_same_shape_raw_tensor_triplet_plan(
    const at::Tensor& first,
    const at::Tensor& second,
    const at::Tensor& third) {
  if (!first.sizes().equals(second.sizes()) ||
      !first.sizes().equals(third.sizes()) || !has_raw_plan_layout(first) ||
      !has_raw_plan_layout(second) || !has_raw_plan_layout(third) ||
      !has_safe_raw_output_overlap(third, first) ||
      !has_safe_raw_output_overlap(third, second)) {
    return std::nullopt;
  }

  RawTensorTripletPlan plan;
  const auto ndim = first.dim();
  if (ndim == 0) {
    return plan;
  }

  plan.inner_size = first.size(ndim - 1);
  for (int64_t dim = 0; dim < ndim - 1; ++dim) {
    plan.outer_count *= first.size(dim);
    plan.outer_sizes.push_back(first.size(dim));
    plan.first_outer_strides.push_back(first.stride(dim));
    plan.second_outer_strides.push_back(second.stride(dim));
    plan.third_outer_strides.push_back(third.stride(dim));
  }
  return plan;
}

template <typename Func>
void for_each_raw_tensor_row(const RawTensorPlan& plan, Func&& func) {
  const int64_t outer_ndim = plan.outer_sizes.size();
  if (outer_ndim == 0) {
    func(0, plan.inner_size);
    return;
  }

  c10::SmallVector<int64_t, 8> indices(outer_ndim, 0);
  int64_t offset = 0;
  for (int64_t outer = 0; outer < plan.outer_count; ++outer) {
    func(offset, plan.inner_size);

    for (int64_t dim = outer_ndim - 1; dim >= 0; --dim) {
      ++indices[dim];
      offset += plan.outer_strides[dim];
      if (indices[dim] < plan.outer_sizes[dim]) {
        break;
      }
      offset -= indices[dim] * plan.outer_strides[dim];
      indices[dim] = 0;
    }
  }
}

template <typename Func>
void for_each_raw_tensor_pair_row(const RawTensorPairPlan& plan, Func&& func) {
  const int64_t outer_ndim = plan.outer_sizes.size();
  if (outer_ndim == 0) {
    func(0, 0, plan.inner_size);
    return;
  }

  c10::SmallVector<int64_t, 8> indices(outer_ndim, 0);
  int64_t first_offset = 0;
  int64_t second_offset = 0;
  for (int64_t outer = 0; outer < plan.outer_count; ++outer) {
    func(first_offset, second_offset, plan.inner_size);

    for (int64_t dim = outer_ndim - 1; dim >= 0; --dim) {
      ++indices[dim];
      first_offset += plan.first_outer_strides[dim];
      second_offset += plan.second_outer_strides[dim];
      if (indices[dim] < plan.outer_sizes[dim]) {
        break;
      }
      first_offset -= indices[dim] * plan.first_outer_strides[dim];
      second_offset -= indices[dim] * plan.second_outer_strides[dim];
      indices[dim] = 0;
    }
  }
}

template <typename Func>
void for_each_raw_tensor_triplet_row(
    const RawTensorTripletPlan& plan,
    Func&& func) {
  const int64_t outer_ndim = plan.outer_sizes.size();
  if (outer_ndim == 0) {
    func(0, 0, 0, plan.inner_size);
    return;
  }

  c10::SmallVector<int64_t, 8> indices(outer_ndim, 0);
  int64_t first_offset = 0;
  int64_t second_offset = 0;
  int64_t third_offset = 0;
  for (int64_t outer = 0; outer < plan.outer_count; ++outer) {
    func(first_offset, second_offset, third_offset, plan.inner_size);

    for (int64_t dim = outer_ndim - 1; dim >= 0; --dim) {
      ++indices[dim];
      first_offset += plan.first_outer_strides[dim];
      second_offset += plan.second_outer_strides[dim];
      third_offset += plan.third_outer_strides[dim];
      if (indices[dim] < plan.outer_sizes[dim]) {
        break;
      }
      first_offset -= indices[dim] * plan.first_outer_strides[dim];
      second_offset -= indices[dim] * plan.second_outer_strides[dim];
      third_offset -= indices[dim] * plan.third_outer_strides[dim];
      indices[dim] = 0;
    }
  }
}

} // namespace at::mcpu::ops
