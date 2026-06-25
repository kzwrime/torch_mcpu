#include "Common.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ExpandUtils.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/where.h>
#include <torch/library.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace at::mcpu {
namespace {

struct TensorViewSpec {
  void* data = nullptr;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  at::TensorOptions options;
};

struct WhereSelfOutArgs {
  TensorViewSpec condition;
  TensorViewSpec self;
  TensorViewSpec other;
  TensorViewSpec out;
  std::vector<const void*> ptrs;
};

struct WhereScalarSelfArgs {
  TensorViewSpec condition;
  at::Scalar self;
  TensorViewSpec other;
  TensorViewSpec out;
  at::ScalarType result_dtype;
  std::vector<const void*> ptrs;
};

struct WhereScalarOtherArgs {
  TensorViewSpec condition;
  TensorViewSpec self;
  at::Scalar other;
  TensorViewSpec out;
  at::ScalarType result_dtype;
  std::vector<const void*> ptrs;
};

struct WhereScalarArgs {
  TensorViewSpec condition;
  at::Scalar self;
  at::Scalar other;
  TensorViewSpec out;
  at::ScalarType result_dtype;
  std::vector<const void*> ptrs;
};

struct WhereCountArgs {
  TensorViewSpec condition;
  int64_t numel = 0;
  int64_t* count = nullptr;
  std::vector<const void*> ptrs;
};

struct WhereWriteArgs {
  TensorViewSpec condition;
  std::vector<TensorViewSpec> result;
  int64_t ndim = 0;
  int64_t numel = 0;
  std::vector<const void*> ptrs;
};

int64_t numel_from_sizes(const std::vector<int64_t>& sizes) {
  int64_t numel = 1;
  for (const auto size : sizes) {
    numel *= size;
  }
  return numel;
}

TensorViewSpec make_cpu_view_spec(const at::Tensor& tensor) {
  TensorViewSpec spec;
  spec.data = tensor.numel() == 0 ? nullptr : tensor.data_ptr();
  spec.sizes = tensor.sizes().vec();
  spec.strides = tensor.strides().vec();
  spec.options = tensor.options().device(c10::DeviceType::CPU);
  return spec;
}

at::Tensor cpu_view_from_spec(const TensorViewSpec& spec) {
  c10::InferenceMode inference_guard(false);
  if (numel_from_sizes(spec.sizes) == 0) {
    return at::empty_strided(spec.sizes, spec.strides, spec.options);
  }
  return at::from_blob(spec.data, spec.sizes, spec.strides, spec.options);
}

std::vector<at::Tensor> cpu_views_from_specs(
    const std::vector<TensorViewSpec>& specs) {
  std::vector<at::Tensor> tensors;
  tensors.reserve(specs.size());
  for (const auto& spec : specs) {
    tensors.push_back(cpu_view_from_spec(spec));
  }
  return tensors;
}

std::vector<const void*> pointer_list(
    std::initializer_list<TensorViewSpec> specs) {
  std::vector<const void*> ptrs;
  ptrs.reserve(specs.size());
  for (const auto& spec : specs) {
    if (spec.data != nullptr) {
      ptrs.push_back(spec.data);
    }
  }
  return ptrs;
}

std::vector<const void*> pointer_list(
    const TensorViewSpec& first,
    const std::vector<TensorViewSpec>& rest) {
  std::vector<const void*> ptrs;
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

template <typename scalar_t>
inline bool is_nonzero_value(const scalar_t& value) {
  if constexpr (c10::is_complex<scalar_t>::value) {
    return value.real() != 0 || value.imag() != 0;
  } else {
    return value != scalar_t(0);
  }
}

void check_where_condition(const at::Tensor& condition) {
  TORCH_CHECK(
      condition.scalar_type() == at::ScalarType::Bool,
      "where expected condition to be a boolean tensor, but got a tensor with dtype ",
      condition.scalar_type());
}

void check_where_out(
    const char* op_name,
    const at::Tensor& out,
    at::IntArrayRef expected_sizes,
    at::ScalarType expected_dtype) {
  ops::check_out_sizes(op_name, out, expected_sizes);
  TORCH_CHECK(
      out.scalar_type() == expected_dtype,
      op_name,
      ": expected out dtype ",
      expected_dtype,
      ", but got ",
      out.scalar_type());
}

std::vector<int64_t> infer_where_sizes(
    at::IntArrayRef condition_sizes,
    at::IntArrayRef self_sizes,
    at::IntArrayRef other_sizes) {
  auto condition_self_sizes = at::infer_size(condition_sizes, self_sizes);
  return at::infer_size(condition_self_sizes, other_sizes);
}

at::Tensor empty_where_mcpu(
    const at::Tensor& condition,
    at::IntArrayRef sizes,
    at::ScalarType dtype) {
  return at::empty(
      sizes,
      condition.options().device(c10::DeviceType::PrivateUse1).dtype(dtype));
}

at::Tensor scalar_cpu_tensor(const at::Scalar& scalar, at::ScalarType dtype) {
  return at::scalar_tensor(
      scalar, at::TensorOptions().device(c10::DeviceType::CPU).dtype(dtype));
}

at::Tensor& where_self_out(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  check_where_condition(condition);
  const auto expected_sizes =
      infer_where_sizes(condition.sizes(), self.sizes(), other.sizes());
  const auto expected_dtype = at::result_type(self, other);
  check_where_out("aten::where.self_out", out, expected_sizes, expected_dtype);

  auto condition_spec = make_cpu_view_spec(condition);
  auto self_spec = make_cpu_view_spec(self);
  auto other_spec = make_cpu_view_spec(other);
  auto out_spec = make_cpu_view_spec(out);
  auto args = std::make_unique<WhereSelfOutArgs>(WhereSelfOutArgs{
      condition_spec,
      self_spec,
      other_spec,
      out_spec,
      pointer_list({condition_spec, self_spec, other_spec, out_spec})});

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::where.self_out", ([args = std::move(args)]), {
        KernelPointerMemoryGuard guard(args->ptrs);
        auto cpu_condition = cpu_view_from_spec(args->condition);
        auto cpu_self = cpu_view_from_spec(args->self);
        auto cpu_other = cpu_view_from_spec(args->other);
        auto cpu_out = cpu_view_from_spec(args->out);
        at::where_out(cpu_out, cpu_condition, cpu_self, cpu_other);
      });
  return out;
}

at::Tensor where_self(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  check_where_condition(condition);
  const auto expected_sizes =
      infer_where_sizes(condition.sizes(), self.sizes(), other.sizes());
  auto out =
      empty_where_mcpu(condition, expected_sizes, at::result_type(self, other));
  where_self_out(condition, self, other, out);
  return out;
}

at::Tensor where_ScalarSelf(
    const at::Tensor& condition,
    const at::Scalar& self,
    const at::Tensor& other) {
  check_where_condition(condition);
  const auto expected_sizes = at::infer_size(condition.sizes(), other.sizes());
  const auto result_dtype = at::result_type(self, other);
  auto out = empty_where_mcpu(condition, expected_sizes, result_dtype);
  auto condition_spec = make_cpu_view_spec(condition);
  auto other_spec = make_cpu_view_spec(other);
  auto out_spec = make_cpu_view_spec(out);
  auto args = std::make_unique<WhereScalarSelfArgs>(WhereScalarSelfArgs{
      condition_spec,
      self,
      other_spec,
      out_spec,
      result_dtype,
      pointer_list({condition_spec, other_spec, out_spec})});

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::where.ScalarSelf", ([args = std::move(args)]), {
        KernelPointerMemoryGuard guard(args->ptrs);
        auto cpu_condition = cpu_view_from_spec(args->condition);
        auto cpu_self_tensor =
            scalar_cpu_tensor(args->self, args->result_dtype);
        auto cpu_other = cpu_view_from_spec(args->other);
        auto cpu_out = cpu_view_from_spec(args->out);
        at::where_out(cpu_out, cpu_condition, cpu_self_tensor, cpu_other);
      });
  return out;
}

at::Tensor where_ScalarOther(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Scalar& other) {
  check_where_condition(condition);
  const auto expected_sizes = at::infer_size(condition.sizes(), self.sizes());
  const auto result_dtype = at::result_type(self, other);
  auto out = empty_where_mcpu(condition, expected_sizes, result_dtype);
  auto condition_spec = make_cpu_view_spec(condition);
  auto self_spec = make_cpu_view_spec(self);
  auto out_spec = make_cpu_view_spec(out);
  auto args = std::make_unique<WhereScalarOtherArgs>(WhereScalarOtherArgs{
      condition_spec,
      self_spec,
      other,
      out_spec,
      result_dtype,
      pointer_list({condition_spec, self_spec, out_spec})});

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::where.ScalarOther", ([args = std::move(args)]), {
        KernelPointerMemoryGuard guard(args->ptrs);
        auto cpu_condition = cpu_view_from_spec(args->condition);
        auto cpu_self = cpu_view_from_spec(args->self);
        auto cpu_other_tensor =
            scalar_cpu_tensor(args->other, args->result_dtype);
        auto cpu_out = cpu_view_from_spec(args->out);
        at::where_out(cpu_out, cpu_condition, cpu_self, cpu_other_tensor);
      });
  return out;
}

at::Tensor where_Scalar(
    const at::Tensor& condition,
    const at::Scalar& self,
    const at::Scalar& other) {
  check_where_condition(condition);
  const auto result_dtype = at::result_type(self, other);
  auto out = empty_where_mcpu(condition, condition.sizes(), result_dtype);
  auto condition_spec = make_cpu_view_spec(condition);
  auto out_spec = make_cpu_view_spec(out);
  auto args = std::make_unique<WhereScalarArgs>(WhereScalarArgs{
      condition_spec,
      self,
      other,
      out_spec,
      result_dtype,
      pointer_list({condition_spec, out_spec})});

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::where.Scalar", ([args = std::move(args)]), {
        KernelPointerMemoryGuard guard(args->ptrs);
        auto cpu_condition = cpu_view_from_spec(args->condition);
        auto cpu_self_tensor =
            scalar_cpu_tensor(args->self, args->result_dtype);
        auto cpu_other_tensor =
            scalar_cpu_tensor(args->other, args->result_dtype);
        auto cpu_out = cpu_view_from_spec(args->out);
        at::where_out(
            cpu_out, cpu_condition, cpu_self_tensor, cpu_other_tensor);
      });
  return out;
}

std::vector<at::Tensor> where(const at::Tensor& condition) {
  const auto condition_spec = make_cpu_view_spec(condition);
  const auto ndim = condition.dim();
  const auto result_count = std::max<int64_t>(ndim, 1);
  const auto numel = condition.numel();
  auto count_tensor = at::empty(
      {}, at::TensorOptions().device(c10::DeviceType::CPU).dtype(at::kLong));
  auto* count_ptr = count_tensor.mutable_data_ptr<int64_t>();
  auto count_args = std::make_unique<WhereCountArgs>(WhereCountArgs{
      condition_spec, numel, count_ptr, pointer_list({condition_spec})});

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::where.count", ([count_args = std::move(count_args)]), {
        KernelPointerMemoryGuard guard(count_args->ptrs);
        auto cpu_condition = cpu_view_from_spec(count_args->condition);
        int64_t count_value = 0;
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            at::ScalarType::Bool,
            cpu_condition.scalar_type(),
            "mcpu_where_count",
            [&] {
              const auto* data = cpu_condition.const_data_ptr<scalar_t>();
              const auto sizes = cpu_condition.sizes().vec();
              const auto strides = cpu_condition.strides().vec();
              const auto ndim = static_cast<int64_t>(sizes.size());
              std::vector<int64_t> index(ndim, 0);
              for (int64_t linear = 0; linear < count_args->numel; ++linear) {
                int64_t offset = 0;
                for (int64_t d = 0; d < ndim; ++d) {
                  offset += index[d] * strides[d];
                }
                if (is_nonzero_value(data[offset])) {
                  ++count_value;
                }
                for (int64_t d = ndim - 1; d >= 0; --d) {
                  if (++index[d] < sizes[d]) {
                    break;
                  }
                  index[d] = 0;
                }
              }
            });
        *count_args->count = count_value;
      });
  at::native::mcpu::synchronize_if_mcpu(condition);
  const auto count = count_tensor.item<int64_t>();

  std::vector<at::Tensor> result;
  result.reserve(result_count);
  for (int64_t dim = 0; dim < result_count; ++dim) {
    result.push_back(at::empty(
        {count},
        condition.options()
            .device(c10::DeviceType::PrivateUse1)
            .dtype(at::kLong)));
  }

  if (count == 0) {
    return result;
  }

  std::vector<TensorViewSpec> result_specs;
  result_specs.reserve(result.size());
  for (const auto& tensor : result) {
    result_specs.push_back(make_cpu_view_spec(tensor));
  }
  auto write_ptrs = pointer_list(condition_spec, result_specs);
  auto write_args = std::make_unique<WhereWriteArgs>(
      WhereWriteArgs{condition_spec, result_specs, ndim, numel, write_ptrs});

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::where", ([write_args = std::move(write_args)]), {
        KernelPointerMemoryGuard guard(write_args->ptrs);
        auto cpu_condition = cpu_view_from_spec(write_args->condition);
        auto cpu_result = cpu_views_from_specs(write_args->result);
        if (write_args->ndim == 0) {
          cpu_result[0].mutable_data_ptr<int64_t>()[0] = 0;
          return;
        }

        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            at::ScalarType::Bool,
            cpu_condition.scalar_type(),
            "mcpu_where_write",
            [&] {
              const auto* data = cpu_condition.const_data_ptr<scalar_t>();
              const auto sizes = cpu_condition.sizes().vec();
              const auto strides = cpu_condition.strides().vec();
              std::vector<int64_t*> out_ptrs;
              out_ptrs.reserve(write_args->ndim);
              for (int64_t dim = 0; dim < write_args->ndim; ++dim) {
                out_ptrs.push_back(cpu_result[dim].mutable_data_ptr<int64_t>());
              }

              int64_t row = 0;
              std::vector<int64_t> index(write_args->ndim, 0);
              for (int64_t linear = 0; linear < write_args->numel; ++linear) {
                int64_t offset = 0;
                for (int64_t d = 0; d < write_args->ndim; ++d) {
                  offset += index[d] * strides[d];
                }
                if (is_nonzero_value(data[offset])) {
                  for (int64_t d = 0; d < write_args->ndim; ++d) {
                    out_ptrs[d][row] = index[d];
                  }
                  ++row;
                }
                for (int64_t d = write_args->ndim - 1; d >= 0; --d) {
                  if (++index[d] < sizes[d]) {
                    break;
                  }
                  index[d] = 0;
                }
              }
            });
      });
  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("where.self", &where_self);
  m.impl("where.self_out", &where_self_out);
  m.impl("where.ScalarSelf", &where_ScalarSelf);
  m.impl("where.ScalarOther", &where_ScalarOther);
  m.impl("where.Scalar", &where_Scalar);
  m.impl("where", &where);
}

} // namespace at::mcpu
