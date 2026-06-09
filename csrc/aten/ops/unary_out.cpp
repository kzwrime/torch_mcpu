#include "Common.h"
#include "runtime/McpuKernelLaunch.h"
#include "runtime/McpuKernelTiming.h"

#include <ATen/ExpandUtils.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/masked_fill.h>
#include <ATen/ops/masked_fill_cpu_dispatch.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/reciprocal.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/zero.h>
#include <torch/library.h>

#include <cmath>

namespace at::mcpu {
namespace {

template <typename scalar_t>
inline bool is_nonzero_value(const scalar_t& value) {
  if constexpr (c10::is_complex<scalar_t>::value) {
    return value.real() != 0 || value.imag() != 0;
  } else {
    return value != scalar_t(0);
  }
}

at::Tensor empty_unary_mcpu(
    const at::Tensor& self,
    at::ScalarType result_dtype) {
  return at::empty_like(
      self, self.options().dtype(result_dtype), at::MemoryFormat::Preserve);
}

at::Tensor& sigmoid_out(const at::Tensor& self, at::Tensor& out) {
  ops::check_out_sizes("aten::sigmoid.out", out, self.sizes());

  launch_timed_kernel(
      "aten::sigmoid_out",
      [self, out](at::mcpu::kernel_timing::Event* timing_event) mutable {
        MCPU_KERNEL_TIMING_SCOPE_EVENT("mcpu::aten::sigmoid", timing_event);
        KernelMemoryGuard guard(self, out);
        auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        at::sigmoid_out(cpu_out, cpu_self);
      });
  return out;
}

at::Tensor sigmoid(const at::Tensor& self) {
  auto result_dtype =
      c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)
      ? at::kFloat
      : self.scalar_type();
  auto out = empty_unary_mcpu(self, result_dtype);
  sigmoid_out(self, out);
  return out;
}

at::Tensor& sigmoid_(at::Tensor& self) {
  launch_kernel(self, [self]() mutable {
    KernelMemoryGuard guard(self);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    at::sigmoid_(cpu_self);
  });
  return self;
}

at::Tensor& silu_out(const at::Tensor& self, at::Tensor& out) {
  ops::check_out_sizes("aten::silu.out", out, self.sizes());

  launch_kernel(out, [self, out]() mutable {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::silu_out(cpu_out, cpu_self);
  });
  return out;
}

at::Tensor silu(const at::Tensor& self) {
  auto out = empty_unary_mcpu(self, self.scalar_type());
  silu_out(self, out);
  return out;
}

at::Tensor& silu_(at::Tensor& self) {
  launch_kernel(self, [self]() mutable {
    KernelMemoryGuard guard(self);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    at::silu_(cpu_self);
  });
  return self;
}

at::Tensor& cos_out(const at::Tensor& self, at::Tensor& out) {
  ops::check_out_sizes("aten::cos.out", out, self.sizes());

  launch_kernel(out, [self, out]() mutable {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::cos_out(cpu_out, cpu_self);
  });
  return out;
}

at::Tensor& sin_out(const at::Tensor& self, at::Tensor& out) {
  ops::check_out_sizes("aten::sin.out", out, self.sizes());

  launch_kernel(out, [self, out]() mutable {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::sin_out(cpu_out, cpu_self);
  });
  return out;
}

at::Tensor& reciprocal_out(const at::Tensor& self, at::Tensor& out) {
  ops::check_out_sizes("aten::reciprocal.out", out, self.sizes());

  launch_kernel(out, [self, out]() mutable {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::reciprocal_out(cpu_out, cpu_self);
  });
  return out;
}

at::Tensor& neg_out(const at::Tensor& self, at::Tensor& out) {
  ops::check_out_sizes("aten::neg.out", out, self.sizes());

  launch_kernel(out, [self, out]() mutable {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::neg_out(cpu_out, cpu_self);
  });
  return out;
}

at::Tensor& clamp_out(
    const at::Tensor& self,
    const std::optional<at::Scalar>& min,
    const std::optional<at::Scalar>& max,
    at::Tensor& out) {
  ops::check_out_sizes("aten::clamp.out", out, self.sizes());

  launch_kernel(out, [self, out, min, max]() mutable {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::clamp_out(cpu_out, cpu_self, min, max);
  });
  return out;
}

at::Tensor& clamp_Tensor_out(
    const at::Tensor& self,
    const std::optional<at::Tensor>& min,
    const std::optional<at::Tensor>& max,
    at::Tensor& out) {
  auto expected_sizes = self.sizes().vec();
  if (min.has_value() && min->defined()) {
    expected_sizes = at::infer_size(expected_sizes, min->sizes());
  }
  if (max.has_value() && max->defined()) {
    expected_sizes = at::infer_size(expected_sizes, max->sizes());
  }
  ops::check_out_sizes("aten::clamp.Tensor_out", out, expected_sizes);

  const at::Tensor min_guard =
      min.has_value() && min->defined() ? *min : at::Tensor();
  const at::Tensor max_guard =
      max.has_value() && max->defined() ? *max : at::Tensor();
  launch_kernel(out, [self, min_guard, max_guard, min, max, out]() mutable {
    KernelMemoryGuard guard(self, min_guard, max_guard, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_min = min.has_value() && min->defined()
        ? std::make_optional(ops::get_cpu_tensor_view_if_needed(*min))
        : std::nullopt;
    auto cpu_max = max.has_value() && max->defined()
        ? std::make_optional(ops::get_cpu_tensor_view_if_needed(*max))
        : std::nullopt;
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::clamp_out(cpu_out, cpu_self, cpu_min, cpu_max);
  });
  return out;
}

at::Tensor& zero_(at::Tensor& self) {
  launch_kernel(self, [self]() mutable {
    KernelMemoryGuard guard(self);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    at::zero_(cpu_self);
  });
  return self;
}

at::Tensor& masked_fill__Scalar(
    at::Tensor& self,
    const at::Tensor& mask,
    const at::Scalar& value) {
  launch_kernel(self, [self, mask, value]() mutable {
    KernelMemoryGuard guard(self, mask);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_mask = ops::get_cpu_tensor_view_if_needed(mask);
    at::cpu::masked_fill_(cpu_self, cpu_mask, value);
  });
  return self;
}

at::Tensor& ne_Scalar_out(
    const at::Tensor& self,
    const at::Scalar& other,
    at::Tensor& out) {
  ops::check_out_sizes("aten::ne.Scalar_out", out, self.sizes());

  launch_kernel(out, [self, out, other]() mutable {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    at::ne_out(cpu_out, cpu_self, other);
  });
  return out;
}

at::Tensor nonzero(const at::Tensor& self) {
  const auto ndim = self.dim();
  const auto numel = self.numel();
  const auto sizes = self.sizes().vec();
  const auto strides = self.strides().vec();
  auto count = at::empty(
      {}, self.options().device(c10::DeviceType::CPU).dtype(at::kLong));
  launch_kernel(self, [self, count, ndim, numel, sizes, strides]() mutable {
    KernelMemoryGuard guard(self);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    int64_t count_value = 0;
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        at::ScalarType::Bool,
        cpu_self.scalar_type(),
        "mcpu_nonzero_count",
        [&] {
          const auto* data = cpu_self.const_data_ptr<scalar_t>();
          std::vector<int64_t> index(ndim, 0);
          for (int64_t linear = 0; linear < numel; ++linear) {
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
    *count.mutable_data_ptr<int64_t>() = count_value;
  });
  at::native::mcpu::synchronize_if_mcpu(self);
  const auto count_value = count.item<int64_t>();
  auto out = at::empty(
      {count_value, ndim},
      self.options().device(c10::DeviceType::PrivateUse1).dtype(at::kLong));

  launch_kernel(out, [self, out, ndim, numel, sizes, strides]() mutable {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self).clone();
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    const auto ndim = cpu_self.dim();
    const auto numel = cpu_self.numel();
    const auto sizes = cpu_self.sizes().vec();
    const auto strides = cpu_self.strides().vec();

    int64_t row = 0;
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        at::ScalarType::Bool,
        cpu_self.scalar_type(),
        "mcpu_nonzero_write",
        [&] {
          const auto* data = cpu_self.const_data_ptr<scalar_t>();
          auto* out_ptr = cpu_out.mutable_data_ptr<int64_t>();
          std::vector<int64_t> index(ndim, 0);
          for (int64_t linear = 0; linear < numel; ++linear) {
            int64_t offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
              offset += index[d] * strides[d];
            }
            if (is_nonzero_value(data[offset])) {
              for (int64_t d = 0; d < ndim; ++d) {
                out_ptr[row * ndim + d] = index[d];
              }
              ++row;
            }
            for (int64_t d = ndim - 1; d >= 0; --d) {
              if (++index[d] < sizes[d]) {
                break;
              }
              index[d] = 0;
            }
          }
        });
  });
  return out;
}

at::Tensor& sum_IntList_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype,
    at::Tensor& out) {
  auto expected_sizes = ops::reduction_sizes(self.sizes(), dim, keepdim);
  ops::check_out_sizes("aten::sum.IntList_out", out, expected_sizes);

  std::optional<std::vector<int64_t>> dim_vec;
  if (dim.has_value()) {
    dim_vec = std::vector<int64_t>(dim->begin(), dim->end());
  }

  launch_kernel(out, [self, out, dim_vec, keepdim, dtype]() mutable {
    KernelMemoryGuard guard(self, out);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
    auto cpu_dim = dim_vec.has_value() ? at::OptionalIntArrayRef(*dim_vec)
                                       : at::OptionalIntArrayRef();
    at::sum_out(cpu_out, cpu_self, cpu_dim, keepdim, dtype);
  });
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("clamp.out", &clamp_out);
  m.impl("clamp.Tensor_out", &clamp_Tensor_out);
  m.impl("cos.out", &cos_out);
  m.impl("masked_fill_.Scalar", &masked_fill__Scalar);
  m.impl("ne.Scalar_out", &ne_Scalar_out);
  m.impl("neg.out", &neg_out);
  m.impl("nonzero", &nonzero);
  m.impl("sigmoid", &sigmoid);
  m.impl("sigmoid.out", &sigmoid_out);
  m.impl("sigmoid_", &sigmoid_);
  m.impl("sin.out", &sin_out);
  m.impl("silu", &silu);
  m.impl("silu.out", &silu_out);
  m.impl("silu_", &silu_);
  m.impl("reciprocal.out", &reciprocal_out);
  m.impl("sum.IntList_out", &sum_IntList_out);
  m.impl("zero_", &zero_);
}

} // namespace at::mcpu
