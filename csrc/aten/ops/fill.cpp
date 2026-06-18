#include "Common.h"
#include "RawPlan.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ops/fill.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

std::optional<ops::RawTensorPlan> make_raw_fill_plan(const at::Tensor& self) {
  if (!ops::is_raw_dtype_supported(self.scalar_type())) {
    return std::nullopt;
  }
  return ops::make_raw_tensor_plan(self);
}

template <typename scalar_t>
void raw_fill_kernel(
    scalar_t* base,
    scalar_t value,
    const ops::RawTensorPlan& plan) {
  ops::for_each_raw_tensor_row(
      plan, [base, value](int64_t offset, int64_t inner_size) {
        auto* row = base + offset;
        for (int64_t i = 0; i < inner_size; ++i) {
          row[i] = value;
        }
      });
}

bool raw_fill__scalar(at::Tensor& self, const at::Scalar& value) {
  auto plan = make_raw_fill_plan(self);
  if (!plan.has_value()) {
    return false;
  }
  if (self.numel() == 0) {
    return true;
  }

  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "mcpu_raw_fill__scalar",
      [&] {
        auto* self_ptr = self.mutable_data_ptr<scalar_t>();
        const auto scalar_value = value.to<scalar_t>();
        MCPU_LAUNCH_TIMED_KERNEL(
            "mcpu::aten::fill_.Scalar.raw",
            ([ self_ptr, scalar_value, plan = *std::move(plan) ]),
            {
              KernelPointerMemoryGuard guard({self_ptr});
              raw_fill_kernel(self_ptr, scalar_value, plan);
            });
      });
  return true;
}

at::Tensor& fill__scalar(at::Tensor& self, const at::Scalar& value) {
  if (raw_fill__scalar(self, value)) {
    return self;
  }

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::fill_.Scalar", ([ self, value ]), {
    KernelMemoryGuard guard(self);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    at::fill_(cpu_self, value);
  });
  return self;
}

at::Tensor& fill__tensor(at::Tensor& self, const at::Tensor& value) {
  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::fill_.Tensor", ([ self, value ]), {
    KernelMemoryGuard guard(self, value);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    auto cpu_value = ops::get_cpu_tensor_view_if_needed(value);
    at::fill_(cpu_self, cpu_value);
  });
  return self;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("fill_.Scalar", &fill__scalar);
  m.impl("fill_.Tensor", &fill__tensor);
}

} // namespace at::mcpu
