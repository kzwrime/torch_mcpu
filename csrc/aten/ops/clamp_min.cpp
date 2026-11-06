#include "Common.h"

#include <ATen/ExpandUtils.h>
#include <ATen/ops/clamp_min.h>
#include <torch/library.h>

#include <memory>
#include <optional>
#include <utility>

// Explicit mcpu implementations of:
//   aten::clamp_min          / clamp_min_          / clamp_min.out (Scalar)
//   aten::clamp_min.Tensor   / clamp_min_.Tensor   / clamp_min.Tensor_out
//   (Tensor)
//
// Shape / dtype are derived on the host with C++ meta-info helpers
// (at::infer_size / at::result_type), the output is pre-allocated on mcpu and
// the async kernel rebuilds temporary CPU views from raw pointers before
// calling the CPU `clamp_min_out`. No at::Tensor is captured by the queued
// task.

namespace at::mcpu {
namespace {

bool is_cpu_scalar_tensor(const at::Tensor& tensor) {
  // ATen permits a zero-dimensional CPU scalar next to an accelerator tensor.
  // It is materialized by value before launch, so no CPU Tensor owner or CPU
  // storage pointer enters the asynchronous task.
  return tensor.is_cpu() && tensor.dim() == 0;
}

// A tensor operand that is either an mcpu tensor view or a zero-dimensional CPU
// scalar tensor materialized by value before launch.
struct OperandSpec {
  ops::TensorViewSpec view;
  std::optional<at::Scalar> scalar;
  at::ScalarType scalar_type = at::kFloat;
};

OperandSpec make_operand_spec(const at::Tensor& tensor) {
  if (is_cpu_scalar_tensor(tensor)) {
    return OperandSpec{{}, tensor.item(), tensor.scalar_type()};
  }
  return OperandSpec{
      ops::make_cpu_view_spec(tensor), std::nullopt, tensor.scalar_type()};
}

at::Tensor cpu_tensor_from_operand_spec(const OperandSpec& spec) {
  if (spec.scalar.has_value()) {
    return at::scalar_tensor(
        *spec.scalar, at::TensorOptions().dtype(spec.scalar_type));
  }
  return ops::cpu_view_from_spec(spec.view);
}

struct ClampMinScalarArgs {
  ops::TensorViewSpec self;
  ops::TensorViewSpec out;
  at::Scalar min;
};

struct ClampMinTensorArgs {
  OperandSpec self;
  OperandSpec min;
  ops::TensorViewSpec out;
};

// ---------------------------------------------------------------------------
// Scalar overloads
// ---------------------------------------------------------------------------

at::ScalarType clamp_min_scalar_result_type(
    const at::Tensor& self,
    const at::Scalar& min) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      !self.is_complex() && !min.isComplex(),
      "clamp is not supported for complex types");
  return at::result_type(self, min);
}

void check_clamp_min_scalar_out_dtype(
    const char* op_name,
    const at::Tensor& self,
    at::ScalarType result_type,
    const at::Tensor& out) {
  if (at::isFloatingType(self.scalar_type())) {
    // Floating self is a borrowing unary op: out must have the same dtype.
    TORCH_CHECK(
        out.scalar_type() == self.scalar_type(),
        op_name,
        ": expected out tensor to have dtype ",
        self.scalar_type(),
        ", but got ",
        out.scalar_type(),
        " instead");
  } else {
    TORCH_CHECK(
        at::canCast(result_type, out.scalar_type()),
        op_name,
        ": result type ",
        result_type,
        " can't be cast to the desired output type ",
        out.scalar_type());
  }
}

at::Tensor& clamp_min_out(
    const at::Tensor& self,
    const at::Scalar& min,
    at::Tensor& out) {
  const auto result_type = clamp_min_scalar_result_type(self, min);
  check_clamp_min_scalar_out_dtype(
      "aten::clamp_min.out", self, result_type, out);
  ops::check_out_sizes("aten::clamp_min.out", out, self.sizes());
  if (out.numel() == 0) {
    return out;
  }

  auto args = std::make_unique<ClampMinScalarArgs>(ClampMinScalarArgs{
      ops::make_cpu_view_spec(self), ops::make_cpu_view_spec(out), min});
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::clamp_min.out", ([args = std::move(args)]), {
        KernelPointerMemoryGuard guard({args->self.data, args->out.data});
        auto cpu_self = ops::cpu_view_from_spec(args->self);
        auto cpu_out = ops::cpu_view_from_spec(args->out);
        at::clamp_min_out(cpu_out, cpu_self, args->min);
      });
  return out;
}

at::Tensor clamp_min(const at::Tensor& self, const at::Scalar& min) {
  const auto result_type = clamp_min_scalar_result_type(self, min);
  auto out = at::empty_like(
      self, self.options().dtype(result_type), at::MemoryFormat::Preserve);
  clamp_min_out(self, min, out);
  return out;
}

at::Tensor& clamp_min_(at::Tensor& self, const at::Scalar& min) {
  const auto result_type = clamp_min_scalar_result_type(self, min);
  check_clamp_min_scalar_out_dtype("aten::clamp_min_", self, result_type, self);
  if (self.numel() == 0) {
    return self;
  }

  auto self_spec = ops::make_cpu_view_spec(self);
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::clamp_min_", ([ self_spec = std::move(self_spec), min ]), {
        KernelPointerMemoryGuard guard({self_spec.data});
        auto cpu_self = ops::cpu_view_from_spec(self_spec);
        at::clamp_min_out(cpu_self, cpu_self, min);
      });
  return self;
}

// ---------------------------------------------------------------------------
// Tensor overloads
// ---------------------------------------------------------------------------

at::ScalarType clamp_min_tensor_result_type(
    const at::Tensor& self,
    const at::Tensor& min) {
  TORCH_CHECK(
      !self.is_complex() && !min.is_complex(),
      "maximum not implemented for complex tensors.");
  return at::result_type(self, min);
}

at::Tensor& clamp_min_Tensor_out_impl(
    const char* op_name,
    const at::Tensor& self,
    const at::Tensor& min,
    at::Tensor& out) {
  const auto result_type = clamp_min_tensor_result_type(self, min);
  TORCH_CHECK(
      at::canCast(result_type, out.scalar_type()),
      op_name,
      ": result type ",
      result_type,
      " can't be cast to the desired output type ",
      out.scalar_type());
  const auto expected_sizes = at::infer_size(self.sizes(), min.sizes());
  ops::check_out_sizes(op_name, out, expected_sizes);
  if (out.numel() == 0) {
    return out;
  }

  auto args = std::make_unique<ClampMinTensorArgs>(ClampMinTensorArgs{
      make_operand_spec(self),
      make_operand_spec(min),
      ops::make_cpu_view_spec(out)});
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::clamp_min.Tensor_out", ([args = std::move(args)]), {
        KernelPointerMemoryGuard guard(
            {args->self.view.data, args->min.view.data, args->out.data});
        auto cpu_self = cpu_tensor_from_operand_spec(args->self);
        auto cpu_min = cpu_tensor_from_operand_spec(args->min);
        auto cpu_out = ops::cpu_view_from_spec(args->out);
        at::clamp_min_out(cpu_out, cpu_self, cpu_min);
      });
  return out;
}

at::Tensor& clamp_min_Tensor_out(
    const at::Tensor& self,
    const at::Tensor& min,
    at::Tensor& out) {
  return clamp_min_Tensor_out_impl(
      "aten::clamp_min.Tensor_out", self, min, out);
}

at::Tensor clamp_min_Tensor(const at::Tensor& self, const at::Tensor& min) {
  const auto result_type = clamp_min_tensor_result_type(self, min);
  // self may be the CPU scalar operand, so take device options from the mcpu
  // operand.
  const auto& device_tensor = is_cpu_scalar_tensor(self) ? min : self;
  const auto out_sizes = at::infer_size(self.sizes(), min.sizes());
  auto out = device_tensor.sizes().equals(out_sizes)
      ? at::empty_like(
            device_tensor,
            device_tensor.options().dtype(result_type),
            at::MemoryFormat::Preserve)
      : at::empty(out_sizes, device_tensor.options().dtype(result_type));
  clamp_min_Tensor_out_impl("aten::clamp_min.Tensor", self, min, out);
  return out;
}

at::Tensor& clamp_min__Tensor(at::Tensor& self, const at::Tensor& min) {
  return clamp_min_Tensor_out_impl("aten::clamp_min_.Tensor", self, min, self);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("clamp_min", &clamp_min);
  m.impl("clamp_min_", &clamp_min_);
  m.impl("clamp_min.out", &clamp_min_out);
  m.impl("clamp_min.Tensor", &clamp_min_Tensor);
  m.impl("clamp_min_.Tensor", &clamp_min__Tensor);
  m.impl("clamp_min.Tensor_out", &clamp_min_Tensor_out);
}

} // namespace at::mcpu
