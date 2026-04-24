#include "Common.h"

#include <ATen/ops/cos.h>
#include <ATen/ops/count_nonzero.h>
#include <ATen/ops/masked_fill.h>
#include <ATen/ops/masked_fill_cpu_dispatch.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/reciprocal.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/zero.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

at::Tensor& cos_out(const at::Tensor& self, at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::cos_out(meta_out, meta_self);
  ops::check_out_sizes("aten::cos.out", out, meta_out);

  at::native::mcpu::MemoryGuard guard(self, out);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::cos_out(cpu_out, cpu_self);
  return out;
}

at::Tensor& sin_out(const at::Tensor& self, at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::sin_out(meta_out, meta_self);
  ops::check_out_sizes("aten::sin.out", out, meta_out);

  at::native::mcpu::MemoryGuard guard(self, out);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::sin_out(cpu_out, cpu_self);
  return out;
}

at::Tensor& reciprocal_out(const at::Tensor& self, at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::reciprocal_out(meta_out, meta_self);
  ops::check_out_sizes("aten::reciprocal.out", out, meta_out);

  at::native::mcpu::MemoryGuard guard(self, out);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::reciprocal_out(cpu_out, cpu_self);
  return out;
}

at::Tensor& zero_(at::Tensor& self) {
  at::native::mcpu::MemoryGuard guard(self);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  at::zero_(cpu_self);
  return self;
}

at::Tensor& masked_fill__Scalar(
    at::Tensor& self,
    const at::Tensor& mask,
    const at::Scalar& value) {
  at::native::mcpu::MemoryGuard guard(self, mask);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_mask = ops::get_cpu_tensor_view_if_needed(mask);
  at::cpu::masked_fill_(cpu_self, cpu_mask, value);
  return self;
}

at::Tensor& ne_Scalar_out(
    const at::Tensor& self,
    const at::Scalar& other,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::ne_out(meta_out, meta_self, other);
  ops::check_out_sizes("aten::ne.Scalar_out", out, meta_out);

  at::native::mcpu::MemoryGuard guard(self, out);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::ne_out(cpu_out, cpu_self, other);
  return out;
}

at::Tensor nonzero(const at::Tensor& self) {
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  const auto count = at::count_nonzero(cpu_self).item<int64_t>();
  auto out = at::empty(
      {count, self.dim()},
      self.options().device(c10::DeviceType::PrivateUse1).dtype(at::kLong));

  at::native::mcpu::MemoryGuard guard(self, out);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::nonzero_out(cpu_out, cpu_self);
  return out;
}

at::Tensor& sum_IntList_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::sum_out(meta_out, meta_self, dim, keepdim, dtype);
  ops::check_out_sizes("aten::sum.IntList_out", out, meta_out);

  at::native::mcpu::MemoryGuard guard(self, out);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::sum_out(cpu_out, cpu_self, dim, keepdim, dtype);
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("cos.out", &cos_out);
  m.impl("masked_fill_.Scalar", &masked_fill__Scalar);
  m.impl("ne.Scalar_out", &ne_Scalar_out);
  m.impl("nonzero", &nonzero);
  m.impl("sin.out", &sin_out);
  m.impl("reciprocal.out", &reciprocal_out);
  m.impl("sum.IntList_out", &sum_IntList_out);
  m.impl("zero_", &zero_);
}

} // namespace at::mcpu
