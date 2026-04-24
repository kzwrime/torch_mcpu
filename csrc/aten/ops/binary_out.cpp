#include "Common.h"

#include <ATen/ops/add.h>
#include <ATen/ops/div.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/sub.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

at::Tensor& add_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_other = ops::to_meta_tensor(other);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::add_out(meta_out, meta_self, meta_other, alpha);
  ops::check_out_sizes("aten::add.out", out, meta_out);

  at::native::mcpu::MemoryGuard guard(self, other, out);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_other = ops::get_cpu_tensor_view_if_needed(other);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::add_out(cpu_out, cpu_self, cpu_other, alpha);
  return out;
}

at::Tensor& div_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_other = ops::to_meta_tensor(other);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::div_out(meta_out, meta_self, meta_other);
  ops::check_out_sizes("aten::div.out", out, meta_out);

  at::native::mcpu::MemoryGuard guard(self, other, out);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_other = ops::get_cpu_tensor_view_if_needed(other);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::div_out(cpu_out, cpu_self, cpu_other);
  return out;
}

at::Tensor& mul_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_other = ops::to_meta_tensor(other);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::mul_out(meta_out, meta_self, meta_other);
  ops::check_out_sizes("aten::mul.out", out, meta_out);

  at::native::mcpu::MemoryGuard guard(self, other, out);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_other = ops::get_cpu_tensor_view_if_needed(other);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::mul_out(cpu_out, cpu_self, cpu_other);
  return out;
}

at::Tensor sub_Tensor(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_other = ops::to_meta_tensor(other);
  auto meta_out = at::empty({0}, self.options().device(c10::DeviceType::Meta));
  at::sub_out(meta_out, meta_self, meta_other, alpha);

  auto out = ops::empty_mcpu_from_meta(meta_out, self.options());
  at::native::mcpu::MemoryGuard guard(self, other, out);
  auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
  auto cpu_other = ops::get_cpu_tensor_view_if_needed(other);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::sub_out(cpu_out, cpu_self, cpu_other, alpha);
  return out;
}

at::Tensor& pow_Scalar_out(
    const at::Scalar& self,
    const at::Tensor& exponent,
    at::Tensor& out) {
  auto meta_exponent = ops::to_meta_tensor(exponent);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::pow_out(meta_out, self, meta_exponent);
  ops::check_out_sizes("aten::pow.Scalar_out", out, meta_out);

  at::native::mcpu::MemoryGuard guard(exponent, out);
  auto cpu_exponent = ops::get_cpu_tensor_view_if_needed(exponent);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::pow_out(cpu_out, self, cpu_exponent);
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.out", &add_out);
  m.impl("div.out", &div_out);
  m.impl("mul.out", &mul_out);
  m.impl("sub.Tensor", &sub_Tensor);
  m.impl("pow.Scalar_out", &pow_Scalar_out);
}

} // namespace at::mcpu
