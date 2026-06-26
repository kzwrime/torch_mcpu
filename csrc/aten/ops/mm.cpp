#include "Common.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ops/addmm.h>
#include <ATen/ops/mm.h>
#include <torch/library.h>

#include <memory>

namespace at::mcpu {
namespace {

at::Tensor& mm_out(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  TORCH_CHECK(self.dim() == 2, "mm: self must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mm: mat2 must be a matrix");
  TORCH_CHECK(
      self.size(1) == mat2.size(0), "mm: self.size(1) must match mat2.size(0)");
  ops::check_out_sizes("aten::mm.out", out, {self.size(0), mat2.size(1)});

  auto self_spec = ops::make_cpu_view_spec(self);
  auto mat2_spec = ops::make_cpu_view_spec(mat2);
  auto out_spec = ops::make_cpu_view_spec(out);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::mm_out",
      ([
        self_spec = std::move(self_spec),
        mat2_spec = std::move(mat2_spec),
        out_spec = std::move(out_spec)
      ]),
      {
        KernelPointerMemoryGuard guard(
            {self_spec.data, mat2_spec.data, out_spec.data});
        auto cpu_self = ops::cpu_view_from_spec(self_spec);
        auto cpu_mat2 = ops::cpu_view_from_spec(mat2_spec);
        auto cpu_out = ops::cpu_view_from_spec(out_spec);
        at::mm_out(cpu_out, cpu_self, cpu_mat2);
      });
  return out;
}

at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
  auto out = at::empty(
      {self.size(0), mat2.size(1)},
      self.options().dtype(at::result_type(self, mat2)));
  mm_out(self, mat2, out);
  return out;
}

at::Tensor& addmm_out(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  TORCH_CHECK(mat1.dim() == 2, "addmm: mat1 must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "addmm: mat2 must be a matrix");
  TORCH_CHECK(
      mat1.size(1) == mat2.size(0),
      "addmm: mat1.size(1) must match mat2.size(0)");
  ops::check_out_sizes("aten::addmm.out", out, {mat1.size(0), mat2.size(1)});

  struct AddmmArgs {
    ops::TensorViewSpec self;
    ops::TensorViewSpec mat1;
    ops::TensorViewSpec mat2;
    ops::TensorViewSpec out;
    at::Scalar beta;
    at::Scalar alpha;
  };
  auto args = std::make_unique<AddmmArgs>(AddmmArgs{
      ops::make_cpu_view_spec(self),
      ops::make_cpu_view_spec(mat1),
      ops::make_cpu_view_spec(mat2),
      ops::make_cpu_view_spec(out),
      beta,
      alpha});

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::addmm_out", ([args = std::move(args)]), {
        KernelPointerMemoryGuard guard(
            {args->self.data,
             args->mat1.data,
             args->mat2.data,
             args->out.data});
        auto cpu_self = ops::cpu_view_from_spec(args->self);
        auto cpu_mat1 = ops::cpu_view_from_spec(args->mat1);
        auto cpu_mat2 = ops::cpu_view_from_spec(args->mat2);
        auto cpu_out = ops::cpu_view_from_spec(args->out);
        at::addmm_out(
            cpu_out, cpu_self, cpu_mat1, cpu_mat2, args->beta, args->alpha);
      });
  return out;
}

at::Tensor addmm(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  auto out = at::empty(
      {mat1.size(0), mat2.size(1)},
      mat1.options().dtype(at::result_type(mat1, mat2)));
  addmm_out(self, mat1, mat2, beta, alpha, out);
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("mm", &mm);
  m.impl("mm.out", &mm_out);
  m.impl("addmm", &addmm);
  m.impl("addmm.out", &addmm_out);
}

} // namespace at::mcpu
