#include "Common.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ops/addmm.h>
#include <ATen/ops/mm.h>
#include <torch/library.h>

#include <memory>
#include <vector>

namespace at::mcpu {
namespace {

struct TensorMeta {
  void* ptr;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  at::TensorOptions options;
};

TensorMeta make_tensor_meta(const at::Tensor& tensor) {
  return TensorMeta{
      tensor.data_ptr(),
      tensor.sizes().vec(),
      tensor.strides().vec(),
      tensor.options().device(c10::DeviceType::CPU)};
}

at::Tensor tensor_from_meta(const TensorMeta& meta) {
  return at::from_blob(meta.ptr, meta.sizes, meta.strides, meta.options);
}

at::Tensor& mm_out(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  TORCH_CHECK(self.dim() == 2, "mm: self must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mm: mat2 must be a matrix");
  TORCH_CHECK(
      self.size(1) == mat2.size(0), "mm: self.size(1) must match mat2.size(0)");
  ops::check_out_sizes("aten::mm.out", out, {self.size(0), mat2.size(1)});

  auto self_meta = make_tensor_meta(self);
  auto mat2_meta = make_tensor_meta(mat2);
  auto out_meta = make_tensor_meta(out);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::mm_out",
      ([
        self_meta = std::move(self_meta),
        mat2_meta = std::move(mat2_meta),
        out_meta = std::move(out_meta)
      ]),
      {
        KernelPointerMemoryGuard guard(
            {self_meta.ptr, mat2_meta.ptr, out_meta.ptr});
        auto cpu_self = tensor_from_meta(self_meta);
        auto cpu_mat2 = tensor_from_meta(mat2_meta);
        auto cpu_out = tensor_from_meta(out_meta);
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
    TensorMeta self;
    TensorMeta mat1;
    TensorMeta mat2;
    TensorMeta out;
    at::Scalar beta;
    at::Scalar alpha;
  };
  auto args = std::make_unique<AddmmArgs>(AddmmArgs{
      make_tensor_meta(self),
      make_tensor_meta(mat1),
      make_tensor_meta(mat2),
      make_tensor_meta(out),
      beta,
      alpha});

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::addmm_out", ([args = std::move(args)]), {
        KernelPointerMemoryGuard guard(
            {args->self.ptr, args->mat1.ptr, args->mat2.ptr, args->out.ptr});
        auto cpu_self = tensor_from_meta(args->self);
        auto cpu_mat1 = tensor_from_meta(args->mat1);
        auto cpu_mat2 = tensor_from_meta(args->mat2);
        auto cpu_out = tensor_from_meta(args->out);
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
