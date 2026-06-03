
#pragma once

#include <ATen/ExpandUtils.h>
#include <ATen/core/TensorBody.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <ATen/EmptyTensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/blob.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/ops/_local_scalar_dense_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/quantize_per_tensor_native.h>
#include <ATen/ops/resize_as_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/view_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/view_native.h>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function_hook.h>

#include "../runtime/McpuKernelLaunch.h"

using namespace torch::aot_inductor;

// Warning: 确保所有 op 没有返回新的
// Tensor，而是通过输出参数（Tensor(a!)）就地修改输入 Tensor。

inline at::Tensor get_cpu_view_from_mcpu_tensor(const at::Tensor& mcpu_tensor) {
  // For MCPU tensors, we can directly create a view on the CPU without copying
  // data, since they share the same underlying storage.
  at::Tensor result = at::from_blob(
      mcpu_tensor.data_ptr(),
      mcpu_tensor.sizes(),
      mcpu_tensor.strides(),
      /*deleter=*/[](void*) {}, // no-op deleter since we don't own the memory
      mcpu_tensor.options().device(c10::DeviceType::CPU));
  return result;
}

inline at::Tensor get_cpu_tensor_view_if_needed(const at::Tensor& tensor) {
  if (tensor.device().is_cpu()) {
    return tensor;
  }
  return get_cpu_view_from_mcpu_tensor(tensor);
}

inline at::Tensor empty_unary_mcpu_result(
    const at::Tensor& self,
    at::ScalarType result_dtype) {
  return at::empty_like(
      self, self.options().dtype(result_dtype), at::MemoryFormat::Preserve);
}

inline at::Tensor empty_binary_mcpu_result(
    const at::Tensor& self,
    const at::Tensor& other) {
  auto out_sizes = at::infer_size(self.sizes(), other.sizes());
  const bool self_is_mcpu =
      self.device().type() == c10::DeviceType::PrivateUse1;
  const bool other_is_mcpu =
      other.device().type() == c10::DeviceType::PrivateUse1;
  const at::Tensor& device_tensor = self_is_mcpu ? self : other;
  auto options = (self_is_mcpu || other_is_mcpu)
      ? device_tensor.options()
      : self.options().device(c10::Device(c10::DeviceType::PrivateUse1, 0));
  return at::empty(out_sizes, options.dtype(at::result_type(self, other)));
}

// clang-format off
inline AOTITorchError aoti_torch_mcpu_view_dtype(AtenTensorHandle self, int32_t dtype, AtenTensorHandle* ret0) {
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        auto tmp_result = at::compositeexplicitautograd::view(
            resolve_tensor_dispatch_flags(self), static_cast<c10::ScalarType>(dtype)
        );
        *ret0 = new_tensor_handle(std::move(tmp_result));
    });
}

inline AOTITorchError aoti_torch_mcpu_mm_out(AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat2) {
    at::Tensor* t_out = tensor_handle_to_tensor_pointer(out);
    at::Tensor* t_self = tensor_handle_to_tensor_pointer(self);
    at::Tensor* t_mat2 = tensor_handle_to_tensor_pointer(mat2);
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        at::mcpu::launch_kernel(*t_out, [out = *t_out, self = *t_self, mat2 = *t_mat2]() mutable {
            at::mcpu::KernelMemoryGuard guard(out, self, mat2);
            at::Tensor cpu_out = get_cpu_view_from_mcpu_tensor(out);
            at::Tensor cpu_self = get_cpu_view_from_mcpu_tensor(self);
            at::Tensor cpu_mat2 = get_cpu_view_from_mcpu_tensor(mat2);
            at::mm_out(cpu_out, cpu_self, cpu_mat2);
        });
    });
}

inline AOTITorchError aoti_torch_mcpu_addmm_out(AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat1, AtenTensorHandle mat2, double beta, double alpha) {
    at::Tensor* t_out = tensor_handle_to_tensor_pointer(out);
    at::Tensor* t_self = tensor_handle_to_tensor_pointer(self);
    at::Tensor* t_mat1 = tensor_handle_to_tensor_pointer(mat1);
    at::Tensor* t_mat2 = tensor_handle_to_tensor_pointer(mat2);
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        at::mcpu::launch_kernel(*t_out, [out = *t_out, self = *t_self, mat1 = *t_mat1, mat2 = *t_mat2, beta, alpha]() mutable {
            at::mcpu::KernelMemoryGuard guard(out, self, mat1, mat2);
            at::Tensor cpu_out = get_cpu_view_from_mcpu_tensor(out);
            at::Tensor cpu_self = get_cpu_view_from_mcpu_tensor(self);
            at::Tensor cpu_mat1 = get_cpu_view_from_mcpu_tensor(mat1);
            at::Tensor cpu_mat2 = get_cpu_view_from_mcpu_tensor(mat2);
            at::addmm_out(cpu_out, cpu_self, cpu_mat1, cpu_mat2, beta, alpha);
        });
    });
}

inline AOTITorchError aoti_torch_mcpu_add_Tensor(AtenTensorHandle self, AtenTensorHandle other, double alpha, AtenTensorHandle* ret0) {
    at::Tensor* t_self = tensor_handle_to_tensor_pointer(self);
    at::Tensor* t_other = tensor_handle_to_tensor_pointer(other);
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        auto out = empty_binary_mcpu_result(*t_self, *t_other);
        auto launch_out = out;
        *ret0 = new_tensor_handle(std::move(out));
        at::mcpu::launch_kernel(launch_out, [out = launch_out, self = *t_self, other = *t_other, alpha]() mutable {
            at::mcpu::KernelMemoryGuard guard(out, self, other);
            at::Tensor cpu_self = get_cpu_tensor_view_if_needed(self);
            at::Tensor cpu_other = get_cpu_tensor_view_if_needed(other);
            at::Tensor cpu_out = get_cpu_view_from_mcpu_tensor(out);
            at::add_out(cpu_out, cpu_self, cpu_other, alpha);
        });
    });
}

inline AOTITorchError aoti_torch_mcpu_mul_Tensor(AtenTensorHandle self, AtenTensorHandle other, AtenTensorHandle* ret0) {
    at::Tensor* t_self = tensor_handle_to_tensor_pointer(self);
    at::Tensor* t_other = tensor_handle_to_tensor_pointer(other);
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        auto out = empty_binary_mcpu_result(*t_self, *t_other);
        auto launch_out = out;
        *ret0 = new_tensor_handle(std::move(out));
        at::mcpu::launch_kernel(launch_out, [out = launch_out, self = *t_self, other = *t_other]() mutable {
            at::mcpu::KernelMemoryGuard guard(out, self, other);
            at::Tensor cpu_self = get_cpu_tensor_view_if_needed(self);
            at::Tensor cpu_other = get_cpu_tensor_view_if_needed(other);
            at::Tensor cpu_out = get_cpu_view_from_mcpu_tensor(out);
            at::mul_out(cpu_out, cpu_self, cpu_other);
        });
    });
}

inline AOTITorchError aoti_torch_mcpu_sigmoid(AtenTensorHandle self, AtenTensorHandle* ret0) {
    at::Tensor* t_self = tensor_handle_to_tensor_pointer(self);
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        auto result_dtype =
            c10::isIntegralType(t_self->scalar_type(), /*includeBool=*/true)
            ? at::kFloat
            : t_self->scalar_type();
        auto out = empty_unary_mcpu_result(*t_self, result_dtype);
        auto launch_out = out;
        *ret0 = new_tensor_handle(std::move(out));
        at::mcpu::launch_kernel(launch_out, [out = launch_out, self = *t_self]() mutable {
            at::mcpu::KernelMemoryGuard guard(out, self);
            at::Tensor cpu_self = get_cpu_view_from_mcpu_tensor(self);
            at::Tensor cpu_out = get_cpu_view_from_mcpu_tensor(out);
            at::sigmoid_out(cpu_out, cpu_self);
        });
    });
}
// clang-format on
