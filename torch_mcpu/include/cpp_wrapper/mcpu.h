
#pragma once

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
#include <ATen/ops/quantize_per_tensor_native.h>
#include <ATen/ops/resize_as_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/view_native.h>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function_hook.h>

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

// clang-format off
inline AOTITorchError aoti_torch_mcpu_mm_out(AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat2) {
    at::Tensor* t_out = tensor_handle_to_tensor_pointer(out);
    at::Tensor* t_self = tensor_handle_to_tensor_pointer(self);
    at::Tensor* t_mat2 = tensor_handle_to_tensor_pointer(mat2);
    at::Tensor cpu_out = get_cpu_view_from_mcpu_tensor(*t_out);
    at::Tensor cpu_self = get_cpu_view_from_mcpu_tensor(*t_self);
    at::Tensor cpu_mat2 = get_cpu_view_from_mcpu_tensor(*t_mat2);
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        // at::cpu::mm_out(
        at::mm_out(cpu_out, cpu_self, cpu_mat2);
    });
}

inline AOTITorchError aoti_torch_mcpu_addmm_out(AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat1, AtenTensorHandle mat2, double beta, double alpha) {
    at::Tensor* t_out = tensor_handle_to_tensor_pointer(out);
    at::Tensor* t_self = tensor_handle_to_tensor_pointer(self);
    at::Tensor* t_mat1 = tensor_handle_to_tensor_pointer(mat1);
    at::Tensor* t_mat2 = tensor_handle_to_tensor_pointer(mat2);
    at::Tensor cpu_out = get_cpu_view_from_mcpu_tensor(*t_out);
    at::Tensor cpu_self = get_cpu_view_from_mcpu_tensor(*t_self);
    at::Tensor cpu_mat1 = get_cpu_view_from_mcpu_tensor(*t_mat1);
    at::Tensor cpu_mat2 = get_cpu_view_from_mcpu_tensor(*t_mat2);
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        at::addmm_out(
            cpu_out, cpu_self, cpu_mat1, cpu_mat2, beta, alpha
        );

    });
}
// clang-format on
