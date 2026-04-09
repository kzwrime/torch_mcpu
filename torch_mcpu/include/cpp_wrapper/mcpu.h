
#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/extension.h>

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

// clang-format off
inline AOTITorchError aoti_torch_mcpu_mm_out(AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat2) {
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        // at::cpu::mm_out(
        at::mm_out(
            *tensor_handle_to_tensor_pointer(out), resolve_tensor_dispatch_flags(self), resolve_tensor_dispatch_flags(mat2)
        );

    });
}

inline AOTITorchError aoti_torch_mcpu_addmm_out(AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat1, AtenTensorHandle mat2, double beta, double alpha) {
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        at::addmm_out(
            *tensor_handle_to_tensor_pointer(out), resolve_tensor_dispatch_flags(self), resolve_tensor_dispatch_flags(mat1), resolve_tensor_dispatch_flags(mat2), beta, alpha
        );

    });
}
// clang-format on
