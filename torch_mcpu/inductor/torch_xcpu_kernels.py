from dataclasses import dataclass

import torch
from torch._ops import OpOverload

from .fusion_layout import (
    Shape,
    Shape3D,
    broadcast_shape_as_3d,
    has_contiguous_last_dim,
    logical_shape,
    output_shape_as_3d,
    same_mcpu_device_and_dtype,
)

__all__ = [
    "Kernel3DPlan",
    "fused_sigmoid_mul_3d_lastdim_op",
    "fused_sigmoid_mul_add_3d_lastdim_op",
    "plan_sigmoid_mul_3d_lastdim",
    "plan_sigmoid_mul_add_3d_lastdim",
]


@dataclass(frozen=True)
class Kernel3DPlan:
    input1_shape: Shape3D
    input2_shape: Shape3D
    input3_shape: Shape3D | None = None
    restore_shape: Shape | None = None


def fused_sigmoid_mul_add_3d_lastdim_op(dtype: torch.dtype) -> OpOverload | None:
    ops = torch.ops.torch_xcpu
    if dtype == torch.bfloat16:
        return ops.fused_sigmoid_mul_add_3d_lastdim_bf16.default
    if dtype == torch.float32:
        return ops.fused_sigmoid_mul_add_3d_lastdim_fp32.default
    return None


def fused_sigmoid_mul_3d_lastdim_op(dtype: torch.dtype) -> OpOverload | None:
    ops = torch.ops.torch_xcpu
    if dtype == torch.bfloat16:
        return ops.fused_sigmoid_mul_3d_lastdim_bf16.default
    if dtype == torch.float32:
        return ops.fused_sigmoid_mul_3d_lastdim_fp32.default
    return None


def plan_sigmoid_mul_add_3d_lastdim(
    input1: torch.Tensor,
    input2: torch.Tensor,
    input3: torch.Tensor,
) -> Kernel3DPlan | None:
    if not same_mcpu_device_and_dtype(input1, input2, input3):
        return None
    if input3.shape != input1.shape:
        return None
    if not has_contiguous_last_dim(input1):
        return None
    if not has_contiguous_last_dim(input2):
        return None
    if not has_contiguous_last_dim(input3):
        return None

    input1_shape = output_shape_as_3d(input1)
    input2_shape = broadcast_shape_as_3d(input2, input1)
    input3_shape = output_shape_as_3d(input3)
    if input1_shape is None or input2_shape is None or input3_shape is None:
        return None

    restore_shape = logical_shape(input1) if input1.dim() == 2 else None
    return Kernel3DPlan(input1_shape, input2_shape, input3_shape, restore_shape)


def plan_sigmoid_mul_3d_lastdim(
    input1: torch.Tensor,
    input2: torch.Tensor,
) -> Kernel3DPlan | None:
    if not same_mcpu_device_and_dtype(input1, input2):
        return None
    if input1.dim() != 3 or input2.dim() != 3 or input1.shape != input2.shape:
        return None
    if not has_contiguous_last_dim(input1) or not has_contiguous_last_dim(input2):
        return None
    return Kernel3DPlan(logical_shape(input1), logical_shape(input2))
