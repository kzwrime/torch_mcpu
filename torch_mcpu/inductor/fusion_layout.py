from typing import Any

import torch


Shape = tuple[Any, ...]
Shape3D = tuple[Any, Any, Any]

__all__ = [
    "Shape",
    "Shape3D",
    "broadcast_shape_as_3d",
    "has_contiguous_last_dim",
    "is_mcpu_device_type",
    "logical_shape",
    "output_shape_as_3d",
    "right_aligned_broadcast_shape",
    "same_mcpu_device_and_dtype",
]


def is_mcpu_device_type(device_type: str) -> bool:
    return device_type in ("mcpu", "privateuseone")


def same_mcpu_device_and_dtype(*values: torch.Tensor) -> bool:
    if not values:
        return False
    dtype = values[0].dtype
    if not is_mcpu_device_type(values[0].device.type):
        return False
    return all(
        value.dtype == dtype and is_mcpu_device_type(value.device.type)
        for value in values[1:]
    )


def has_contiguous_last_dim(tensor: torch.Tensor) -> bool:
    return tensor.dim() >= 1 and tensor.stride(-1) == 1


def logical_shape(tensor: torch.Tensor) -> Shape:
    return tuple(tensor.size(i) for i in range(tensor.dim()))


def output_shape_as_3d(tensor: torch.Tensor) -> Shape3D | None:
    shape = logical_shape(tensor)
    if len(shape) == 2:
        return shape[0], 1, shape[1]
    if len(shape) == 3:
        return shape
    return None


def right_aligned_broadcast_shape(
    tensor: torch.Tensor,
    output: torch.Tensor,
) -> Shape | None:
    input_shape = logical_shape(tensor)
    output_shape = logical_shape(output)
    if not input_shape or len(input_shape) > len(output_shape):
        return None

    padded = (1,) * (len(output_shape) - len(input_shape)) + input_shape
    broadcast_dims = zip(padded, output_shape)
    if all(input_dim in (1, output_dim) for input_dim, output_dim in broadcast_dims):
        return padded
    return None


def broadcast_shape_as_3d(
    tensor: torch.Tensor,
    output: torch.Tensor,
) -> Shape3D | None:
    broadcast_shape = right_aligned_broadcast_shape(tensor, output)
    if broadcast_shape is None:
        return None
    if len(broadcast_shape) == 2:
        return broadcast_shape[0], 1, broadcast_shape[1]
    if len(broadcast_shape) == 3:
        return broadcast_shape
    return None
