from __future__ import annotations

import torch
from torch import _C
from torch.overrides import handle_torch_function, has_torch_function_unary
from torch.utils.dlpack import DLDeviceType


def patch_mcpu_dlpack() -> None:
    if getattr(torch.Tensor, "_mcpu_dlpack_patched", False):
        return

    torch.Tensor._mcpu_original_dlpack = torch.Tensor.__dlpack__
    torch.Tensor._mcpu_original_dlpack_device = torch.Tensor.__dlpack_device__

    torch.Tensor.__dlpack__ = _mcpu_tensor_dlpack
    torch.Tensor.__dlpack_device__ = _mcpu_tensor_dlpack_device
    torch.Tensor._mcpu_dlpack_patched = True


def _mcpu_tensor_dlpack(
    self,
    *,
    stream=-1,
    max_version=None,
    dl_device=None,
    copy=None,
):
    if has_torch_function_unary(self):
        args = (self,)
        kwargs = {
            "stream": stream,
            "max_version": max_version,
            "dl_device": dl_device,
            "copy": copy,
        }
        return handle_torch_function(torch.Tensor.__dlpack__, (self,), *args, **kwargs)

    if self.requires_grad:
        raise BufferError(
            "Can't export tensors that require gradient, use tensor.detach()"
        )
    if self.is_conj():
        raise BufferError("Can't export tensors with the conjugate bit set")
    if self.layout != torch.strided:
        raise BufferError("Can't export tensors with layout other than torch.strided")

    if self.device.type == "cuda" and self.device.index != torch.cuda.current_device():
        raise BufferError(
            "Can't export tensors on a different CUDA device index. "
            f"Expected: {self.device.index}. "
            f"Current device: {torch.cuda.current_device()}."
        )

    if stream is not None and type(stream) is not int:
        raise TypeError("stream must be ``int`` or ``none``")
    elif self.device.type == "cuda" and stream != -1:
        is_rocm = torch.version.hip is not None
        is_cuda = not is_rocm

        if stream is None or (is_rocm and stream == 0) or (is_cuda and stream == 1):
            stream = torch.cuda.default_stream()
        else:
            if is_cuda and stream == 2:
                raise BufferError("per-thread default stream is not supported.")

            device_str = "CUDA" if is_cuda else "ROCm"
            assert (is_cuda and stream != 0) or (
                is_rocm and stream not in (1, 2)
            ), f"unsupported stream on {device_str}: {stream}."

            stream = torch.cuda.ExternalStream(stream)

        current_stream = torch.cuda.current_stream()
        if stream != current_stream:
            event = torch.cuda.Event()
            event.record(current_stream)
            stream.wait_event(event)
    elif self.device.type in ("cpu", "mcpu"):
        assert stream is None or stream == -1, "stream should be None on cpu."

    if self.device.type == "xla":
        import torch_xla
        import torch_xla.utils.dlpack as xla_dlpack

        if (
            len(torch_xla.real_devices()) <= 0
            or "cuda" not in torch_xla.real_devices()[0].lower()
        ):
            raise RuntimeError("Can't export to dlpack an XLA tensor that is not on CUDA.")

        return xla_dlpack.to_dlpack(self)

    if self.device.type == "mcpu" and dl_device is not None:
        target_device_type, _ = dl_device
        if target_device_type == DLDeviceType.kDLCPU:
            cpu_view = torch.mcpu.get_cpu_view_from_mcpu_tensor(self)
            return torch.Tensor._mcpu_original_dlpack(
                cpu_view,
                stream=stream,
                max_version=max_version,
                dl_device=dl_device,
                copy=copy,
            )

    if max_version is None or max_version[0] < 1:
        return _C._to_dlpack(self, dl_device=dl_device, copy=copy)

    return _C._to_dlpack_versioned(self, dl_device=dl_device, copy=copy)


def _mcpu_tensor_dlpack_device(self):
    if has_torch_function_unary(self):
        return handle_torch_function(torch.Tensor.__dlpack_device__, (self,), self)

    device = self.device
    idx = device.index if device.index is not None else 0
    torch_device_type = device.type
    if torch_device_type == "cuda" and torch.version.hip is not None:
        device_type = DLDeviceType.kDLROCM
    elif torch_device_type == "cpu" and self.is_pinned():
        device_type = DLDeviceType.kDLCUDAHost
    elif torch_device_type == "cuda":
        device_type = DLDeviceType.kDLCUDA
    elif torch_device_type == "cpu":
        device_type = DLDeviceType.kDLCPU
    elif torch_device_type == "xpu":
        device_type = DLDeviceType.kDLOneAPI
    elif torch_device_type in ("privateuse1", "mcpu"):
        device_type = DLDeviceType.kDLExtDev
    elif torch_device_type == "xla":
        import torch_xla

        if (
            len(torch_xla.real_devices()) <= 0
            or "cuda" not in torch_xla.real_devices()[0].lower()
        ):
            raise ValueError(f"Unknown device type {torch_device_type} for Dlpack")

        device_type = DLDeviceType.kDLCUDA
    elif torch_device_type == "mps":
        device_type = DLDeviceType.kDLMetal
    else:
        raise ValueError(f"Unknown device type {torch_device_type} for Dlpack")
    return (device_type, idx)


_mcpu_tensor_dlpack.__module__ = "torch"
_mcpu_tensor_dlpack_device.__module__ = "torch"
