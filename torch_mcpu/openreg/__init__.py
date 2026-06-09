import os
import time

import torch

import torch_mcpu._C  # type: ignore[misc]

from . import meta  # noqa: F401
from .amp import get_amp_supported_dtype  # noqa: F401
from typing import Optional, Any

_initialized = False
_TORCH_STREAM_BASE = torch.Stream
_TORCH_EVENT_BASE = torch.Event
_SYNC_DELAY_SECONDS_ENV = "TORCH_MCPU_SYNC_DELAY_SECONDS"
_DISABLE_SYNC_DELAY_ENV = "TORCH_MCPU_DISABLE_SYNC_DELAY"

# Defaults:
# - TORCH_MCPU_DISABLE_SYNC_DELAY: enabled by default, so no sync delay is added.
# - TORCH_MCPU_SYNC_DELAY_SECONDS: 10.0 seconds when sync delay is explicitly enabled.
_DEFAULT_DISABLE_SYNC_DELAY = "1"
_DEFAULT_SYNC_DELAY_SECONDS = "10.0"


def _env_flag_enabled(value: str, default: bool) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _sync_delay_seconds() -> float:
    disable_delay = _env_flag_enabled(
        os.getenv(_DISABLE_SYNC_DELAY_ENV, _DEFAULT_DISABLE_SYNC_DELAY),
        default=True,
    )
    if disable_delay:
        return 0.0

    raw_delay = os.getenv(_SYNC_DELAY_SECONDS_ENV, _DEFAULT_SYNC_DELAY_SECONDS)
    try:
        return max(0.0, float(raw_delay))
    except ValueError:
        return float(_DEFAULT_SYNC_DELAY_SECONDS)


def _is_mcpu_sync_target(obj: Any) -> bool:
    device = getattr(obj, "device", None)
    device_type = getattr(device, "type", None)
    return device_type in {"mcpu", "privateuseone"}


def _maybe_delay_sync(obj: Any) -> None:
    if not _is_mcpu_sync_target(obj):
        return

    delay_seconds = _sync_delay_seconds()
    if delay_seconds > 0:
        time.sleep(delay_seconds)


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = torch.accelerator._get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch_mcpu._C._exchangeDevice(self.idx)

    def __exit__(self, type, value, traceback):
        self.idx = torch_mcpu._C._set_device(self.prev_idx)
        return False

class Stream(_TORCH_STREAM_BASE):
    def __new__(cls, device=None, priority=0, **kwargs):
        if device is None:
            return super().__new__(cls, priority=priority, **kwargs)
        else:
            with torch.mcpu.device(device):
                return super().__new__(cls, priority=priority, **kwargs)

    def query(self):
        _maybe_delay_sync(self)
        return super().query()

    def synchronize(self) -> None:
        _maybe_delay_sync(self)
        return super().synchronize()

    def wait_event(self, event) -> None:
        _maybe_delay_sync(self)
        return super().wait_event(event)

    def wait_stream(self, stream) -> None:
        _maybe_delay_sync(self)
        return super().wait_stream(stream)

    @classmethod
    def priority_range(cls):
        """Return (least_priority, greatest_priority) for this device."""
        return torch_mcpu._C._get_stream_priority_range()

    @property
    def priority(self):
        """Return the priority of this stream (0=normal, 1=high)."""
        # Decode priority from stream_id encoding:
        # stream_id bits: [si | stream_type (3 bits) | native (1 bit)]
        # stream_type 0x6 = DEFAULT (treated as priority 0)
        stream_id = self.stream_id
        if not (stream_id & 1):  # external stream
            return 0
        kStreamTypeBits = 3
        stream_type = (stream_id >> 1) & ((1 << kStreamTypeBits) - 1)
        if stream_type == 0x6:  # DEFAULT stream
            return 0
        return stream_type


class Event(_TORCH_EVENT_BASE):
    def query(self):
        _maybe_delay_sync(self)
        return super().query()

    def synchronize(self) -> None:
        _maybe_delay_sync(self)
        return super().synchronize()

    def wait(self, stream=None) -> None:
        _maybe_delay_sync(self)
        if stream is None:
            return super().wait()
        return super().wait(stream)

    def elapsed_time(self, end_event) -> float:
        _maybe_delay_sync(self)
        return super().elapsed_time(end_event)

class StreamContext:
    r"""Context-manager that selects a given stream.

    All CUDA kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """

    cur_stream: Optional[Stream]

    def __init__(self, stream: Optional[Stream]):
        self.stream = stream
        self.idx = torch.accelerator._get_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                # pyrefly: ignore [bad-assignment]
                self.idx = -1

        self.src_prev_stream = (
            None if not torch.jit.is_scripting() else torch.accelerator.current_stream(None)
        )
        self.dst_prev_stream = (
            None if not torch.jit.is_scripting() else torch.accelerator.current_stream(None)
        )

    def __enter__(self):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None or CUDA device not available
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = torch.accelerator.current_stream(None)

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with torch.mcpu.device(cur_stream.device):
                self.dst_prev_stream = torch.accelerator.current_stream(cur_stream.device)
        torch.accelerator.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no CUDA device available, return
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            torch.accelerator.set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        torch.accelerator.set_stream(self.src_prev_stream)  # type: ignore[arg-type]

def stream(stream: Optional[Stream]) -> StreamContext:
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note::
        In eager mode stream is of type Stream class while in JIT it is
        an object of the custom class ``torch.classes.cuda.Stream``.
    """
    return StreamContext(stream)


def current_stream(device=None) -> torch.Stream:
    return torch.accelerator.current_stream(device)


def default_stream(device=None) -> torch.Stream:
    device_index = torch.accelerator._get_device_index(device, optional=True)
    if device_index is None:
        device_index = current_device()
    stream_id, device_index, device_type = torch_mcpu._C._get_default_stream(
        device_index
    )
    return torch.Stream(
        stream_id=stream_id,
        device_type=device_type,
        device_index=device_index,
    )


def set_stream(stream: Optional[Stream]) -> None:
    torch.accelerator.set_stream(stream)


def synchronize(device=None) -> None:
    delay_target = current_stream(device)
    _maybe_delay_sync(delay_target)
    torch.accelerator.synchronize(device)


def set_op_timing_enabled(enabled: bool) -> None:
    torch_mcpu._C._set_op_timing_enabled(enabled)


def reset_op_timing() -> None:
    torch_mcpu._C._reset_op_timing()


def get_op_timing() -> list[dict[str, Any]]:
    return torch_mcpu._C._get_op_timing()


def set_kernel_timing_enabled(enabled: bool) -> None:
    torch_mcpu._C._set_kernel_timing_enabled(enabled)


def reset_kernel_timing() -> None:
    torch_mcpu._C._reset_kernel_timing()


def get_kernel_timing() -> list[dict[str, Any]]:
    return torch_mcpu._C._get_kernel_timing()


def read_kernel_timing_tsc() -> int:
    return torch_mcpu._C._read_kernel_timing_tsc()


def is_available():
    return True


try:
    from torch_mcpu.profiler import install as _install_profiler

    _install_profiler()
    del _install_profiler
except Exception:
    pass


def empty_cache() -> None:
    """Release all unoccupied cached memory back to the OS."""
    synchronize()
    torch_mcpu._C._empty_cache()


def memory_stats(device=None) -> dict:
    """Return a dict of memory allocator statistics for the given device."""
    if device is None:
        device = current_device()
    else:
        device = torch.accelerator._get_device_index(device, optional=True)
    return torch_mcpu._C._memory_stats(device)


def memory_allocated(device=None) -> int:
    return memory_stats(device).get("allocated_bytes.all.current", 0)


def memory_reserved(device=None) -> int:
    return memory_stats(device).get("reserved_bytes.all.current", 0)


def max_memory_allocated(device=None) -> int:
    return memory_stats(device).get("allocated_bytes.all.peak", 0)


def max_memory_reserved(device=None) -> int:
    return memory_stats(device).get("reserved_bytes.all.peak", 0)


def reset_peak_memory_stats(device=None) -> None:
    """Reset peak memory usage statistics for the given device."""
    if device is None:
        device = current_device()
    else:
        device = torch.accelerator._get_device_index(device, optional=True)
    torch_mcpu._C._reset_peak_memory_stats(device)


def get_mcpu_view_from_cpu_tensor(cpu_tensor: "torch.Tensor") -> "torch.Tensor":
    """Return an mcpu tensor that is a view of *cpu_tensor*'s memory.

    For pinned input the returned tensor shares the same physical memory
    (writes from either side are immediately visible on the other).  For
    non-pinned input a contiguous copy is pinned first, so only the initial
    values are shared — subsequent mutations to *cpu_tensor* are not reflected.

    Args:
        cpu_tensor: A CPU tensor (pinned or unpinned).

    Returns:
        An mcpu tensor backed by the same (or a pinned copy of the) memory.
    """
    if not isinstance(cpu_tensor, torch.Tensor):
        raise TypeError("Expected cpu_tensor to be a torch.Tensor")
    if cpu_tensor.device.type != "cpu":
        raise ValueError("Expected cpu_tensor to be on CPU")
    return torch.ops.mcpu.get_mcpu_view_from_cpu_tensor(cpu_tensor)


# def get_cpu_view_from_mcpu_tensor(mcpu_tensor: "torch.Tensor") -> "torch.Tensor":
#     """Return a CPU tensor that is a view of *mcpu_tensor*'s memory.
#
#     Unlike the CPU->mcpu direction, this path does not require pinned memory.
#
#     Args:
#         mcpu_tensor: An mcpu tensor.
#
#     Returns:
#         A CPU tensor backed by the same memory.
#     """
#     if not isinstance(mcpu_tensor, torch.Tensor):
#         raise TypeError("Expected mcpu_tensor to be a torch.Tensor")
#     if mcpu_tensor.device.type != "mcpu":
#         raise ValueError("Expected mcpu_tensor to be on mcpu")
#     return torch.ops.mcpu.get_cpu_view_from_mcpu_tensor(mcpu_tensor)


def get_unprotected_cpu_view_from_mcpu_tensor(
    mcpu_tensor: "torch.Tensor",
) -> "torch.Tensor":
    """Return a CPU view of *mcpu_tensor* after disabling page protection.

    This is an explicit opt-in escape hatch for Python code that needs direct
    CPU access to mcpu backing memory.  The strict
    :func:`get_cpu_view_from_mcpu_tensor` API keeps page protection intact, so
    it can still catch accidental host-side memory accesses while porting
    third-party kernels.
    """
    if not isinstance(mcpu_tensor, torch.Tensor):
        raise TypeError("Expected mcpu_tensor to be a torch.Tensor")
    if mcpu_tensor.device.type != "mcpu":
        raise ValueError("Expected mcpu_tensor to be on mcpu")
    synchronize(mcpu_tensor.device)
    return torch.ops.mcpu.get_unprotected_cpu_view_from_mcpu_tensor(mcpu_tensor)


def reset_accumulated_memory_stats(device=None) -> None:
    """Reset accumulated memory usage statistics for the given device."""
    if device is None:
        device = current_device()
    else:
        device = torch.accelerator._get_device_index(device, optional=True)
    torch_mcpu._C._reset_accumulated_memory_stats(device)


def get_memory_info(device=None) -> tuple[int, int]:
    return torch.accelerator.get_memory_info(device)


mem_get_info = get_memory_info


def device_count() -> int:
    return torch_mcpu._C._get_device_count()


def current_device():
    return torch_mcpu._C._get_device()


# LITERALINCLUDE START: PYTHON SET DEVICE FUNCTION
def set_device(device: torch.device | int) -> None:
    if isinstance(device, torch.device):
        if device.type != "mcpu":
            raise ValueError("Expected a torch.device with type 'mcpu', but got %s." % device)  # noqa
        device = device.index
    assert isinstance(device, int)
    if device < 0:
        raise ValueError("Expected a non-negative integer device index, but got %d." % device)  # noqa
    torch_mcpu._C._set_device(device)

# LITERALINCLUDE END: PYTHON SET DEVICE FUNCTION


def init():
    _lazy_init()


def is_initialized():
    return _initialized


def _lazy_init():
    global _initialized
    if is_initialized():
        return
    torch_mcpu._C._init()
    _initialized = True


from .random import *  # noqa: F403


__all__ = [
    "device",
    "Stream",
    "StreamContext",
    "stream",
    "current_stream",
    "default_stream",
    "set_stream",
    "synchronize",
    "set_op_timing_enabled",
    "reset_op_timing",
    "get_op_timing",
    "set_kernel_timing_enabled",
    "reset_kernel_timing",
    "get_kernel_timing",
    "read_kernel_timing_tsc",
    "device_count",
    "current_device",
    "set_device",
    "initial_seed",
    "is_available",
    "init",
    "is_initialized",
    "empty_cache",
    "get_memory_info",
    "mem_get_info",
    "get_mcpu_view_from_cpu_tensor",
    "get_cpu_view_from_mcpu_tensor",
    "get_unprotected_cpu_view_from_mcpu_tensor",
    "memory_stats",
    "memory_allocated",
    "memory_reserved",
    "max_memory_allocated",
    "max_memory_reserved",
    "reset_peak_memory_stats",
    "reset_accumulated_memory_stats",
    "random",
    "manual_seed",
    "manual_seed_all",
    "get_rng_state",
    "set_rng_state",
]
