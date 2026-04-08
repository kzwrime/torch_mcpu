import torch

import torch_mcpu._C  # type: ignore[misc]

from . import meta  # noqa: F401
from .amp import get_amp_supported_dtype  # noqa: F401
from typing import Optional, Any

_initialized = False


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

class Stream(torch.Stream):
    def __new__(cls, device=None, priority=0, **kwargs):
        if device is None:
            return super().__new__(cls, priority=priority, **kwargs)
        else:
            with torch.mcpu.device(device):
                return super().__new__(cls, priority=priority, **kwargs)

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

def is_available():
    return True


def empty_cache() -> None:
    """Release all unoccupied cached memory back to the OS."""
    torch_mcpu._C._empty_cache()


def memory_stats(device=None) -> dict:
    """Return a dict of memory allocator statistics for the given device."""
    if device is None:
        device = current_device()
    return torch_mcpu._C._memory_stats(device)


def reset_peak_memory_stats(device=None) -> None:
    """Reset peak memory usage statistics for the given device."""
    if device is None:
        device = current_device()
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
    return torch_mcpu._C._get_mcpu_view_from_cpu_tensor(cpu_tensor)


def reset_accumulated_memory_stats(device=None) -> None:
    """Reset accumulated memory usage statistics for the given device."""
    if device is None:
        device = current_device()
    torch_mcpu._C._reset_accumulated_memory_stats(device)


def device_count() -> int:
    return torch_mcpu._C._get_device_count()


def current_device():
    return torch_mcpu._C._get_device()


# LITERALINCLUDE START: PYTHON SET DEVICE FUNCTION
def set_device(device) -> None:
    if device >= 0:
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
    "device_count",
    "current_device",
    "set_device",
    "initial_seed",
    "is_available",
    "init",
    "is_initialized",
    "empty_cache",
    "get_mcpu_view_from_cpu_tensor",
    "memory_stats",
    "reset_peak_memory_stats",
    "reset_accumulated_memory_stats",
    "random",
    "manual_seed",
    "manual_seed_all",
    "get_rng_state",
    "set_rng_state",
]
