import os

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.device_interface import (
    CpuInterface,
    register_interface_for_device,
)
from torch._inductor.codegen import cpp_utils
from torch._inductor.codegen.common import register_backend_for_device

import torch_mcpu._C  # type: ignore[misc]
import torch_mcpu.openreg
from torch_mcpu.inductor.extension_codegen_backend import (
    McpuCppWrapperCodegen,
    McpuScheduling,
    McpuWrapperCodegen,
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class McpuInterface(CpuInterface):
    """Dynamo device interface for the mcpu PrivateUse1 backend."""

    class device(torch_mcpu.openreg.device):
        pass

    class Stream(torch_mcpu.openreg.Stream):
        pass

    class Worker(CpuInterface.Worker):
        @staticmethod
        def set_device(device: int) -> None:
            torch_mcpu.openreg.set_device(device)

        @staticmethod
        def current_device() -> int:
            return torch_mcpu.openreg.current_device()

    @staticmethod
    def current_device() -> int:
        return torch_mcpu.openreg.current_device()

    @staticmethod
    def set_device(device: torch.types.Device) -> None:
        torch_mcpu.openreg.set_device(device)

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        return torch_mcpu._C._exchangeDevice(device)

    @staticmethod
    def exchange_device(device: int) -> int:
        return torch_mcpu._C._exchangeDevice(device)

    @staticmethod
    def device_count() -> int:
        return torch_mcpu.openreg.device_count()

    @staticmethod
    def is_available() -> bool:
        return torch_mcpu.openreg.is_available()

    @staticmethod
    def stream(stream: torch.Stream):
        return torch_mcpu.openreg.stream(stream)

    @staticmethod
    def current_stream(device: torch.types.Device = None) -> torch.Stream:
        return torch_mcpu.openreg.current_stream(device)

    @staticmethod
    def set_stream(stream: torch.Stream) -> None:
        torch_mcpu.openreg.set_stream(stream)

    @staticmethod
    def synchronize(device: torch.types.Device = None) -> None:
        torch_mcpu.openreg.synchronize(device)

    @staticmethod
    def memory_allocated(device: torch.types.Device = None) -> int:
        return torch_mcpu.openreg.memory_allocated(device)

    @staticmethod
    def memory_reserved(device: torch.types.Device = None) -> int:
        return torch_mcpu.openreg.memory_reserved(device)

    @staticmethod
    def max_memory_allocated(device: torch.types.Device = None) -> int:
        return torch_mcpu.openreg.max_memory_allocated(device)

    @staticmethod
    def max_memory_reserved(device: torch.types.Device = None) -> int:
        return torch_mcpu.openreg.max_memory_reserved(device)


def setup_mcpu_compile() -> None:
    """Register mcpu with Dynamo and Inductor."""
    inductor_config.fallback_by_default = _env_flag(
        "TORCH_MCPU_INDUCTOR_FALLBACK_BY_DEFAULT"
    )
    register_interface_for_device("mcpu", McpuInterface)
    register_interface_for_device("privateuseone", McpuInterface)
    register_backend_for_device(
        "mcpu",
        McpuScheduling,
        McpuWrapperCodegen,
        McpuCppWrapperCodegen,
    )
    cpp_utils.DEVICE_TO_ATEN["mcpu"] = "at::kPrivateUse1"
