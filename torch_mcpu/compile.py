import os
from pathlib import Path

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
from torch_mcpu.inductor.torch_xcpu_fusion import (
    McpuTorchXcpuFusionPass,
    append_post_grad_pass,
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _append_env_flags(name: str, flags: str) -> None:
    existing = os.environ.get(name, "")
    parts = existing.split()
    missing = [flag for flag in flags.split() if flag not in parts]
    if missing:
        os.environ[name] = " ".join([existing, *missing]).strip()


def _setup_aoti_link_flags() -> None:
    lib_dir = Path(__file__).resolve().parent / "lib"
    _append_env_flags(
        "AOTI_EXTRA_LDFLAGS",
        f"-L{lib_dir} -Wl,-rpath,{lib_dir} -ltorch_mcpu -lopenreg",
    )


def _disable_inductor_compute_kernels() -> None:
    """Force every elementwise/reduction aten op to use its eager kernel.

    mcpu runs ops asynchronously via ``launch_kernel`` on a worker thread and its
    device memory is not host-accessible. Inductor's generated C++ loop kernels
    (``cpp_fused_*``) read tensor data pointers directly on the host thread, which
    bypasses that model and segfaults. Inductor only emits such loop kernels for
    ``pointwise``/``reduction`` tagged ops, so registering an aten fallback for
    each of them disables fused-kernel generation generically while leaving view,
    factory and matmul ops to Inductor's normal (data-free) handling.
    """
    from torch._inductor import lowering

    for op in list(lowering.lowerings):
        if (
            isinstance(op, torch._ops.OpOverload)
            and op.namespace == "aten"
            and op not in lowering.fallbacks
            and (torch.Tag.pointwise in op.tags or torch.Tag.reduction in op.tags)
        ):
            lowering.make_fallback(op, warn=False, override_decomp=True)


def _register_mcpu_aoti_fallback_shims() -> None:
    from torchgen.aoti.fallback_ops import inductor_fallback_ops

    inductor_fallback_ops.setdefault("aten.sigmoid.default", {})


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
    def default_stream(device: torch.types.Device = None) -> torch.Stream:
        return torch_mcpu.openreg.default_stream(device)

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
    _setup_aoti_link_flags()
    _disable_inductor_compute_kernels()
    _register_mcpu_aoti_fallback_shims()
    if _env_flag("TORCH_MCPU_ENABLE_TORCH_XCPU_FUSIONS", True):
        inductor_config.post_grad_custom_post_pass = append_post_grad_pass(
            inductor_config.post_grad_custom_post_pass,
            McpuTorchXcpuFusionPass(),
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
