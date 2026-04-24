import sys

import torch


if sys.platform == "win32":
    from ._utils import _load_dll_libraries

    _load_dll_libraries()
    del _load_dll_libraries

import torch._inductor.config as inductor_config
from torch._inductor.codegen import cpp_utils
from torch._inductor.codegen.common import (
    get_scheduling_for_device,
    get_wrapper_codegen_for_device,
    register_backend_for_device,
)

import torch_mcpu._C  # type: ignore[misc]
import torch_mcpu.distributed
import torch_mcpu.openreg

from torch_mcpu.inductor.extension_codegen_backend import (
    McpuCppWrapperCodegen,
    McpuScheduling,
    McpuWrapperCodegen,
)

torch.utils.rename_privateuse1_backend("mcpu")
torch._register_device_module("mcpu", torch_mcpu.openreg)

# Patch vllm's torch_triton_utils with mcpu C++ implementations when available.
try:
    from vllm.utils.mcpu_triton_utils import patch_torch_triton_utils
    patch_torch_triton_utils()
except ImportError:
    pass
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

def _setup_mcpu_inductor():
    """Register mcpu with torch.inductor (idempotent)."""
    register_backend_for_device(
        "mcpu",
        McpuScheduling,
        McpuWrapperCodegen,
        McpuCppWrapperCodegen,
    )
    cpp_utils.DEVICE_TO_ATEN["mcpu"] = "at::kPrivateUse1"

_setup_mcpu_inductor()
torch_mcpu.distributed.patch_mcpu_distributed()

# LITERALINCLUDE START: AUTOLOAD
def _autoload():
    # It is a placeholder function here to be registered as an entry point.
    pass


# LITERALINCLUDE END: AUTOLOAD
