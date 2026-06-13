import sys


# LITERALINCLUDE START: AUTOLOAD
def _autoload():
    # It is a placeholder function here to be registered as an entry point.
    pass


# LITERALINCLUDE END: AUTOLOAD

import torch


if sys.platform == "win32":
    from ._utils import _load_dll_libraries

    _load_dll_libraries()
    del _load_dll_libraries

import torch_mcpu._C  # type: ignore[misc]
import torch_mcpu.dlpack
import torch_mcpu.openreg
from torch_mcpu.compile import setup_mcpu_compile
from torch_mcpu.compile_flags import (
    get_compile_config,
    get_compile_definitions,
    get_compile_flags,
)
from torch_mcpu.paths import get_include, get_library_dir

torch.utils.rename_privateuse1_backend("mcpu")
torch._register_device_module("mcpu", torch_mcpu.openreg)
import torch_mcpu.distributed
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
torch.Stream = torch_mcpu.openreg.Stream
torch.Event = torch_mcpu.openreg.Event
torch_mcpu.dlpack.patch_mcpu_dlpack()
setup_mcpu_compile()
