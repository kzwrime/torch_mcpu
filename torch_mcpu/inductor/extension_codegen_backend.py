"""
Torch Inductor codegen backend for the mcpu device.

Mirrors docs/ref/extension_backends/cpp/extension_codegen_backend.py and
registers mcpu as a CPU-based inductor backend so that torch.compile() can
generate and execute code for mcpu tensors.

Usage::

    from torch_mcpu.inductor.extension_codegen_backend import (
        McpuCppWrapperCodegen,
        McpuScheduling,
        McpuWrapperCodegen,
    )
    from torch._inductor.codegen.common import register_backend_for_device
    from torch._inductor.codegen import cpp_utils

    register_backend_for_device(
        "mcpu",
        McpuScheduling,
        McpuWrapperCodegen,
        McpuCppWrapperCodegen,
    )
    cpp_utils.DEVICE_TO_ATEN["mcpu"] = "at::kPrivateUse1"
"""
import os
from pathlib import Path

from torch._inductor.codegen import cpp, cpp_wrapper_cpu, wrapper
from torch._inductor.codegen.common import (
    DeviceOpOverrides,
    register_device_op_overrides,
)
from torch._inductor.codegen.cpu_device_op_overrides import CpuDeviceOpOverrides
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.virtualized import V


class McpuDeviceOpOverrides(CpuDeviceOpOverrides):
    """Device-op overrides for mcpu.

    mcpu is a CPU-emulated device so the CPU overrides apply directly
    (set_device / synchronize / device_guard are all no-ops).
    """


class McpuWrapperCodegen(wrapper.PythonWrapperCodegen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def create(is_subgraph, subgraph_name, parent_wrapper, partition_signatures=None):
        # The base-class create() hard-codes PythonWrapperCodegen(); override it
        # so the correct type is instantiated.
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return wrapper.SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return McpuWrapperCodegen()

    def _generate_kernel_call_helper(self, kernel_name, call_args, *, device=None, triton=True, **kwargs):
        # mcpu is CPU-backed: treat non-triton kernel calls the same as "cpu"
        device = device or V.graph.get_current_device_or_throw()
        if not triton and device.type == "mcpu":
            self.writeline(self.wrap_kernel_call(kernel_name, call_args))
            return
        super()._generate_kernel_call_helper(
            kernel_name, call_args, device=device, triton=triton, **kwargs
        )


class McpuCppWrapperCodegen(cpp_wrapper_cpu.CppWrapperCpu):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def create(is_subgraph, subgraph_name, parent_wrapper, partition_signatures=None):
        # The base-class create() hard-codes CppWrapperCpu(); override it
        # so the correct type (McpuCppWrapperCodegen) is instantiated.
        return McpuCppWrapperCodegen()

    @staticmethod
    def get_device_include_path(device: str) -> str:
        base_dir = Path(__file__).resolve().parent
        header = base_dir.parent / "include" / "cpp_wrapper" / "mcpu.h"
        
        if not header.exists():
            raise FileNotFoundError(f"Failed to find header file: {header}")

        return f'#include "{header}"'

        # mcpu is CPU-backed; reuse the CPU wrapper header
        # if device == "mcpu":
        #     device = "cpu"
        # return cpp_wrapper_cpu.CppWrapperCpu.get_device_include_path(device)


class McpuScheduling(BaseScheduling):
    """Scheduling for the mcpu device.

    Delegates to CppScheduling since mcpu runs on CPU hardware.
    """

    def __init__(self, scheduler):
        super().__init__(scheduler)
        self._scheduling = cpp.CppScheduling(scheduler)

    def can_fuse_vertical(self, node1, node2):
        return True

    def can_fuse_horizontal(self, node1, node2):
        return True

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_template(self, template_node, epilogue_nodes, prologue_nodes=()):
        pass

    def codegen_node(self, node):
        self._scheduling.codegen_node(node)

    def codegen_sync(self):
        pass

    def flush(self):
        self._scheduling.flush()


register_device_op_overrides("mcpu", McpuDeviceOpOverrides())
