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
from textwrap import dedent

import torch

from torch._inductor.codegen import cpp, cpp_wrapper_cpu, wrapper
from torch._inductor.codegen import cpu_device_op_overrides as _cpu_device_op_overrides  # noqa: F401
from torch._inductor.codegen.common import (
    DeviceOpOverrides,
    register_device_op_overrides,
)
from torch._inductor.custom_graph_pass import (
    CustomGraphModulePass,
    get_hash_for_files,
)
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.virtualized import V

def _meta_uses_mcpu(value) -> bool:
    if isinstance(value, torch.Tensor):
        return value.device.type == "mcpu"
    if isinstance(value, (list, tuple)):
        return any(_meta_uses_mcpu(item) for item in value)
    if isinstance(value, dict):
        return any(_meta_uses_mcpu(item) for item in value.values())
    return False


def _graph_uses_mcpu(gm) -> bool:
    for node in gm.graph.nodes:
        if _meta_uses_mcpu(node.meta.get("val")) or _meta_uses_mcpu(
            node.meta.get("example_value")
        ):
            return True
    return False


def _is_mcpu_compute_op(node) -> bool:
    target = node.target
    return (
        node.op == "call_function"
        and isinstance(target, torch._ops.OpOverload)
        and target.namespace == "aten"
        and (
            torch.Tag.pointwise in target.tags
            or torch.Tag.reduction in target.tags
        )
        and (
            _meta_uses_mcpu(node.meta.get("val"))
            or _meta_uses_mcpu(node.meta.get("example_value"))
        )
    )


class McpuDisableComputeFusionPass(CustomGraphModulePass):
    """Selectively lower mcpu compute ops through ATen fallback.

    mcpu is used to model an accelerator with device memory that is not directly
    host-accessible. Inductor's generated ``cpp_fused_*`` kernels execute raw
    loops on the host thread over tensor data pointers, so they bypass the
    launch-kernel/page-protection model. Marking pointwise and reduction ATen
    nodes with ``should_fallback`` uses Inductor's selective lowering mechanism
    to keep graph compilation and AOTI wrapper generation while avoiding CPU
    fused-loop codegen for mcpu compute.
    """

    def __call__(self, gm: torch.fx.GraphModule) -> None:
        if not _graph_uses_mcpu(gm):
            return
        for node in gm.graph.nodes:
            if _is_mcpu_compute_op(node):
                node.meta["should_fallback"] = True

    def uuid(self):
        return get_hash_for_files((__file__,), extra=self.__class__.__name__)


class McpuDeviceOpOverrides(DeviceOpOverrides):
    """Device-op overrides for mcpu.

    mcpu is a PrivateUse1 accelerator simulation. Generated wrappers should use
    mcpu device/stream APIs instead of inheriting CPU no-op behavior.
    """

    def import_get_raw_stream_as(self, name: str) -> str:
        return dedent(
            f"""
            def {name}(device_idx):
                import torch
                return torch.mcpu.current_stream(device_idx).stream_id
            """
        )

    def set_device(self, device_idx: int) -> str:
        return f"torch.mcpu.set_device({device_idx})"

    def synchronize(self) -> str:
        return "torch.mcpu.synchronize()"

    def device_guard(self, device_idx: int) -> str:
        return f"torch.mcpu.device({device_idx})"

    def cpp_device_guard(self) -> str:
        return "at::DeviceGuard"

    def cpp_aoti_device_guard(self) -> str:
        return "AOTIMcpuGuard"

    def cpp_stream_guard(self) -> str:
        return "c10::OptionalStreamGuard"

    def cpp_aoti_stream_guard(self) -> str:
        return "AOTIMcpuStreamGuard"

    def cpp_getStreamFromExternal(self) -> str:
        return "c10::mcpu::getStreamFromExternal"

    def kernel_header(self) -> str:
        return ""

    def kernel_driver(self) -> str:
        return ""

    def cpp_stream_type(self) -> str:
        return "orStream_t"

    def aoti_get_stream(self) -> str:
        return "aoti_torch_mcpu_get_current_stream"

    def cpp_kernel_type(self) -> str:
        return "void*"

    def cpp_device_ptr(self) -> str:
        return "void*"

    def tma_descriptor_helpers(self) -> str:
        return ""

    def cpp_scratch(self, idx: int, workspace, prefix=None):
        return None


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

    def codegen_int_array_var(
        self,
        int_array,
        writeline,
        known_statically=False,
        graph=None,
    ):
        return self._codegen_int_array_var_impl(
            int_array, writeline, known_statically
        )

    @staticmethod
    def create(is_subgraph, subgraph_name, parent_wrapper, partition_signatures=None):
        # The base-class create() hard-codes CppWrapperCpu(); override it
        # so the correct type (McpuCppWrapperCodegen) is instantiated.
        return McpuCppWrapperCodegen()

    @staticmethod
    def get_device_include_path(device: str) -> str:
        base_dir = Path(__file__).resolve().parent
        package_dir = base_dir.parent
        header = base_dir.parent / "include" / "cpp_wrapper" / "mcpu.h"
        
        if not header.exists():
            raise FileNotFoundError(f"Failed to find header file: {header}")

        aoti_flags = [
            f"-I{package_dir / 'include'}",
            "-DTORCH_MCPU_ENABLE_MEMORY_PROTECTION=1",
            "-DTORCH_MCPU_KERNEL_TIMING_USE_TSC=0",
        ]
        extra_cflags = os.environ.get("AOTI_EXTRA_CFLAGS", "")
        existing_flags = extra_cflags.split()
        missing_flags = [flag for flag in aoti_flags if flag not in existing_flags]
        if missing_flags:
            os.environ["AOTI_EXTRA_CFLAGS"] = " ".join(
                [*missing_flags, *existing_flags]
            )

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
        return self._scheduling.can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(self, node1, node2):
        return self._scheduling.can_fuse_horizontal(node1, node2)

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

    def benchmark_fused_nodes(self, nodes):
        # mcpu does not implement Inductor combo-kernel codegen. If benchmarked,
        # report fusion as unavailable instead of letting the scheduler create a
        # ForeachKernelSchedulerNode that only SIMD/CUDA backends can lower.
        return float("inf"), ""

    def benchmark_combo_kernel(self, node_list):
        # See benchmark_fused_nodes: combo kernels are not a supported mcpu
        # lowering path today.
        return float("inf"), 0.0, []


register_device_op_overrides("mcpu", McpuDeviceOpOverrides())
