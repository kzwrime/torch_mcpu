"""
Tests for torch.compile support on the mcpu backend.

Mirrors docs/ref/extension_backends/cpp/test_extension_backend.py:
verifies that registering mcpu with torch._inductor allows torch.compile()
to generate and execute fused C++ kernels for mcpu tensors.
"""

import os
import unittest

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from torch._inductor.codegen import cpp_utils
from torch._inductor.codegen.common import (
    get_scheduling_for_device,
    get_wrapper_codegen_for_device,
    register_backend_for_device,
)
from torch._inductor.utils import (
    add_scheduler_init_hook,
    run_and_get_code,
    run_and_get_cpp_code,
    run_and_get_kernels,
    run_and_get_triton_code,
    run_fw_bw_and_get_code,
    triton_version_uses_attrs_dict,
)

import torch_mcpu  # noqa: F401 – registers the mcpu backend with PyTorch
from torch_mcpu.inductor.extension_codegen_backend import (
    McpuCppWrapperCodegen,
    McpuScheduling,
    McpuWrapperCodegen,
)


def _setup_mcpu_inductor():
    """Register mcpu with torch.inductor (idempotent)."""
    register_backend_for_device(
        "mcpu",
        McpuScheduling,
        McpuWrapperCodegen,
        McpuCppWrapperCodegen,
    )
    cpp_utils.DEVICE_TO_ATEN["mcpu"] = "at::kPrivateUse1"


class TestMcpuInductorRegistration(unittest.TestCase):
    """Verify that the inductor backend classes are registered correctly."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _setup_mcpu_inductor()

    def test_scheduling_registered(self):
        self.assertIs(get_scheduling_for_device("mcpu"), McpuScheduling)

    def test_python_wrapper_registered(self):
        self.assertIs(get_wrapper_codegen_for_device("mcpu"), McpuWrapperCodegen)

    def test_cpp_wrapper_registered(self):
        self.assertIs(
            get_wrapper_codegen_for_device("mcpu", cpp_wrapper=True),
            McpuCppWrapperCodegen,
        )


class TestMcpuCompile(unittest.TestCase):
    """Verify that torch.compile() produces correct results for mcpu tensors."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _setup_mcpu_inductor()

    def setUp(self):
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_tensors(self, rows=2, cols=16):
        device = torch.device("mcpu")
        x = torch.empty(rows, cols, device=device).fill_(1)
        y = torch.empty(rows, cols, device=device).fill_(2)
        z = torch.empty(rows, cols, device=device).fill_(3)
        ref = torch.empty(rows, cols).fill_(5)  # 1*2 + 3 = 5
        return x, y, z, ref

    @staticmethod
    def fn(a, b, c):
        return a * b + c

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_compile_python_wrapper(self):
        """Basic torch.compile with Python wrapper (default mode)."""
        x, y, z, ref = self._make_tensors()
        opt_fn = torch.compile(self.fn)
        res = opt_fn(x, y, z)
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu")))

    def test_compile_cpp_wrapper(self):
        """torch.compile with cpp_wrapper=True (AOT C++ code generation)."""
        x, y, z, ref = self._make_tensors()
        with inductor_config.patch({"cpp_wrapper": True}):
            opt_fn = torch.compile(self.fn)
            res, code = run_and_get_cpp_code(opt_fn, x, y, z)
            # res = opt_fn(x, y, z)
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu")))

    def test_compile_repeated_calls(self):
        """Compiled function produces consistent results across multiple calls."""
        x, y, z, ref = self._make_tensors()
        opt_fn = torch.compile(self.fn)
        for _ in range(3):
            res = opt_fn(x, y, z)
            self.assertTrue(torch.allclose(ref, res.to("cpu")))

    def test_compile_larger_tensor(self):
        """Compiled function works on larger tensors."""
        x, y, z, ref = self._make_tensors(rows=64, cols=128)
        opt_fn = torch.compile(self.fn)
        res = opt_fn(x, y, z)
        self.assertTrue(torch.allclose(ref, res.to("cpu")))


if __name__ == "__main__":
    unittest.main()
