"""
Tests for torch.compile support on the mcpu backend.

Mirrors docs/ref/extension_backends/cpp/test_extension_backend.py:
verifies that registering mcpu with torch._inductor allows torch.compile()
to generate and execute fused C++ kernels for mcpu tensors.
"""

import os
import unittest
from importlib.util import find_spec
from itertools import count
from pathlib import Path
from unittest.mock import patch

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from torch.utils._ordered_set import OrderedSet
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
from torch_mcpu.inductor.torch_xcpu_fusion import McpuTorchXcpuFusionPass


def _setup_mcpu_inductor():
    """Register mcpu with torch.inductor (idempotent)."""
    register_backend_for_device(
        "mcpu",
        McpuScheduling,
        McpuWrapperCodegen,
        McpuCppWrapperCodegen,
    )
    cpp_utils.DEVICE_TO_ATEN["mcpu"] = "at::kPrivateUse1"


def _torch_xcpu_aoti_env():
    spec = find_spec("torch_xcpu")
    if spec is None or not spec.submodule_search_locations:
        raise unittest.SkipTest("torch_xcpu is not available")

    install_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
    header = install_dir / "include" / "aoti_torch_xcpu.h"
    so_files = sorted(install_dir.glob("_C*.so"))
    if not header.is_file() or not so_files:
        raise unittest.SkipTest("torch_xcpu AOTI bits are not available")

    so_file = so_files[0]
    return {
        "AOTI_EXTRA_CFLAGS": f"-include {header}",
        "AOTI_EXTRA_LDFLAGS": f"-Wl,-rpath,{so_file.parent} {so_file}",
        "TORCHINDUCTOR_DIRECT_DISPATCH_PREFIXES": "torch_xcpu",
    }


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

    @staticmethod
    def _make_cpp_codegen_for_int_array_tests():
        codegen = object.__new__(McpuCppWrapperCodegen)
        codegen.int_array_id = count()
        codegen.declared_int_array_vars = OrderedSet()
        codegen.codegen_int_array_var_cache = {}
        return codegen

    def test_cpp_wrapper_does_not_cache_dynamic_int_arrays(self):
        codegen = self._make_cpp_codegen_for_int_array_tests()
        lines = []

        first = codegen.codegen_int_array_var(
            "{s0, 2L, 128L}", lines.append, known_statically=False
        )
        second = codegen.codegen_int_array_var(
            "{s0, 2L, 128L}", lines.append, known_statically=False
        )

        self.assertNotEqual(first, second)
        self.assertEqual(len(lines), 2)

    def test_cpp_wrapper_does_not_cache_static_int_arrays(self):
        codegen = self._make_cpp_codegen_for_int_array_tests()
        lines = []

        first = codegen.codegen_int_array_var(
            "{1L, 2L, 128L}", lines.append, known_statically=True
        )
        second = codegen.codegen_int_array_var(
            "{1L, 2L, 128L}", lines.append, known_statically=True
        )

        self.assertNotEqual(first, second)
        self.assertEqual(len(lines), 2)


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

    def test_cpp_wrapper_uses_mcpu_aten_fallback_not_cpp_fused_loop(self):
        """mcpu cpp_wrapper must not emit raw Inductor C++ fused loops."""

        def pointwise(a, b, c):
            return a[:, None].sigmoid() * b + c

        a = torch.empty(4, device="mcpu", dtype=torch.bfloat16).fill_(1)
        b = torch.empty(4, 8, device="mcpu", dtype=torch.bfloat16).fill_(2)
        c = torch.empty(4, 8, device="mcpu", dtype=torch.bfloat16).fill_(3)
        ref = pointwise(a, b, c).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "fallback_by_default": True,
            "post_grad_custom_post_pass": None,
        }):
            opt_fn = torch.compile(pointwise)
            res, code = run_and_get_cpp_code(opt_fn, a, b, c)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu")))
        self.assertNotIn("cpp_fused", code_text)
        self.assertIn("aoti_torch_mcpu_mul_Tensor", code_text)
        self.assertIn("aoti_torch_mcpu_add_Tensor", code_text)

    def test_cpp_wrapper_fuses_sigmoid_and_add_with_torch_xcpu(self):
        """mcpu post-grad pass replaces add/mul/sigmoid with torch_xcpu op."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def pointwise(a, b, c):
            return b * a[:, None].sigmoid() + c

        a = torch.empty(4, device="mcpu", dtype=torch.bfloat16).fill_(1)
        b = torch.empty(4, 8, device="mcpu", dtype=torch.bfloat16).fill_(2)
        c = torch.empty(4, 8, device="mcpu", dtype=torch.bfloat16).fill_(3)
        ref = pointwise(a, b, c).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(pointwise)
            res, code = run_and_get_cpp_code(opt_fn, a, b, c)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu")))
        self.assertNotIn("cpp_fused_add_mul_sigmoid", code_text)
        self.assertIn(
            "torch_xcpu::fused_sigmoid_mul_add_3d_lastdim_bf16",
            code_text,
        )

    def test_cpp_wrapper_fuses_sigmoid_and_mul_with_torch_xcpu(self):
        """mcpu post-grad pass replaces mul/sigmoid with torch_xcpu op."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def pointwise(a, b):
            return b * a.sigmoid()

        a = torch.empty(2, 3, 8, device="mcpu", dtype=torch.bfloat16).fill_(1)
        b = torch.empty(2, 3, 8, device="mcpu", dtype=torch.bfloat16).fill_(2)
        ref = pointwise(a, b).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(pointwise)
            res, code = run_and_get_cpp_code(opt_fn, a, b)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu")))
        self.assertNotIn("cpp_fused_mul_sigmoid", code_text)
        self.assertIn(
            "torch_xcpu::fused_sigmoid_mul_3d_lastdim_bf16",
            code_text,
        )

    def test_cpp_wrapper_direct_dispatch_fusions_with_dynamic_shapes(self):
        """Cover the vLLM path: dynamic dims, direct dispatch, AOTI C++ wrapper."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def pointwise(input2_2d, input1_2d, input3_2d, input2_3d, input1_3d):
            add_out = input1_2d * input2_2d.sigmoid() + input3_2d
            mul_out = input1_3d * input2_3d.sigmoid()
            return add_out, mul_out

        input2_2d = torch.empty(
            4, 1, device="mcpu", dtype=torch.bfloat16
        ).fill_(1)
        input1_2d = torch.empty(
            4, 8, device="mcpu", dtype=torch.bfloat16
        ).fill_(2)
        input3_2d = torch.empty(
            4, 8, device="mcpu", dtype=torch.bfloat16
        ).fill_(3)
        input2_3d = torch.empty(
            4, 2, 8, device="mcpu", dtype=torch.bfloat16
        ).fill_(1)
        input1_3d = torch.empty(
            4, 2, 8, device="mcpu", dtype=torch.bfloat16
        ).fill_(2)

        ref_add, ref_mul = pointwise(
            input2_2d, input1_2d, input3_2d, input2_3d, input1_3d
        )
        torch._dynamo.mark_dynamic(input2_2d, 0)
        torch._dynamo.mark_dynamic(input1_2d, 0)
        torch._dynamo.mark_dynamic(input3_2d, 0)
        torch._dynamo.mark_dynamic(input2_3d, 0)
        torch._dynamo.mark_dynamic(input1_3d, 0)

        with (
            patch.dict(os.environ, _torch_xcpu_aoti_env(), clear=False),
            inductor_config.patch({"cpp_wrapper": True}),
        ):
            opt_fn = torch.compile(pointwise, dynamic=True)
            res, code = run_and_get_cpp_code(
                opt_fn, input2_2d, input1_2d, input3_2d, input2_3d, input1_3d
            )

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertTrue(torch.allclose(ref_add.to("cpu"), res[0].to("cpu")))
        self.assertTrue(torch.allclose(ref_mul.to("cpu"), res[1].to("cpu")))
        self.assertNotIn("cpp_fused_add_mul_sigmoid", code_text)
        self.assertNotIn("cpp_fused_mul_sigmoid", code_text)
        self.assertIn(
            "aoti_torch_mcpu_fused_sigmoid_mul_add_3d_lastdim_bf16",
            code_text,
        )
        self.assertIn(
            "aoti_torch_mcpu_fused_sigmoid_mul_3d_lastdim_bf16",
            code_text,
        )

    def test_torch_xcpu_fusion_fuses_3d_broadcast_add(self):
        """Cover MoE-style [tokens, 1, hidden] * [tokens, 1, 1] + residual."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def pointwise(input2, input1, input3):
            return input1 * input2.sigmoid() + input3

        input2 = torch.empty(4, 1, 1, device="mcpu", dtype=torch.bfloat16).fill_(1)
        input1 = torch.empty(4, 1, 8, device="mcpu", dtype=torch.bfloat16).fill_(2)
        input3 = torch.empty(4, 1, 8, device="mcpu", dtype=torch.bfloat16).fill_(3)
        ref = pointwise(input2, input1, input3).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(pointwise)
            res, code = run_and_get_cpp_code(opt_fn, input2, input1, input3)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu")))
        self.assertIn(
            "torch_xcpu::fused_sigmoid_mul_add_3d_lastdim_bf16",
            code_text,
        )

    def test_torch_xcpu_fusion_follows_pytorch_2d_broadcast_for_3d_add(self):
        """A [seq, hidden] gate broadcasts to [1, seq, hidden], not [seq, 1, hidden]."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def pointwise(input2, input1, input3):
            return input1 * input2.sigmoid() + input3

        input2_cpu = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 16
        input2 = input2_cpu.to(torch.bfloat16).to("mcpu")
        input1 = torch.empty(4, 4, 8, device="mcpu", dtype=torch.bfloat16).fill_(1)
        input3 = torch.empty(4, 4, 8, device="mcpu", dtype=torch.bfloat16).fill_(0)
        ref = pointwise(input2, input1, input3).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(pointwise)
            res, code = run_and_get_cpp_code(opt_fn, input2, input1, input3)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu"), atol=1e-2, rtol=1e-2))
        self.assertIn(
            "torch_xcpu::fused_sigmoid_mul_add_3d_lastdim_bf16",
            code_text,
        )

    def test_torch_xcpu_fusion_follows_pytorch_1d_broadcast_for_3d_add(self):
        """A [hidden] gate stays right-aligned even when batch == hidden."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def pointwise(input2, input1, input3):
            return input1 * input2.sigmoid() + input3

        input2_cpu = torch.arange(4, dtype=torch.float32) / 4
        input2 = input2_cpu.to(torch.bfloat16).to("mcpu")
        input1 = torch.empty(4, 2, 4, device="mcpu", dtype=torch.bfloat16).fill_(1)
        input3 = torch.empty(4, 2, 4, device="mcpu", dtype=torch.bfloat16).fill_(0)
        ref = pointwise(input2, input1, input3).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(pointwise)
            res, code = run_and_get_cpp_code(opt_fn, input2, input1, input3)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu"), atol=1e-2, rtol=1e-2))
        self.assertIn(
            "torch_xcpu::fused_sigmoid_mul_add_3d_lastdim_bf16",
            code_text,
        )

    def test_torch_xcpu_fusion_fuses_3d_lastdim_contiguous_slice_mul(self):
        """Cover attention-style gate slices whose last dimension is contiguous."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def pointwise(input2, input1):
            return input1 * input2.sigmoid()

        input1 = torch.empty(4, 8, 256, device="mcpu", dtype=torch.bfloat16).fill_(2)
        input2_base = torch.empty(
            4, 8, 512, device="mcpu", dtype=torch.bfloat16
        ).fill_(1)
        input2 = input2_base[:, :, 256:]
        ref = pointwise(input2, input1).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(pointwise)
            res, code = run_and_get_cpp_code(opt_fn, input2, input1)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu")))
        self.assertIn(
            "torch_xcpu::fused_sigmoid_mul_3d_lastdim_bf16",
            code_text,
        )

    def test_torch_xcpu_fusion_skips_unsupported_stride(self):
        """The custom pass only selects kernels matching their layout names."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def noncontig_mul(a, b):
            return b * a.sigmoid()

        a_base = torch.empty(2, 3, 16, device="mcpu", dtype=torch.bfloat16).fill_(1)
        b_base = torch.empty(2, 3, 16, device="mcpu", dtype=torch.bfloat16).fill_(2)
        a = a_base[:, :, ::2]
        b = b_base[:, :, ::2]
        ref = noncontig_mul(a, b).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(noncontig_mul)
            res, code = run_and_get_cpp_code(opt_fn, a, b)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu")))
        self.assertNotIn(
            "torch_xcpu::fused_sigmoid_mul_3d_lastdim",
            code_text,
        )

    def test_torch_xcpu_fusion_skips_mixed_dtype_add(self):
        """Do not reinterpret mixed dtype tensors as a single fused-op dtype."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def mixed_dtype_add(a, b, c):
            return b * a[:, None].sigmoid() + c

        a = torch.empty(4, device="mcpu", dtype=torch.bfloat16).fill_(1)
        b = torch.empty(4, 8, device="mcpu", dtype=torch.bfloat16).fill_(2)
        c = torch.empty(4, 8, device="mcpu", dtype=torch.float32).fill_(3)
        ref = mixed_dtype_add(a, b, c).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(mixed_dtype_add)
            res, code = run_and_get_cpp_code(opt_fn, a, b, c)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu"), atol=1e-2, rtol=1e-2))
        self.assertNotIn(
            "torch_xcpu::fused_sigmoid_mul_add_3d_lastdim",
            code_text,
        )

    def test_torch_xcpu_fusion_skips_mixed_dtype_mul(self):
        """Do not fuse sigmoid/mul when the two operands have different dtypes."""

        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            self.skipTest("torch_xcpu is not available")

        def mixed_dtype_mul(a, b):
            return b * a.sigmoid()

        a = torch.empty(2, 3, 8, device="mcpu", dtype=torch.float32).fill_(1)
        b = torch.empty(2, 3, 8, device="mcpu", dtype=torch.bfloat16).fill_(2)
        ref = mixed_dtype_mul(a, b).to("cpu")

        with inductor_config.patch({
            "cpp_wrapper": True,
            "post_grad_custom_post_pass": McpuTorchXcpuFusionPass(),
        }):
            opt_fn = torch.compile(mixed_dtype_mul)
            res, code = run_and_get_cpp_code(opt_fn, a, b)

        code_text = "\n".join(code) if isinstance(code, (list, tuple)) else code
        self.assertEqual(res.device.type, "mcpu")
        self.assertTrue(torch.allclose(ref, res.to("cpu"), atol=1e-2, rtol=1e-2))
        self.assertNotIn(
            "torch_xcpu::fused_sigmoid_mul_3d_lastdim",
            code_text,
        )

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
