"""
Regression test for mcpu Python-wrapper Inductor C++ pybinding kernels.

Run with:

    export TORCHINDUCTOR_CACHE_DIR="$PWD/torch_compile_cache"
    python -m pytest -q tests/test_python_wrapper_cpp_pybinding_repro.py

The generated wrapper files are left under TORCHINDUCTOR_CACHE_DIR so they can
be opened directly after the test.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run_repro(cache_dir, repro):
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(Path.cwd()), os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
        "TORCHINDUCTOR_CACHE_DIR": str(cache_dir),
    }
    return subprocess.run(
        [sys.executable, "-c", repro],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _read_generated_python(cache_dir):
    return {
        path: path.read_text(errors="ignore")
        for path in cache_dir.rglob("*.py")
    }


def _make_cache_dir():
    cache_dir = Path(
        os.environ.get("TORCHINDUCTOR_CACHE_DIR", Path.cwd() / "torch_compile_cache")
    )
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def test_python_wrapper_does_not_emit_raw_cpp_pybinding_zero_kernel_for_mcpu():
    cache_dir = _make_cache_dir()
    repro = r"""
import torch
import torch._dynamo
import torch._inductor.config as inductor_config
import torch_mcpu  # noqa: F401


def split_then_zero(x):
    left, _ = torch.split(
        x,
        [x.shape[1] // 2, x.shape[1] - x.shape[1] // 2],
        dim=1,
    )
    return torch.zeros_like(left)


x = torch.empty(8, 64, device="mcpu", dtype=torch.bfloat16).fill_(1)
torch._dynamo.mark_dynamic(x, 0)

with inductor_config.patch({
    "cpp_wrapper": False,
    "post_grad_custom_post_pass": None,
}):
    compiled = torch.compile(split_then_zero, dynamic=True, backend="inductor")
    compiled(x)
"""

    result = _run_repro(cache_dir, repro)
    generated_python = _read_generated_python(cache_dir)
    generated = []
    for path, text in generated_python.items():
        if (
            "async_compile.cpp_pybinding" in text
            and "cpp_fused_zeros_like" in text
        ):
            generated.append(path)

    assert result.returncode == 0, (
        "Expected compiled mcpu Python-wrapper graph to run without executing "
        "a raw host-side cpp_pybinding kernel. "
        f"Subprocess return code: {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert not generated, (
        "Expected mcpu Python-wrapper Inductor to route zeros_like through "
        f"ATen fallback instead of emitting raw cpp_pybinding kernels: {generated}"
    )


def test_python_wrapper_does_not_emit_raw_cpp_pybinding_add_mul_for_mcpu():
    cache_dir = _make_cache_dir()
    repro = r"""
import torch
import torch._inductor.config as inductor_config
import torch_mcpu  # noqa: F401


def pointwise(a, b, c):
    return a * b + c


a = torch.empty(8, 64, device="mcpu", dtype=torch.bfloat16).fill_(1)
b = torch.empty(8, 64, device="mcpu", dtype=torch.bfloat16).fill_(2)
c = torch.empty(8, 64, device="mcpu", dtype=torch.bfloat16).fill_(3)

with inductor_config.patch({
    "cpp_wrapper": False,
    "post_grad_custom_post_pass": None,
}):
    compiled = torch.compile(pointwise, backend="inductor")
    out = compiled(a, b, c)

assert out.device.type == "mcpu"
assert torch.allclose(
    out.to("cpu"),
    torch.empty(8, 64, dtype=torch.bfloat16).fill_(5),
)
"""

    result = _run_repro(cache_dir, repro)
    generated_python = _read_generated_python(cache_dir)
    raw_kernels = [
        path
        for path, text in generated_python.items()
        if "async_compile.cpp_pybinding" in text or "cpp_fused" in text
    ]

    assert result.returncode == 0, (
        "Expected compiled mcpu Python-wrapper add/mul graph to run through "
        "ATen fallback. "
        f"Subprocess return code: {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert not raw_kernels, (
        "Expected mcpu Python-wrapper Inductor to route add/mul through "
        f"ATen fallback instead of emitting raw fused kernels: {raw_kernels}"
    )
