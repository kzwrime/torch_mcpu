#!/usr/bin/env python
"""
Minimal MLP example with custom mysilu_out and torch.compile
Demonstrating Dynamic Shapes for varying token counts (vLLM style)
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_mcpu  # noqa: F401 – registers the mcpu backend with PyTorch

ddevice = "mcpu"


def silu(x):
    """PyTorch-native implementation equivalent to forward()."""
    return F.silu(x)


class MLP(nn.Module):
    def __init__(self, in_out_dim=4, hidden_dim=8):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(hidden_dim, in_out_dim) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(in_out_dim, hidden_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(in_out_dim))

    def forward(self, x):
        # First mm
        hidden = torch.mm(x, self.W1.t()) + self.b1
        # SiLU activation
        # activated = torch.empty_like(hidden)
        # ops.mysilu_out(hidden, activated)
        activated = silu(hidden)
        # Second mm
        out = torch.mm(activated, self.W2.t()) + self.b2
        return out


class MultiMLP(nn.Module):
    def __init__(self, num_mlps=3, in_out_dim=4, hidden_dim=8):
        super().__init__()
        self.mlps = nn.ModuleList([
            MLP(in_out_dim, hidden_dim) for _ in range(num_mlps)
        ])

    def forward(self, x):
        outputs = []
        for mlp in self.mlps:
            outputs.append(mlp(x))
        return outputs


@pytest.fixture
def mlp():
    """Create a single MLP model."""
    return MLP(in_out_dim=4, hidden_dim=8).to(ddevice).eval()


@pytest.fixture
def multi_mlp():
    """Create a MultiMLP model."""
    return MultiMLP(num_mlps=20, in_out_dim=4, hidden_dim=8).to(ddevice).eval()


@pytest.fixture
def multi_mlp_compiled(multi_mlp):
    """Create a compiled MultiMLP model."""
    compiled = torch.compile(multi_mlp)
    return compiled


def test_mlp_forward(mlp):
    """Test basic MLP forward pass."""
    x = torch.randn(5, 4).to(ddevice)
    with torch.inference_mode():
        out = mlp(x)
    assert out.shape == (5, 4)


def test_multi_mlp_forward(multi_mlp):
    """Test MultiMLP forward pass."""
    x = torch.randn(5, 4).to(ddevice)
    with torch.inference_mode():
        outputs = multi_mlp(x)
    assert len(outputs) == 20
    for out in outputs:
        assert out.shape == (5, 4)


def test_compile_mlp_dynamic_shapes(multi_mlp, multi_mlp_compiled):
    """Test torch.compile with dynamic shapes across varying token counts."""
    x_init = torch.randn(5, 4).to(ddevice)

    with torch.inference_mode():
        # Mark initial input as dynamic
        torch._dynamo.mark_dynamic(x_init, 0)

        # Warmup
        for _ in range(3):
            _ = multi_mlp_compiled(x_init)

        # Test dynamic shapes
        shapes_to_test = [(12, 4), (1, 4), (1024, 4), (33, 4)]

        for shape in shapes_to_test:
            x_new = torch.randn(*shape).to(ddevice)
            torch._dynamo.mark_dynamic(x_new, 0)

            out_eager = multi_mlp(x_new)
            out_compiled = multi_mlp_compiled(x_new)

            assert len(out_eager) == len(out_compiled)
            for orig, comp in zip(out_eager, out_compiled):
                assert torch.allclose(orig, comp, rtol=1e-4), \
                    f"Outputs differ for shape {shape}"