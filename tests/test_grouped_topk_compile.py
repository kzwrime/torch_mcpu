"""
Regression tests for the DeepSeek grouped_topk routing pattern used by vLLM.

The copied function below intentionally mirrors the Python implementation in
vllm.model_executor.layers.fused_moe.router.grouped_topk_router.grouped_topk.
It isolates the torch.compile failure on mcpu without starting a vLLM engine.
"""

import pytest
import torch

import torch_mcpu  # noqa: F401 - registers the mcpu backend with PyTorch


def copied_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values

    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def _make_inputs(dtype: torch.dtype = torch.bfloat16):
    torch.manual_seed(0)
    num_tokens = 8
    hidden_size = 16
    num_experts = 16
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype)
    gating_output = torch.randn(num_tokens, num_experts, dtype=dtype)
    return hidden_states, gating_output


def test_copied_grouped_topk_eager_mcpu_matches_cpu():
    hidden_states, gating_output = _make_inputs()

    expected_weights, expected_ids = copied_grouped_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=2,
        renormalize=True,
        num_expert_group=4,
        topk_group=2,
    )
    actual_weights, actual_ids = copied_grouped_topk(
        hidden_states=hidden_states.to("mcpu"),
        gating_output=gating_output.to("mcpu"),
        topk=2,
        renormalize=True,
        num_expert_group=4,
        topk_group=2,
    )

    torch.testing.assert_close(actual_weights.to("cpu"), expected_weights)
    torch.testing.assert_close(actual_ids.to("cpu"), expected_ids)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mcpu Inductor currently cannot compile the dynamic grouped_topk "
        "reduction/topk pattern copied from vLLM."
    ),
)
def test_copied_grouped_topk_compile_mcpu_dynamic_shapes():
    hidden_states, gating_output = _make_inputs()
    hidden_states = hidden_states.to("mcpu")
    gating_output = gating_output.to("mcpu")
    torch._dynamo.mark_dynamic(hidden_states, 0)
    torch._dynamo.mark_dynamic(gating_output, 0)

    compiled_grouped_topk = torch.compile(
        copied_grouped_topk,
        dynamic=True,
        backend="inductor",
    )
    actual_weights, actual_ids = compiled_grouped_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=2,
        renormalize=True,
        num_expert_group=4,
        topk_group=2,
    )

    expected_weights, expected_ids = copied_grouped_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=2,
        renormalize=True,
        num_expert_group=4,
        topk_group=2,
    )
    torch.testing.assert_close(actual_weights.to("cpu"), expected_weights.to("cpu"))
    torch.testing.assert_close(actual_ids.to("cpu"), expected_ids.to("cpu"))
