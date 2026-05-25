import hashlib
import inspect
import operator
from collections.abc import Callable
from typing import Any

import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._ops import OpOverload


def _node_tensor(node: fx.Node) -> torch.Tensor | None:
    value = node.meta.get("val", node.meta.get("example_value"))
    return value if isinstance(value, torch.Tensor) else None


def _is_mcpu_tensor(node: fx.Node) -> bool:
    value = _node_tensor(node)
    return value is not None and value.device.type in ("mcpu", "privateuseone")


def _is_call(node: fx.Node, target: object) -> bool:
    return node.op == "call_function" and node.target == target


def _is_add(node: fx.Node) -> bool:
    return _is_call(node, torch.ops.aten.add.Tensor) and node.kwargs.get("alpha", 1) == 1


def _is_mul(node: fx.Node) -> bool:
    return _is_call(node, torch.ops.aten.mul.Tensor)


def _is_sigmoid(node: fx.Node) -> bool:
    return _is_call(node, torch.ops.aten.sigmoid.default)


def _match_sigmoid_mul(node: fx.Node) -> tuple[fx.Node, fx.Node] | None:
    if not _is_mul(node) or len(node.args) < 2:
        return None
    lhs, rhs = node.args[:2]
    if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
        return None
    if _is_sigmoid(lhs) and lhs.args and isinstance(lhs.args[0], fx.Node):
        return lhs.args[0], rhs
    if _is_sigmoid(rhs) and rhs.args and isinstance(rhs.args[0], fx.Node):
        return rhs.args[0], lhs
    return None


def _fused_sigmoid_and_add_op(dtype: torch.dtype) -> OpOverload | None:
    if dtype == torch.bfloat16:
        return torch.ops.torch_xcpu.fused_sigmoid_and_add_bf16.default
    if dtype == torch.float32:
        return torch.ops.torch_xcpu.fused_sigmoid_and_add_fp32.default
    return None


def _fused_sigmoid_and_mul_op(dtype: torch.dtype) -> OpOverload | None:
    if dtype == torch.bfloat16:
        return torch.ops.torch_xcpu.fused_sigmoid_and_mul_bf16.default
    if dtype == torch.float32:
        return torch.ops.torch_xcpu.fused_sigmoid_and_mul_fp32.default
    return None


def _replace_with_auto_functionalized(
    graph: fx.Graph,
    anchor: fx.Node,
    fused_op: OpOverload,
    output_like: fx.Node,
    kwargs: dict[str, fx.Node],
) -> fx.Node:
    output_value = _node_tensor(output_like)
    with graph.inserting_before(anchor):
        output = graph.call_function(torch.ops.aten.empty_like.default, (output_like,))
        if output_value is not None:
            output.meta["val"] = output_value
        result = graph.call_function(
            auto_functionalized,
            (fused_op,),
            {"output": output, **kwargs},
        )
        getitem = graph.call_function(operator.getitem, (result, 1))
        if output_value is not None:
            getitem.meta["val"] = output_value
        return getitem


def _fuse_sigmoid_and_add(graph: fx.Graph) -> tuple[int, set[fx.Node]]:
    count = 0
    consumed_mul_nodes: set[fx.Node] = set()
    for node in list(graph.nodes):
        if not _is_add(node) or len(node.args) < 2:
            continue

        lhs, rhs = node.args[:2]
        if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
            continue

        match = _match_sigmoid_mul(lhs)
        mul_node, other = lhs, rhs
        if match is None:
            match = _match_sigmoid_mul(rhs)
            mul_node, other = rhs, lhs
        if match is None or not isinstance(mul_node, fx.Node):
            continue

        gate, input_ = match
        input_value = _node_tensor(input_)
        gate_value = _node_tensor(gate)
        other_value = _node_tensor(other)
        if input_value is None or gate_value is None or other_value is None:
            continue
        if not _is_mcpu_tensor(input_) or input_value.dim() != 2:
            continue
        if other_value.shape != input_value.shape:
            continue
        if gate_value.dim() not in (1, 2) or gate_value.shape[0] != input_value.shape[0]:
            continue
        if gate_value.dim() == 2 and gate_value.shape[1] not in (1, input_value.shape[1]):
            continue

        fused_op = _fused_sigmoid_and_add_op(input_value.dtype)
        if fused_op is None:
            continue

        replacement = _replace_with_auto_functionalized(
            graph,
            node,
            fused_op,
            input_,
            {"gate": gate, "input": input_, "other": other},
        )
        node.replace_all_uses_with(replacement)
        consumed_mul_nodes.add(mul_node)
        count += 1
    return count, consumed_mul_nodes


def _fuse_sigmoid_and_mul(graph: fx.Graph, skip_nodes: set[fx.Node]) -> int:
    count = 0
    for node in list(graph.nodes):
        if node in skip_nodes:
            continue
        match = _match_sigmoid_mul(node)
        if match is None:
            continue

        gate, input_ = match
        input_value = _node_tensor(input_)
        gate_value = _node_tensor(gate)
        if input_value is None or gate_value is None:
            continue
        if not _is_mcpu_tensor(input_) or input_value.dim() != 3:
            continue
        if gate_value.shape != input_value.shape:
            continue

        fused_op = _fused_sigmoid_and_mul_op(input_value.dtype)
        if fused_op is None:
            continue

        replacement = _replace_with_auto_functionalized(
            graph,
            node,
            fused_op,
            input_,
            {"input": input_, "gate": gate},
        )
        node.replace_all_uses_with(replacement)
        count += 1
    return count


class McpuTorchXcpuFusionPass:
    """Replace small mcpu pointwise chains with torch_xcpu fused ops."""

    def __call__(self, graph: fx.Graph) -> None:
        try:
            import torch_xcpu  # noqa: F401
        except ImportError:
            return

        _, consumed_mul_nodes = _fuse_sigmoid_and_add(graph)
        _fuse_sigmoid_and_mul(graph, consumed_mul_nodes)
        graph.eliminate_dead_code()

    def uuid(self) -> str:
        hasher = hashlib.sha256()
        for item in (
            McpuTorchXcpuFusionPass,
            _fuse_sigmoid_and_add,
            _fuse_sigmoid_and_mul,
        ):
            hasher.update(inspect.getsource(item).encode("utf-8"))
        return hasher.hexdigest()


class ChainedPostGradPass:
    def __init__(self, *passes: Callable[[fx.Graph], None]) -> None:
        self.passes = passes

    def __call__(self, graph: fx.Graph) -> None:
        for pass_ in self.passes:
            pass_(graph)

    def uuid(self) -> str:
        state: list[str] = []
        for pass_ in self.passes:
            uuid_fn = getattr(pass_, "uuid", None)
            if callable(uuid_fn):
                state.append(str(uuid_fn()))
            else:
                state.append(repr(pass_))
        return hashlib.sha256(repr(state).encode("utf-8")).hexdigest()


def append_post_grad_pass(
    existing: Any,
    new_pass: Callable[[fx.Graph], None],
) -> Callable[[fx.Graph], None]:
    if existing is None:
        return new_pass
    return ChainedPostGradPass(new_pass, existing)
