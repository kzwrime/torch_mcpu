import sys
from functools import wraps

import torch


class _McpuDistributedWork:
    """Proxy work handle that keeps CPU views alive for async collectives."""

    def __init__(self, work, *tensors):
        self._work = work
        self._tensors = tensors

    def __getattr__(self, name):
        return getattr(self._work, name)


def _mcpu_to_cpu_view_tensor(tensor):
    if isinstance(tensor, torch.Tensor) and tensor.device.type == "mcpu":
        return torch.mcpu.get_cpu_view_from_mcpu_tensor(tensor)
    return tensor


def _mcpu_to_cpu_view_tensors(tensors):
    return [_mcpu_to_cpu_view_tensor(tensor) for tensor in tensors]


def _mcpu_wrap_async_work(work, *tensors):
    if work is not None:
        return _McpuDistributedWork(work, *tensors)
    return work


def _patch_imported_function_aliases(name, *original_functions, replacement):
    for mod in list(sys.modules.values()):
        try:
            current = getattr(mod, name, None)
        except Exception:
            continue
        if any(current is fn for fn in original_functions):
            try:
                setattr(mod, name, replacement)
            except Exception:
                continue


def patch_mcpu_distributed():
    try:
        import torch.distributed as dist
        import torch.distributed.distributed_c10d as dist_c10d
    except ImportError:
        return

    if getattr(dist_c10d, "_mcpu_collectives_patched", False):
        return

    original_dist_all_reduce = dist.all_reduce
    original_dist_all_gather = dist.all_gather
    original_dist_all_gather_into_tensor = dist.all_gather_into_tensor
    original_dist_gather = dist.gather
    original_dist_send = dist.send
    original_dist_recv = dist.recv
    original_dist_isend = dist.isend
    original_dist_irecv = dist.irecv
    original_dist_broadcast = dist.broadcast
    original_all_reduce = dist_c10d.all_reduce
    original_all_gather = dist_c10d.all_gather
    original_all_gather_into_tensor = dist_c10d.all_gather_into_tensor
    original_gather = dist_c10d.gather
    original_send = dist_c10d.send
    original_recv = dist_c10d.recv
    original_isend = dist_c10d.isend
    original_irecv = dist_c10d.irecv
    original_broadcast = dist_c10d.broadcast

    @wraps(original_all_reduce)
    def _mcpu_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "mcpu":
            cpu_tensor = _mcpu_to_cpu_view_tensor(tensor)
            work = original_all_reduce(
                cpu_tensor,
                op=op,
                group=group,
                async_op=async_op,
            )
            if async_op and work is not None:
                return _mcpu_wrap_async_work(work, tensor, cpu_tensor)
            return work
        return original_all_reduce(
            tensor,
            op=op,
            group=group,
            async_op=async_op,
        )

    @wraps(original_all_gather)
    def _mcpu_all_gather(tensor_list, tensor, group=None, async_op=False):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "mcpu":
            cpu_tensor = _mcpu_to_cpu_view_tensor(tensor)
            cpu_tensor_list = _mcpu_to_cpu_view_tensors(tensor_list)
            work = original_all_gather(
                cpu_tensor_list,
                cpu_tensor,
                group=group,
                async_op=async_op,
            )
            if async_op and work is not None:
                return _mcpu_wrap_async_work(
                    work,
                    tensor,
                    cpu_tensor,
                    *tensor_list,
                    *cpu_tensor_list,
                )
            return work
        return original_all_gather(
            tensor_list,
            tensor,
            group=group,
            async_op=async_op,
        )

    @wraps(original_all_gather_into_tensor)
    def _mcpu_all_gather_into_tensor(
        output_tensor,
        input_tensor,
        group=None,
        async_op=False,
    ):
        if (
            isinstance(input_tensor, torch.Tensor)
            and input_tensor.device.type == "mcpu"
        ):
            cpu_input_tensor = _mcpu_to_cpu_view_tensor(input_tensor)
            cpu_output_tensor = _mcpu_to_cpu_view_tensor(output_tensor)
            work = original_all_gather_into_tensor(
                cpu_output_tensor,
                cpu_input_tensor,
                group=group,
                async_op=async_op,
            )
            if async_op and work is not None:
                return _mcpu_wrap_async_work(
                    work,
                    output_tensor,
                    cpu_output_tensor,
                    input_tensor,
                    cpu_input_tensor,
                )
            return work
        return original_all_gather_into_tensor(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )

    @wraps(original_gather)
    def _mcpu_gather(
        tensor,
        gather_list=None,
        dst=None,
        group=None,
        async_op=False,
        group_dst=None,
    ):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "mcpu":
            cpu_tensor = _mcpu_to_cpu_view_tensor(tensor)
            cpu_gather_list = None
            keepalive_tensors = [tensor, cpu_tensor]
            if gather_list:
                cpu_gather_list = _mcpu_to_cpu_view_tensors(gather_list)
                keepalive_tensors.extend(gather_list)
                keepalive_tensors.extend(cpu_gather_list)
            work = original_gather(
                cpu_tensor,
                gather_list=cpu_gather_list,
                dst=dst,
                group=group,
                async_op=async_op,
                group_dst=group_dst,
            )
            if async_op and work is not None:
                return _mcpu_wrap_async_work(work, *keepalive_tensors)
            return work
        return original_gather(
            tensor,
            gather_list=gather_list,
            dst=dst,
            group=group,
            async_op=async_op,
            group_dst=group_dst,
        )

    @wraps(original_send)
    def _mcpu_send(tensor, dst=None, group=None, tag=0, group_dst=None):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "mcpu":
            return original_send(
                _mcpu_to_cpu_view_tensor(tensor),
                dst=dst,
                group=group,
                tag=tag,
                group_dst=group_dst,
            )
        return original_send(
            tensor,
            dst=dst,
            group=group,
            tag=tag,
            group_dst=group_dst,
        )

    @wraps(original_recv)
    def _mcpu_recv(tensor, src=None, group=None, tag=0, group_src=None):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "mcpu":
            return original_recv(
                _mcpu_to_cpu_view_tensor(tensor),
                src=src,
                group=group,
                tag=tag,
                group_src=group_src,
            )
        return original_recv(
            tensor,
            src=src,
            group=group,
            tag=tag,
            group_src=group_src,
        )

    @wraps(original_isend)
    def _mcpu_isend(tensor, dst=None, group=None, tag=0, group_dst=None):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "mcpu":
            cpu_tensor = _mcpu_to_cpu_view_tensor(tensor)
            work = original_isend(
                cpu_tensor,
                dst=dst,
                group=group,
                tag=tag,
                group_dst=group_dst,
            )
            return _mcpu_wrap_async_work(work, tensor, cpu_tensor)
        return original_isend(
            tensor,
            dst=dst,
            group=group,
            tag=tag,
            group_dst=group_dst,
        )

    @wraps(original_irecv)
    def _mcpu_irecv(tensor, src=None, group=None, tag=0, group_src=None):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "mcpu":
            cpu_tensor = _mcpu_to_cpu_view_tensor(tensor)
            work = original_irecv(
                cpu_tensor,
                src=src,
                group=group,
                tag=tag,
                group_src=group_src,
            )
            return _mcpu_wrap_async_work(work, tensor, cpu_tensor)
        return original_irecv(
            tensor,
            src=src,
            group=group,
            tag=tag,
            group_src=group_src,
        )

    @wraps(original_broadcast)
    def _mcpu_broadcast(
        tensor,
        src=None,
        group=None,
        async_op=False,
        group_src=None,
    ):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "mcpu":
            cpu_tensor = _mcpu_to_cpu_view_tensor(tensor)
            work = original_broadcast(
                cpu_tensor,
                src=src,
                group=group,
                async_op=async_op,
                group_src=group_src,
            )
            if async_op and work is not None:
                return _mcpu_wrap_async_work(work, tensor, cpu_tensor)
            return work
        return original_broadcast(
            tensor,
            src=src,
            group=group,
            async_op=async_op,
            group_src=group_src,
        )

    dist_c10d.all_reduce = _mcpu_all_reduce
    dist_c10d.all_gather = _mcpu_all_gather
    dist_c10d.all_gather_into_tensor = _mcpu_all_gather_into_tensor
    dist_c10d.gather = _mcpu_gather
    dist_c10d.send = _mcpu_send
    dist_c10d.recv = _mcpu_recv
    dist_c10d.isend = _mcpu_isend
    dist_c10d.irecv = _mcpu_irecv
    dist_c10d.broadcast = _mcpu_broadcast
    dist.all_reduce = _mcpu_all_reduce
    dist.all_gather = _mcpu_all_gather
    dist.all_gather_into_tensor = _mcpu_all_gather_into_tensor
    dist.gather = _mcpu_gather
    dist.send = _mcpu_send
    dist.recv = _mcpu_recv
    dist.isend = _mcpu_isend
    dist.irecv = _mcpu_irecv
    dist.broadcast = _mcpu_broadcast
    _patch_imported_function_aliases(
        "all_reduce",
        original_dist_all_reduce,
        original_all_reduce,
        replacement=_mcpu_all_reduce,
    )
    _patch_imported_function_aliases(
        "all_gather",
        original_dist_all_gather,
        original_all_gather,
        replacement=_mcpu_all_gather,
    )
    _patch_imported_function_aliases(
        "all_gather_into_tensor",
        original_dist_all_gather_into_tensor,
        original_all_gather_into_tensor,
        replacement=_mcpu_all_gather_into_tensor,
    )
    _patch_imported_function_aliases(
        "gather",
        original_dist_gather,
        original_gather,
        replacement=_mcpu_gather,
    )
    _patch_imported_function_aliases(
        "send",
        original_dist_send,
        original_send,
        replacement=_mcpu_send,
    )
    _patch_imported_function_aliases(
        "recv",
        original_dist_recv,
        original_recv,
        replacement=_mcpu_recv,
    )
    _patch_imported_function_aliases(
        "isend",
        original_dist_isend,
        original_isend,
        replacement=_mcpu_isend,
    )
    _patch_imported_function_aliases(
        "irecv",
        original_dist_irecv,
        original_irecv,
        replacement=_mcpu_irecv,
    )
    _patch_imported_function_aliases(
        "broadcast",
        original_dist_broadcast,
        original_broadcast,
        replacement=_mcpu_broadcast,
    )
    dist_c10d._mcpu_collectives_patched = True
