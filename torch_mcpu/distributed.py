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


def patch_mcpu_distributed():
    try:
        import torch.distributed as dist
        import torch.distributed.distributed_c10d as dist_c10d
    except ImportError:
        return

    if getattr(dist_c10d, "_mcpu_collectives_patched", False):
        return

    original_all_reduce = dist_c10d.all_reduce
    original_all_gather = dist_c10d.all_gather
    original_all_gather_into_tensor = dist_c10d.all_gather_into_tensor
    original_gather = dist_c10d.gather

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

    dist_c10d.all_reduce = _mcpu_all_reduce
    dist_c10d.all_gather = _mcpu_all_gather
    dist_c10d.all_gather_into_tensor = _mcpu_all_gather_into_tensor
    dist_c10d.gather = _mcpu_gather
    dist.all_reduce = _mcpu_all_reduce
    dist.all_gather = _mcpu_all_gather
    dist.all_gather_into_tensor = _mcpu_all_gather_into_tensor
    dist.gather = _mcpu_gather
    dist_c10d._mcpu_collectives_patched = True
