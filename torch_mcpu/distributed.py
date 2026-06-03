import sys
from functools import wraps

import torch
import torch_mcpu._C  # type: ignore[misc]


def _is_mcpu_tensor(obj):
    return isinstance(obj, torch.Tensor) and obj.device.type == "mcpu"


def _contains_mcpu_tensor(obj):
    if _is_mcpu_tensor(obj):
        return True
    if isinstance(obj, (list, tuple)):
        return any(_contains_mcpu_tensor(item) for item in obj)
    return False


def _get_group_or_default(group):
    if group is not None:
        return group
    import torch.distributed.distributed_c10d as dist_c10d

    return dist_c10d._get_default_group()


def _ensure_mcpu_process_group_backend(group=None):
    group = _get_group_or_default(group)

    mcpu_device = torch.device("mcpu")
    try:
        group._get_backend(mcpu_device)
        return group
    except Exception:
        pass

    cpu_backend = group._get_backend(torch.device("cpu"))
    mcpu_backend = torch_mcpu._C._make_mcpu_process_group_backend(cpu_backend)
    backend_type = torch.distributed.ProcessGroup.BackendType.CUSTOM
    group._register_backend(mcpu_device, backend_type, mcpu_backend)
    return group


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
    original_dist_reduce = dist.reduce
    original_dist_gather = dist.gather
    original_dist_send = dist.send
    original_dist_recv = dist.recv
    original_dist_isend = dist.isend
    original_dist_irecv = dist.irecv
    original_dist_broadcast = dist.broadcast
    original_all_reduce = dist_c10d.all_reduce
    original_all_gather = dist_c10d.all_gather
    original_all_gather_into_tensor = dist_c10d.all_gather_into_tensor
    original_reduce = dist_c10d.reduce
    original_gather = dist_c10d.gather
    original_send = dist_c10d.send
    original_recv = dist_c10d.recv
    original_isend = dist_c10d.isend
    original_irecv = dist_c10d.irecv
    original_broadcast = dist_c10d.broadcast

    @wraps(original_all_reduce)
    def _mcpu_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if _contains_mcpu_tensor(tensor):
            _ensure_mcpu_process_group_backend(group)
        return original_all_reduce(tensor, op=op, group=group, async_op=async_op)

    @wraps(original_all_gather)
    def _mcpu_all_gather(tensor_list, tensor, group=None, async_op=False):
        if _contains_mcpu_tensor(tensor) or _contains_mcpu_tensor(tensor_list):
            _ensure_mcpu_process_group_backend(group)
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
        if _contains_mcpu_tensor(input_tensor) or _contains_mcpu_tensor(output_tensor):
            _ensure_mcpu_process_group_backend(group)
        return original_all_gather_into_tensor(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )

    @wraps(original_reduce)
    def _mcpu_reduce(
        tensor,
        dst=None,
        op=dist.ReduceOp.SUM,
        group=None,
        async_op=False,
        group_dst=None,
    ):
        if _contains_mcpu_tensor(tensor):
            _ensure_mcpu_process_group_backend(group)
        return original_reduce(
            tensor,
            dst=dst,
            op=op,
            group=group,
            async_op=async_op,
            group_dst=group_dst,
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
        if _contains_mcpu_tensor(tensor) or _contains_mcpu_tensor(gather_list):
            _ensure_mcpu_process_group_backend(group)
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
        if _contains_mcpu_tensor(tensor):
            _ensure_mcpu_process_group_backend(group)
        return original_send(
            tensor,
            dst=dst,
            group=group,
            tag=tag,
            group_dst=group_dst,
        )

    @wraps(original_recv)
    def _mcpu_recv(tensor, src=None, group=None, tag=0, group_src=None):
        if _contains_mcpu_tensor(tensor):
            _ensure_mcpu_process_group_backend(group)
        return original_recv(
            tensor,
            src=src,
            group=group,
            tag=tag,
            group_src=group_src,
        )

    @wraps(original_isend)
    def _mcpu_isend(tensor, dst=None, group=None, tag=0, group_dst=None):
        if _contains_mcpu_tensor(tensor):
            _ensure_mcpu_process_group_backend(group)
        return original_isend(
            tensor,
            dst=dst,
            group=group,
            tag=tag,
            group_dst=group_dst,
        )

    @wraps(original_irecv)
    def _mcpu_irecv(tensor, src=None, group=None, tag=0, group_src=None):
        if _contains_mcpu_tensor(tensor):
            _ensure_mcpu_process_group_backend(group)
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
        if _contains_mcpu_tensor(tensor):
            _ensure_mcpu_process_group_backend(group)
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
    dist_c10d.reduce = _mcpu_reduce
    dist_c10d.gather = _mcpu_gather
    dist_c10d.send = _mcpu_send
    dist_c10d.recv = _mcpu_recv
    dist_c10d.isend = _mcpu_isend
    dist_c10d.irecv = _mcpu_irecv
    dist_c10d.broadcast = _mcpu_broadcast
    dist.all_reduce = _mcpu_all_reduce
    dist.all_gather = _mcpu_all_gather
    dist.all_gather_into_tensor = _mcpu_all_gather_into_tensor
    dist.reduce = _mcpu_reduce
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
        "reduce",
        original_dist_reduce,
        original_reduce,
        replacement=_mcpu_reduce,
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
