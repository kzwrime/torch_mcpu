"""mcpu distributed backend registration."""

from __future__ import annotations

import torch
import torch.distributed as dist

import torch_mcpu._C  # type: ignore[misc]


_MCPU_BACKEND = "mcpu"


def register() -> None:
    backend_name = _MCPU_BACKEND
    if dist.is_backend_available(backend_name):
        return

    dist.Backend.register_backend(
        backend_name,
        torch_mcpu._C.create_process_group_mcpu,
        devices=[torch._C._get_privateuse1_backend_name()],
    )


register()
