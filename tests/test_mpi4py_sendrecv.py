#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch_mcpu  # noqa: F401
from mpi4py import MPI


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_size != 2:
        raise RuntimeError(
            f"test_mpi4py_sendrecv.py expects exactly 2 ranks, got {world_size}"
        )

    count = 16
    dtype = torch.float32

    if rank == 0:
        send = torch.arange(count, dtype=dtype, device="mcpu")
        recv = torch.empty(count, dtype=dtype, device="mcpu")

        send_req = comm.Isend(send, dest=1, tag=11)
        comm.Recv(recv, source=1, tag=22)
        send_req.Wait()

        expected = torch.arange(count, dtype=dtype) + 1000
        torch.testing.assert_close(recv.cpu(), expected)
    else:
        recv = torch.empty(count, dtype=dtype, device="mcpu")
        send = torch.arange(count, dtype=dtype, device="mcpu") + 1000

        recv_req = comm.Irecv(recv, source=0, tag=11)
        recv_req.Wait()
        comm.Send(send, dest=0, tag=22)

        expected = torch.arange(count, dtype=dtype)
        torch.testing.assert_close(recv.cpu(), expected)

    comm.Barrier()
    if rank == 0:
        print("mpi4py mcpu send/recv passed")


if __name__ == "__main__":
    main()
