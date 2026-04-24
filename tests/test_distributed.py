# Owner(s): ["module: PrivateUse1"]

import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_mcpu
from torch.testing._internal.common_utils import run_tests, TestCase


def _collective_worker(rank: int, world_size: int, init_file: str, queue) -> None:
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )

        group = dist.new_group(list(range(world_size)))

        reduced = torch.tensor([float(rank + 1)], device="mcpu")
        work = dist.all_reduce(reduced, group=group, async_op=True)
        if work is not None:
            work.wait()

        gathered = [torch.empty(1, device="mcpu") for _ in range(world_size)]
        work = dist.all_gather(
            gathered,
            torch.tensor([float(rank + 10)], device="mcpu"),
            group=group,
            async_op=True,
        )
        if work is not None:
            work.wait()

        gathered_into = torch.empty(world_size, device="mcpu")
        work = dist.all_gather_into_tensor(
            gathered_into,
            torch.tensor([float(rank + 30)], device="mcpu"),
            group=group,
            async_op=True,
        )
        if work is not None:
            work.wait()

        if rank == 0:
            gather_list = [torch.empty(1, device="mcpu") for _ in range(world_size)]
        else:
            gather_list = None
        work = dist.gather(
            torch.tensor([float(rank + 20)], device="mcpu"),
            gather_list=gather_list,
            group=group,
            group_dst=0,
            async_op=True,
        )
        if work is not None:
            work.wait()

        queue.put(
            {
                "rank": rank,
                "all_reduce": float(reduced.cpu().item()),
                "all_reduce_device": reduced.device.type,
                "all_gather": [float(t.cpu().item()) for t in gathered],
                "all_gather_devices": [t.device.type for t in gathered],
                "all_gather_into_tensor": [float(v) for v in gathered_into.cpu().tolist()],
                "all_gather_into_tensor_device": gathered_into.device.type,
                "gather": None if gather_list is None else [float(t.cpu().item()) for t in gather_list],
                "gather_devices": None if gather_list is None else [t.device.type for t in gather_list],
            }
        )
    except Exception as exc:
        queue.put({"rank": rank, "error": repr(exc)})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestDistributed(TestCase):
    @unittest.skipIf(os.name == "nt", "file init + spawn test is Linux-oriented")
    def test_collectives_on_mcpu_use_cpu_views(self):
        world_size = 2
        fd, init_file = tempfile.mkstemp(prefix="mcpu_dist_")
        os.close(fd)
        os.unlink(init_file)
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        procs = [
            ctx.Process(
                target=_collective_worker,
                args=(rank, world_size, init_file, queue),
            )
            for rank in range(world_size)
        ]

        for proc in procs:
            proc.start()

        results = [queue.get(timeout=30) for _ in range(world_size)]

        for proc in procs:
            proc.join(timeout=30)

        for proc in procs:
            self.assertEqual(proc.exitcode, 0)

        by_rank = {result["rank"]: result for result in results}
        self.assertEqual(set(by_rank.keys()), {0, 1})

        for rank, result in by_rank.items():
            self.assertNotIn("error", result, f"rank {rank} failed: {result.get('error')}")
            self.assertEqual(result["all_reduce"], 3.0)
            self.assertEqual(result["all_reduce_device"], "mcpu")
            self.assertEqual(result["all_gather"], [10.0, 11.0])
            self.assertEqual(result["all_gather_devices"], ["mcpu", "mcpu"])
            self.assertEqual(result["all_gather_into_tensor"], [30.0, 31.0])
            self.assertEqual(result["all_gather_into_tensor_device"], "mcpu")

        self.assertEqual(by_rank[0]["gather"], [20.0, 21.0])
        self.assertEqual(by_rank[0]["gather_devices"], ["mcpu", "mcpu"])
        self.assertIsNone(by_rank[1]["gather"])
        self.assertIsNone(by_rank[1]["gather_devices"])


if __name__ == "__main__":
    run_tests()
