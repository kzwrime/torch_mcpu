# Owner(s): ["module: PrivateUse1"]

import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_mcpu
from torch.testing._internal.common_utils import run_tests, TestCase


def _collective_worker(rank: int, world_size: int, init_file: str, conn) -> None:
    try:
        dist.init_process_group(
            "cpu:gloo,mcpu:mcpu",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )

        group = dist.new_group(list(range(world_size)))

        reduced = torch.tensor([float(rank + 1)], device="mcpu")
        work = dist.all_reduce(reduced, group=group, async_op=True)
        if work is not None:
            work.wait()

        reduce_to_0 = torch.tensor([float(rank + 1)], device="mcpu")
        work = dist.reduce(
            reduce_to_0,
            group=group,
            group_dst=0,
            async_op=True,
        )
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

        broadcasted = torch.tensor([float(rank + 40)], device="mcpu")
        work = dist.broadcast(
            broadcasted,
            group=group,
            group_src=0,
            async_op=True,
        )
        if work is not None:
            work.wait()

        send_recv = torch.empty(1, device="mcpu")
        if rank == 0:
            dist.send(
                torch.tensor([50.0], device="mcpu"),
                group=group,
                group_dst=1,
            )
            send_recv_source = None
        else:
            send_recv_source = dist.recv(send_recv, group=group)

        async_recv_from_0 = torch.empty(1, device="mcpu")
        async_recv_from_1 = torch.empty(1, device="mcpu")
        if rank == 0:
            send_work = dist.isend(
                torch.tensor([60.0], device="mcpu"),
                group=group,
                group_dst=1,
            )
            recv_work = dist.irecv(async_recv_from_1, group=group, group_src=1)
            if send_work is not None:
                send_work.wait()
            if recv_work is not None:
                recv_work.wait()
        else:
            recv_work = dist.irecv(async_recv_from_0, group=group, group_src=0)
            send_work = dist.isend(
                torch.tensor([70.0], device="mcpu"),
                group=group,
                group_dst=0,
            )
            if recv_work is not None:
                recv_work.wait()
            if send_work is not None:
                send_work.wait()

        conn.send(
            {
                "rank": rank,
                "all_reduce": float(reduced.cpu().item()),
                "all_reduce_device": reduced.device.type,
                "reduce": float(reduce_to_0.cpu().item()),
                "reduce_device": reduce_to_0.device.type,
                "all_gather": [float(t.cpu().item()) for t in gathered],
                "all_gather_devices": [t.device.type for t in gathered],
                "all_gather_into_tensor": [
                    float(v) for v in gathered_into.cpu().tolist()
                ],
                "all_gather_into_tensor_device": gathered_into.device.type,
                "gather": (
                    None
                    if gather_list is None
                    else [float(t.cpu().item()) for t in gather_list]
                ),
                "gather_devices": (
                    None
                    if gather_list is None
                    else [t.device.type for t in gather_list]
                ),
                "broadcast": float(broadcasted.cpu().item()),
                "broadcast_device": broadcasted.device.type,
                "send_recv": None if rank == 0 else float(send_recv.cpu().item()),
                "send_recv_device": None if rank == 0 else send_recv.device.type,
                "send_recv_source": send_recv_source,
                "async_recv_from_0": (
                    None if rank == 0 else float(async_recv_from_0.cpu().item())
                ),
                "async_recv_from_0_device": (
                    None if rank == 0 else async_recv_from_0.device.type
                ),
                "async_recv_from_1": (
                    None if rank == 1 else float(async_recv_from_1.cpu().item())
                ),
                "async_recv_from_1_device": (
                    None if rank == 1 else async_recv_from_1.device.type
                ),
            }
        )
        os._exit(0)
    except Exception as exc:
        conn.send({"rank": rank, "error": repr(exc)})
        os._exit(1)


def _async_submit_worker(
    rank: int,
    world_size: int,
    init_file: str,
    conn,
) -> None:
    try:
        dist.init_process_group(
            "cpu:gloo,mcpu:mcpu",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )

        group = dist.new_group(list(range(world_size)))
        tensor = torch.tensor([float(rank + 1)], device="mcpu")
        work = dist.all_reduce(tensor, group=group, async_op=True)
        completed_before_wait = work.is_completed()
        work.wait()

        conn.send(
            {
                "rank": rank,
                "completed_before_wait": completed_before_wait,
                "device": tensor.device.type,
                "value": float(tensor.cpu().item()),
            }
        )
        os._exit(0)
    except Exception as exc:
        conn.send({"rank": rank, "error": repr(exc)})
        os._exit(1)


def _bidirectional_p2p_worker(
    rank: int,
    world_size: int,
    init_file: str,
    conn,
) -> None:
    try:
        dist.init_process_group(
            "cpu:gloo,mcpu:mcpu",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )

        group = dist.new_group(list(range(world_size)))
        peer = 1 - rank

        send_tensor = torch.tensor([float(rank + 1)], device="mcpu")
        recv_tensor = torch.empty(1, device="mcpu")
        send_work = dist.isend(send_tensor, group=group, group_dst=peer)
        recv_work = dist.irecv(recv_tensor, group=group, group_src=peer)
        send_work.wait()
        recv_work.wait()

        batch_send = torch.tensor([float(rank + 10)], device="mcpu")
        batch_recv = torch.empty(1, device="mcpu")
        works = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, batch_send, peer, group=group),
                dist.P2POp(dist.irecv, batch_recv, peer, group=group),
            ]
        )
        for work in works:
            work.wait()

        conn.send(
            {
                "rank": rank,
                "recv": float(recv_tensor.cpu().item()),
                "batch_recv": float(batch_recv.cpu().item()),
            }
        )
        os._exit(0)
    except Exception as exc:
        conn.send({"rank": rank, "error": repr(exc)})
        os._exit(1)


class TestDistributed(TestCase):
    def test_mcpu_backend_is_registered(self):
        self.assertTrue(dist.is_backend_available("mcpu"))
        self.assertEqual(
            dist.get_default_backend_for_device(torch.device("mcpu")),
            "mcpu",
        )

    @unittest.skipIf(os.name == "nt", "file init + spawn test is Linux-oriented")
    def test_distributed_ops_on_mcpu_use_cpu_views(self):
        world_size = 2
        fd, init_file = tempfile.mkstemp(prefix="mcpu_dist_")
        os.close(fd)
        os.unlink(init_file)
        ctx = mp.get_context("fork")
        parent_conns = []
        procs = []
        for rank in range(world_size):
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            parent_conns.append(parent_conn)
            procs.append(
                ctx.Process(
                    target=_collective_worker,
                    args=(rank, world_size, init_file, child_conn),
                )
            )

        for proc in procs:
            proc.start()

        results = []
        for parent_conn in parent_conns:
            self.assertTrue(parent_conn.poll(30), "timed out waiting for result")
            results.append(parent_conn.recv())

        for proc in procs:
            proc.join(timeout=5)
            if proc.exitcode is None:
                proc.terminate()
                proc.join(timeout=5)

        by_rank = {result["rank"]: result for result in results}
        self.assertEqual(set(by_rank.keys()), {0, 1})

        for rank, result in by_rank.items():
            self.assertNotIn(
                "error",
                result,
                f"rank {rank} failed: {result.get('error')}",
            )
            self.assertEqual(result["all_reduce"], 3.0)
            self.assertEqual(result["all_reduce_device"], "mcpu")
            self.assertEqual(result["reduce_device"], "mcpu")
            self.assertEqual(result["all_gather"], [10.0, 11.0])
            self.assertEqual(result["all_gather_devices"], ["mcpu", "mcpu"])
            self.assertEqual(result["all_gather_into_tensor"], [30.0, 31.0])
            self.assertEqual(result["all_gather_into_tensor_device"], "mcpu")
            self.assertEqual(result["broadcast"], 40.0)
            self.assertEqual(result["broadcast_device"], "mcpu")

        self.assertEqual(by_rank[0]["gather"], [20.0, 21.0])
        self.assertEqual(by_rank[0]["gather_devices"], ["mcpu", "mcpu"])
        self.assertIsNone(by_rank[1]["gather"])
        self.assertIsNone(by_rank[1]["gather_devices"])
        self.assertIsNone(by_rank[0]["send_recv"])
        self.assertIsNone(by_rank[0]["send_recv_device"])
        self.assertIsNone(by_rank[0]["send_recv_source"])
        self.assertEqual(by_rank[1]["send_recv"], 50.0)
        self.assertEqual(by_rank[1]["send_recv_device"], "mcpu")
        self.assertEqual(by_rank[1]["send_recv_source"], 0)
        self.assertIsNone(by_rank[0]["async_recv_from_0"])
        self.assertIsNone(by_rank[0]["async_recv_from_0_device"])
        self.assertEqual(by_rank[1]["async_recv_from_0"], 60.0)
        self.assertEqual(by_rank[1]["async_recv_from_0_device"], "mcpu")
        self.assertEqual(by_rank[0]["async_recv_from_1"], 70.0)
        self.assertEqual(by_rank[0]["async_recv_from_1_device"], "mcpu")
        self.assertIsNone(by_rank[1]["async_recv_from_1"])
        self.assertIsNone(by_rank[1]["async_recv_from_1_device"])
        self.assertEqual(by_rank[0]["reduce"], 3.0)

    @unittest.skipIf(os.name == "nt", "file init + spawn test is Linux-oriented")
    def test_async_allreduce_returns_work_and_completes(self):
        world_size = 2
        fd, init_file = tempfile.mkstemp(prefix="mcpu_dist_stream_")
        os.close(fd)
        os.unlink(init_file)
        ctx = mp.get_context("fork")
        parent_conns = []
        procs = []
        for rank in range(world_size):
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            parent_conns.append(parent_conn)
            procs.append(
                ctx.Process(
                    target=_async_submit_worker,
                    args=(rank, world_size, init_file, child_conn),
                )
            )

        for proc in procs:
            proc.start()

        results = []
        for parent_conn in parent_conns:
            self.assertTrue(parent_conn.poll(180), "timed out waiting for result")
            results.append(parent_conn.recv())

        for proc in procs:
            proc.join(timeout=5)
            if proc.exitcode is None:
                proc.terminate()
                proc.join(timeout=5)

        by_rank = {result["rank"]: result for result in results}
        self.assertEqual(set(by_rank.keys()), {0, 1})

        for rank, result in by_rank.items():
            self.assertNotIn(
                "error",
                result,
                f"rank {rank} failed: {result.get('error')}",
            )
            self.assertEqual(result["device"], "mcpu")
            self.assertEqual(result["value"], 3.0)
            self.assertIsInstance(result["completed_before_wait"], bool)

    @unittest.skipIf(os.name == "nt", "file init + spawn test is Linux-oriented")
    def test_bidirectional_p2p_does_not_deadlock(self):
        world_size = 2
        fd, init_file = tempfile.mkstemp(prefix="mcpu_dist_p2p_")
        os.close(fd)
        os.unlink(init_file)
        ctx = mp.get_context("fork")
        parent_conns = []
        procs = []
        for rank in range(world_size):
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            parent_conns.append(parent_conn)
            procs.append(
                ctx.Process(
                    target=_bidirectional_p2p_worker,
                    args=(rank, world_size, init_file, child_conn),
                )
            )

        for proc in procs:
            proc.start()

        results = []
        for parent_conn in parent_conns:
            self.assertTrue(parent_conn.poll(30), "timed out waiting for result")
            results.append(parent_conn.recv())

        for proc in procs:
            proc.join(timeout=5)
            if proc.exitcode is None:
                proc.terminate()
                proc.join(timeout=5)

        by_rank = {result["rank"]: result for result in results}
        self.assertEqual(set(by_rank.keys()), {0, 1})
        for rank, result in by_rank.items():
            self.assertNotIn(
                "error",
                result,
                f"rank {rank} failed: {result.get('error')}",
            )

        self.assertEqual(by_rank[0]["recv"], 2.0)
        self.assertEqual(by_rank[1]["recv"], 1.0)
        self.assertEqual(by_rank[0]["batch_recv"], 11.0)
        self.assertEqual(by_rank[1]["batch_recv"], 10.0)


if __name__ == "__main__":
    run_tests()
