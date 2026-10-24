# Owner(s): ["module: PrivateUse1"]

"""Multi-rank xmalloc tests in the real launch mode.

Every rank is a process of one MPI job, so nothing here spawns a worker and
the tests also apply where a job manager lets only its own processes use the
device. Ranks pair up into domains of two that map one shared pool, as with
VLLM_XCPU_SHARED_KV_GROUP_SIZE=2; four ranks give two concurrent domains.
The startup sequence follows vllm_xcpu_plugin.shared_kv_cache: agree on a job
id, allocate identically, probe with a nonce, let the owner unlink and clear.

    cd /tmp
    mpirun -np 2 python -m pytest -s -p no:cacheprovider -x -q \
        /path/to/torch_mcpu/tests/test_xmalloc_mpi.py

Use -s: an aborting rank is killed before pytest would show captured output.

A job-managed region is used as provided (TORCH_MCPU_SHARED_SIZE plus BASE or
SIM_FILE); otherwise a file under /dev/shm simulates it. Without an MPI
launcher the module is skipped.

All ranks must run the same tests in the same order. Failed expectations are
collected and raised on every rank together, and any other exception aborts
the job, so no rank is left waiting in a collective its peer never reaches.
"""

import functools
import os
import secrets
import sys
import time
import traceback
import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_mcpu  # noqa: F401
from torch_mcpu.shared_memory import xmalloc

# Do not initialize MPI in a process that was not launched by an MPI runner.
LAUNCHED = any(name in os.environ for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE"))
if LAUNCHED:
    from mpi4py import MPI

    try:
        import torch_mpi_ext
    except ImportError:
        torch_mpi_ext = None

DOMAIN_SIZE = 2
SIM_REGION_BYTES = 1 << 20
POOL_BYTES = 16 * 1024
PROBE_BYTES = 256
BLOCK_BYTES = 1024
FENCE_DELAY = 0.3


def block_range(index):
    start = PROBE_BYTES + index * BLOCK_BYTES
    return slice(start, start + BLOCK_BYTES)


def collective(test):
    def wrapper(self):
        self.failures = []
        try:
            test(self)
        except BaseException:
            traceback.print_exc()
            sys.stderr.flush()
            MPI.COMM_WORLD.Abort(1)
        failures = [line for rank in self.comm.allgather(self.failures) for line in rank]
        if failures:
            self.fail("\n".join(failures))

    return functools.wraps(test)(wrapper)


@unittest.skipUnless(LAUNCHED, "run under mpirun -np 2 (or 4)")
@unittest.skipIf(LAUNCHED and torch_mpi_ext is None, "torch_mpi_ext is required")
class TestSharedXmallocRanks(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.comm = MPI.COMM_WORLD
        cls.rank, cls.size = cls.comm.Get_rank(), cls.comm.Get_size()
        if cls.size % DOMAIN_SIZE:
            raise unittest.SkipTest(f"needs a multiple of {DOMAIN_SIZE} ranks")
        cls.domain, cls.local = divmod(cls.rank, DOMAIN_SIZE)
        cls.owner = cls.local == 0
        # The fence spans every rank, as the plugin's spans the whole TP group.
        cls.fence_comm = cls.comm.Dup()
        cls.fence_handle = torch.tensor([cls.fence_comm.py2f()], dtype=torch.int64)
        cls.fence_token = torch.zeros(1, device="mcpu")
        cls.path = None
        try:
            error = cls.attach_and_probe()
        except (RuntimeError, ValueError) as caught:
            error = str(caught)  # a configuration problem every rank reports
        except BaseException:
            traceback.print_exc()
            sys.stderr.flush()
            MPI.COMM_WORLD.Abort(1)
        errors = cls.comm.allgather(error)
        if any(errors):
            raise unittest.SkipTest(f"no usable shared region: {errors}")

    @classmethod
    def attach_and_probe(cls):
        job = cls.comm.bcast(secrets.token_hex(8) if cls.rank == 0 else None, root=0)
        if "TORCH_MCPU_SHARED_SIZE" not in os.environ:
            os.environ["TORCH_MCPU_SHARED_SIZE"] = str(SIM_REGION_BYTES)
            path = f"/dev/shm/xcpu_shared_kv.test.{job}"
        else:
            path = os.getenv("TORCH_MCPU_SHARED_SIM_FILE")
        if path:
            cls.path = f"{path}.dp0.pcp0.tp{cls.domain}"
            os.environ["TORCH_MCPU_SHARED_SIM_FILE"] = cls.path
        if cls.local != 0:
            time.sleep(FENCE_DELAY)  # attach after the owner created the region
        cls.pool = xmalloc(POOL_BYTES, torch.mcpu.current_device()).view(torch.int8)
        cls.guard = xmalloc(PROBE_BYTES)
        # Probe: the owner stamps a nonce and every rank must read it back.
        nonce = os.urandom(8) if cls.owner else None
        if nonce is not None:
            cls.pool[:8].copy_(torch.frombuffer(bytearray(nonce), dtype=torch.int8))
        cls.host_fence()
        seen = bytes(cls.pool[:8].cpu().numpy())
        cls.probe_records = cls.comm.allgather((cls.domain, nonce, seen))
        # The allgather also orders every read before the owner clears the pool.
        if cls.owner:
            if cls.path:
                os.unlink(cls.path)
            cls.pool.zero_()
        cls.host_fence()
        # Recorded now: later tests, which run in name order, reuse the pool.
        cls.pool_cleared = not cls.pool.cpu().any().item()

    @classmethod
    def host_fence(cls):
        torch.mcpu.synchronize()
        cls.comm.Barrier()

    @classmethod
    def device_fence(cls):
        """Queue an all-reduce on the current stream; the host does not wait."""
        output = torch.empty_like(cls.fence_token)
        torch_mpi_ext.ops.all_reduce_out_wrapper(
            output, cls.fence_token, cls.fence_handle
        )

    def expect(self, condition, message):
        if not condition:
            self.failures.append(f"rank {self.rank}: {message}")

    def expect_bytes(self, tensor, expected, message):
        actual = bytes(tensor.cpu().numpy())
        self.expect(actual == bytes(expected), f"{message}: got {actual[:16]!r}...")

    def domain_pattern(self, salt, length=BLOCK_BYTES):
        """Bytes every rank of a domain computes identically; domains differ."""
        seed = (1 + self.domain) * 53 + salt * 7
        return bytes((seed + 3 * i) & 0x7F for i in range(length))

    @collective
    def test_startup_probe_maps_one_pool_per_domain(self):
        by_domain = {}
        for domain, nonce, _ in self.probe_records:
            if nonce is not None:
                by_domain[domain] = nonce
        self.expect(len(by_domain) == self.size // DOMAIN_SIZE, "one owner per domain")
        self.expect(len(set(by_domain.values())) == len(by_domain), "nonces not unique")
        for domain, _, seen in self.probe_records:
            self.expect(seen == by_domain[domain], f"domain {domain} read {seen!r}")
        # Identical allocation order gives identical layout on every rank.
        gaps = self.comm.allgather(self.guard.data_ptr() - self.pool.data_ptr())
        self.expect(set(gaps) == {POOL_BYTES}, f"layout differs across ranks: {gaps}")
        if self.path:
            # The owner removed the file once every rank mapped it.
            self.expect(not os.path.exists(self.path), "simulation file left behind")
        self.expect(self.pool_cleared, "pool not cleared after the probe")

    @collective
    def test_replicated_writes_without_ordering_are_read_back(self):
        # Every rank of a domain writes identical bytes, reads back what it
        # wrote, and never synchronizes with its peers inside a step.
        jitter = torch.Generator().manual_seed(self.rank)
        expected = {}
        for step in range(24):
            block = step % 4
            expected[block] = self.domain_pattern(step)
            time.sleep(float(torch.rand(1, generator=jitter)) * 0.004)
            window = self.pool[block_range(block)]
            window.copy_(torch.frombuffer(bytearray(expected[block]), dtype=torch.int8))
            self.expect_bytes(window, expected[block], f"read-back at step {step}")
            self.device_fence()
        self.host_fence()
        # All writers agree, and the other domain's pool did not leak in.
        for block, content in expected.items():
            self.expect_bytes(
                self.pool[block_range(block)], content, f"block {block} at the end"
            )

    @collective
    def test_disjoint_writers_are_visible_to_peers_after_fence(self):
        stripe = BLOCK_BYTES // DOMAIN_SIZE
        mine = bytes([(self.rank + 1) * 17 & 0x7F]) * stripe
        window = self.pool[block_range(6)]
        window[self.local * stripe : (self.local + 1) * stripe].copy_(
            torch.frombuffer(bytearray(mine), dtype=torch.int8)
        )
        self.host_fence()
        first = self.domain * DOMAIN_SIZE
        expected = b"".join(
            bytes([(first + local + 1) * 17 & 0x7F]) * stripe
            for local in range(DOMAIN_SIZE)
        )
        self.expect_bytes(window, expected, "stripes written by every rank")

    @collective
    def test_fence_makes_a_peers_rewrite_visible(self):
        # A late peer rewrites a block before the fence. The fence completes
        # only after every rank reaches it, so the early rank must read the new
        # bytes afterwards and must have waited for the late one.
        window = self.pool[block_range(4)]
        old, new = self.domain_pattern(100), self.domain_pattern(101)
        window.copy_(torch.frombuffer(bytearray(old), dtype=torch.int8))
        self.host_fence()
        started = time.monotonic()
        if self.local == 1:
            time.sleep(FENCE_DELAY)
            window.zero_()
            window.copy_(torch.frombuffer(bytearray(new), dtype=torch.int8))
        self.device_fence()
        snapshot = window.clone()
        torch.mcpu.synchronize()
        waited = time.monotonic() - started
        self.expect_bytes(snapshot, new, "read after the fence missed the rewrite")
        if self.local == 0:
            self.expect(waited >= 0.7 * FENCE_DELAY, f"fence returned early: {waited:.3f}s")
        self.host_fence()

    @collective
    def test_fence_orders_a_late_read_before_a_peers_rewrite(self):
        # A late rank reads a block that its peer will rewrite. The peer's
        # rewrite is queued behind the fence, so the late read must see the
        # old bytes even though the peer reached the fence long before.
        window = self.pool[block_range(5)]
        old, new = self.domain_pattern(200), self.domain_pattern(201)
        window.copy_(torch.frombuffer(bytearray(old), dtype=torch.int8))
        self.host_fence()
        snapshot = None
        if self.local == 0:
            time.sleep(FENCE_DELAY)
            snapshot = window.clone()
        self.device_fence()
        if self.local == 1:
            window.copy_(torch.frombuffer(bytearray(new), dtype=torch.int8))
        torch.mcpu.synchronize()
        if snapshot is not None:
            self.expect_bytes(snapshot, old, "late read saw the peer's rewrite")
        self.host_fence()
        self.expect_bytes(window, new, "rewrite lost")


if __name__ == "__main__":
    run_tests()
