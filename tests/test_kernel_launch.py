# Owner(s): ["module: PrivateUse1"]

import gc
import os
import signal
import subprocess
import sys
import textwrap

import torch
import torch_mcpu  # noqa: F401
from torch.testing._internal.common_utils import TestCase, run_tests


def _run_child(code: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )


class TestKernelLaunchProtection(TestCase):
    def test_direct_cpu_view_write_is_rejected_by_page_protection(self):
        result = _run_child(
            """
            import torch
            import torch_mcpu  # noqa: F401

            x = torch.empty(8, dtype=torch.float32, device="mcpu")
            y = torch.mcpu.get_cpu_view_from_mcpu_tensor(x)
            y.fill_(1.0)
            """
        )

        self.assertNotEqual(result.returncode, 0)
        if os.name != "nt":
            if result.returncode == 1:
                self.assertIn(
                    "get_cpu_view_from_mcpu_tensor is not supported",
                    result.stderr,
                )
            else:
                self.assertEqual(result.returncode, -signal.SIGSEGV)

    def test_launched_fill_is_async_and_synchronizes(self):
        stream = torch.Stream(device="mcpu:0")
        x = torch.empty(8, dtype=torch.int64, device="mcpu")

        with stream:
            torch.ops.mcpu.stream_sleep_fill_(x, 23, 500)

        self.assertFalse(stream.query())
        stream.synchronize()
        self.assertTrue(torch.equal(x.cpu(), torch.full((8,), 23, dtype=torch.int64)))

    def _run_zero_kv_blocks_with_metadata_lifetime(
        self, *, keepalive: bool, pressure_allocations: int
    ):
        page_size = 16
        kv = torch.full((page_size,), 9, dtype=torch.int32, device="mcpu")
        seg_addrs = torch.tensor([kv.data_ptr()], dtype=torch.uint64).to("mcpu")
        blocker = torch.empty(1, dtype=torch.int64, device="mcpu")
        stream = torch.Stream(device="mcpu:0")
        junk = []

        with stream:
            torch.ops.mcpu.stream_sleep_fill_(blocker, 1, 100)
            if keepalive:
                block_ids = torch.tensor([0], dtype=torch.int64).to("mcpu")
                torch.ops.mcpu.zero_kv_blocks_kernel_impl(
                    seg_addrs, block_ids, 1, 1, page_size
                )
            else:
                torch.ops.mcpu.zero_kv_blocks_kernel_impl(
                    seg_addrs,
                    torch.tensor([0], dtype=torch.int64).to("mcpu"),
                    1,
                    1,
                    page_size,
                )

            for _ in range(pressure_allocations):
                junk.append(torch.full((1,), 7, dtype=torch.int64, device="mcpu"))

        gc.collect()
        stream.synchronize()
        self.assertTrue(torch.equal(kv.cpu(), torch.zeros_like(kv.cpu())))

    def test_kernel_all_memory_guard_keeps_temporary_metadata_accessible(self):
        cases = [
            ("keepalive", True, 0),
            ("temporary", False, 0),
            ("temporary_same_stream_pressure", False, 100),
        ]

        for name, keepalive, pressure_allocations in cases:
            with self.subTest(name=name):
                self._run_zero_kv_blocks_with_metadata_lifetime(
                    keepalive=keepalive,
                    pressure_allocations=pressure_allocations,
                )

    def test_kernel_all_memory_guard_temporary_metadata_reuse_loop(self):
        for _ in range(10):
            self._run_zero_kv_blocks_with_metadata_lifetime(
                keepalive=False,
                pressure_allocations=16,
            )

    def test_host_to_mcpu_copy_is_stream_ordered_and_blocking(self):
        stream = torch.Stream(device="mcpu:0")
        cpu = torch.arange(16, dtype=torch.float32)

        with stream:
            blocker = torch.empty(1, dtype=torch.int64, device="mcpu")
            torch.ops.mcpu.stream_sleep_fill_(blocker, 1, 100)
            device = cpu.to("mcpu")

        self.assertTrue(stream.query())
        self.assertTrue(torch.equal(device.cpu(), cpu))

    def test_mcpu_to_host_copy_is_stream_ordered_and_blocking(self):
        stream = torch.Stream(device="mcpu:0")
        device = torch.zeros(16, dtype=torch.float32, device="mcpu")

        with stream:
            blocker = torch.empty(1, dtype=torch.int64, device="mcpu")
            torch.ops.mcpu.stream_sleep_fill_(blocker, 1, 100)
            device.fill_(5)
            host = device.cpu()

        self.assertTrue(stream.query())
        self.assertTrue(torch.equal(host, torch.full_like(host, 5)))

    def test_mcpu_to_mcpu_copy_is_stream_ordered_and_async(self):
        stream = torch.Stream(device="mcpu:0")
        src = torch.zeros(16, dtype=torch.int64, device="mcpu")
        dst = torch.zeros(16, dtype=torch.int64, device="mcpu")

        with stream:
            blocker = torch.empty(1, dtype=torch.int64, device="mcpu")
            torch.ops.mcpu.stream_sleep_fill_(blocker, 1, 100)
            torch.ops.mcpu.stream_sleep_fill_(src, 7, 0)
            dst.copy_(src)

        self.assertFalse(stream.query())
        stream.synchronize()
        self.assertTrue(torch.equal(dst.cpu(), torch.full_like(dst.cpu(), 7)))


if __name__ == "__main__":
    run_tests()
