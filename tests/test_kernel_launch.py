# Owner(s): ["module: PrivateUse1"]

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
            self.assertEqual(result.returncode, -signal.SIGSEGV)

    def test_launched_fill_is_async_and_synchronizes(self):
        stream = torch.Stream(device="mcpu:0")
        x = torch.empty(8, dtype=torch.int64, device="mcpu")

        with stream:
            torch.ops.mcpu.stream_sleep_fill_(x, 23, 500)

        self.assertFalse(stream.query())
        stream.synchronize()
        self.assertTrue(torch.equal(x.cpu(), torch.full((8,), 23, dtype=torch.int64)))


if __name__ == "__main__":
    run_tests()
