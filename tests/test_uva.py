import torch
import torch_mcpu  # noqa: F401
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase

class TestGetMcpuViewFromCpuTensor(TestCase):
    """Tests for torch.mcpu.get_mcpu_view_from_cpu_tensor.

    The mcpu backend is a CPU-emulated device, so its "device" pointers are
    ordinary CPU addresses.  For pinned input the returned mcpu tensor shares
    the same physical memory as the CPU tensor (UVA-like bidirectional
    visibility).  For non-pinned input a contiguous pinned copy is created
    first, so only the initial values are transferred.
    """

    # ------------------------------------------------------------------
    # Basic correctness
    # ------------------------------------------------------------------

    def test_returns_mcpu_tensor(self):
        cpu = torch.zeros(4, 4, dtype=torch.float32)
        mcpu = torch.mcpu.get_mcpu_view_from_cpu_tensor(cpu)
        self.assertEqual(mcpu.device.type, "mcpu")

    def test_empty_tensor(self):
        cpu = torch.zeros(0, dtype=torch.float32)
        mcpu = torch.mcpu.get_mcpu_view_from_cpu_tensor(cpu)
        self.assertEqual(mcpu.device.type, "mcpu")
        self.assertEqual(mcpu.numel(), 0)

    def test_preserves_dtype(self):
        for dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
            cpu = torch.zeros(10, dtype=dtype)
            mcpu = torch.mcpu.get_mcpu_view_from_cpu_tensor(cpu)
            self.assertEqual(mcpu.dtype, dtype)

    def test_preserves_shape(self):
        cpu = torch.zeros(3, 4, 5)
        mcpu = torch.mcpu.get_mcpu_view_from_cpu_tensor(cpu)
        self.assertEqual(mcpu.shape, cpu.shape)

    # ------------------------------------------------------------------
    # Pinned tensor: bidirectional memory sharing
    # ------------------------------------------------------------------

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pinned_cpu_write_visible_on_mcpu(self):
        """Writing to the pinned CPU tensor must be visible via the mcpu view."""
        cpu = torch.zeros(10, 10, dtype=torch.int32, pin_memory=True)
        mcpu = torch.mcpu.get_mcpu_view_from_cpu_tensor(cpu)

        self.assertEqual(int(mcpu[0, 0]), 0)
        self.assertEqual(int(mcpu[2, 3]), 0)

        cpu[0, 0] = 1
        cpu[2, 3] = 2

        self.assertEqual(int(mcpu[0, 0]), 1)
        self.assertEqual(int(mcpu[2, 3]), 2)

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pinned_mcpu_write_visible_on_cpu(self):
        """Writing to the mcpu view must be visible in the pinned CPU tensor."""
        cpu = torch.zeros(10, 10, dtype=torch.int32, pin_memory=True)
        mcpu = torch.mcpu.get_mcpu_view_from_cpu_tensor(cpu)

        mcpu[0, 0] = 3
        mcpu[2, 3] = 4

        self.assertEqual(int(cpu[0, 0]), 3)
        self.assertEqual(int(cpu[2, 3]), 4)

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pinned_mcpu_inplace_visible_on_cpu(self):
        """In-place mcpu operations must be reflected in the CPU tensor."""
        cpu = torch.zeros(10, 10, dtype=torch.int32, pin_memory=True)
        mcpu = torch.mcpu.get_mcpu_view_from_cpu_tensor(cpu)

        cpu[0, 0] = 1
        cpu[2, 3] = 2
        cpu[4, 5] = -1

        mcpu.mul_(2)

        self.assertEqual(int(cpu[0, 0]), 2)
        self.assertEqual(int(cpu[2, 3]), 4)
        self.assertEqual(int(cpu[4, 5]), -2)

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pinned_same_data_ptr(self):
        """The mcpu view must share the same data pointer as the pinned tensor."""
        cpu = torch.zeros(100, pin_memory=True)
        mcpu = torch.mcpu.get_mcpu_view_from_cpu_tensor(cpu)
        self.assertEqual(cpu.data_ptr(), mcpu.data_ptr())

    # ------------------------------------------------------------------
    # Non-pinned tensor: snapshot copy
    # ------------------------------------------------------------------

    def test_non_pinned_initial_values_correct(self):
        """Non-pinned path: initial values must match the source tensor."""
        cpu = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        mcpu = torch.mcpu.get_mcpu_view_from_cpu_tensor(cpu)
        self.assertTrue(torch.equal(cpu, mcpu.cpu()))

    def test_non_pinned_returns_mcpu_tensor(self):
        cpu = torch.randn(8, 8)
        mcpu = torch.mcpu.get_mcpu_view_from_cpu_tensor(cpu)
        self.assertEqual(mcpu.device.type, "mcpu")

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_rejects_non_cpu_tensor(self):
        mcpu_tensor = torch.zeros(4, device="mcpu")
        with self.assertRaises(Exception):
            torch.mcpu.get_mcpu_view_from_cpu_tensor(mcpu_tensor)

    def test_rejects_non_tensor(self):
        with self.assertRaises(Exception):
            torch.mcpu.get_mcpu_view_from_cpu_tensor(42)  # type: ignore[arg-type]


class TestGetCpuViewFromMcpuTensor(TestCase):
    def test_returns_cpu_tensor(self):
        mcpu = torch.arange(8, dtype=torch.float32, device="mcpu")
        cpu = torch.mcpu.get_cpu_view_from_mcpu_tensor(mcpu)
        self.assertEqual(cpu.device.type, "cpu")

    def test_empty_tensor(self):
        mcpu = torch.empty(0, device="mcpu")
        cpu = torch.mcpu.get_cpu_view_from_mcpu_tensor(mcpu)
        self.assertEqual(cpu.device.type, "cpu")
        self.assertEqual(cpu.numel(), 0)

    def test_preserves_metadata(self):
        mcpu = torch.arange(24, dtype=torch.int64, device="mcpu").reshape(2, 3, 4)
        cpu = torch.mcpu.get_cpu_view_from_mcpu_tensor(mcpu)
        self.assertEqual(cpu.shape, mcpu.shape)
        self.assertEqual(cpu.dtype, mcpu.dtype)

    def test_mcpu_write_visible_on_cpu(self):
        mcpu = torch.zeros(4, 4, dtype=torch.int32, device="mcpu")
        cpu = torch.mcpu.get_cpu_view_from_mcpu_tensor(mcpu)

        mcpu[1, 2] = 7
        self.assertEqual(int(cpu[1, 2]), 7)

    def test_cpu_write_visible_on_mcpu(self):
        mcpu = torch.zeros(4, 4, dtype=torch.int32, device="mcpu")
        cpu = torch.mcpu.get_cpu_view_from_mcpu_tensor(mcpu)

        cpu[2, 1] = 9
        self.assertEqual(int(mcpu[2, 1]), 9)

    def test_same_data_ptr(self):
        mcpu = torch.arange(16, dtype=torch.float32, device="mcpu")
        cpu = torch.mcpu.get_cpu_view_from_mcpu_tensor(mcpu)
        self.assertEqual(cpu.data_ptr(), mcpu.data_ptr())

    def test_rejects_non_mcpu_tensor(self):
        cpu = torch.zeros(4)
        with self.assertRaises(Exception):
            torch.mcpu.get_cpu_view_from_mcpu_tensor(cpu)

    def test_rejects_non_tensor(self):
        with self.assertRaises(Exception):
            torch.mcpu.get_cpu_view_from_mcpu_tensor(42)  # type: ignore[arg-type]


if __name__ == "__main__":
    run_tests()
