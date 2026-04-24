# Owner(s): ["module: PrivateUse1"]

import unittest

import torch
import torch_mcpu  # noqa: F401
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.dlpack import DLDeviceType


class TestDLPack(TestCase):
    def test_open_device_dlpack(self):
        """Test DLPack conversion for mcpu device"""
        x_in = torch.randn(2, 3).to("mcpu")
        capsule = torch.utils.dlpack.to_dlpack(x_in)
        x_out = torch.from_dlpack(capsule)
        self.assertTrue(x_out.device == x_in.device)

        x_in = x_in.to("cpu")
        x_out = x_out.to("cpu")
        self.assertEqual(x_in, x_out)

    def test_dlpack_roundtrip(self):
        """Test DLPack roundtrip conversion"""
        x = torch.randn(2, 3, device="mcpu")
        capsule = torch.utils.dlpack.to_dlpack(x)
        y = torch.from_dlpack(capsule)

        self.assertEqual(x.device, y.device)
        self.assertEqual(x, y)

    def test_dlpack_tensor_protocol_roundtrip(self):
        """Test __dlpack__ / __dlpack_device__ protocol for mcpu tensors."""
        x = torch.randn(2, 3, device="mcpu")

        self.assertEqual(
            x.__dlpack_device__(),
            (DLDeviceType.kDLExtDev, x.device.index or 0),
        )

        y = torch.from_dlpack(x)

        self.assertEqual(y.device, x.device)
        self.assertEqual(y, x)

        y.add_(1)
        self.assertEqual(y, x)

    def test_dlpack_cpu_export_shares_storage(self):
        """Test exporting an mcpu tensor as a CPU DLPack capsule without copies."""
        x = torch.arange(6, dtype=torch.float32, device="mcpu").reshape(2, 3)
        cpu_view = torch.mcpu.get_cpu_view_from_mcpu_tensor(x)

        capsule = x.__dlpack__(dl_device=(DLDeviceType.kDLCPU, 0))
        y = torch.from_dlpack(capsule)

        self.assertEqual(y.device.type, "cpu")
        self.assertEqual(y.data_ptr(), cpu_view.data_ptr())
        self.assertEqual(y, cpu_view)

        y.add_(10)
        self.assertEqual(y, cpu_view)
        self.assertEqual(y, x.cpu())

    def test_dlpack_different_shapes(self):
        """Test DLPack with different tensor shapes"""
        shapes = [(1,), (2, 3), (4, 5, 6), (1, 2, 3, 4)]

        for shape in shapes:
            x = torch.randn(*shape, device="mcpu")
            capsule = torch.utils.dlpack.to_dlpack(x)
            y = torch.from_dlpack(capsule)

            self.assertEqual(x.shape, y.shape)
            self.assertEqual(x, y)

    @unittest.skip("Abs kernel only supports float type when assertEuqal")
    def test_dlpack_different_dtypes(self):
        """Test DLPack with different dtypes"""
        dtypes = [torch.float32, torch.float16, torch.int32, torch.int64]

        for dtype in dtypes:
            x = torch.randn(2, 3, device="mcpu", dtype=dtype)
            capsule = torch.utils.dlpack.to_dlpack(x)
            y = torch.from_dlpack(capsule)

            self.assertEqual(x.dtype, y.dtype)
            self.assertEqual(x, y)

    def test_dlpack_cross_device(self):
        """Test DLPack conversion across devices"""
        x_cpu = torch.randn(2, 3)
        x_mcpu = x_cpu.to("mcpu")

        capsule = torch.utils.dlpack.to_dlpack(x_mcpu)
        y = torch.from_dlpack(capsule)

        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(x_cpu, y.cpu())

    def test_dlpack_non_contiguous(self):
        """Test DLPack with non-contiguous tensors"""
        x = torch.randn(3, 4, device="mcpu")
        x_t = x.t()  # Transpose creates non-contiguous tensor

        capsule = torch.utils.dlpack.to_dlpack(x_t)
        y = torch.from_dlpack(capsule)

        self.assertEqual(x_t.shape, y.shape)
        self.assertEqual(x_t, y)


if __name__ == "__main__":
    run_tests()
