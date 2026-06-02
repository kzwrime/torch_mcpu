# Owner(s): ["module: PrivateUse1"]

import unittest

import torch
import torch_mcpu  # noqa: F401
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.dlpack import DLDeviceType


class TestDLPack(TestCase):
    def _mcpu_default_stream(self):
        current = torch.mcpu.current_stream()
        return torch.Stream(
            stream_id=(0x6 << 1) | 1,
            device_type=current.device_type,
            device_index=current.device_index,
        )

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
        self.assertEqual(torch.Tensor.__dlpack_device__.__module__, "torch")
        self.assertEqual(torch.Tensor.__dlpack__.__module__, "torch")

        self.assertEqual(
            x.__dlpack_device__(),
            (DLDeviceType.kDLExtDev, x.device.index or 0),
        )

        y = torch.from_dlpack(x)

        self.assertEqual(y.device, x.device)
        self.assertEqual(y, x)

        y.add_(1)
        self.assertEqual(y, x)

    def test_dlpack_cpu_export_copies_storage(self):
        """Test exporting an mcpu tensor to CPU DLPack follows copy semantics."""
        x = torch.arange(6, dtype=torch.float32, device="mcpu").reshape(2, 3)
        cpu_view = torch.mcpu.get_unprotected_cpu_view_from_mcpu_tensor(x)

        capsule = x.__dlpack__(dl_device=(DLDeviceType.kDLCPU, 0))
        y = torch.from_dlpack(capsule)

        self.assertEqual(y.device.type, "cpu")
        self.assertEqual(y, cpu_view)
        self.assertNotEqual(y.data_ptr(), cpu_view.data_ptr())

        y.add_(10)
        self.assertNotEqual(y, cpu_view)
        self.assertEqual(cpu_view, x.cpu())

    def test_dlpack_cpu_export_copy_false_errors(self):
        x = torch.arange(6, dtype=torch.float32, device="mcpu")
        torch.mcpu.synchronize()

        with self.assertRaisesRegex(
            ValueError,
            "cannot move .* tensor from mcpu:0 to cpu without copying",
        ):
            x.__dlpack__(dl_device=(DLDeviceType.kDLCPU, 0), copy=False)

    def test_dlpack_cpu_export_copy_true_copies_storage(self):
        x = torch.arange(6, dtype=torch.float32, device="mcpu")
        cpu_view = torch.mcpu.get_unprotected_cpu_view_from_mcpu_tensor(x)

        capsule = x.__dlpack__(dl_device=(DLDeviceType.kDLCPU, 0), copy=True)
        y = torch.from_dlpack(capsule)

        self.assertEqual(y.device.type, "cpu")
        self.assertEqual(y, cpu_view)
        self.assertNotEqual(y.data_ptr(), cpu_view.data_ptr())

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

    def test_dlpack_waits_for_current_mcpu_stream_after_mm(self):
        producer = torch.Stream(device="mcpu:0")
        consumer = torch.Stream(device="mcpu:0")
        blocker = torch.zeros(1, dtype=torch.int64, device="mcpu")
        a = torch.full((256, 256), 2.0, device="mcpu")
        b = torch.full((256, 256), 3.0, device="mcpu")
        out = torch.zeros((256, 256), device="mcpu")
        expected = torch.full((256, 256), 1536.0)
        torch.mcpu.synchronize()

        with producer:
            torch.ops.mcpu.stream_sleep_fill_(blocker, 1, 300)
            torch.mm(a, b, out=out)
            self.assertFalse(producer.query())

            capsule = out.__dlpack__(stream=consumer.stream_id)
            y = torch.from_dlpack(capsule)

            self.assertFalse(producer.query())

        consumer.synchronize()
        raw_y = torch.ops.mcpu.get_unprotected_cpu_view_from_mcpu_tensor(y)
        self.assertEqual(raw_y, expected)

    def test_dlpack_explicit_mcpu_stream_waits_on_producer_event(self):
        producer = torch.Stream(device="mcpu:0")
        consumer = torch.Stream(device="mcpu:0")
        x = torch.zeros(8, dtype=torch.int64, device="mcpu")
        torch.mcpu.synchronize()

        with producer:
            torch.ops.mcpu.stream_sleep_fill_(x, 19, 300)
            capsule = x.__dlpack__(stream=consumer.stream_id)

        y = torch.from_dlpack(capsule)
        with consumer:
            consumer.synchronize()
            raw_y = torch.ops.mcpu.get_unprotected_cpu_view_from_mcpu_tensor(y)

        self.assertEqual(raw_y[0].item(), 19)

    def test_dlpack_none_stream_waits_on_default_stream(self):
        default_stream = self._mcpu_default_stream()
        torch.mcpu.set_stream(default_stream)
        producer = torch.Stream(device="mcpu:0")
        x = torch.zeros(8, dtype=torch.int64, device="mcpu")
        torch.mcpu.synchronize()

        with producer:
            torch.ops.mcpu.stream_sleep_fill_(x, 31, 300)
            capsule = x.__dlpack__(stream=None)
            y = torch.from_dlpack(capsule)

            self.assertFalse(producer.query())

        default_stream.synchronize()
        raw_y = torch.ops.mcpu.get_unprotected_cpu_view_from_mcpu_tensor(y)
        self.assertEqual(raw_y[0].item(), 31)

    def test_dlpack_stream_minus_one_does_not_insert_dependency(self):
        producer = torch.Stream(device="mcpu:0")
        x = torch.zeros(8, dtype=torch.int64, device="mcpu")
        torch.mcpu.synchronize()

        with producer:
            torch.ops.mcpu.stream_sleep_fill_(x, 37, 500)
            capsule = x.__dlpack__(stream=-1)
            y = torch.from_dlpack(capsule)

            self.assertFalse(producer.query())
            raw_y = torch.ops.mcpu.get_unprotected_cpu_view_from_mcpu_tensor(y)
            self.assertEqual(raw_y[0].item(), 0)

        producer.synchronize()
        raw_y = torch.ops.mcpu.get_unprotected_cpu_view_from_mcpu_tensor(y)
        self.assertEqual(raw_y[0].item(), 37)

    def test_dlpack_stream_zero_is_unsupported(self):
        x = torch.zeros(8, dtype=torch.int64, device="mcpu")

        with self.assertRaisesRegex(AssertionError, "unsupported stream on MCPU: 0"):
            x.__dlpack__(stream=0)

    def test_dlpack_versioned_export_uses_stream_dependency(self):
        producer = torch.Stream(device="mcpu:0")
        consumer = torch.Stream(device="mcpu:0")
        x = torch.zeros(8, dtype=torch.int64, device="mcpu")
        torch.mcpu.synchronize()

        with producer:
            torch.ops.mcpu.stream_sleep_fill_(x, 41, 300)
            capsule = x.__dlpack__(
                stream=consumer.stream_id,
                max_version=(1, 0),
            )
            y = torch.from_dlpack(capsule)

            self.assertFalse(producer.query())

        consumer.synchronize()
        raw_y = torch.ops.mcpu.get_unprotected_cpu_view_from_mcpu_tensor(y)
        self.assertEqual(raw_y[0].item(), 41)

    def test_dlpack_cpu_export_uses_mcpu_stream_dependency(self):
        producer = torch.Stream(device="mcpu:0")
        blocker = torch.zeros(1, dtype=torch.int64, device="mcpu")
        a = torch.full((128, 128), 2.0, device="mcpu")
        b = torch.full((128, 128), 3.0, device="mcpu")
        out = torch.zeros((128, 128), device="mcpu")
        expected = torch.full((128, 128), 768.0)
        torch.mcpu.synchronize()

        with producer:
            torch.ops.mcpu.stream_sleep_fill_(blocker, 1, 300)
            torch.mm(a, b, out=out)
            self.assertFalse(producer.query())

            capsule = out.__dlpack__(dl_device=(DLDeviceType.kDLCPU, 0))
            y = torch.from_dlpack(capsule)

            self.assertTrue(producer.query())

        self.assertEqual(y, expected)
        cpu_view = torch.ops.mcpu.get_unprotected_cpu_view_from_mcpu_tensor(out)
        self.assertNotEqual(y.data_ptr(), cpu_view.data_ptr())


if __name__ == "__main__":
    run_tests()
