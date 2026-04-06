# Owner(s): ["module: PrivateUse1"]

import types
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestBackendModule(TestCase):
    def test_backend_module_name(self):
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "mcpu")
        # backend can be renamed to the same name multiple times
        torch.utils.rename_privateuse1_backend("mcpu")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            torch.utils.rename_privateuse1_backend("dev")

    def test_backend_module_registration(self):
        def generate_faked_module():
            return types.ModuleType("fake_module")

        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):
            torch._register_device_module("dev", generate_faked_module())
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            torch._register_device_module("mcpu", generate_faked_module())

    def test_backend_module_function(self):
        with self.assertRaisesRegex(RuntimeError, "Try to call torch.mcpu"):
            torch.utils.backend_registration._get_custom_mod_func("func_name_")
        self.assertTrue(
            torch.utils.backend_registration._get_custom_mod_func("device_count")() == 2
        )


class TestBackendProperty(TestCase):
    def test_backend_generate_methods(self):
        with self.assertRaisesRegex(RuntimeError, "The custom device module of"):
            torch.utils.generate_methods_for_privateuse1_backend()

        self.assertTrue(hasattr(torch.Tensor, "is_mcpu"))
        self.assertTrue(hasattr(torch.Tensor, "mcpu"))
        self.assertTrue(hasattr(torch.TypedStorage, "is_mcpu"))
        self.assertTrue(hasattr(torch.TypedStorage, "mcpu"))
        self.assertTrue(hasattr(torch.UntypedStorage, "is_mcpu"))
        self.assertTrue(hasattr(torch.UntypedStorage, "mcpu"))
        self.assertTrue(hasattr(torch.nn.Module, "mcpu"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "is_mcpu"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "mcpu"))

    def test_backend_tensor_methods(self):
        x = torch.empty(4, 4)
        self.assertFalse(x.is_mcpu)

        y = x.mcpu(torch.device("mcpu"))
        self.assertTrue(y.is_mcpu)
        z = x.mcpu(torch.device("mcpu:0"))
        self.assertTrue(z.is_mcpu)
        n = x.mcpu(0)
        self.assertTrue(n.is_mcpu)

    @unittest.skip("Need to support Parameter in mcpu")
    def test_backend_module_methods(self):
        class FakeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self):
                pass

        module = FakeModule()
        self.assertEqual(module.x.device.type, "cpu")
        module.mcpu()  # type: ignore[misc]
        self.assertEqual(module.x.device.type, "mcpu")

    @unittest.skip("Need to support untyped_storage in mcpu")
    def test_backend_storage_methods(self):
        x = torch.empty(4, 4)

        x_cpu = x.storage()
        self.assertFalse(x_cpu.is_mcpu)
        x_mcpu = x_cpu.mcpu()
        self.assertTrue(x_mcpu.is_mcpu)

        y = torch.empty(4, 4)

        y_cpu = y.untyped_storage()
        self.assertFalse(y_cpu.is_mcpu)
        y_mcpu = y_cpu.mcpu()
        self.assertTrue(y_mcpu.is_mcpu)

    def test_backend_packed_sequence_methods(self):
        x = torch.rand(5, 3)
        y = torch.tensor([1, 1, 1, 1, 1])

        z_cpu = torch.nn.utils.rnn.PackedSequence(x, y)
        self.assertFalse(z_cpu.is_mcpu)

        z_mcpu = z_cpu.mcpu()
        self.assertTrue(z_mcpu.is_mcpu)


class TestTensorType(TestCase):
    def test_backend_tensor_type(self):
        dtypes_map = {
            torch.bool: "torch.mcpu.BoolTensor",
            torch.double: "torch.mcpu.DoubleTensor",
            torch.float32: "torch.mcpu.FloatTensor",
            torch.half: "torch.mcpu.HalfTensor",
            torch.int32: "torch.mcpu.IntTensor",
            torch.int64: "torch.mcpu.LongTensor",
            torch.int8: "torch.mcpu.CharTensor",
            torch.short: "torch.mcpu.ShortTensor",
            torch.uint8: "torch.mcpu.ByteTensor",
        }

        for dtype, str in dtypes_map.items():
            x = torch.empty(4, 4, dtype=dtype, device="mcpu")
            self.assertTrue(x.type() == str)

    # Note that all dtype-d Tensor objects here are only for legacy reasons
    # and should NOT be used.
    @skipIfTorchDynamo()
    def test_backend_type_methods(self):
        # Tensor
        tensor_cpu = torch.randn([8]).float()
        self.assertEqual(tensor_cpu.type(), "torch.FloatTensor")

        tensor_mcpu = tensor_cpu.mcpu()
        self.assertEqual(tensor_mcpu.type(), "torch.mcpu.FloatTensor")

        # Storage
        storage_cpu = tensor_cpu.storage()
        self.assertEqual(storage_cpu.type(), "torch.FloatStorage")

        tensor_mcpu = tensor_cpu.mcpu()
        storage_mcpu = tensor_mcpu.storage()
        self.assertEqual(storage_mcpu.type(), "torch.storage.TypedStorage")

        class CustomFloatStorage:
            @property
            def __module__(self):
                return "torch." + torch._C._get_privateuse1_backend_name()

            @property
            def __name__(self):
                return "FloatStorage"

        try:
            torch.mcpu.FloatStorage = CustomFloatStorage()
            self.assertEqual(storage_mcpu.type(), "torch.mcpu.FloatStorage")

            # test custom int storage after defining FloatStorage
            tensor_mcpu = tensor_cpu.int().mcpu()
            storage_mcpu = tensor_mcpu.storage()
            self.assertEqual(storage_mcpu.type(), "torch.storage.TypedStorage")
        finally:
            torch.mcpu.FloatStorage = None


if __name__ == "__main__":
    run_tests()
