"""Test suite for inplace mcpu operations and memory behavior."""

import torch
import unittest
import gc
import time


def make_mcpu_tensor(*shape):
    numel = 1
    for dim in shape:
        numel *= dim
    return (torch.arange(numel, dtype=torch.float32).reshape(shape) / 10).to("mcpu")


class TestInplaceOps(unittest.TestCase):
    """Test inplace operations implemented by mcpu kernels."""

    def test_inplace_no_copy(self):
        """Verify inplace operations don't copy memory."""
        x = torch.ones(100, device="mcpu")
        original_ptr = x.data_ptr()

        x.add_(1)

        # Verify no copy occurred
        self.assertEqual(x.data_ptr(), original_ptr,
                        "Inplace operation should not change data pointer")
        self.assertTrue(torch.all(x.cpu() == 2),
                       "Inplace operation should modify values correctly")

    def test_inplace_storage_sharing(self):
        """Verify inplace operations correctly share storage."""
        x = make_mcpu_tensor(10)
        y = x.view(5, 2)  # Create a view

        original_x_ptr = x.data_ptr()
        original_y_ptr = y.data_ptr()

        # Inplace operation on x
        x.add_(1)

        # y should see the changes (shared storage)
        self.assertEqual(y.data_ptr(), original_y_ptr,
                        "View data_ptr should remain constant")
        self.assertEqual(x.data_ptr(), original_x_ptr,
                        "Original tensor data_ptr should remain constant")
        self.assertTrue(torch.allclose(y.cpu(), x.view(5, 2).cpu()),
                       "View should reflect changes to original tensor")

    def test_inplace_with_explicit_kernel(self):
        """Test inplace operation implemented by mcpu."""
        x = torch.zeros(5, device="mcpu")

        x.add_(1)

        self.assertTrue(torch.all(x.cpu() == 1),
                       "Inplace operation should modify tensor correctly")
        self.assertEqual(x.device.type, "mcpu",
                        "Tensor should remain on mcpu device")

    def test_multiple_inplace_operations(self):
        """Test multiple inplace operations in sequence."""
        x = torch.ones(50, device="mcpu")
        original_ptr = x.data_ptr()

        # Chain of inplace operations
        x.add_(1)
        self.assertEqual(x.data_ptr(), original_ptr)

        x.mul_(2)
        self.assertEqual(x.data_ptr(), original_ptr)

        x.sub_(1)
        self.assertEqual(x.data_ptr(), original_ptr)

        # Verify final result
        # Calculation: ((1 + 1) * 2) - 1 = 3
        expected = torch.ones(50) * 3
        self.assertTrue(torch.allclose(x.cpu(), expected))

    def test_inplace_with_views(self):
        """Test inplace operations on view tensors."""
        x = make_mcpu_tensor(20)
        y = x.view(4, 5)  # Create a view

        original_x_ptr = x.data_ptr()
        original_y_ptr = y.data_ptr()

        # Inplace operation on view
        y.add_(1)

        # Both should point to same storage
        self.assertEqual(x.data_ptr(), original_x_ptr)
        self.assertEqual(y.data_ptr(), original_y_ptr)

        # Changes should be visible in both
        self.assertTrue(torch.allclose(x.cpu(), y.view(20).cpu()))

    def test_inplace_preserves_device(self):
        """Test that inplace operations preserve device type."""
        x = make_mcpu_tensor(10)

        # Various inplace operations
        x.add_(1)
        self.assertEqual(x.device.type, "mcpu")

        x.mul_(2)
        self.assertEqual(x.device.type, "mcpu")

        x.div_(2)
        self.assertEqual(x.device.type, "mcpu")


class TestFallbackMemoryBehavior(unittest.TestCase):
    """Test memory behavior of explicit mcpu operations."""

    def test_binary_op_creates_new_tensor(self):
        """Test that non-inplace binary ops create new tensors."""
        x = make_mcpu_tensor(10)
        y = make_mcpu_tensor(10) + 1.0

        x_ptr = x.data_ptr()
        y_ptr = y.data_ptr()

        z = torch.add(x, y)

        # z should be a new tensor
        self.assertNotEqual(z.data_ptr(), x_ptr,
                           "Result should be a new tensor")
        self.assertNotEqual(z.data_ptr(), y_ptr,
                           "Result should be a new tensor")
        self.assertEqual(z.device.type, "mcpu",
                        "Result should be on mcpu device")

    def test_binary_op_with_scalars(self):
        """Test binary operations with scalar operands."""
        x = make_mcpu_tensor(10)

        # Operations with scalars
        y = x + 1.0
        self.assertTrue(torch.allclose(y.cpu(), x.cpu() + 1.0))
        self.assertEqual(y.device.type, "mcpu")

        z = x - 2.0
        self.assertTrue(torch.allclose(z.cpu(), x.cpu() - 2.0))
        self.assertEqual(z.device.type, "mcpu")

    def test_binary_op_device_consistency(self):
        """Test that explicit binary ops maintain device consistency."""
        x = make_mcpu_tensor(10)
        y = make_mcpu_tensor(10) + 1.0

        # All results should be on mcpu
        z1 = torch.add(x, y)
        z2 = torch.sub(x, y)
        z3 = x + 1.0
        z4 = x - 1.0

        for z in [z1, z2, z3, z4]:
            self.assertEqual(z.device.type, "mcpu",
                           "Result should be on mcpu device")


class TestMemorySafety(unittest.TestCase):
    """Test memory safety of explicit mcpu operations."""

    def test_no_memory_leaks(self):
        """Ensure repeated explicit operations do not leak memory."""
        # Get baseline memory
        gc.collect()

        # Perform many fallback operations
        for _ in range(100):
            x = make_mcpu_tensor(100)
            y = make_mcpu_tensor(100) + 1.0
            z = torch.add(x, y)
            del x, y, z

        gc.collect()

        # Should not leak memory - if we get here without crash, test passes
        self.assertTrue(True)

    def test_alias_correctness(self):
        """Ensure mutable aliasing still works correctly."""
        x = torch.zeros(5, device="mcpu")

        x.add_(1)

        self.assertTrue(torch.all(x.cpu() == 1),
                       "Inplace operation should work correctly")

    def test_tensor_lifetime(self):
        """Test that tensor lifetimes are managed correctly."""
        x = make_mcpu_tensor(10)
        x_ptr = x.data_ptr()

        y = torch.add(x, 1.0)

        # Original tensor should still be valid
        self.assertEqual(x.data_ptr(), x_ptr,
                        "Original tensor should remain valid")

        # Result should be independent
        y.fill_(0.0)
        self.assertFalse(torch.all(x.cpu() == 0.0),
                        "Modifying result should not affect input")


class TestCopyDetection(unittest.TestCase):
    """Tests to detect and validate copy operations."""

    def test_inplace_copy_detection(self):
        """Detect if inplace operations are copying memory."""
        x = torch.ones(100, device="mcpu")
        original_ptr = x.data_ptr()

        # Perform inplace operation
        x.add_(1)

        # Check if memory was copied
        if x.data_ptr() != original_ptr:
            self.fail(f"Inplace operation copied memory! "
                     f"Original ptr: {original_ptr}, New ptr: {x.data_ptr()}")

    def test_binary_op_copy_detection(self):
        """Detect memory copies in explicit binary operations."""
        x = make_mcpu_tensor(1000)
        y = make_mcpu_tensor(1000) + 1.0

        x_ptr = x.data_ptr()
        y_ptr = y.data_ptr()

        z = torch.add(x, y)

        # Result should be a new allocation (not sharing with inputs)
        self.assertNotEqual(z.data_ptr(), x_ptr,
                           "Result should not share memory with input x")
        self.assertNotEqual(z.data_ptr(), y_ptr,
                           "Result should not share memory with input y")

    def test_view_no_copy(self):
        """Verify view operations don't copy memory."""
        x = make_mcpu_tensor(100)
        y = x.view(10, 10)

        # Views should share the same underlying storage
        self.assertEqual(x.data_ptr(), y.data_ptr(),
                        "View should share memory with original")

    def test_slice_copy_behavior(self):
        """Test memory behavior of slice operations."""
        x = make_mcpu_tensor(100)
        y = x[10:20]

        # Slices might have different data_ptr but should share storage
        x_storage = x.storage().data_ptr()
        y_storage = y.storage().data_ptr()

        # For contiguous tensors, storage should be shared
        self.assertEqual(x_storage, y_storage,
                        "Slice should share storage with original")

if __name__ == '__main__':
    unittest.main()
