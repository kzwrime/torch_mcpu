# Owner(s): ["module: PrivateUse1"]

import torch
import torch_mcpu  # noqa: F401
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestStream(TestCase):
    def _worker_policy(self, stream):
        return torch_mcpu._C._get_stream_worker_policy(
            stream.stream_id,
            stream.device_index,
            stream.device_type,
        )

    def test_00_stream_priority_selects_worker_idle_policy(self):
        low, high = torch.Stream.priority_range()
        low_stream = torch.Stream(device="mcpu:0", priority=low)
        high_stream = torch.Stream(device="mcpu:0", priority=high)

        low_policy = self._worker_policy(low_stream)
        high_policy = self._worker_policy(high_stream)

        self.assertEqual(low, 1)
        self.assertEqual(high, 0)

        self.assertEqual(low_policy["priority"], low)
        self.assertEqual(low_policy["idle_policy"], "block")
        self.assertEqual(low_policy["is_default_stream"], 0)

        self.assertEqual(high_policy["priority"], high)
        self.assertEqual(high_policy["idle_policy"], "busy")
        self.assertEqual(high_policy["is_default_stream"], 0)

    def test_00_default_stream_worker_is_busy(self):
        default_stream = torch.mcpu.default_stream()
        policy = self._worker_policy(default_stream)
        self.assertEqual(policy["is_default_stream"], 1)
        self.assertEqual(policy["idle_policy"], "busy")

    def test_00_stream_worker_policy_rejects_external_stream_id(self):
        default_stream = torch.mcpu.default_stream()
        with self.assertRaisesRegex(RuntimeError, "external streams"):
            torch_mcpu._C._get_stream_worker_policy(
                0,
                default_stream.device_index,
                default_stream.device_type,
            )

    def _stream_test_tensor(self, value=0, size=8):
        tensor = torch.full((size,), value, dtype=torch.int64, device="mcpu")
        torch.mcpu.synchronize()
        return tensor

    @skipIfTorchDynamo()
    def test_stream_create(self):
        """Test stream creation with different methods"""
        stream = torch.Stream(device="mcpu")
        self.assertEqual(stream.device_index, torch.mcpu.current_device())
        stream = torch.Stream(device="mcpu:0")
        self.assertEqual(stream.device.type, "mcpu")
        self.assertEqual(stream.device_index, 0)

        stream = torch.Stream(0)
        self.assertEqual(stream.device.type, "mcpu")
        self.assertEqual(stream.device_index, 0)

        stream1 = torch.Stream(
            stream_id=stream.stream_id,
            device_type=stream.device_type,
            device_index=stream.device_index,
        )
        self.assertEqual(stream, stream1)

    @skipIfTorchDynamo()
    def test_first_pool_stream_does_not_alias_default_slot(self):
        """The first user-created stream should not reuse default's pool slot."""
        default_stream = torch.mcpu.default_stream()
        stream = torch.Stream(device="mcpu:0")
        self.assertEqual(default_stream, torch.mcpu.current_stream())
        self.assertNotEqual(stream, default_stream)
        # Native stream ids encode the pool index above the low 4 bits. Index 0
        # is reserved as the OpenReg backing stream for MCPU's emulated default
        # stream, so user-created priority-0 streams must start at index 1.
        self.assertNotEqual(stream.stream_id >> 4, 0)

    @skipIfTorchDynamo()
    def test_stream_context(self):
        """Test stream context manager"""
        with torch.Stream(device="mcpu:0") as stream:
            self.assertEqual(torch.accelerator.current_stream(), stream)

    def test_stream_context_exception_restore(self):
        prev = torch.accelerator.current_stream()
        inner_stream = torch.Stream(device="mcpu:0")
        try:
            with inner_stream:
                # inside the context we should be on the inner stream
                self.assertEqual(torch.accelerator.current_stream(), inner_stream)
                raise RuntimeError("forced")
        except RuntimeError:
            pass
        # After the exception, the current stream should be restored.
        self.assertEqual(torch.accelerator.current_stream(), prev)

    @skipIfTorchDynamo()
    def test_stream_switch(self):
        """Test switching between streams"""
        stream1 = torch.Stream(device="mcpu:0")
        torch.accelerator.set_stream(stream1)
        current_stream = torch.accelerator.current_stream()
        self.assertEqual(current_stream, stream1)

        stream2 = torch.Stream(device="mcpu:0")
        current_stream = torch.accelerator.current_stream()
        self.assertEqual(current_stream, stream1)
        torch.accelerator.set_stream(stream2)
        current_stream = torch.accelerator.current_stream()
        self.assertEqual(current_stream, stream2)

    @skipIfTorchDynamo()
    def test_stream_synchronize(self):
        """Test stream synchronization"""
        stream = torch.Stream(device="mcpu:0")
        self.assertEqual(True, stream.query())

        event = torch.Event()
        event.record(stream)
        stream.synchronize()
        self.assertEqual(True, stream.query())

    @skipIfTorchDynamo()
    def test_stream_repr(self):
        """Test stream string representation"""
        stream = torch.Stream(device="mcpu:0")
        self.assertTrue(
            "torch.Stream device_type=mcpu, device_index=0" in repr(stream)
        )

    @skipIfTorchDynamo()
    def test_stream_wait_stream(self):
        """Test stream waiting on another stream"""
        stream_1 = torch.Stream(device="mcpu:0")
        stream_2 = torch.Stream(device="mcpu:0")
        stream_2.wait_stream(stream_1)

    @skipIfTorchDynamo()
    def test_stream_async_native_op(self):
        stream = torch.Stream(device="mcpu:0")
        tensor = self._stream_test_tensor(value=0)

        with stream:
            torch.ops.mcpu.stream_sleep_fill_(tensor, 7, 1000)

        if not stream.query():
            self.assertEqual(torch.ops.mcpu.first_element_int(tensor), 0)

        stream.synchronize()
        self.assertTrue(stream.query())
        self.assertEqual(torch.ops.mcpu.first_element_int(tensor), 7)

    @skipIfTorchDynamo()
    def test_cpu_fallback_orders_with_current_stream_launches(self):
        stream = torch.Stream(device="mcpu:0")
        tensor = self._stream_test_tensor(value=0)

        with stream:
            torch.ops.mcpu.stream_sleep_fill_(tensor, 2, 100)
            fallback_result = torch.erf(tensor)
            # sigmoid is an explicit mcpu launch. It must observe the fallback
            # output copy queued immediately before it on this same stream.
            launched_result = torch.sigmoid(fallback_result)
            self.assertEqual(torch.mcpu.current_stream(), stream)

        stream.synchronize()
        expected = torch.sigmoid(torch.erf(torch.full((8,), 2, dtype=torch.int64)))
        self.assertEqual(launched_result.cpu(), expected)

    @skipIfTorchDynamo()
    def test_cpu_fallback_batches_multiple_inputs_on_one_stream(self):
        stream = torch.Stream(device="mcpu:0")
        lhs = self._stream_test_tensor(value=0)
        rhs = self._stream_test_tensor(value=0)

        with stream:
            torch.ops.mcpu.stream_sleep_fill_(lhs, 2, 50)
            torch.ops.mcpu.stream_sleep_fill_(rhs, 4, 50)
            result = torch.atan2(lhs, rhs)
            self.assertEqual(torch.mcpu.current_stream(), stream)

        stream.synchronize()
        expected = torch.atan2(
            torch.full((8,), 2, dtype=torch.int64),
            torch.full((8,), 4, dtype=torch.int64),
        )
        self.assertEqual(result.cpu(), expected)

    @skipIfTorchDynamo()
    def test_cpu_fallback_honors_explicit_cross_stream_dependencies(self):
        producer_lhs = torch.Stream(device="mcpu:0")
        producer_rhs = torch.Stream(device="mcpu:0")
        consumer = torch.Stream(device="mcpu:0")
        lhs = self._stream_test_tensor(value=0)
        rhs = self._stream_test_tensor(value=0)

        with producer_lhs:
            torch.ops.mcpu.stream_sleep_fill_(lhs, 3, 100)
        with producer_rhs:
            torch.ops.mcpu.stream_sleep_fill_(rhs, 5, 100)

        with consumer:
            consumer.wait_stream(producer_lhs)
            consumer.wait_stream(producer_rhs)
            fallback_result = torch.atan2(lhs, rhs)
            launched_result = torch.sigmoid(fallback_result)

        consumer.synchronize()
        expected = torch.sigmoid(
            torch.atan2(
                torch.full((8,), 3, dtype=torch.int64),
                torch.full((8,), 5, dtype=torch.int64),
            )
        )
        self.assertEqual(launched_result.cpu(), expected)

    @skipIfTorchDynamo()
    def test_sync_kernel_launch_build_option(self):
        definitions = torch_mcpu.get_compile_definitions()
        if definitions.get("TORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH") != "1":
            self.skipTest("synchronous kernel launch is disabled in this build")

        stream = torch.Stream(device="mcpu:0")
        src = self._stream_test_tensor(value=0)
        dst = self._stream_test_tensor(value=-1)

        with stream:
            torch.ops.mcpu.stream_sleep_fill_(src, 19, 100)
            self.assertTrue(stream.query())
            self.assertEqual(torch.ops.mcpu.first_element_int(src), 19)

            torch.ops.mcpu.stream_sleep_copy_(dst, src, 100)
            self.assertTrue(stream.query())
            self.assertEqual(torch.ops.mcpu.first_element_int(dst), 19)

    @skipIfTorchDynamo()
    def test_device_synchronize_waits_for_stream_native_op(self):
        stream = torch.Stream(device="mcpu:0")
        tensor = self._stream_test_tensor(value=0)

        with stream:
            torch.ops.mcpu.stream_sleep_fill_(tensor, 13, 500)

        self.assertEqual(torch.ops.mcpu.first_element_int(tensor), 0)

        torch.mcpu.synchronize()

        self.assertTrue(stream.query())
        self.assertEqual(torch.ops.mcpu.first_element_int(tensor), 13)

    @skipIfTorchDynamo()
    def test_dynamo_device_interface_synchronize_waits_for_stream_native_op(self):
        from torch._dynamo.device_interface import get_interface_for_device

        stream = torch.Stream(device="mcpu:0")
        tensor = self._stream_test_tensor(value=0)

        with stream:
            torch.ops.mcpu.stream_sleep_fill_(tensor, 17, 500)

        self.assertEqual(torch.ops.mcpu.first_element_int(tensor), 0)

        get_interface_for_device("mcpu").synchronize(torch.device("mcpu:0"))

        self.assertTrue(stream.query())
        self.assertEqual(torch.ops.mcpu.first_element_int(tensor), 17)

    @skipIfTorchDynamo()
    def test_stream_wait_stream_orders_native_ops(self):
        producer = torch.Stream(device="mcpu:0")
        consumer = torch.Stream(device="mcpu:0")
        src = self._stream_test_tensor(value=0)
        dst = self._stream_test_tensor(value=-1)

        with producer:
            torch.ops.mcpu.stream_sleep_fill_(src, 11, 2000)

        with consumer:
            consumer.wait_stream(producer)
            torch.ops.mcpu.stream_sleep_copy_(dst, src, 0)

        self.assertEqual(torch.ops.mcpu.first_element_int(dst), -1)

        consumer.synchronize()

        self.assertEqual(torch.ops.mcpu.first_element_int(dst), 11)
        self.assertEqual(torch.ops.mcpu.first_element_int(src), 11)

    @skipIfTorchDynamo()
    def test_stream_wait_event(self):
        """Test stream waiting on event"""
        s1 = torch.Stream(device="mcpu")
        s2 = torch.Stream(device="mcpu")
        e = s1.record_event()
        s2.wait_event(e)

    @skipIfTorchDynamo()
    def test_stream_equality(self):
        """Test stream equality comparison"""
        stream1 = torch.Stream(device="mcpu:0")
        stream2 = torch.Stream(device="mcpu:0")

        # Different streams should not be equal
        self.assertNotEqual(stream1, stream2)

        # Same stream should be equal to itself
        self.assertEqual(stream1, stream1)

        # Stream created with same parameters should be equal
        stream3 = torch.Stream(
            stream_id=stream1.stream_id,
            device_type=stream1.device_type,
            device_index=stream1.device_index,
        )
        self.assertEqual(stream1, stream3)

    @skipIfTorchDynamo()
    def test_stream_multiple_devices(self):
        """Test streams on multiple devices"""
        # Skip this test in single-device configuration
        if torch.mcpu.device_count() < 2:
            self.skipTest("This test requires 2 devices, but only 1 is available")

        stream0 = torch.Stream(device="mcpu:0")
        stream1 = torch.Stream(device="mcpu:1")

        self.assertEqual(stream0.device_index, 0)
        self.assertEqual(stream1.device_index, 1)

        # Set current stream for each device
        torch.accelerator.set_device_index(0)
        torch.accelerator.set_stream(stream0)
        self.assertEqual(torch.accelerator.current_stream(), stream0)

        torch.accelerator.set_device_index(1)
        torch.accelerator.set_stream(stream1)
        self.assertEqual(torch.accelerator.current_stream(), stream1)

    @skipIfTorchDynamo()
    def test_stream_context_nested(self):
        """Test nested stream contexts"""
        stream1 = torch.Stream(device="mcpu:0")
        stream2 = torch.Stream(device="mcpu:0")

        with stream1:
            self.assertEqual(torch.accelerator.current_stream(), stream1)
            with stream2:
                self.assertEqual(torch.accelerator.current_stream(), stream2)
            # Should restore to stream1
            self.assertEqual(torch.accelerator.current_stream(), stream1)

    @skipIfTorchDynamo()
    def test_stream_record_event(self):
        """Test recording events on streams"""
        stream = torch.Stream(device="mcpu")
        event = stream.record_event()

        self.assertIsNotNone(event)
        self.assertEqual(event.device.type, "mcpu")
        stream.synchronize()
        self.assertTrue(event.query())


if __name__ == "__main__":
    run_tests()
