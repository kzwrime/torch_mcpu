# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestEvent(TestCase):
    @skipIfTorchDynamo()
    def test_event_create(self):
        event = torch.Event(device="mcpu")
        self.assertEqual(event.device.type, "mcpu")
        self.assertEqual(event.device.index, None)
        self.assertEqual(event.event_id, 0)

        event = torch.Event(device="mcpu:0")
        self.assertEqual(event.device.type, "mcpu")
        self.assertEqual(event.device.index, None)
        self.assertEqual(event.event_id, 0)

        event = torch.Event()
        self.assertEqual(event.device.type, "mcpu")
        self.assertEqual(event.device.index, None)
        self.assertEqual(event.event_id, 0)

        stream = torch.Stream(device="mcpu:0")
        event = stream.record_event()
        self.assertEqual(event.device.type, "mcpu")
        self.assertEqual(event.device.index, 0)
        self.assertNotEqual(event.event_id, 0)

    @skipIfTorchDynamo()
    def test_event_query(self):
        event = torch.Event()
        self.assertTrue(event.query())

        stream = torch.Stream(device="mcpu:0")
        event = stream.record_event()
        event.synchronize()
        self.assertTrue(event.query())

    @skipIfTorchDynamo()
    def test_event_record(self):
        stream = torch.Stream(device="mcpu:0")
        event1 = stream.record_event()
        self.assertNotEqual(0, event1.event_id)

        event2 = stream.record_event()
        self.assertNotEqual(0, event2.event_id)

        self.assertNotEqual(event1.event_id, event2.event_id)

    @skipIfTorchDynamo()
    def test_event_re_record_waits_for_latest_version(self):
        stream1 = torch.Stream(device="mcpu:0")
        stream2 = torch.Stream(device="mcpu:0")
        marker1 = torch.zeros(1, dtype=torch.int64, device="mcpu")
        marker2 = torch.zeros(1, dtype=torch.int64, device="mcpu")
        event = torch.Event(device="mcpu:0")
        torch.mcpu.synchronize()

        with stream1:
            torch.ops.mcpu.stream_sleep_fill_(marker1, 1, 100)
            event.record(stream1)

        with stream2:
            torch.ops.mcpu.stream_sleep_fill_(marker2, 1, 2000)
            event.record(stream2)

        stream1.synchronize()
        self.assertTrue(stream1.query())
        self.assertFalse(event.query())

        event.synchronize()
        self.assertTrue(event.query())
        self.assertEqual(torch.ops.mcpu.first_element_int(marker2), 1)

    @skipIfTorchDynamo()
    def test_stream_wait_event_waits_for_record_version_at_call(self):
        producer = torch.Stream(device="mcpu:0")
        consumer = torch.Stream(device="mcpu:0")
        src = torch.zeros(8, dtype=torch.int64, device="mcpu")
        dst = torch.full((8,), -1, dtype=torch.int64, device="mcpu")
        event = torch.Event(device="mcpu:0")
        torch.mcpu.synchronize()

        with producer:
            torch.ops.mcpu.stream_sleep_fill_(src, 23, 300)
            event.record(producer)

        with consumer:
            consumer.wait_event(event)
            torch.ops.mcpu.stream_sleep_copy_(dst, src, 0)

        self.assertFalse(consumer.query())
        consumer.synchronize()
        self.assertEqual(torch.ops.mcpu.first_element_int(dst), 23)

    @skipIfTorchDynamo()
    def test_event_elapsed_time(self):
        stream = torch.Stream(device="mcpu:0")

        event1 = torch.Event(device="mcpu:0", enable_timing=True)
        event1.record(stream)
        event2 = torch.Event(device="mcpu:0", enable_timing=True)
        event2.record(stream)

        stream.synchronize()
        self.assertTrue(event1.query())
        self.assertTrue(event2.query())

        ms = event1.elapsed_time(event2)
        self.assertTrue(ms > 0)

    @skipIfTorchDynamo()
    def test_event_wait_stream(self):
        stream1 = torch.Stream(device="mcpu")
        stream2 = torch.Stream(device="mcpu")

        event = stream1.record_event()
        stream2.wait_event(event)


if __name__ == "__main__":
    run_tests()
