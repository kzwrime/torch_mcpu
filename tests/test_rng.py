# Owner(s): ["module: PrivateUse1"]

import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestRNG(TestCase):
    def test_generator(self):
        """Test generator creation on mcpu device"""
        generator = torch.Generator(device="mcpu:1")
        self.assertEqual(generator.device.type, "mcpu")
        self.assertEqual(generator.device.index, 1)

    def test_rng_state(self):
        """Test RNG state get and set"""
        state = torch.mcpu.get_rng_state(0)
        torch.mcpu.set_rng_state(state, 0)

    def test_manual_seed(self):
        """Test manual seed setting"""
        torch.mcpu.manual_seed_all(2024)
        self.assertEqual(torch.mcpu.initial_seed(), 2024)

    def test_generator_seed(self):
        """Test generator seed setting"""
        generator = torch.Generator(device="mcpu:0")
        generator.manual_seed(42)
        self.assertEqual(generator.initial_seed(), 42)

        generator = torch.Generator(device="mcpu:1")
        generator.manual_seed(100)
        self.assertEqual(generator.initial_seed(), 100)

    @unittest.skip("mcpu backend does not implement per-device RNG yet")
    def test_generator_state(self):
        """Test generator state get/set"""
        generator = torch.Generator(device="mcpu:0")
        state = generator.get_state()

        # Generate some random numbers
        x1 = torch.randn(10, device="mcpu:0", generator=generator)

        # Set state back
        generator.set_state(state)
        x2 = torch.randn(10, device="mcpu:0", generator=generator)

        # Should produce same sequence
        self.assertEqual(x1, x2)

    @unittest.skip("mcpu backend does not implement per-device RNG yet")
    def test_rng_state_consistency(self):
        """Test RNG state consistency across devices"""
        state0 = torch.mcpu.get_rng_state(0)
        state1 = torch.mcpu.get_rng_state(1)

        # States should be different for different devices
        self.assertNotEqual(state0, state1)

        # Setting state should work
        torch.mcpu.set_rng_state(state0, 0)
        restored_state = torch.mcpu.get_rng_state(0)
        self.assertEqual(state0, restored_state)

    def test_manual_seed_all(self):
        """Test manual_seed_all sets seed for all devices"""
        torch.mcpu.manual_seed_all(1234)

        # Check that seed is set
        seed = torch.mcpu.initial_seed()
        self.assertEqual(seed, 1234)

        # Test with different seed
        torch.mcpu.manual_seed_all(5678)
        seed = torch.mcpu.initial_seed()
        self.assertEqual(seed, 5678)

    def test_uniform_reserves_generator_state_at_submission(self):
        generator = torch.Generator(device="mcpu:0").manual_seed(123)
        state_before = generator.get_state()
        blocker = torch.empty(1, device="mcpu:0", dtype=torch.int64)

        torch.ops.mcpu.stream_sleep_fill_(blocker, 1, 100)
        torch.empty(16, device="mcpu:0").uniform_(generator=generator)
        state_after_submission = generator.get_state()

        self.assertNotEqual(state_before, state_after_submission)
        torch.mcpu.synchronize()

    def test_uniform_reseed_does_not_change_submitted_work(self):
        expected_generator = torch.Generator(device="mcpu:0").manual_seed(123)
        expected = torch.empty(16, device="mcpu:0")
        expected.uniform_(generator=expected_generator)
        torch.mcpu.synchronize()

        generator = torch.Generator(device="mcpu:0").manual_seed(123)
        blocker = torch.empty(1, device="mcpu:0", dtype=torch.int64)
        actual = torch.empty_like(expected)

        torch.ops.mcpu.stream_sleep_fill_(blocker, 1, 100)
        actual.uniform_(generator=generator)
        generator.manual_seed(456)
        torch.mcpu.synchronize()

        self.assertEqual(actual.cpu(), expected.cpu())

    def test_uniform_generator_state_restore(self):
        generator = torch.Generator(device="mcpu:0").manual_seed(123)
        state = generator.get_state()
        first = torch.empty(257, device="mcpu:0")
        repeat = torch.empty_like(first)

        first.uniform_(-3.0, 5.0, generator=generator)
        torch.mcpu.synchronize()
        generator.set_state(state)
        repeat.uniform_(-3.0, 5.0, generator=generator)
        torch.mcpu.synchronize()

        self.assertEqual(first.cpu(), repeat.cpu())

    def test_uniform_is_independent_of_parallel_schedule(self):
        original_threads = torch.get_num_threads()
        try:
            generator = torch.Generator(device="mcpu:0").manual_seed(123)
            torch.set_num_threads(1)
            serial = torch.empty(1_000_003, device="mcpu:0")
            serial.uniform_(generator=generator)
            torch.mcpu.synchronize()

            generator.manual_seed(123)
            torch.set_num_threads(16)
            parallel = torch.empty_like(serial)
            parallel.uniform_(generator=generator)
            torch.mcpu.synchronize()
        finally:
            torch.set_num_threads(original_threads)

        self.assertEqual(serial.cpu(), parallel.cpu())

    def test_uniform_empty_does_not_advance_generator(self):
        generator = torch.Generator(device="mcpu:0").manual_seed(123)
        state = generator.get_state()

        torch.empty(0, device="mcpu:0").uniform_(generator=generator)

        self.assertEqual(generator.get_state(), state)

    def test_uniform_rejects_cpu_generator(self):
        generator = torch.Generator(device="cpu").manual_seed(123)
        with self.assertRaisesRegex(RuntimeError, "Expected an mcpu generator"):
            torch.empty(4, device="mcpu:0").uniform_(generator=generator)

    def test_uniform_invalid_bounds_fail_before_submission(self):
        generator = torch.Generator(device="mcpu:0").manual_seed(123)
        state = generator.get_state()

        with self.assertRaisesRegex(RuntimeError, "from=2.*> to=1"):
            torch.empty(4, device="mcpu:0").uniform_(
                2.0, 1.0, generator=generator
            )

        self.assertEqual(generator.get_state(), state)

    def test_uniform_noncontiguous_view_updates_base(self):
        base = torch.full((17, 18), -9.0, device="mcpu:0")
        view = base[:, ::2]
        generator = torch.Generator(device="mcpu:0").manual_seed(123)

        view.uniform_(-2.0, 3.0, generator=generator)
        torch.mcpu.synchronize()
        result = base.cpu()

        self.assertTrue((result[:, ::2] >= -2.0).all())
        self.assertTrue((result[:, ::2] < 3.0).all())
        self.assertEqual(result[:, 1::2], torch.full((17, 9), -9.0))

    def test_uniform_supported_dtypes_and_bounds(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            generator = torch.Generator(device="mcpu:0").manual_seed(123)
            result = torch.empty(4099, device="mcpu:0", dtype=dtype)

            result.uniform_(-0.75, 1.25, generator=generator)
            torch.mcpu.synchronize()
            cpu_result = result.cpu()

            self.assertTrue(torch.isfinite(cpu_result).all())
            self.assertTrue((cpu_result >= -0.75).all())
            self.assertTrue((cpu_result < 1.25).all())

    def test_uniform_low_precision_uses_rounded_bounds(self):
        for dtype in (torch.float16, torch.bfloat16):
            result = torch.empty(1_000_003, device="mcpu:0", dtype=dtype)
            result.uniform_(0.1, 0.2)
            torch.mcpu.synchronize()
            cpu_result = result.cpu()
            rounded_from = torch.tensor(0.1, dtype=dtype)
            rounded_to = torch.tensor(0.2, dtype=dtype)

            self.assertTrue((cpu_result >= rounded_from).all())
            self.assertTrue((cpu_result < rounded_to).all())

    @unittest.skip("mcpu backend does not implement per-device RNG yet")
    def test_generator_different_devices(self):
        """Test generators on different devices"""
        gen0 = torch.Generator(device="mcpu:0")
        gen1 = torch.Generator(device="mcpu:1")

        gen0.manual_seed(1)
        gen1.manual_seed(1)

        x0 = torch.randn(10, device="mcpu:0", generator=gen0)
        x1 = torch.randn(10, device="mcpu:1", generator=gen1)

        # Should produce same sequence with same seed
        self.assertEqual(x0.cpu(), x1.cpu())


if __name__ == "__main__":
    run_tests()
