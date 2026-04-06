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
