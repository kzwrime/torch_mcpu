# Owner(s): ["module: PrivateUse1"]

import torch
import torch_mcpu  # noqa: F401
from torch.testing._internal.common_utils import TestCase, run_tests


class TestRopeKernel(TestCase):
    def test_prepare_rope_positions_matches_prefill_decode_and_strides(self):
        positions = torch.full((3, 8), -99, dtype=torch.int64, device="mcpu")
        prefill_positions = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                [110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
                [120, 121, 122, 123, 124, 125, 126, 127, 128, 129],
                [130, 131, 132, 133, 134, 135, 136, 137, 138, 139],
            ],
            dtype=torch.int32,
            device="mcpu",
        )
        idx_mapping = torch.tensor([1, 0], dtype=torch.int32, device="mcpu")
        query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32, device="mcpu")
        prefill_lens = torch.tensor([8, 3], dtype=torch.int32, device="mcpu")
        num_computed = torch.tensor([4, 3], dtype=torch.int32, device="mcpu")
        prefill_delta = torch.tensor([4, 7], dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.prepare_rope_positions_kernel_impl(
            positions,
            positions.stride(0),
            prefill_positions,
            3 * prefill_positions.stride(0),
            prefill_positions.stride(0),
            prefill_delta,
            idx_mapping,
            query_start_loc,
            prefill_lens,
            num_computed,
            3,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            positions.cpu(),
            torch.tensor(
                [
                    [10, 11, 14, 15, 16, -99, -99, -99],
                    [10, 11, 24, 25, 26, -99, -99, -99],
                    [10, 11, 34, 35, 36, -99, -99, -99],
                ],
                dtype=torch.int64,
            ),
        )


if __name__ == "__main__":
    run_tests()
