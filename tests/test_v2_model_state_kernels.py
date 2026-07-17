# Owner(s): ["module: PrivateUse1"]

import torch
import torch_mcpu  # noqa: F401
from torch.testing._internal.common_utils import TestCase, run_tests


class TestV2ModelStateKernels(TestCase):
    def test_scatter_num_accepted_skips_filtered_rows_and_clamps_to_one(self):
        idx_mapping = torch.tensor([2, -1, 0, 3], dtype=torch.int32, device="mcpu")
        num_sampled = torch.tensor([0, 5, -3, 6], dtype=torch.int32, device="mcpu")
        num_accepted = torch.full((4,), 77, dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_scatter_num_accepted(
            idx_mapping,
            num_sampled,
            num_accepted,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            num_accepted.cpu(),
            torch.tensor([1, 77, 1, 6], dtype=torch.int32),
        )


if __name__ == "__main__":
    run_tests()
