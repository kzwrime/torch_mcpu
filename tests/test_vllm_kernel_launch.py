# Owner(s): ["module: PrivateUse1"]

import math

import torch
import torch_mcpu  # noqa: F401
from torch.testing._internal.common_utils import TestCase, run_tests


class TestVllmKernelLaunch(TestCase):
    def test_temperature_kernel(self):
        logits = torch.tensor(
            [[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]], device="mcpu"
        )
        idx = torch.tensor([0, 1], dtype=torch.int32, device="mcpu")
        temperature = torch.tensor([2.0, 1.0], dtype=torch.float32, device="mcpu")

        torch.ops.mcpu.vllm_temperature_kernel(logits, idx, temperature, 3)
        torch.mcpu.synchronize()

        self.assertEqual(
            logits.cpu(),
            torch.tensor([[1.0, 2.0, 3.0], [1.0, 3.0, 5.0]]),
        )

    def test_min_p_kernel(self):
        logits = torch.tensor(
            [[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]], device="mcpu"
        )
        idx = torch.tensor([0, 1], dtype=torch.int32, device="mcpu")
        min_p = torch.tensor([math.exp(-1.5), 0.0], dtype=torch.float32, device="mcpu")

        torch.ops.mcpu.vllm_min_p_kernel(logits, idx, min_p, 3)
        torch.mcpu.synchronize()

        expected = torch.tensor(
            [[float("-inf"), 1.0, 2.0], [3.0, 2.0, 1.0]]
        )
        self.assertEqual(logits.cpu(), expected)

    def test_prompt_logprobs_token_ids(self):
        out = torch.empty(3, dtype=torch.int64, device="mcpu")
        query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32, device="mcpu")
        idx_mapping = torch.tensor([1, 0], dtype=torch.int32, device="mcpu")
        num_computed = torch.tensor([0, 1], dtype=torch.int32, device="mcpu")
        all_token_ids = torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23]],
            dtype=torch.int32,
            device="mcpu",
        )

        torch.ops.mcpu.vllm_prompt_logprobs_token_ids(
            out, query_start_loc, idx_mapping, num_computed, all_token_ids
        )
        torch.mcpu.synchronize()

        self.assertEqual(out.cpu(), torch.tensor([22, 23, 11], dtype=torch.int64))

    def test_get_num_sampled_and_rejected_returns_num_rejected(self):
        num_sampled = torch.tensor([1, 2, 3], dtype=torch.int32, device="mcpu")
        seq_lens = torch.tensor([5, 2, 8], dtype=torch.int32, device="mcpu")
        cu_num_logits = torch.tensor([0, 4, 7, 12], dtype=torch.int32, device="mcpu")
        idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32, device="mcpu")
        prefill_len = torch.tensor([4, 6, 8], dtype=torch.int32, device="mcpu")

        num_rejected = torch.ops.mcpu.vllm_get_num_sampled_and_rejected(
            num_sampled,
            seq_lens,
            cu_num_logits,
            idx_mapping,
            prefill_len,
        )
        torch.mcpu.synchronize()

        self.assertEqual(num_sampled.cpu(), torch.tensor([1, 0, 3], dtype=torch.int32))
        self.assertEqual(num_rejected.cpu(), torch.tensor([3, 0, 2], dtype=torch.int32))


if __name__ == "__main__":
    run_tests()
