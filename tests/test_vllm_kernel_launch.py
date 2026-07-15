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

        num_rejected = torch.empty_like(num_sampled)
        torch.ops.mcpu.vllm_get_num_sampled_and_rejected(
            num_sampled,
            num_rejected,
            seq_lens,
            cu_num_logits,
            idx_mapping,
            prefill_len,
        )
        torch.mcpu.synchronize()

        self.assertEqual(num_sampled.cpu(), torch.tensor([1, 0, 3], dtype=torch.int32))
        self.assertEqual(num_rejected.cpu(), torch.tensor([3, 0, 2], dtype=torch.int32))

    def test_apply_grammar_bitmask_masks_zero_bits(self):
        logits = torch.arange(70, dtype=torch.float32).reshape(2, 35)
        logits_mcpu = logits.to("mcpu")
        logits_indices = torch.tensor([1], dtype=torch.int32, device="mcpu")
        # Token 1 and token 33 are rejected (zero bits). All other valid
        # vocabulary positions are allowed (one bits).
        bitmask = torch.tensor(
            [[-3, 0b101]], dtype=torch.int32, device="mcpu"
        )

        torch.ops.mcpu.vllm_apply_grammar_bitmask(
            logits_mcpu, logits_indices, bitmask, 35
        )
        torch.mcpu.synchronize()

        expected = logits.clone()
        expected[1, 1] = float("-inf")
        expected[1, 33] = float("-inf")
        self.assertEqual(logits_mcpu.cpu(), expected)

    def test_apply_grammar_bitmask_handles_signed_high_bit(self):
        logits = torch.zeros((1, 32), dtype=torch.bfloat16, device="mcpu")
        logits_indices = torch.tensor([0], dtype=torch.int32, device="mcpu")
        # Bits 0-30 are one; signed bit 31 is zero and must be masked.
        bitmask = torch.tensor(
            [[0x7FFFFFFF]], dtype=torch.int32, device="mcpu"
        )

        torch.ops.mcpu.vllm_apply_grammar_bitmask(
            logits, logits_indices, bitmask, 32
        )
        torch.mcpu.synchronize()

        expected = torch.zeros(32, dtype=torch.bfloat16)
        expected[31] = float("-inf")
        self.assertEqual(logits.cpu()[0], expected)

    def test_logit_bias_preserves_all_allowed_tokens(self):
        vocab_size = 1100
        for num_allowed in (513, 1024):
            with self.subTest(num_allowed=num_allowed):
                logits_cpu = torch.arange(vocab_size, dtype=torch.float32).reshape(1, -1)
                logits = logits_cpu.to("mcpu")
                expanded_idx_mapping = torch.tensor(
                    [0], dtype=torch.int32, device="mcpu"
                )
                num_allowed_token_ids = torch.tensor(
                    [num_allowed], dtype=torch.int32, device="mcpu"
                )
                allowed_token_ids = torch.arange(
                    1024, dtype=torch.int32, device="mcpu"
                ).reshape(1, -1)
                num_logit_bias = torch.zeros(1, dtype=torch.int32, device="mcpu")
                bias_token_ids = torch.zeros(
                    (1, 1024), dtype=torch.int32, device="mcpu"
                )
                bias = torch.zeros((1, 1024), dtype=torch.float32, device="mcpu")
                pos = torch.zeros(1, dtype=torch.int64, device="mcpu")
                min_lens = torch.zeros(1, dtype=torch.int32, device="mcpu")
                num_stop_token_ids = torch.zeros(
                    1, dtype=torch.int32, device="mcpu"
                )
                stop_token_ids = torch.zeros(
                    (1, 128), dtype=torch.int32, device="mcpu"
                )

                torch.ops.mcpu.vllm_bias_kernel(
                    logits,
                    vocab_size,
                    expanded_idx_mapping,
                    num_allowed_token_ids,
                    allowed_token_ids,
                    num_logit_bias,
                    bias_token_ids,
                    bias,
                    pos,
                    min_lens,
                    num_stop_token_ids,
                    stop_token_ids,
                )
                torch.mcpu.synchronize()

                expected = logits_cpu.clone()
                expected[:, num_allowed:] = float("-inf")
                self.assertEqual(logits.cpu(), expected)

    def test_fill_logprob_token_ids_custom_overrides_topk(self):
        out_ids = torch.full((3, 5), -99, dtype=torch.int64, device="mcpu")
        valid_mask = torch.zeros((3, 5), dtype=torch.bool, device="mcpu")
        sampled = torch.tensor([10, 20, 30], dtype=torch.int64, device="mcpu")
        topk = torch.tensor(
            [[1, 2], [3, 4], [5, 6]], dtype=torch.int32, device="mcpu"
        )
        mapping = torch.tensor([2, 0, 1], dtype=torch.int32, device="mcpu")
        num_custom = torch.tensor([0, 3, 0], dtype=torch.int32, device="mcpu")
        custom = torch.tensor(
            [[40, 41, 42, 43], [7, 8, 9, 10], [50, 51, 52, 53]],
            dtype=torch.int32,
            device="mcpu",
        )

        torch.ops.mcpu.vllm_fill_logprob_token_ids_kernel(
            out_ids,
            out_ids.stride(0),
            valid_mask,
            valid_mask.stride(0),
            sampled,
            topk,
            topk.stride(0),
            mapping,
            num_custom,
            custom,
            custom.stride(0),
            2,
            4,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            out_ids.cpu(),
            torch.tensor(
                [[10, 1, 2, -99, -99], [20, 3, 4, -99, -99], [30, 7, 8, 9, -99]],
                dtype=torch.int64,
            ),
        )
        self.assertEqual(
            valid_mask.cpu(),
            torch.tensor(
                [
                    [True, True, True, False, False],
                    [True, True, True, False, False],
                    [True, True, True, True, False],
                ]
            ),
        )

    def test_fill_logprob_token_ids_num_topk_zero(self):
        out_ids = torch.full((2, 4), -1, dtype=torch.int64, device="mcpu")
        valid_mask = torch.zeros((2, 4), dtype=torch.bool, device="mcpu")
        sampled = torch.tensor([11, 22], dtype=torch.int64, device="mcpu")
        topk_unused = torch.zeros((2, 1), dtype=torch.int32, device="mcpu")
        mapping = torch.tensor([0, 1], dtype=torch.int32, device="mcpu")
        num_custom = torch.tensor([0, 2], dtype=torch.int32, device="mcpu")
        custom = torch.tensor(
            [[0, 0, 0], [70, 71, 72]], dtype=torch.int32, device="mcpu"
        )

        torch.ops.mcpu.vllm_fill_logprob_token_ids_kernel(
            out_ids,
            out_ids.stride(0),
            valid_mask,
            valid_mask.stride(0),
            sampled,
            topk_unused,
            topk_unused.stride(0),
            mapping,
            num_custom,
            custom,
            custom.stride(0),
            0,
            2,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            out_ids.cpu(),
            torch.tensor([[11, -1, -1, -1], [22, 70, 71, -1]], dtype=torch.int64),
        )
        self.assertEqual(
            valid_mask.cpu(),
            torch.tensor(
                [[True, False, False, False], [True, True, True, False]]
            ),
        )

    def test_fill_logprob_token_ids_respects_padded_cols(self):
        out_ids = torch.full((1, 5), -1, dtype=torch.int64, device="mcpu")
        valid_mask = torch.zeros((1, 5), dtype=torch.bool, device="mcpu")
        sampled = torch.tensor([11], dtype=torch.int64, device="mcpu")
        topk = torch.tensor([[1, 2, 3]], dtype=torch.int32, device="mcpu")
        mapping = torch.tensor([0], dtype=torch.int32, device="mcpu")
        num_custom = torch.tensor([3], dtype=torch.int32, device="mcpu")
        custom = torch.tensor([[70, 71, 72]], dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_fill_logprob_token_ids_kernel(
            out_ids,
            out_ids.stride(0),
            valid_mask,
            valid_mask.stride(0),
            sampled,
            topk,
            topk.stride(0),
            mapping,
            num_custom,
            custom,
            custom.stride(0),
            3,
            2,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            out_ids.cpu(),
            torch.tensor([[11, 70, 71, -1, -1]], dtype=torch.int64),
        )
        self.assertEqual(
            valid_mask.cpu(),
            torch.tensor([[True, True, True, False, False]]),
        )

    def test_combine_sampled_and_draft_tokens_supports_zero_or_one_sampled(self):
        idx_mapping = torch.tensor([1, 0], dtype=torch.int32, device="mcpu")
        last_sampled = torch.tensor([100, 200], dtype=torch.int64, device="mcpu")
        query_start = torch.tensor([0, 3, 6], dtype=torch.int32, device="mcpu")
        seq_lens = torch.tensor([5, 5], dtype=torch.int32, device="mcpu")
        prefill_len = torch.tensor([2, 2], dtype=torch.int32, device="mcpu")
        drafts = torch.tensor(
            [[10, 11, 12], [20, 21, 22]], dtype=torch.int64, device="mcpu"
        )

        cases = (
            (1, [0, 200, 20, 100, 10, 11]),
            (0, [0, 20, 21, 10, 11, 12]),
        )
        for num_new_sampled, expected_input in cases:
            with self.subTest(num_new_sampled=num_new_sampled):
                input_ids = torch.zeros(6, dtype=torch.int32, device="mcpu")
                cu_num_logits = torch.tensor(
                    [0, 2, 5], dtype=torch.int32, device="mcpu"
                )
                logits_indices = torch.empty(5, dtype=torch.int64, device="mcpu")

                torch.ops.mcpu.vllm_combine_sampled_and_draft_tokens(
                    input_ids,
                    idx_mapping,
                    last_sampled,
                    query_start,
                    seq_lens,
                    prefill_len,
                    drafts,
                    drafts.stride(0),
                    cu_num_logits,
                    logits_indices,
                    num_new_sampled,
                )
                torch.mcpu.synchronize()

                self.assertEqual(
                    input_ids.cpu(), torch.tensor(expected_input, dtype=torch.int32)
                )
                self.assertEqual(
                    logits_indices.cpu(), torch.tensor([1, 2, 3, 4, 5])
                )

    def test_combine_sampled_and_draft_tokens_skips_prefill_writes(self):
        input_ids = torch.full((3,), 9, dtype=torch.int32, device="mcpu")
        idx_mapping = torch.tensor([0], dtype=torch.int32, device="mcpu")
        last_sampled = torch.tensor([100], dtype=torch.int64, device="mcpu")
        query_start = torch.tensor([0, 3], dtype=torch.int32, device="mcpu")
        seq_lens = torch.tensor([2], dtype=torch.int32, device="mcpu")
        prefill_len = torch.tensor([2], dtype=torch.int32, device="mcpu")
        drafts = torch.tensor([[10, 11]], dtype=torch.int64, device="mcpu")
        cu_num_logits = torch.tensor([0, 2], dtype=torch.int32, device="mcpu")
        logits_indices = torch.empty(2, dtype=torch.int64, device="mcpu")

        torch.ops.mcpu.vllm_combine_sampled_and_draft_tokens(
            input_ids,
            idx_mapping,
            last_sampled,
            query_start,
            seq_lens,
            prefill_len,
            drafts,
            drafts.stride(0),
            cu_num_logits,
            logits_indices,
            0,
        )
        torch.mcpu.synchronize()

        self.assertEqual(input_ids.cpu(), torch.full((3,), 9, dtype=torch.int32))
        self.assertEqual(logits_indices.cpu(), torch.tensor([1, 2]))

    def test_post_update_skips_negative_mapping_and_updates_state(self):
        idx_mapping = torch.tensor([-1, 1, 0], dtype=torch.int32, device="mcpu")
        num_computed = torch.tensor([10, 20], dtype=torch.int32, device="mcpu")
        # vLLM 0.24 stores this pointer-compatible buffer as [max_reqs, 1].
        last_sampled = torch.full((2, 1), -1, dtype=torch.int64, device="mcpu")
        output_counts = torch.zeros((2, 10), dtype=torch.int32, device="mcpu")
        sampled = torch.tensor(
            [[99, 98], [3, 4], [5, 6]], dtype=torch.int64, device="mcpu"
        )
        num_sampled = torch.tensor([1, 2, 1], dtype=torch.int32, device="mcpu")
        num_rejected = torch.tensor([9, 1, 0], dtype=torch.int32, device="mcpu")
        query_start = torch.tensor([0, 2, 5, 6], dtype=torch.int32, device="mcpu")
        all_token_ids = torch.full((2, 8), -1, dtype=torch.int32, device="mcpu")
        total_len = torch.tensor([1, 2], dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_post_update(
            idx_mapping,
            num_computed,
            last_sampled,
            output_counts,
            output_counts.stride(0),
            sampled,
            sampled.stride(0),
            num_sampled,
            num_rejected,
            query_start,
            all_token_ids,
            all_token_ids.stride(0),
            total_len,
        )
        torch.mcpu.synchronize()

        self.assertEqual(num_computed.cpu(), torch.tensor([11, 22], dtype=torch.int32))
        self.assertEqual(
            last_sampled.cpu(), torch.tensor([[5], [4]], dtype=torch.int64)
        )
        self.assertEqual(total_len.cpu(), torch.tensor([2, 4], dtype=torch.int32))
        expected_ids = torch.full((2, 8), -1, dtype=torch.int32)
        expected_ids[0, 1] = 5
        expected_ids[1, 2:4] = torch.tensor([3, 4], dtype=torch.int32)
        self.assertEqual(all_token_ids.cpu(), expected_ids)
        expected_counts = torch.zeros((2, 10), dtype=torch.int32)
        expected_counts[0, 5] = 1
        expected_counts[1, 3] = 1
        expected_counts[1, 4] = 1
        self.assertEqual(output_counts.cpu(), expected_counts)

    def test_post_update_supports_optional_query_and_counts(self):
        idx_mapping = torch.tensor([0], dtype=torch.int32, device="mcpu")
        num_computed = torch.tensor([10], dtype=torch.int32, device="mcpu")
        last_sampled = torch.tensor([-1], dtype=torch.int64, device="mcpu")
        sampled = torch.tensor([[7]], dtype=torch.int64, device="mcpu")
        num_sampled = torch.tensor([0], dtype=torch.int32, device="mcpu")
        num_rejected = torch.tensor([2], dtype=torch.int32, device="mcpu")
        all_token_ids = torch.full((1, 4), -1, dtype=torch.int32, device="mcpu")
        total_len = torch.tensor([1], dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_post_update(
            idx_mapping,
            num_computed,
            last_sampled,
            None,
            0,
            sampled,
            sampled.stride(0),
            num_sampled,
            num_rejected,
            None,
            all_token_ids,
            all_token_ids.stride(0),
            total_len,
        )
        torch.mcpu.synchronize()

        self.assertEqual(num_computed.cpu(), torch.tensor([8], dtype=torch.int32))
        self.assertEqual(last_sampled.cpu(), torch.tensor([-1], dtype=torch.int64))
        self.assertEqual(total_len.cpu(), torch.tensor([1], dtype=torch.int32))
        self.assertEqual(
            all_token_ids.cpu(), torch.full((1, 4), -1, dtype=torch.int32)
        )

    def test_ranks_matches_triton_comparison_and_handles_padded_ids(self):
        logits = torch.tensor(
            [[1.0, 4.0, 2.0, 4.0], [3.0, 2.0, 1.0, 0.0]], device="mcpu"
        )
        token_ids = torch.tensor([1, -1], dtype=torch.int64, device="mcpu")
        output = torch.empty(2, dtype=torch.int64, device="mcpu")

        torch.ops.mcpu.vllm_ranks_kernel(output, logits, token_ids, 4)
        torch.mcpu.synchronize()

        # Rank counts ties with >=, matching the v0.24 Triton kernel. Invalid
        # padded ids are outside its contract and are made harmless here.
        self.assertEqual(output.cpu(), torch.tensor([2, 0], dtype=torch.int64))

    def test_apply_write_single_supports_staged_write_dtypes(self):
        for dtype in (torch.int32, torch.int64, torch.float32):
            with self.subTest(dtype=dtype):
                output = torch.zeros((3, 6), dtype=dtype, device="mcpu")
                indices = torch.tensor([2, 0, 2], dtype=torch.int32, device="mcpu")
                starts = torch.tensor([1, 3, 4], dtype=torch.int32, device="mcpu")
                contents = torch.tensor([10, 11, 20, 30, 31], dtype=dtype, device="mcpu")
                cu_lens = torch.tensor([2, 3, 5], dtype=torch.int32, device="mcpu")

                torch.ops.mcpu.vllm_apply_write_single(
                    output,
                    output.stride(0),
                    indices,
                    starts,
                    contents,
                    cu_lens,
                )
                torch.mcpu.synchronize()

                expected = torch.zeros((3, 6), dtype=dtype)
                expected[2, 1:3] = torch.tensor([10, 11], dtype=dtype)
                expected[0, 3] = 20
                expected[2, 4:6] = torch.tensor([30, 31], dtype=dtype)
                self.assertEqual(output.cpu(), expected)

    def test_apply_write_multi_resolves_output_pointer_and_stride(self):
        outputs = [
            torch.zeros((2, 5), dtype=torch.int32, device="mcpu"),
            torch.zeros((3, 4), dtype=torch.int32, device="mcpu"),
        ]
        output_ptrs = torch.tensor(
            [output.data_ptr() for output in outputs],
            dtype=torch.uint64,
            device="mcpu",
        )
        output_strides = torch.tensor(
            [output.stride(0) for output in outputs],
            dtype=torch.int64,
            device="mcpu",
        )
        indices = torch.tensor([2, 1, 0], dtype=torch.int32, device="mcpu")
        starts = torch.tensor([1, 2, 0], dtype=torch.int32, device="mcpu")
        contents = torch.tensor([10, 11, 20, 30, 31], dtype=torch.int32, device="mcpu")
        cu_lens = torch.tensor([2, 3, 5], dtype=torch.int32, device="mcpu")
        group_ids = torch.tensor([1, 0, 1], dtype=torch.int32, device="mcpu")

        torch.ops.mcpu.vllm_apply_write_multi(
            output_ptrs,
            output_strides,
            indices,
            starts,
            contents,
            cu_lens,
            group_ids,
        )
        torch.mcpu.synchronize()

        expected0 = torch.zeros((2, 5), dtype=torch.int32)
        expected0[1, 2] = 20
        expected1 = torch.zeros((3, 4), dtype=torch.int32)
        expected1[2, 1:3] = torch.tensor([10, 11], dtype=torch.int32)
        expected1[0, 0:2] = torch.tensor([30, 31], dtype=torch.int32)
        self.assertEqual(outputs[0].cpu(), expected0)
        self.assertEqual(outputs[1].cpu(), expected1)

    def test_gather_block_tables_pointer_abi_preserves_active_tail(self):
        sources = [
            torch.tensor(
                [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
                dtype=torch.int32,
                device="mcpu",
            ),
            torch.tensor(
                [[40, 41, 42], [50, 51, 52], [60, 61, 62]],
                dtype=torch.int32,
                device="mcpu",
            ),
        ]
        destinations = [
            torch.full((4, 4), 99, dtype=torch.int32, device="mcpu"),
            torch.full((4, 3), 99, dtype=torch.int32, device="mcpu"),
        ]
        src_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in sources],
            dtype=torch.uint64,
            device="mcpu",
        )
        dst_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in destinations],
            dtype=torch.uint64,
            device="mcpu",
        )
        strides = torch.tensor([4, 3], dtype=torch.int64, device="mcpu")
        idx_mapping = torch.tensor([2, 0], dtype=torch.int32, device="mcpu")
        num_blocks = torch.tensor(
            [[1, 4, 2], [2, 3, 1]], dtype=torch.int32, device="mcpu"
        )

        torch.ops.mcpu.vllm_gather_block_tables_kernel(
            idx_mapping,
            src_ptrs,
            dst_ptrs,
            strides,
            num_blocks,
            num_blocks.stride(0),
            2,
            2,
            3,
        )
        torch.mcpu.synchronize()

        expected0 = torch.tensor(
            [[30, 31, 99, 99], [10, 99, 99, 99], [0, 0, 0, 0], [99, 99, 99, 99]],
            dtype=torch.int32,
        )
        expected1 = torch.tensor(
            [[60, 99, 99], [40, 41, 99], [0, 0, 0], [99, 99, 99]],
            dtype=torch.int32,
        )
        self.assertEqual(destinations[0].cpu(), expected0)
        self.assertEqual(destinations[1].cpu(), expected1)


    def test_compute_slot_mappings_pointer_abi_and_tail_padding(self):
        block_tables = [
            torch.tensor(
                [[10, 11, 12], [20, 21, 22]], dtype=torch.int32, device="mcpu"
            ),
            torch.tensor(
                [[30, 31, 32, 33, 34], [40, 41, 42, 43, 44]],
                dtype=torch.int32,
                device="mcpu",
            ),
        ]
        table_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in block_tables],
            dtype=torch.uint64,
            device="mcpu",
        )
        table_strides = torch.tensor([3, 5], dtype=torch.int64, device="mcpu")
        block_sizes = torch.tensor([4, 2], dtype=torch.int32, device="mcpu")
        idx_mapping = torch.tensor([1, 0], dtype=torch.int32, device="mcpu")
        query_start = torch.tensor([0, 2, 5], dtype=torch.int32, device="mcpu")
        positions = torch.tensor([0, 5, 2, 3, 8], dtype=torch.int64, device="mcpu")
        slots = torch.full((2, 8), 999, dtype=torch.int64, device="mcpu")

        torch.ops.mcpu.vllm_compute_slot_mappings_kernel(
            8,
            idx_mapping,
            query_start,
            positions,
            table_ptrs,
            table_strides,
            block_sizes,
            slots,
            slots.stride(0),
            0,
            1,
            1,
            -1,
            2,
            2,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            slots.cpu(),
            torch.tensor(
                [[80, 85, 42, 43, 48, -1, -1, -1], [80, 85, 62, 63, 68, -1, -1, -1]],
                dtype=torch.int64,
            ),
        )

    def test_compute_slot_mappings_context_parallel_formula(self):
        block_table = torch.tensor(
            [[10]], dtype=torch.int32, device="mcpu"
        )
        table_ptrs = torch.tensor(
            [block_table.data_ptr()], dtype=torch.uint64, device="mcpu"
        )
        table_strides = torch.tensor([1], dtype=torch.int64, device="mcpu")
        block_sizes = torch.tensor([4], dtype=torch.int32, device="mcpu")
        idx_mapping = torch.tensor([0], dtype=torch.int32, device="mcpu")
        query_start = torch.tensor([0, 8], dtype=torch.int32, device="mcpu")
        positions = torch.arange(8, dtype=torch.int64, device="mcpu")
        slots = torch.empty((1, 8), dtype=torch.int64, device="mcpu")

        torch.ops.mcpu.vllm_compute_slot_mappings_kernel(
            8,
            idx_mapping,
            query_start,
            positions,
            table_ptrs,
            table_strides,
            block_sizes,
            slots,
            slots.stride(0),
            1,
            2,
            1,
            -1,
            1,
            1,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            slots.cpu()[0],
            torch.tensor([-1, 40, -1, 41, -1, 42, -1, 43], dtype=torch.int64),
        )

    def test_compute_slot_mapping_v1_abi_context_parallel_and_tail_padding(self):
        block_table = torch.tensor(
            [[10], [20]], dtype=torch.int32, device="mcpu"
        )
        query_start = torch.tensor([0, 4, 8], dtype=torch.int32, device="mcpu")
        positions = torch.arange(8, dtype=torch.int64, device="mcpu")
        slots = torch.full((10,), 999, dtype=torch.int64, device="mcpu")

        torch.ops.mcpu.vllm_compute_slot_mapping_kernel(
            8,
            10,
            query_start,
            positions,
            block_table,
            block_table.stride(0),
            4,
            slots,
            2,
            1,
            1,
            -1,
        )
        torch.mcpu.synchronize()

        self.assertEqual(
            slots.cpu(),
            torch.tensor(
                [-1, 40, -1, 41, -1, 82, -1, 83, -1, -1],
                dtype=torch.int64,
            ),
        )


    def test_zero_kv_blocks_uses_segment_addresses(self):
        segments = [
            torch.arange(32, dtype=torch.int32, device="mcpu").reshape(4, 8),
            torch.arange(100, 132, dtype=torch.int32, device="mcpu").reshape(4, 8),
        ]
        expected = [segment.cpu() for segment in segments]
        for tensor in expected:
            tensor[1].zero_()
            tensor[3].zero_()
        seg_addrs = torch.tensor(
            [segment.data_ptr() for segment in segments],
            dtype=torch.uint64,
            device="mcpu",
        )
        block_ids = torch.tensor([1, 3], dtype=torch.int64, device="mcpu")

        torch.ops.mcpu.zero_kv_blocks_kernel_impl(
            seg_addrs, block_ids, 2, 2, 8
        )
        torch.mcpu.synchronize()

        self.assertEqual(segments[0].cpu(), expected[0])
        self.assertEqual(segments[1].cpu(), expected[1])


if __name__ == "__main__":
    run_tests()
