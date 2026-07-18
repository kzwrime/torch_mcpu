// SPDX-License-Identifier: Apache-2.0
//
// C++ kernels for vllm/v1/spec_decode/utils.py padded EAGLE launches.

#include "common.h"
#include "runtime/McpuKernelLaunch.h"

#include <algorithm>

namespace {

void vllm_eagle_prepare_next_token_padded_impl(
    const at::Tensor& sampled_token_ids,
    const at::Tensor& discard_request_mask,
    const at::Tensor& backup_next_token_ids,
    at::Tensor& next_token_ids,
    at::Tensor& valid_sampled_tokens_count,
    int64_t vocab_size,
    int64_t num_sampled_tokens_per_req,
    int64_t num_reqs) {
  VLLM_MCPU_CHECK_DIM(sampled_token_ids, 2, "sampled_token_ids");
  VLLM_MCPU_CHECK_DTYPE(sampled_token_ids, at::kInt, "sampled_token_ids");
  VLLM_MCPU_CHECK_DIM(discard_request_mask, 1, "discard_request_mask");
  VLLM_MCPU_CHECK_DTYPE(
      discard_request_mask, at::kBool, "discard_request_mask");
  VLLM_MCPU_CHECK_DTYPE(
      backup_next_token_ids, at::kInt, "backup_next_token_ids");
  VLLM_MCPU_CHECK_DTYPE(next_token_ids, at::kInt, "next_token_ids");
  VLLM_MCPU_CHECK_DTYPE(
      valid_sampled_tokens_count, at::kInt, "valid_sampled_tokens_count");
  VLLM_MCPU_CHECK(
      num_reqs >= 0 && num_reqs <= sampled_token_ids.size(0),
      "num_reqs exceeds sampled_token_ids rows");
  VLLM_MCPU_CHECK(
      num_sampled_tokens_per_req >= 0 &&
          num_sampled_tokens_per_req <= sampled_token_ids.size(1),
      "num_sampled_tokens_per_req exceeds sampled_token_ids columns");
  VLLM_MCPU_CHECK(
      discard_request_mask.numel() >= num_reqs &&
          backup_next_token_ids.numel() >= num_reqs &&
          next_token_ids.numel() >= num_reqs &&
          valid_sampled_tokens_count.numel() >= num_reqs,
      "EAGLE next-token tensors are shorter than num_reqs");

  const int32_t* sampled = sampled_token_ids.data_ptr<int32_t>();
  const bool* discard = discard_request_mask.data_ptr<bool>();
  const int32_t* backup = backup_next_token_ids.data_ptr<int32_t>();
  int32_t* next = next_token_ids.data_ptr<int32_t>();
  int32_t* valid = valid_sampled_tokens_count.data_ptr<int32_t>();
  const int64_t sampled_stride = sampled_token_ids.stride(0);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::vllm_eagle_prepare_next_token_padded",
      ([
        sampled,
        discard,
        backup,
        next,
        valid,
        sampled_stride,
        vocab_size,
        num_sampled_tokens_per_req,
        num_reqs
      ]),
      {
        at::mcpu::KernelPointerMemoryGuard guard(
            {sampled, discard, backup, next, valid});
        for (int64_t req = 0; req < num_reqs; ++req) {
          if (discard[req]) {
            next[req] = backup[req];
            valid[req] = 0;
            continue;
          }

          int32_t valid_count = 0;
          int32_t last_valid_token = 0;
          const int32_t* row = sampled + req * sampled_stride;
          for (int64_t pos = 0; pos < num_sampled_tokens_per_req; ++pos) {
            const int32_t token = row[pos];
            // Match Triton exactly: negative values other than -1 are valid
            // when they are below vocab_size.
            if (token != -1 && token < vocab_size) {
              ++valid_count;
              last_valid_token = token;
            }
          }
          next[req] = valid_count > 0 ? last_valid_token : backup[req];
          valid[req] = valid_count;
        }
      });
}

void vllm_eagle_prepare_inputs_padded_impl(
    const at::Tensor& cu_num_draft_tokens,
    const at::Tensor& valid_sampled_tokens_count,
    const at::Tensor& query_start_loc,
    at::Tensor& token_indices_to_sample,
    at::Tensor& num_rejected_tokens,
    int64_t num_reqs) {
  const at::Tensor* tensors[] = {
      &cu_num_draft_tokens,
      &valid_sampled_tokens_count,
      &query_start_loc,
      &token_indices_to_sample,
      &num_rejected_tokens};
  for (const at::Tensor* tensor : tensors) {
    VLLM_MCPU_CHECK_DIM(*tensor, 1, "EAGLE prepare-input tensor");
    VLLM_MCPU_CHECK_DTYPE(*tensor, at::kInt, "EAGLE prepare-input tensor");
  }
  VLLM_MCPU_CHECK(
      num_reqs >= 0 && cu_num_draft_tokens.numel() >= num_reqs &&
          valid_sampled_tokens_count.numel() >= num_reqs &&
          query_start_loc.numel() >= num_reqs + 1 &&
          token_indices_to_sample.numel() >= num_reqs &&
          num_rejected_tokens.numel() >= num_reqs,
      "invalid num_reqs for EAGLE prepare-input tensors");

  const int32_t* cu = cu_num_draft_tokens.data_ptr<int32_t>();
  const int32_t* valid = valid_sampled_tokens_count.data_ptr<int32_t>();
  const int32_t* query = query_start_loc.data_ptr<int32_t>();
  int32_t* indices = token_indices_to_sample.data_ptr<int32_t>();
  int32_t* rejected = num_rejected_tokens.data_ptr<int32_t>();

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::vllm_eagle_prepare_inputs_padded",
      ([ cu, valid, query, indices, rejected, num_reqs ]),
      {
        at::mcpu::KernelPointerMemoryGuard guard(
            {cu, valid, query, indices, rejected});
        for (int64_t req = 0; req < num_reqs; ++req) {
          const int32_t start = req == 0 ? 0 : cu[req - 1];
          const int32_t num_draft = cu[req] - start;
          const int32_t num_rejected =
              num_draft > 0 ? num_draft + 1 - valid[req] : 0;
          rejected[req] = num_rejected;
          indices[req] = query[req + 1] - 1 - num_rejected;
        }
      });
}

void vllm_eagle_step_slot_mapping_metadata_impl(
    const at::Tensor& positions,
    const at::Tensor& block_table,
    at::Tensor& seq_lens,
    at::Tensor& out_clamped_positions,
    at::Tensor& out_slot_mapping,
    int64_t block_size,
    int64_t max_model_len,
    int64_t n_blocks_per_req,
    int64_t pad_id,
    int64_t batch_size) {
  VLLM_MCPU_CHECK_DTYPE(positions, at::kLong, "positions");
  VLLM_MCPU_CHECK_DTYPE(block_table, at::kInt, "block_table");
  VLLM_MCPU_CHECK_DTYPE(seq_lens, at::kInt, "seq_lens");
  VLLM_MCPU_CHECK_DTYPE(
      out_clamped_positions, at::kLong, "out_clamped_positions");
  VLLM_MCPU_CHECK_DTYPE(out_slot_mapping, at::kLong, "out_slot_mapping");
  VLLM_MCPU_CHECK(
      block_size > 0 && max_model_len > 0 && n_blocks_per_req > 0,
      "EAGLE slot-mapping constants must be positive");
  VLLM_MCPU_CHECK(
      batch_size >= 0 && positions.numel() >= batch_size &&
          seq_lens.numel() >= batch_size &&
          out_clamped_positions.numel() >= batch_size &&
          block_table.size(0) >= batch_size &&
          block_table.size(1) >= n_blocks_per_req,
      "invalid batch or block-table shape");

  const int64_t* pos = positions.data_ptr<int64_t>();
  const int32_t* table = block_table.data_ptr<int32_t>();
  int32_t* seq = seq_lens.data_ptr<int32_t>();
  int64_t* out_pos = out_clamped_positions.data_ptr<int64_t>();
  int64_t* slots = out_slot_mapping.data_ptr<int64_t>();
  const int64_t table_stride = block_table.stride(0);
  const int64_t input_batch_size = out_slot_mapping.numel();

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::vllm_eagle_step_slot_mapping_metadata",
      ([
        pos,
        table,
        seq,
        out_pos,
        slots,
        table_stride,
        input_batch_size,
        block_size,
        max_model_len,
        n_blocks_per_req,
        pad_id,
        batch_size
      ]),
      {
        at::mcpu::KernelPointerMemoryGuard guard(
            {pos, table, seq, out_pos, slots});
        for (int64_t req = 0; req < input_batch_size; ++req) {
          if (req >= batch_size) {
            slots[req] = pad_id;
            continue;
          }
          const int64_t new_position = pos[req] + 1;
          const bool exceeds_max = new_position >= max_model_len;
          const int64_t clamped_position = exceeds_max ? 0 : new_position;
          const int64_t block_number = std::min<int64_t>(
              clamped_position / block_size, n_blocks_per_req - 1);
          const int32_t block_id = table[req * table_stride + block_number];
          out_pos[req] = clamped_position;
          slots[req] = exceeds_max
              ? pad_id
              : block_id * block_size + clamped_position % block_size;
          seq[req] = static_cast<int32_t>(std::min<int64_t>(
              exceeds_max ? 1 : static_cast<int64_t>(seq[req]) + 1,
              max_model_len));
        }
      });
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_eagle_prepare_next_token_padded("
      "Tensor sampled_token_ids, Tensor discard_request_mask, "
      "Tensor backup_next_token_ids, Tensor(a!) next_token_ids, "
      "Tensor(b!) valid_sampled_tokens_count, int vocab_size, "
      "int num_sampled_tokens_per_req, int num_reqs) -> ()");
  m.def(
      "vllm_eagle_prepare_inputs_padded("
      "Tensor cu_num_draft_tokens, Tensor valid_sampled_tokens_count, "
      "Tensor query_start_loc, Tensor(a!) token_indices_to_sample, "
      "Tensor(b!) num_rejected_tokens, int num_reqs) -> ()");
  m.def(
      "vllm_eagle_step_slot_mapping_metadata("
      "Tensor positions, Tensor block_table, Tensor(a!) seq_lens, "
      "Tensor(b!) out_clamped_positions, Tensor(c!) out_slot_mapping, "
      "int block_size, int max_model_len, int n_blocks_per_req, "
      "int pad_id, int batch_size) -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl(
      "vllm_eagle_prepare_next_token_padded",
      &vllm_eagle_prepare_next_token_padded_impl);
  m.impl(
      "vllm_eagle_prepare_inputs_padded",
      &vllm_eagle_prepare_inputs_padded_impl);
  m.impl(
      "vllm_eagle_step_slot_mapping_metadata",
      &vllm_eagle_step_slot_mapping_metadata_impl);
}
