// SPDX-License-Identifier: Apache-2.0
//
// C++ kernel for the autoregressive draft speculator prefill preparation in
// vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py.

#include "common.h"
#include "runtime/McpuKernelLaunch.h"

#include <algorithm>
#include <cstdint>

namespace {

void vllm_autoregressive_prepare_prefill_inputs_impl(
    at::Tensor& last_token_indices,
    at::Tensor& draft_current_step,
    at::Tensor& draft_input_ids,
    at::Tensor& draft_positions,
    at::Tensor& draft_query_start_loc,
    at::Tensor& draft_seq_lens,
    const at::Tensor& target_input_ids,
    const at::Tensor& target_positions,
    const at::Tensor& idx_mapping,
    const at::Tensor& last_sampled,
    const at::Tensor& next_prefill_tokens,
    const at::Tensor& num_sampled,
    const at::Tensor& num_rejected,
    const at::Tensor& query_start_loc,
    const at::Tensor& seq_lens,
    int64_t max_num_reqs) {
  const at::Tensor* int32_tensors[] = {
      &draft_input_ids,
      &draft_query_start_loc,
      &draft_seq_lens,
      &target_input_ids,
      &idx_mapping,
      &next_prefill_tokens,
      &num_sampled,
      &num_rejected,
      &query_start_loc,
      &seq_lens};
  for (const at::Tensor* tensor : int32_tensors) {
    VLLM_MCPU_CHECK_DIM(*tensor, 1, "autoregressive prefill int32 tensor");
    VLLM_MCPU_CHECK_DTYPE(
        *tensor, at::kInt, "autoregressive prefill int32 tensor");
  }
  const at::Tensor* int64_tensors[] = {
      &last_token_indices, &draft_positions, &target_positions};
  for (const at::Tensor* tensor : int64_tensors) {
    VLLM_MCPU_CHECK_DIM(*tensor, 1, "autoregressive prefill int64 tensor");
    VLLM_MCPU_CHECK_DTYPE(
        *tensor, at::kLong, "autoregressive prefill int64 tensor");
  }
  VLLM_MCPU_CHECK_DTYPE(draft_current_step, at::kLong, "draft_current_step");
  VLLM_MCPU_CHECK(
      draft_current_step.numel() == 1,
      "draft_current_step must contain one value");
  VLLM_MCPU_CHECK_DTYPE(last_sampled, at::kLong, "last_sampled");
  VLLM_MCPU_CHECK(
      last_sampled.dim() == 1 ||
          (last_sampled.dim() == 2 && last_sampled.size(1) == 1),
      "last_sampled must have shape [N] or [N, 1]");
  VLLM_MCPU_CHECK(
      last_sampled.is_contiguous(), "last_sampled must be contiguous");

  const int64_t num_reqs = idx_mapping.numel();
  VLLM_MCPU_CHECK(num_reqs > 0, "autoregressive prefill requires requests");
  VLLM_MCPU_CHECK(
      max_num_reqs >= num_reqs,
      "max_num_reqs must be at least the active request count");
  VLLM_MCPU_CHECK(
      last_token_indices.numel() >= max_num_reqs &&
          draft_current_step.numel() == 1 &&
          draft_query_start_loc.numel() >= max_num_reqs + 1 &&
          draft_seq_lens.numel() >= max_num_reqs,
      "autoregressive prefill output buffers are too small");
  VLLM_MCPU_CHECK(
      num_sampled.numel() >= num_reqs && num_rejected.numel() >= num_reqs &&
          query_start_loc.numel() >= num_reqs + 1 &&
          seq_lens.numel() >= num_reqs,
      "autoregressive prefill request inputs are too small");
  VLLM_MCPU_CHECK(
      draft_input_ids.numel() >= target_input_ids.numel() &&
          draft_positions.numel() >= target_positions.numel(),
      "autoregressive prefill draft token buffers are too small");

  int64_t* last_indices = last_token_indices.data_ptr<int64_t>();
  int64_t* current_step = draft_current_step.data_ptr<int64_t>();
  int32_t* draft_ids = draft_input_ids.data_ptr<int32_t>();
  int64_t* draft_pos = draft_positions.data_ptr<int64_t>();
  int32_t* draft_query = draft_query_start_loc.data_ptr<int32_t>();
  int32_t* draft_seq = draft_seq_lens.data_ptr<int32_t>();
  const int32_t* target_ids = target_input_ids.data_ptr<int32_t>();
  const int64_t* target_pos = target_positions.data_ptr<int64_t>();
  const int32_t* mapping = idx_mapping.data_ptr<int32_t>();
  const int64_t* sampled = last_sampled.data_ptr<int64_t>();
  const int32_t* prefill = next_prefill_tokens.data_ptr<int32_t>();
  const int32_t* sampled_count = num_sampled.data_ptr<int32_t>();
  const int32_t* rejected_count = num_rejected.data_ptr<int32_t>();
  const int32_t* query = query_start_loc.data_ptr<int32_t>();
  const int32_t* target_seq = seq_lens.data_ptr<int32_t>();

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::vllm_autoregressive_prepare_prefill_inputs",
      ([
        last_indices,
        current_step,
        draft_ids,
        draft_pos,
        draft_query,
        draft_seq,
        target_ids,
        target_pos,
        mapping,
        sampled,
        prefill,
        sampled_count,
        rejected_count,
        query,
        target_seq,
        num_reqs,
        max_num_reqs
      ]),
      {
        at::mcpu::KernelPointerMemoryGuard guard(
            {last_indices,
             current_step,
             draft_ids,
             draft_pos,
             draft_query,
             draft_seq,
             target_ids,
             target_pos,
             mapping,
             sampled,
             prefill,
             sampled_count,
             rejected_count,
             query,
             target_seq});

        for (int64_t req = 0; req < num_reqs; ++req) {
          const int32_t req_state_idx = mapping[req];
          const int32_t query_start = query[req];
          const int32_t query_end = query[req + 1];
          const int32_t query_len =
              query_end - query_start - rejected_count[req];
          const int32_t next_token = sampled_count[req] > 0
              ? static_cast<int32_t>(sampled[req_state_idx])
              : prefill[req_state_idx];

          // Match Triton: remove the first target token by shifting the
          // remaining valid query tokens left, then append the next token.
          for (int32_t pos = 1; pos < query_len; ++pos) {
            draft_ids[query_start + pos - 1] = target_ids[query_start + pos];
          }
          const int32_t last_token_index = query_start + query_len - 1;
          last_indices[req] = last_token_index;
          draft_ids[last_token_index] = next_token;

          for (int32_t pos = 0; pos < query_len; ++pos) {
            draft_pos[query_start + pos] = target_pos[query_start + pos];
          }
          draft_query[req] = query_start;
          draft_seq[req] = target_seq[req];
        }

        // The last Triton program performs the graph-padding writes.
        current_step[0] = 0;
        const int32_t final_query_end = query[num_reqs];
        for (int64_t req = num_reqs; req <= max_num_reqs; ++req) {
          draft_query[req] = final_query_end;
        }
        for (int64_t req = num_reqs; req < max_num_reqs; ++req) {
          draft_seq[req] = 0;
          last_indices[req] = 0;
        }
      });
}

void vllm_autoregressive_prepare_decode_inputs_impl(
    const at::Tensor& draft_tokens,
    int64_t draft_tokens_stride,
    const at::Tensor& target_seq_lens,
    const at::Tensor& num_rejected,
    at::Tensor& input_ids,
    at::Tensor& positions,
    at::Tensor& query_start_loc,
    at::Tensor& seq_lens,
    int64_t max_model_len,
    int64_t max_num_reqs,
    bool advance_draft_positions) {
  VLLM_MCPU_CHECK_DTYPE(draft_tokens, at::kLong, "draft_tokens");
  VLLM_MCPU_CHECK_DIM(target_seq_lens, 1, "target_seq_lens");
  VLLM_MCPU_CHECK_DIM(num_rejected, 1, "num_rejected");
  VLLM_MCPU_CHECK_DIM(input_ids, 1, "input_ids");
  VLLM_MCPU_CHECK_DIM(positions, 1, "positions");
  VLLM_MCPU_CHECK_DIM(query_start_loc, 1, "query_start_loc");
  VLLM_MCPU_CHECK_DIM(seq_lens, 1, "seq_lens");
  VLLM_MCPU_CHECK_DTYPE(target_seq_lens, at::kInt, "target_seq_lens");
  VLLM_MCPU_CHECK_DTYPE(num_rejected, at::kInt, "num_rejected");
  VLLM_MCPU_CHECK_DTYPE(input_ids, at::kInt, "input_ids");
  VLLM_MCPU_CHECK_DTYPE(positions, at::kLong, "positions");
  VLLM_MCPU_CHECK_DTYPE(query_start_loc, at::kInt, "query_start_loc");
  VLLM_MCPU_CHECK_DTYPE(seq_lens, at::kInt, "seq_lens");

  const int64_t num_reqs = draft_tokens.size(0);
  VLLM_MCPU_CHECK(
      draft_tokens.dim() == 1 || draft_tokens.dim() == 2,
      "draft_tokens must be 1D or 2D");
  VLLM_MCPU_CHECK(
      draft_tokens_stride == draft_tokens.stride(0),
      "draft_tokens_stride mismatch");
  VLLM_MCPU_CHECK(
      max_num_reqs >= num_reqs,
      "max_num_reqs must be at least the active request count");
  VLLM_MCPU_CHECK(
      input_ids.numel() >= max_num_reqs && positions.numel() >= max_num_reqs &&
          query_start_loc.numel() >= max_num_reqs + 1 &&
          seq_lens.numel() >= max_num_reqs,
      "autoregressive decode output buffers are too small");
  VLLM_MCPU_CHECK(
      !advance_draft_positions ||
          (max_model_len > 0 && target_seq_lens.numel() >= num_reqs &&
           num_rejected.numel() >= num_reqs),
      "invalid position-advance inputs");

  const int64_t* draft = draft_tokens.data_ptr<int64_t>();
  const int32_t* target_seq = target_seq_lens.data_ptr<int32_t>();
  const int32_t* rejected = num_rejected.data_ptr<int32_t>();
  int32_t* ids = input_ids.data_ptr<int32_t>();
  int64_t* pos = positions.data_ptr<int64_t>();
  int32_t* query = query_start_loc.data_ptr<int32_t>();
  int32_t* seq = seq_lens.data_ptr<int32_t>();

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::vllm_autoregressive_prepare_decode_inputs",
      ([
        draft,
        target_seq,
        rejected,
        ids,
        pos,
        query,
        seq,
        draft_tokens_stride,
        num_reqs,
        max_model_len,
        max_num_reqs,
        advance_draft_positions
      ]),
      {
        at::mcpu::KernelPointerMemoryGuard guard(
            {draft, target_seq, rejected, ids, pos, query, seq});
        for (int64_t req = 0; req < num_reqs; ++req) {
          ids[req] = static_cast<int32_t>(draft[req * draft_tokens_stride]);
          if (advance_draft_positions) {
            pos[req] = std::min<int64_t>(pos[req] + 1, max_model_len - 1);
            seq[req] = static_cast<int32_t>(std::min<int64_t>(
                static_cast<int64_t>(target_seq[req]) - rejected[req] + 1,
                max_model_len));
          }
        }

        // Match the extra Triton program used for graph padding.
        for (int64_t req = 0; req <= max_num_reqs; ++req) {
          query[req] = static_cast<int32_t>(std::min<int64_t>(req, num_reqs));
        }
        for (int64_t req = num_reqs; req < max_num_reqs; ++req) {
          seq[req] = 0;
        }
      });
}

void vllm_autoregressive_update_draft_inputs_impl(
    at::Tensor& output_draft_tokens,
    int64_t output_draft_tokens_stride,
    at::Tensor& next_input_hidden_states,
    int64_t next_input_hidden_states_stride,
    at::Tensor& input_ids,
    at::Tensor& positions,
    at::Tensor& seq_lens,
    const at::Tensor& draft_tokens,
    const at::Tensor& current_draft_step,
    const at::Tensor& hidden_states,
    int64_t hidden_states_stride,
    int64_t hidden_size,
    int64_t max_model_len,
    int64_t num_speculative_steps,
    bool advance_draft_positions) {
  VLLM_MCPU_CHECK_DIM(output_draft_tokens, 2, "output_draft_tokens");
  VLLM_MCPU_CHECK_DIM(next_input_hidden_states, 2, "next_input_hidden_states");
  VLLM_MCPU_CHECK_DIM(input_ids, 1, "input_ids");
  VLLM_MCPU_CHECK_DIM(positions, 1, "positions");
  VLLM_MCPU_CHECK_DIM(seq_lens, 1, "seq_lens");
  VLLM_MCPU_CHECK_DIM(draft_tokens, 1, "draft_tokens");
  VLLM_MCPU_CHECK_DIM(hidden_states, 2, "hidden_states");
  VLLM_MCPU_CHECK_DTYPE(output_draft_tokens, at::kLong, "output_draft_tokens");
  VLLM_MCPU_CHECK_DTYPE(input_ids, at::kInt, "input_ids");
  VLLM_MCPU_CHECK_DTYPE(positions, at::kLong, "positions");
  VLLM_MCPU_CHECK_DTYPE(seq_lens, at::kInt, "seq_lens");
  VLLM_MCPU_CHECK_DTYPE(draft_tokens, at::kLong, "draft_tokens");
  VLLM_MCPU_CHECK_DTYPE(current_draft_step, at::kLong, "current_draft_step");
  VLLM_MCPU_CHECK(
      current_draft_step.numel() == 1,
      "current_draft_step must contain one value");
  VLLM_MCPU_CHECK(
      hidden_states.scalar_type() == next_input_hidden_states.scalar_type(),
      "hidden state input and output dtypes must match");

  const int64_t num_reqs = draft_tokens.numel();
  VLLM_MCPU_CHECK(
      output_draft_tokens_stride == output_draft_tokens.stride(0),
      "output_draft_tokens_stride mismatch");
  VLLM_MCPU_CHECK(
      output_draft_tokens.stride(1) == 1,
      "output_draft_tokens columns must be contiguous");
  VLLM_MCPU_CHECK(
      next_input_hidden_states_stride == next_input_hidden_states.stride(0),
      "next_input_hidden_states_stride mismatch");
  VLLM_MCPU_CHECK(
      hidden_states_stride == hidden_states.stride(0),
      "hidden_states_stride mismatch");
  VLLM_MCPU_CHECK(
      next_input_hidden_states.stride(1) == 1 && hidden_states.stride(1) == 1,
      "hidden state feature dimensions must be contiguous");
  VLLM_MCPU_CHECK(
      hidden_size == hidden_states.size(1),
      "hidden_size does not match hidden_states");
  VLLM_MCPU_CHECK(
      num_speculative_steps > 0 &&
          output_draft_tokens.size(1) >= num_speculative_steps,
      "output_draft_tokens is too narrow");
  VLLM_MCPU_CHECK(
      output_draft_tokens.size(0) >= num_reqs &&
          next_input_hidden_states.size(0) >= num_reqs &&
          next_input_hidden_states.size(1) >= hidden_size &&
          hidden_states.size(0) >= num_reqs && input_ids.numel() >= num_reqs &&
          positions.numel() >= num_reqs && seq_lens.numel() >= num_reqs,
      "autoregressive update buffers are too small");
  VLLM_MCPU_CHECK(
      !advance_draft_positions || max_model_len > 0,
      "max_model_len must be positive when advancing positions");

  int64_t* output_tokens = output_draft_tokens.data_ptr<int64_t>();
  void* next_hidden = next_input_hidden_states.data_ptr();
  int32_t* ids = input_ids.data_ptr<int32_t>();
  int64_t* pos = positions.data_ptr<int64_t>();
  int32_t* seq = seq_lens.data_ptr<int32_t>();
  const int64_t* draft = draft_tokens.data_ptr<int64_t>();
  const int64_t* current_step = current_draft_step.data_ptr<int64_t>();
  const void* hidden = hidden_states.data_ptr();
  const int64_t element_size = hidden_states.element_size();

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::vllm_autoregressive_update_draft_inputs",
      ([
        output_tokens,
        next_hidden,
        ids,
        pos,
        seq,
        draft,
        current_step,
        hidden,
        output_draft_tokens_stride,
        next_input_hidden_states_stride,
        hidden_states_stride,
        hidden_size,
        element_size,
        num_reqs,
        max_model_len,
        num_speculative_steps,
        advance_draft_positions
      ]),
      {
        at::mcpu::KernelPointerMemoryGuard guard(
            {output_tokens,
             next_hidden,
             ids,
             pos,
             seq,
             draft,
             current_step,
             hidden});
        const int64_t step = current_step[0];
        for (int64_t req = 0; req < num_reqs; ++req) {
          const int64_t draft_token = draft[req];
          output_tokens[req * output_draft_tokens_stride + step] = draft_token;

          // Triton returns after recording the token on the final draft step.
          if (step >= num_speculative_steps - 1) {
            continue;
          }

          ids[req] = static_cast<int32_t>(draft_token);
          auto* dst = static_cast<uint8_t*>(next_hidden) +
              req * next_input_hidden_states_stride * element_size;
          const auto* src = static_cast<const uint8_t*>(hidden) +
              req * hidden_states_stride * element_size;
          for (int64_t i = 0; i < hidden_size * element_size; ++i) {
            dst[i] = src[i];
          }

          if (advance_draft_positions) {
            pos[req] = std::min<int64_t>(pos[req] + 1, max_model_len - 1);
            seq[req] = static_cast<int32_t>(std::min<int64_t>(
                static_cast<int64_t>(seq[req]) + 1, max_model_len));
          }
        }
      });
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_autoregressive_prepare_prefill_inputs("
      "Tensor(a!) last_token_indices, Tensor(b!) draft_current_step, "
      "Tensor(c!) draft_input_ids, Tensor(d!) draft_positions, "
      "Tensor(e!) draft_query_start_loc, Tensor(f!) draft_seq_lens, "
      "Tensor target_input_ids, Tensor target_positions, Tensor idx_mapping, "
      "Tensor last_sampled, Tensor next_prefill_tokens, Tensor num_sampled, "
      "Tensor num_rejected, Tensor query_start_loc, Tensor seq_lens, "
      "int max_num_reqs) -> ()");
  m.def(
      "vllm_autoregressive_prepare_decode_inputs("
      "Tensor draft_tokens, int draft_tokens_stride, Tensor target_seq_lens, "
      "Tensor num_rejected, Tensor(a!) input_ids, Tensor(b!) positions, "
      "Tensor(c!) query_start_loc, Tensor(d!) seq_lens, int max_model_len, "
      "int max_num_reqs, bool advance_draft_positions) -> ()");
  m.def(
      "vllm_autoregressive_update_draft_inputs("
      "Tensor(a!) output_draft_tokens, int output_draft_tokens_stride, "
      "Tensor(b!) next_input_hidden_states, "
      "int next_input_hidden_states_stride, Tensor(c!) input_ids, "
      "Tensor(d!) positions, Tensor(e!) seq_lens, Tensor draft_tokens, "
      "Tensor current_draft_step, Tensor hidden_states, "
      "int hidden_states_stride, int hidden_size, int max_model_len, "
      "int num_speculative_steps, bool advance_draft_positions) -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl(
      "vllm_autoregressive_prepare_prefill_inputs",
      &vllm_autoregressive_prepare_prefill_inputs_impl);
  m.impl(
      "vllm_autoregressive_prepare_decode_inputs",
      &vllm_autoregressive_prepare_decode_inputs_impl);
  m.impl(
      "vllm_autoregressive_update_draft_inputs",
      &vllm_autoregressive_update_draft_inputs_impl);
}
