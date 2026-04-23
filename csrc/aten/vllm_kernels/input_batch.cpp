// SPDX-License-Identifier: Apache-2.0
//
// C++ kernels for vllm/v1/worker/gpu/input_batch.py Triton kernels:
//   prepare_prefill_inputs
//   prepare_pos_seq_lens
//   combine_sampled_and_draft_tokens
//   get_num_sampled_and_rejected        (returns (num_sampled, num_rejected))
//   post_update
//   post_update_pool
//   expand_idx_mapping                  (returns (expanded_idx, local_pos))

#include <ATen/ATen.h>
#include <torch/library.h>
#include <cstdint>
#include "common.h"

namespace {

// ---------------------------------------------------------------------------
// prepare_prefill_inputs
// input_ids[qstart:qend] = all_token_ids[req, ncomp:ncomp+qlen]
// if ncomp+qlen < plen: next_prefill_tokens[req] = all_token_ids[req,
// ncomp+qlen]
// ---------------------------------------------------------------------------
void vllm_prepare_prefill_inputs_impl(
    at::Tensor& input_ids, // [max_num_tokens], int32
    at::Tensor& next_prefill_tokens, // [max_num_reqs], int32
    const at::Tensor& idx_mapping, // [num_reqs], int32
    const at::Tensor& query_start_loc, // [num_reqs+1], int32
    const at::Tensor& all_token_ids, // [max_num_reqs, max_seq_len], int32
    const at::Tensor& prefill_len, // [max_num_reqs], int32
    const at::Tensor& num_computed_tokens) { // [max_num_reqs], int32

  int64_t num_reqs = idx_mapping.size(0);
  int64_t atids_stride = all_token_ids.stride(0);

  int32_t* iids_ptr = input_ids.data_ptr<int32_t>();
  int32_t* npt_ptr = next_prefill_tokens.data_ptr<int32_t>();
  const int32_t* idx_ptr = idx_mapping.data_ptr<int32_t>();
  const int32_t* qs_ptr = query_start_loc.data_ptr<int32_t>();
  const int32_t* atids_ptr = all_token_ids.data_ptr<int32_t>();
  const int32_t* plen_ptr = prefill_len.data_ptr<int32_t>();
  const int32_t* ncomp_ptr = num_computed_tokens.data_ptr<int32_t>();

  for (int64_t r = 0; r < num_reqs; r++) {
    int32_t req = idx_ptr[r];
    int32_t plen = plen_ptr[req];
    int32_t ncomp = ncomp_ptr[req];
    if (ncomp >= plen)
      continue;

    int32_t qstart = qs_ptr[r];
    int32_t qend = qs_ptr[r + 1];
    int32_t qlen = qend - qstart;
    const int32_t* req_tids = atids_ptr + (int64_t)req * atids_stride;

    for (int32_t i = 0; i < qlen; i++) {
      iids_ptr[qstart + i] = req_tids[ncomp + i];
    }
    int32_t next_pos = ncomp + qlen;
    if (next_pos < plen) {
      npt_ptr[req] = req_tids[next_pos];
    }
  }
}

// ---------------------------------------------------------------------------
// prepare_pos_seq_lens
// seq_lens[r] = ncomp + qlen
// pos[qstart:qend] = [ncomp, ncomp+1, ..., ncomp+qlen-1]
// Pads seq_lens[num_reqs:max_num_reqs] = 0
// ---------------------------------------------------------------------------
void vllm_prepare_pos_seq_lens_impl(
    const at::Tensor& idx_mapping, // [num_reqs], int32
    const at::Tensor& query_start_loc, // [num_reqs+1], int32
    const at::Tensor& num_computed_tokens, // [max_num_reqs], int32
    at::Tensor& pos, // [max_num_tokens], int64
    at::Tensor& seq_lens) { // [max_num_reqs], int32

  int64_t num_reqs = idx_mapping.size(0);
  int64_t max_num_reqs = seq_lens.size(0);

  const int32_t* idx_ptr = idx_mapping.data_ptr<int32_t>();
  const int32_t* qs_ptr = query_start_loc.data_ptr<int32_t>();
  const int32_t* ncomp_ptr = num_computed_tokens.data_ptr<int32_t>();
  int64_t* pos_ptr = pos.data_ptr<int64_t>();
  int32_t* sl_ptr = seq_lens.data_ptr<int32_t>();

  for (int64_t r = 0; r < num_reqs; r++) {
    int32_t req = idx_ptr[r];
    int32_t ncomp = ncomp_ptr[req];
    int32_t qstart = qs_ptr[r];
    int32_t qend = qs_ptr[r + 1];
    int32_t qlen = qend - qstart;
    sl_ptr[r] = ncomp + qlen;
    for (int32_t i = 0; i < qlen; i++) {
      pos_ptr[qstart + i] = (int64_t)(ncomp + i);
    }
  }
  // Pad unused entries for CUDA-graph compatibility
  for (int64_t r = num_reqs; r < max_num_reqs; r++) {
    sl_ptr[r] = 0;
  }
}

// ---------------------------------------------------------------------------
// combine_sampled_and_draft_tokens  -> logits_indices [num_logits] is filled
// in-place
// ---------------------------------------------------------------------------
void vllm_combine_sampled_and_draft_tokens_impl(
    at::Tensor& input_ids, // [max_num_tokens], int32
    const at::Tensor& idx_mapping, // [num_reqs], int32
    const at::Tensor& last_sampled_tokens, // [max_num_reqs], int32
    const at::Tensor& query_start_loc, // [num_reqs+1], int32
    const at::Tensor& seq_lens, // [num_reqs], int32
    const at::Tensor& prefill_len, // [max_num_reqs], int32
    const at::Tensor& draft_tokens, // [max_num_reqs, max_draft], int32
    const at::Tensor& cu_num_logits, // [num_reqs+1], int32
    int64_t num_logits,
    at::Tensor& logits_indices) {
  int64_t num_reqs = idx_mapping.size(0);

  VLLM_MCPU_CHECK(
      logits_indices.numel() == num_logits,
      "logits_indices must have num_logits elements, got ",
      logits_indices.numel(),
      ", expected ",
      num_logits);
  VLLM_MCPU_CHECK_DIM(logits_indices, 1, "logits_indices");
  VLLM_MCPU_CHECK(
      logits_indices.scalar_type() == at::kLong,
      "logits_indices must be int64, got ",
      logits_indices.scalar_type());

  int32_t* iids_ptr = input_ids.data_ptr<int32_t>();
  const int32_t* idx_ptr = idx_mapping.data_ptr<int32_t>();
  // last_sampled_tokens and draft_tokens are int64 (token IDs from sampler)
  const int64_t* lst_ptr = last_sampled_tokens.data_ptr<int64_t>();
  const int32_t* qs_ptr = query_start_loc.data_ptr<int32_t>();
  const int32_t* sl_ptr = seq_lens.data_ptr<int32_t>();
  const int32_t* plen_ptr = prefill_len.data_ptr<int32_t>();
  const int64_t* draft_ptr = draft_tokens.data_ptr<int64_t>();
  const int32_t* cunl_ptr = cu_num_logits.data_ptr<int32_t>();
  int64_t draft_stride = draft_tokens.stride(0);

  int64_t* li_ptr = logits_indices.data_ptr<int64_t>();

  for (int64_t r = 0; r < num_reqs; r++) {
    int32_t req = idx_ptr[r];
    int32_t ls = cunl_ptr[r];
    int32_t le = cunl_ptr[r + 1];
    int32_t n_logits = le - ls;
    int32_t n_draft = n_logits - 1;
    int32_t qend = qs_ptr[r + 1];
    int32_t pos_start = qend - n_logits;

    for (int32_t i = 0; i < n_logits; i++) {
      li_ptr[ls + i] = (int64_t)(pos_start + i);
    }

    int32_t seq_len = sl_ptr[r];
    int32_t plen = plen_ptr[req];
    if (seq_len <= plen)
      continue; // chunked prefill

    iids_ptr[qend - n_logits] = (int32_t)lst_ptr[req];
    if (n_draft > 0) {
      const int64_t* dr = draft_ptr + (int64_t)req * draft_stride;
      for (int32_t i = 0; i < n_draft; i++) {
        iids_ptr[qend - n_draft + i] = (int32_t)dr[i];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// get_num_sampled_and_rejected  → (num_sampled, num_rejected)
// ---------------------------------------------------------------------------
void vllm_get_num_sampled_and_rejected_impl(
    at::Tensor& num_sampled, // [num_reqs], int32  (in/out)
    at::Tensor& num_rejected, // [num_reqs], int32 (out)
    const at::Tensor& seq_lens, // [num_reqs], int32
    const at::Tensor& cu_num_logits, // [num_reqs+1], int32
    const at::Tensor& idx_mapping, // [num_reqs], int32
    const at::Tensor& prefill_len) { // [max_num_reqs], int32

  int64_t num_reqs = idx_mapping.size(0);
  VLLM_MCPU_CHECK_DIM(num_rejected, 1, "num_rejected");
  VLLM_MCPU_CHECK_DTYPE(num_rejected, at::kInt, "num_rejected");
  VLLM_MCPU_CHECK(
      num_rejected.size(0) == num_sampled.size(0),
      "num_rejected must have ",
      num_sampled.size(0),
      " elements, got ",
      num_rejected.size(0));

  int32_t* ns_ptr = num_sampled.data_ptr<int32_t>();
  int32_t* nr_ptr = num_rejected.data_ptr<int32_t>();
  const int32_t* sl_ptr = seq_lens.data_ptr<int32_t>();
  const int32_t* cunl_ptr = cu_num_logits.data_ptr<int32_t>();
  const int32_t* idx_ptr = idx_mapping.data_ptr<int32_t>();
  const int32_t* plen_ptr = prefill_len.data_ptr<int32_t>();

  for (int64_t r = 0; r < num_reqs; r++) {
    int32_t req = idx_ptr[r];
    int32_t sl = sl_ptr[r];
    int32_t plen = plen_ptr[req];
    bool is_chunked = (sl < plen);
    int32_t n_logits = cunl_ptr[r + 1] - cunl_ptr[r];
    if (is_chunked) {
      ns_ptr[r] = 0;
      nr_ptr[r] = 0;
    } else {
      nr_ptr[r] = n_logits - ns_ptr[r];
    }
  }
}

// ---------------------------------------------------------------------------
// post_update
// ---------------------------------------------------------------------------
void vllm_post_update_impl(
    const at::Tensor& idx_mapping, // [num_reqs], int32
    at::Tensor& num_computed_tokens, // [max_num_reqs], int32
    at::Tensor& last_sampled_tokens, // [max_num_reqs, 1], int64
    const std::optional<at::Tensor>&
        output_bin_counts, // [max_num_reqs, vocab_size] or nullopt
    const at::Tensor& sampled_tokens, // [num_reqs, max_spec+1], int64
    const at::Tensor& num_sampled, // [num_reqs], int32
    const at::Tensor& num_rejected, // [num_reqs], int32
    const at::Tensor& query_start_loc, // [num_reqs+1], int32
    at::Tensor& all_token_ids, // [max_num_reqs, max_seq_len], int32
    at::Tensor& total_len) { // [max_num_reqs], int32

  int64_t num_reqs = idx_mapping.size(0);
  int64_t atids_stride = all_token_ids.stride(0);
  int64_t st_stride = sampled_tokens.stride(0);
  // last_sampled_tokens has shape [max_num_reqs, 1]; stride to offset into it
  int64_t lst_stride = last_sampled_tokens.stride(0);

  const int32_t* idx_ptr = idx_mapping.data_ptr<int32_t>();
  int32_t* ncomp_ptr = num_computed_tokens.data_ptr<int32_t>();
  // last_sampled_tokens and sampled_tokens are int64 (token IDs from sampler)
  int64_t* lst_ptr = last_sampled_tokens.data_ptr<int64_t>();
  const int64_t* st_ptr = sampled_tokens.data_ptr<int64_t>();
  const int32_t* ns_ptr = num_sampled.data_ptr<int32_t>();
  const int32_t* nr_ptr = num_rejected.data_ptr<int32_t>();
  const int32_t* qs_ptr = query_start_loc.data_ptr<int32_t>();
  int32_t* atids_ptr =
      all_token_ids.data_ptr<int32_t>(); // all_token_ids is int32
  int32_t* tlen_ptr = total_len.data_ptr<int32_t>();

  int32_t* obc_ptr = nullptr;
  int64_t obc_stride = 0;
  if (output_bin_counts.has_value() && output_bin_counts->defined()) {
    obc_ptr = output_bin_counts->data_ptr<int32_t>();
    obc_stride = output_bin_counts->stride(0);
  }
  int64_t obc_vocab =
      output_bin_counts.has_value() && output_bin_counts->defined()
      ? output_bin_counts->size(1)
      : 0;

  for (int64_t r = 0; r < num_reqs; r++) {
    int32_t req = idx_ptr[r];
    int32_t n = ns_ptr[r];
    int32_t tlen = tlen_ptr[req];

    if (n > 0) {
      const int64_t* st_row = st_ptr + r * st_stride;
      lst_ptr[(int64_t)req * lst_stride] = st_row[n - 1];
      tlen_ptr[req] = tlen + n;
      int32_t* dst = atids_ptr + (int64_t)req * atids_stride + tlen;
      for (int32_t i = 0; i < n; i++) {
        dst[i] = (int32_t)st_row[i]; // cast int64 → int32 for all_token_ids
        if (obc_ptr != nullptr) {
          int32_t tok = (int32_t)st_row[i];
          if (tok >= 0 && tok < (int32_t)obc_vocab) {
            obc_ptr[(int64_t)req * obc_stride + tok]++;
          }
        }
      }
    }

    int32_t qstart = qs_ptr[r];
    int32_t qend = qs_ptr[r + 1];
    int32_t qlen = qend - qstart;
    int32_t nr = nr_ptr[r];
    ncomp_ptr[req] = ncomp_ptr[req] + qlen - nr;
  }
}

// ---------------------------------------------------------------------------
// post_update_pool
// ---------------------------------------------------------------------------
void vllm_post_update_pool_impl(
    const at::Tensor& idx_mapping, // [num_reqs], int32
    at::Tensor& num_computed_tokens, // [max_num_reqs], int32
    const at::Tensor& query_start_loc) { // [num_reqs+1], int32

  int64_t num_reqs = idx_mapping.size(0);
  const int32_t* idx_ptr = idx_mapping.data_ptr<int32_t>();
  int32_t* ncomp_ptr = num_computed_tokens.data_ptr<int32_t>();
  const int32_t* qs_ptr = query_start_loc.data_ptr<int32_t>();

  for (int64_t r = 0; r < num_reqs; r++) {
    int32_t req = idx_ptr[r];
    int32_t qlen = qs_ptr[r + 1] - qs_ptr[r];
    ncomp_ptr[req] += qlen;
  }
}

// ---------------------------------------------------------------------------
// expand_idx_mapping  → (expanded_idx_mapping, expanded_local_pos)
// ---------------------------------------------------------------------------
void vllm_expand_idx_mapping_impl(
    const at::Tensor& idx_mapping, // [num_reqs], int32
    at::Tensor& expanded_idx_mapping, // [total_num_logits], int32
    at::Tensor& expanded_local_pos, // [total_num_logits], int32
    int64_t total_num_logits,
    const at::Tensor& cu_num_logits, // [num_reqs+1], int32
    int64_t /*max_expand_len*/) {
  int64_t num_reqs = idx_mapping.size(0);
  VLLM_MCPU_CHECK_DIM(expanded_idx_mapping, 1, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(expanded_idx_mapping, at::kInt, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DIM(expanded_local_pos, 1, "expanded_local_pos");
  VLLM_MCPU_CHECK_DTYPE(expanded_local_pos, at::kInt, "expanded_local_pos");
  VLLM_MCPU_CHECK(
      expanded_idx_mapping.size(0) == total_num_logits,
      "expanded_idx_mapping must have ",
      total_num_logits,
      " elements, got ",
      expanded_idx_mapping.size(0));
  VLLM_MCPU_CHECK(
      expanded_local_pos.size(0) == total_num_logits,
      "expanded_local_pos must have ",
      total_num_logits,
      " elements, got ",
      expanded_local_pos.size(0));

  const int32_t* idx_ptr = idx_mapping.data_ptr<int32_t>();
  const int32_t* cunl_ptr = cu_num_logits.data_ptr<int32_t>();
  int32_t* exp_ptr = expanded_idx_mapping.data_ptr<int32_t>();
  int32_t* lp_ptr = expanded_local_pos.data_ptr<int32_t>();

  for (int64_t r = 0; r < num_reqs; r++) {
    int32_t start = cunl_ptr[r];
    int32_t end = cunl_ptr[r + 1];
    int32_t n = end - start;
    int32_t req = idx_ptr[r];
    for (int32_t i = 0; i < n; i++) {
      exp_ptr[start + i] = req;
      lp_ptr[start + i] = i;
    }
  }
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_prepare_prefill_inputs("
      "Tensor(a!) input_ids, "
      "Tensor(b!) next_prefill_tokens, "
      "Tensor idx_mapping, "
      "Tensor query_start_loc, "
      "Tensor all_token_ids, "
      "Tensor prefill_len, "
      "Tensor num_computed_tokens"
      ") -> ()");
  m.def(
      "vllm_prepare_pos_seq_lens("
      "Tensor idx_mapping, "
      "Tensor query_start_loc, "
      "Tensor num_computed_tokens, "
      "Tensor(a!) pos, "
      "Tensor(b!) seq_lens"
      ") -> ()");
  m.def(
      "vllm_combine_sampled_and_draft_tokens("
      "Tensor(a!) input_ids, "
      "Tensor idx_mapping, "
      "Tensor last_sampled_tokens, "
      "Tensor query_start_loc, "
      "Tensor seq_lens, "
      "Tensor prefill_len, "
      "Tensor draft_tokens, "
      "Tensor cu_num_logits, "
      "int num_logits, "
      "Tensor(a!) logits_indices"
      ") -> ()");
  m.def(
      "vllm_get_num_sampled_and_rejected("
      "Tensor(a!) num_sampled, "
      "Tensor(b!) num_rejected, "
      "Tensor seq_lens, "
      "Tensor cu_num_logits, "
      "Tensor idx_mapping, "
      "Tensor prefill_len"
      ") -> ()");
  m.def(
      "vllm_post_update("
      "Tensor idx_mapping, "
      "Tensor(a!) num_computed_tokens, "
      "Tensor(b!) last_sampled_tokens, "
      "Tensor? output_bin_counts, "
      "Tensor sampled_tokens, "
      "Tensor num_sampled, "
      "Tensor num_rejected, "
      "Tensor query_start_loc, "
      "Tensor(c!) all_token_ids, "
      "Tensor(d!) total_len"
      ") -> ()");
  m.def(
      "vllm_post_update_pool("
      "Tensor idx_mapping, "
      "Tensor(a!) num_computed_tokens, "
      "Tensor query_start_loc"
      ") -> ()");
  m.def(
      "vllm_expand_idx_mapping("
      "Tensor idx_mapping, "
      "Tensor(a!) expanded_idx_mapping, "
      "Tensor(b!) expanded_local_pos, "
      "int total_num_logits, "
      "Tensor cu_num_logits, "
      "int max_expand_len"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_prepare_prefill_inputs", &vllm_prepare_prefill_inputs_impl);
  m.impl("vllm_prepare_pos_seq_lens", &vllm_prepare_pos_seq_lens_impl);
  m.impl(
      "vllm_combine_sampled_and_draft_tokens",
      &vllm_combine_sampled_and_draft_tokens_impl);
  m.impl(
      "vllm_get_num_sampled_and_rejected",
      &vllm_get_num_sampled_and_rejected_impl);
  m.impl("vllm_post_update", &vllm_post_update_impl);
  m.impl("vllm_post_update_pool", &vllm_post_update_pool_impl);
  m.impl("vllm_expand_idx_mapping", &vllm_expand_idx_mapping_impl);
}
