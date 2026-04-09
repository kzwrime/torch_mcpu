// SPDX-License-Identifier: Apache-2.0
//
// C++ kernels for vllm/v1/worker/gpu/sample/penalties.py:
//   _penalties_kernel  — apply repetition / frequency / presence penalties
//   _bincount_kernel   — populate prompt_bin_mask and output_bin_counts

#include "common.h"

namespace {

// ---------------------------------------------------------------------------
// _penalties_kernel
// ---------------------------------------------------------------------------
template <typename scalar_t>
static void vllm_penalties_kernel_typed(
    scalar_t* logits_ptr,
    int64_t num_tokens,
    int64_t logits_stride,
    int64_t vocab_size,
    const int32_t* idx_ptr,
    const int32_t* tids_ptr,
    const int32_t* lpos_ptr,
    const float* rep_ptr,
    const float* freq_ptr,
    const float* pres_ptr,
    const int32_t* pbm_ptr,
    int64_t pbm_stride,
    const int32_t* obc_ptr,
    int64_t obc_stride,
    int64_t max_spec_len) {

  for (int64_t tok = 0; tok < num_tokens; tok++) {
    int32_t req = idx_ptr[tok];
    float rep_pen = rep_ptr[req];
    float freq_pen = freq_ptr[req];
    float pres_pen = pres_ptr[req];

    if (rep_pen == 1.0f && freq_pen == 0.0f && pres_pen == 0.0f) continue;

    scalar_t* row = logits_ptr + tok * logits_stride;
    const int32_t* obc_row = obc_ptr + (int64_t)req * obc_stride;
    const int32_t* pbm_row = pbm_ptr + (int64_t)req * pbm_stride;

    int32_t pos = lpos_ptr[tok];
    int64_t start_idx = tok - pos;

    for (int64_t v = 0; v < vocab_size; v++) {
      int32_t cnt = obc_row[v];

      if (pos > 0 && max_spec_len > 0) {
        for (int32_t prev = 0; prev < pos; prev++) {
          int32_t draft_tok = tids_ptr[start_idx + prev + 1];
          if (draft_tok == (int32_t)v) cnt++;
        }
      }

      bool in_output = (cnt > 0);
      float logit;
      if constexpr (std::is_same_v<scalar_t, float>) {
        logit = row[v];
      } else {
        logit = static_cast<float>(row[v]);
      }

      if (rep_pen != 1.0f) {
        int64_t word = v / 32;
        int bit = (int)(v % 32);
        bool in_prompt = (pbm_row[word] >> bit) & 1;
        if (in_prompt || in_output) {
          if (logit > 0.0f) logit = logit / rep_pen;
          else               logit = logit * rep_pen;
        }
      }

      logit -= freq_pen * (float)cnt;
      logit -= pres_pen * (in_output ? 1.0f : 0.0f);

      if constexpr (std::is_same_v<scalar_t, float>) {
        row[v] = logit;
      } else {
        row[v] = static_cast<scalar_t>(logit);
      }
    }
  }
}

void vllm_penalties_kernel_impl(
    at::Tensor& logits,                      // [num_tokens, vocab_size]
    const at::Tensor& expanded_idx_mapping,  // [num_tokens], int32
    const at::Tensor& token_ids,             // [num_tokens], int32
    const at::Tensor& expanded_local_pos,    // [num_tokens], int32
    const at::Tensor& repetition_penalty,    // [max_num_reqs], float32
    const at::Tensor& frequency_penalty,     // [max_num_reqs], float32
    const at::Tensor& presence_penalty,      // [max_num_reqs], float32
    const at::Tensor& prompt_bin_mask,       // [max_num_reqs, cdiv(vocab_size,32)], int32
    const at::Tensor& output_bin_counts,     // [max_num_reqs, vocab_size], int32
    int64_t vocab_size,
    int64_t max_spec_len) {

  VLLM_MCPU_CHECK_DIM(logits, 2, "logits");
  VLLM_MCPU_CHECK_FLOAT(logits, "logits");
  VLLM_MCPU_CHECK_DTYPE(expanded_idx_mapping, at::kInt, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(token_ids, at::kInt, "token_ids");
  VLLM_MCPU_CHECK_DTYPE(expanded_local_pos, at::kInt, "expanded_local_pos");
  VLLM_MCPU_CHECK_DTYPE(repetition_penalty, at::kFloat, "repetition_penalty");
  VLLM_MCPU_CHECK_DTYPE(frequency_penalty, at::kFloat, "frequency_penalty");
  VLLM_MCPU_CHECK_DTYPE(presence_penalty, at::kFloat, "presence_penalty");

  int64_t num_tokens = logits.size(0);
  int64_t logits_stride = logits.stride(0);
  int64_t pbm_stride = prompt_bin_mask.stride(0);
  int64_t obc_stride = output_bin_counts.stride(0);

  const int32_t* idx_ptr = expanded_idx_mapping.data_ptr<int32_t>();
  const int32_t* tids_ptr = token_ids.data_ptr<int32_t>();
  const int32_t* lpos_ptr = expanded_local_pos.data_ptr<int32_t>();
  const float* rep_ptr = repetition_penalty.data_ptr<float>();
  const float* freq_ptr = frequency_penalty.data_ptr<float>();
  const float* pres_ptr = presence_penalty.data_ptr<float>();
  const int32_t* pbm_ptr = prompt_bin_mask.data_ptr<int32_t>();
  const int32_t* obc_ptr = output_bin_counts.data_ptr<int32_t>();

  VLLM_MCPU_DISPATCH_FLOAT(logits, "vllm_penalties_kernel", {
    vllm_penalties_kernel_typed<scalar_t>(
        logits.data_ptr<scalar_t>(), num_tokens, logits_stride, vocab_size,
        idx_ptr, tids_ptr, lpos_ptr,
        rep_ptr, freq_ptr, pres_ptr,
        pbm_ptr, pbm_stride, obc_ptr, obc_stride, max_spec_len);
  });
}

// ---------------------------------------------------------------------------
// _bincount_kernel — no logits, no float dispatch needed
// ---------------------------------------------------------------------------
void vllm_bincount_kernel_impl(
    const at::Tensor& expanded_idx_mapping,  // [num_reqs], int32
    const at::Tensor& all_token_ids,         // [max_num_reqs, max_seq_len], int32
    const at::Tensor& prompt_len,            // [max_num_reqs], int32
    const at::Tensor& prefill_len,           // [max_num_reqs], int32
    at::Tensor& prompt_bin_mask,             // [max_num_reqs, packed_cols], int32 (output)
    at::Tensor& output_bin_counts) {         // [max_num_reqs, vocab_size], int32  (output)

  VLLM_MCPU_CHECK_DTYPE(expanded_idx_mapping, at::kInt, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(all_token_ids, at::kInt, "all_token_ids");
  VLLM_MCPU_CHECK_DTYPE(prompt_len, at::kInt, "prompt_len");
  VLLM_MCPU_CHECK_DTYPE(prefill_len, at::kInt, "prefill_len");
  VLLM_MCPU_CHECK_DTYPE(prompt_bin_mask, at::kInt, "prompt_bin_mask");
  VLLM_MCPU_CHECK_DTYPE(output_bin_counts, at::kInt, "output_bin_counts");

  int64_t num_reqs = expanded_idx_mapping.size(0);
  int64_t tids_stride = all_token_ids.stride(0);
  int64_t packed_cols = prompt_bin_mask.size(1);
  int64_t vocab_size = output_bin_counts.size(1);

  const int32_t* idx_ptr = expanded_idx_mapping.data_ptr<int32_t>();
  const int32_t* atids_ptr = all_token_ids.data_ptr<int32_t>();
  const int32_t* plen_ptr = prompt_len.data_ptr<int32_t>();
  const int32_t* flen_ptr = prefill_len.data_ptr<int32_t>();
  int32_t* pbm_ptr = prompt_bin_mask.data_ptr<int32_t>();
  int32_t* obc_ptr = output_bin_counts.data_ptr<int32_t>();
  int64_t pbm_stride = prompt_bin_mask.stride(0);
  int64_t obc_stride = output_bin_counts.stride(0);

  for (int64_t r = 0; r < num_reqs; r++) {
    int32_t req = idx_ptr[r];
    int32_t plen = plen_ptr[req];
    int32_t flen = flen_ptr[req];
    const int32_t* tids = atids_ptr + (int64_t)req * tids_stride;
    int32_t* pbm_row = pbm_ptr + (int64_t)req * pbm_stride;
    int32_t* obc_row = obc_ptr + (int64_t)req * obc_stride;

    for (int32_t pos = 0; pos < plen; pos++) {
      int32_t tid = tids[pos];
      int64_t word = tid / 32;
      int bit = tid % 32;
      if (word >= 0 && word < packed_cols) pbm_row[word] |= (1 << bit);
    }

    for (int32_t pos = plen; pos < flen; pos++) {
      int32_t tid = tids[pos];
      if (tid >= 0 && tid < (int32_t)vocab_size) obc_row[tid]++;
    }
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_penalties_kernel("
      "Tensor(a!) logits, "
      "Tensor expanded_idx_mapping, "
      "Tensor token_ids, "
      "Tensor expanded_local_pos, "
      "Tensor repetition_penalty, "
      "Tensor frequency_penalty, "
      "Tensor presence_penalty, "
      "Tensor prompt_bin_mask, "
      "Tensor output_bin_counts, "
      "int vocab_size, "
      "int max_spec_len"
      ") -> ()");
  m.def(
      "vllm_bincount_kernel("
      "Tensor expanded_idx_mapping, "
      "Tensor all_token_ids, "
      "Tensor prompt_len, "
      "Tensor prefill_len, "
      "Tensor(a!) prompt_bin_mask, "
      "Tensor(b!) output_bin_counts"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_penalties_kernel", &vllm_penalties_kernel_impl);
  m.impl("vllm_bincount_kernel", &vllm_bincount_kernel_impl);
}
