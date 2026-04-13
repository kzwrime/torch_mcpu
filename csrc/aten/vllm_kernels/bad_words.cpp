// SPDX-License-Identifier: Apache-2.0
//
// C++ kernel for vllm/v1/worker/gpu/sample/bad_words.py::_bad_words_kernel
//
// For each token, check all bad-word sequences. If the context ends with the
// bad-word prefix, set that bad word's final token logit to -inf.

#include "common.h"

namespace {

template <typename scalar_t>
static void vllm_bad_words_kernel_typed(
    scalar_t* logits_ptr,
    int64_t num_tokens,
    int64_t logits_stride,
    int64_t vocab_size,
    const int32_t* idx_ptr,
    const int32_t* bw_ids_ptr,
    const int32_t* bw_off_ptr,
    int64_t bw_ids_stride,
    int64_t bw_off_stride,
    const int32_t* n_bw_ptr,
    const int32_t* atids_ptr,
    int64_t token_ids_stride,
    const int32_t* plen_ptr,
    const int32_t* tlen_ptr,
    const int32_t* input_ids_ptr,
    const int32_t* local_pos_ptr) {
  const scalar_t neg_inf =
      static_cast<scalar_t>(-std::numeric_limits<float>::infinity());

  for (int64_t tok = 0; tok < num_tokens; tok++) {
    int32_t req = idx_ptr[tok];
    int32_t n_bw = n_bw_ptr[req];
    if (n_bw == 0)
      continue;

    int32_t pos = local_pos_ptr[tok];
    int64_t cur_req_first = tok - pos;
    int32_t plen = plen_ptr[req];
    int32_t tlen = tlen_ptr[req];
    int32_t output_len = tlen - plen;
    int32_t effective_len = output_len + pos;

    const int32_t* off_row = bw_off_ptr + (int64_t)req * bw_off_stride;
    const int32_t* bw_row = bw_ids_ptr + (int64_t)req * bw_ids_stride;
    const int32_t* output_base =
        atids_ptr + (int64_t)req * token_ids_stride + plen;
    scalar_t* row = logits_ptr + tok * logits_stride;

    for (int32_t bw = 0; bw < n_bw; bw++) {
      int32_t start = off_row[bw];
      int32_t end = off_row[bw + 1];
      int32_t prefix_len = end - start - 1;

      if (prefix_len > effective_len)
        continue;

      int32_t last_token = bw_row[end - 1];

      bool match = true;
      for (int32_t i = 0; i < prefix_len && match; i++) {
        int32_t expected = bw_row[start + i];
        int32_t actual_pos = effective_len - prefix_len + i;
        int32_t actual;
        if (actual_pos >= output_len) {
          int32_t spec_off = actual_pos - output_len;
          actual = input_ids_ptr[cur_req_first + spec_off];
        } else {
          actual = output_base[actual_pos];
        }
        if (expected != actual)
          match = false;
      }

      if (match && last_token >= 0 && last_token < (int32_t)vocab_size) {
        row[last_token] = neg_inf;
      }
    }
  }
}

void vllm_bad_words_kernel_impl(
    at::Tensor& logits, // [num_tokens, vocab_size]
    const at::Tensor& expanded_idx_mapping, // [num_tokens], int32
    const at::Tensor& bad_word_token_ids, // [max_num_reqs, max_bw_toks], int32
    const at::Tensor&
        bad_word_offsets, // [max_num_reqs, max_bad_words+1], int32
    const at::Tensor& num_bad_words, // [max_num_reqs], int32
    const at::Tensor& all_token_ids, // [max_num_reqs, max_seq_len], int32
    const at::Tensor& prompt_len, // [max_num_reqs], int32
    const at::Tensor& total_len, // [max_num_reqs], int32
    const at::Tensor& input_ids, // [num_tokens], int32  (draft tokens)
    const at::Tensor& expanded_local_pos) { // [num_tokens], int32

  VLLM_MCPU_CHECK(
      logits.dim() == 2, "logits must be 2D, got ", logits.dim(), "D");
  VLLM_MCPU_CHECK(
      logits.scalar_type() == at::kFloat ||
          logits.scalar_type() == at::kBFloat16 ||
          logits.scalar_type() == at::kHalf,
      "logits must be float32, float16, or bfloat16");
  VLLM_MCPU_CHECK(
      expanded_idx_mapping.scalar_type() == at::kInt,
      "expanded_idx_mapping must be int32");
  VLLM_MCPU_CHECK(
      expanded_local_pos.scalar_type() == at::kInt,
      "expanded_local_pos must be int32");

  int64_t num_tokens = logits.size(0);
  int64_t logits_stride = logits.stride(0);
  int64_t bw_ids_stride = bad_word_token_ids.stride(0);
  int64_t bw_off_stride = bad_word_offsets.stride(0);
  int64_t token_ids_stride = all_token_ids.stride(0);
  int64_t vocab_size = logits.size(1);

  const int32_t* idx_ptr = expanded_idx_mapping.data_ptr<int32_t>();
  const int32_t* bw_ids_ptr = bad_word_token_ids.data_ptr<int32_t>();
  const int32_t* bw_off_ptr = bad_word_offsets.data_ptr<int32_t>();
  const int32_t* n_bw_ptr = num_bad_words.data_ptr<int32_t>();
  const int32_t* atids_ptr = all_token_ids.data_ptr<int32_t>();
  const int32_t* plen_ptr = prompt_len.data_ptr<int32_t>();
  const int32_t* tlen_ptr = total_len.data_ptr<int32_t>();
  const int32_t* input_ids_ptr = input_ids.data_ptr<int32_t>();
  const int32_t* local_pos_ptr = expanded_local_pos.data_ptr<int32_t>();

  VLLM_MCPU_DISPATCH_FLOAT(logits, "vllm_bad_words_kernel", {
    vllm_bad_words_kernel_typed<scalar_t>(
        logits.data_ptr<scalar_t>(),
        num_tokens,
        logits_stride,
        vocab_size,
        idx_ptr,
        bw_ids_ptr,
        bw_off_ptr,
        bw_ids_stride,
        bw_off_stride,
        n_bw_ptr,
        atids_ptr,
        token_ids_stride,
        plen_ptr,
        tlen_ptr,
        input_ids_ptr,
        local_pos_ptr);
  });
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_bad_words_kernel("
      "Tensor(a!) logits, "
      "Tensor expanded_idx_mapping, "
      "Tensor bad_word_token_ids, "
      "Tensor bad_word_offsets, "
      "Tensor num_bad_words, "
      "Tensor all_token_ids, "
      "Tensor prompt_len, "
      "Tensor total_len, "
      "Tensor input_ids, "
      "Tensor expanded_local_pos"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_bad_words_kernel", &vllm_bad_words_kernel_impl);
}
