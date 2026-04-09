// SPDX-License-Identifier: Apache-2.0
//
// C++ kernel for vllm/v1/worker/gpu/sample/prompt_logprob.py::_prompt_logprobs_token_ids_kernel
//
// For each request batch_idx and each query position i:
//   prompt_logprobs_token_ids[query_start + i] =
//       all_token_ids[req, num_computed + 1 + i]

#include "common.h"

namespace {

void vllm_prompt_logprobs_token_ids_impl(
    at::Tensor& prompt_logprobs_token_ids,  // [total_query_len], int64  (output)
    const at::Tensor& query_start_loc,      // [num_reqs+1], int32
    const at::Tensor& idx_mapping,          // [num_reqs], int32
    const at::Tensor& num_computed_tokens,  // [max_num_reqs], int32
    const at::Tensor& all_token_ids) {      // [max_num_reqs, max_seq_len], int32

  VLLM_MCPU_CHECK_DTYPE(prompt_logprobs_token_ids, at::kLong, "prompt_logprobs_token_ids");
  VLLM_MCPU_CHECK_DTYPE(query_start_loc, at::kInt, "query_start_loc");
  VLLM_MCPU_CHECK_DTYPE(idx_mapping, at::kInt, "idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(num_computed_tokens, at::kInt, "num_computed_tokens");
  VLLM_MCPU_CHECK_DTYPE(all_token_ids, at::kInt, "all_token_ids");

  int64_t num_reqs = idx_mapping.size(0);
  int64_t token_ids_stride = all_token_ids.stride(0);

  int64_t* out_ptr = prompt_logprobs_token_ids.data_ptr<int64_t>();
  const int32_t* qs_ptr = query_start_loc.data_ptr<int32_t>();
  const int32_t* idx_ptr = idx_mapping.data_ptr<int32_t>();
  const int32_t* ncomp_ptr = num_computed_tokens.data_ptr<int32_t>();
  const int32_t* tids_ptr = all_token_ids.data_ptr<int32_t>();

  for (int64_t batch_idx = 0; batch_idx < num_reqs; batch_idx++) {
    int32_t req = idx_ptr[batch_idx];
    int32_t query_start = qs_ptr[batch_idx];
    int32_t query_end = qs_ptr[batch_idx + 1];
    int32_t query_len = query_end - query_start;
    int32_t num_computed = ncomp_ptr[req];
    const int32_t* req_tokens = tids_ptr + (int64_t)req * token_ids_stride;

    for (int32_t i = 0; i < query_len; i++) {
      out_ptr[query_start + i] = (int64_t)req_tokens[num_computed + 1 + i];
    }
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_prompt_logprobs_token_ids("
      "Tensor(a!) prompt_logprobs_token_ids, "
      "Tensor query_start_loc, "
      "Tensor idx_mapping, "
      "Tensor num_computed_tokens, "
      "Tensor all_token_ids"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_prompt_logprobs_token_ids", &vllm_prompt_logprobs_token_ids_impl);
}
