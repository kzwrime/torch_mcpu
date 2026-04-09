// SPDX-License-Identifier: Apache-2.0
//
// C++ kernel for vllm/v1/worker/gpu/sample/min_p.py::_min_p_kernel
//
// For each token: threshold = max_logit + log(min_p).
// All logits below threshold are set to -inf.

#include "common.h"

namespace {

template <typename scalar_t>
static void vllm_min_p_kernel_typed(
    scalar_t* logits_ptr,
    int64_t num_tokens,
    int64_t logits_stride,
    const int32_t* idx_ptr,
    const float* min_p_ptr,
    int64_t vocab_size) {

  const scalar_t neg_inf = static_cast<scalar_t>(-std::numeric_limits<float>::infinity());

  for (int64_t tok = 0; tok < num_tokens; tok++) {
    int32_t req_idx = idx_ptr[tok];
    float min_p_val = min_p_ptr[req_idx];
    if (min_p_val == 0.0f) continue;

    scalar_t* row = logits_ptr + tok * logits_stride;

    // Find max logit.
    float max_val = -std::numeric_limits<float>::infinity();
    if constexpr (std::is_same_v<scalar_t, float>) {
      for (int64_t i = 0; i < vocab_size; i++) {
        if (row[i] > max_val) max_val = row[i];
      }
    } else {
      for (int64_t i = 0; i < vocab_size; i++) {
        float v = static_cast<float>(row[i]);
        if (v > max_val) max_val = v;
      }
    }

    float threshold = max_val + std::log(min_p_val);
    if constexpr (std::is_same_v<scalar_t, float>) {
      for (int64_t i = 0; i < vocab_size; i++) {
        if (row[i] < threshold) row[i] = neg_inf;
      }
    } else {
      for (int64_t i = 0; i < vocab_size; i++) {
        if (static_cast<float>(row[i]) < threshold) row[i] = neg_inf;
      }
    }
  }
}

void vllm_min_p_kernel_impl(
    at::Tensor& logits,                      // [num_tokens, vocab_size]
    const at::Tensor& expanded_idx_mapping,  // [num_tokens], int32
    const at::Tensor& min_p,                 // [max_num_reqs], float32
    int64_t vocab_size) {

  VLLM_MCPU_CHECK_DIM(logits, 2, "logits");
  VLLM_MCPU_CHECK_FLOAT(logits, "logits");
  VLLM_MCPU_CHECK_DIM(expanded_idx_mapping, 1, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(expanded_idx_mapping, at::kInt, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DIM(min_p, 1, "min_p");
  VLLM_MCPU_CHECK_DTYPE(min_p, at::kFloat, "min_p");
  VLLM_MCPU_CHECK(vocab_size > 0 && vocab_size <= logits.size(1),
      "vocab_size ", vocab_size, " out of range [1, ", logits.size(1), "]");

  int64_t num_tokens = logits.size(0);
  int64_t logits_stride = logits.stride(0);
  const int32_t* idx_ptr = expanded_idx_mapping.data_ptr<int32_t>();
  const float* min_p_ptr = min_p.data_ptr<float>();

  VLLM_MCPU_DISPATCH_FLOAT(logits, "vllm_min_p_kernel", {
    vllm_min_p_kernel_typed<scalar_t>(
        logits.data_ptr<scalar_t>(), num_tokens, logits_stride,
        idx_ptr, min_p_ptr, vocab_size);
  });
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_min_p_kernel("
      "Tensor(a!) logits, "
      "Tensor expanded_idx_mapping, "
      "Tensor min_p, "
      "int vocab_size"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_min_p_kernel", &vllm_min_p_kernel_impl);
}
