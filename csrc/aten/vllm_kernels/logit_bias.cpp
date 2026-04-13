// SPDX-License-Identifier: Apache-2.0
//
// C++ kernel for vllm/v1/worker/gpu/sample/logit_bias.py::_bias_kernel
//
// For each token:
//   1. If num_allowed > 0: set all logits to -inf, then restore allowed ones.
//   2. Add logit bias values for specified token IDs.
//   3. If pos < min_len: set stop token IDs to -inf.

#include "common.h"

namespace {

template <typename scalar_t>
static void vllm_bias_kernel_typed(
    scalar_t* logits_ptr,
    int64_t num_tokens,
    int64_t logits_stride,
    int64_t vocab_size,
    const int32_t* idx_ptr,
    const int32_t* n_allowed_ptr,
    const int32_t* allowed_ptr,
    int64_t allowed_stride,
    const int32_t* n_bias_ptr,
    const int32_t* bias_ids_ptr,
    int64_t bias_ids_stride,
    const float* bias_ptr,
    const int64_t* pos_ptr,
    const int32_t* min_lens_ptr,
    const int32_t* n_stop_ptr,
    const int32_t* stop_ptr,
    int64_t stop_stride) {
  const scalar_t neg_inf =
      static_cast<scalar_t>(-std::numeric_limits<float>::infinity());

  for (int64_t tok = 0; tok < num_tokens; tok++) {
    int32_t req = idx_ptr[tok];
    scalar_t* row = logits_ptr + tok * logits_stride;

    // 1. Allowed token IDs
    int32_t n_allowed = n_allowed_ptr[req];
    if (n_allowed > 0) {
      const int32_t* allowed_row = allowed_ptr + (int64_t)req * allowed_stride;
      scalar_t saved[512];
      int32_t saved_ids[512];
      int32_t n_save = (n_allowed <= 512) ? n_allowed : 512;
      for (int32_t k = 0; k < n_save; k++) {
        int32_t id = allowed_row[k];
        if (id < 0)
          id = 0;
        if (id >= (int32_t)vocab_size)
          id = (int32_t)vocab_size - 1;
        saved_ids[k] = id;
        saved[k] = row[id];
      }
      for (int64_t i = 0; i < vocab_size; i++)
        row[i] = neg_inf;
      for (int32_t k = 0; k < n_save; k++)
        row[saved_ids[k]] = saved[k];
    }

    // 2. Logit bias
    int32_t n_bias = n_bias_ptr[req];
    if (n_bias > 0) {
      const int32_t* bid_row = bias_ids_ptr + (int64_t)req * bias_ids_stride;
      const float* bval_row = bias_ptr + (int64_t)req * bias_ids_stride;
      for (int32_t k = 0; k < n_bias; k++) {
        int32_t id = bid_row[k];
        if (id < 0 || id >= (int32_t)vocab_size)
          continue;
        if constexpr (std::is_same_v<scalar_t, float>) {
          row[id] += bval_row[k];
        } else {
          row[id] =
              static_cast<scalar_t>(static_cast<float>(row[id]) + bval_row[k]);
        }
      }
    }

    // 3. Min tokens: mask stop tokens if pos < min_len
    int32_t n_stop = n_stop_ptr[req];
    if (n_stop > 0) {
      int64_t cur_pos = pos_ptr[tok];
      int32_t min_len = min_lens_ptr[req];
      if (cur_pos < min_len) {
        const int32_t* stop_row = stop_ptr + (int64_t)req * stop_stride;
        for (int32_t k = 0; k < n_stop; k++) {
          int32_t id = stop_row[k];
          if (id < 0 || id >= (int32_t)vocab_size)
            continue;
          row[id] = neg_inf;
        }
      }
    }
  }
}

void vllm_bias_kernel_impl(
    at::Tensor& logits, // [num_tokens, vocab_size]
    int64_t vocab_size,
    const at::Tensor& expanded_idx_mapping, // [num_tokens], int32
    const at::Tensor& num_allowed_token_ids, // [max_num_reqs], int32
    const at::Tensor& allowed_token_ids, // [max_num_reqs, max_allowed], int32
    const at::Tensor& num_logit_bias, // [max_num_reqs], int32
    const at::Tensor& bias_token_ids, // [max_num_reqs, max_bias], int32
    const at::Tensor& bias, // [max_num_reqs, max_bias], float32
    const at::Tensor& pos, // [num_tokens], int32
    const at::Tensor& min_lens, // [max_num_reqs], int32
    const at::Tensor& num_stop_token_ids, // [max_num_reqs], int32
    const at::Tensor& stop_token_ids) { // [max_num_reqs, max_stop], int32

  VLLM_MCPU_CHECK_DIM(logits, 2, "logits");
  VLLM_MCPU_CHECK_FLOAT(logits, "logits");
  VLLM_MCPU_CHECK_DIM(expanded_idx_mapping, 1, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(expanded_idx_mapping, at::kInt, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(bias, at::kFloat, "bias");
  VLLM_MCPU_CHECK_DTYPE(pos, at::kLong, "pos");

  int64_t num_tokens = logits.size(0);
  int64_t logits_stride = logits.stride(0);
  int64_t allowed_stride = allowed_token_ids.stride(0);
  int64_t bias_ids_stride = bias_token_ids.stride(0);
  int64_t stop_stride = stop_token_ids.stride(0);

  const int32_t* idx_ptr = expanded_idx_mapping.data_ptr<int32_t>();
  const int32_t* n_allowed_ptr = num_allowed_token_ids.data_ptr<int32_t>();
  const int32_t* allowed_ptr = allowed_token_ids.data_ptr<int32_t>();
  const int32_t* n_bias_ptr = num_logit_bias.data_ptr<int32_t>();
  const int32_t* bias_ids_ptr = bias_token_ids.data_ptr<int32_t>();
  const float* bias_ptr = bias.data_ptr<float>();
  const int64_t* pos_ptr = pos.data_ptr<int64_t>();
  const int32_t* min_lens_ptr = min_lens.data_ptr<int32_t>();
  const int32_t* n_stop_ptr = num_stop_token_ids.data_ptr<int32_t>();
  const int32_t* stop_ptr = stop_token_ids.data_ptr<int32_t>();

  VLLM_MCPU_DISPATCH_FLOAT(logits, "vllm_bias_kernel", {
    vllm_bias_kernel_typed<scalar_t>(
        logits.data_ptr<scalar_t>(),
        num_tokens,
        logits_stride,
        vocab_size,
        idx_ptr,
        n_allowed_ptr,
        allowed_ptr,
        allowed_stride,
        n_bias_ptr,
        bias_ids_ptr,
        bias_ids_stride,
        bias_ptr,
        pos_ptr,
        min_lens_ptr,
        n_stop_ptr,
        stop_ptr,
        stop_stride);
  });
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_bias_kernel("
      "Tensor(a!) logits, "
      "int vocab_size, "
      "Tensor expanded_idx_mapping, "
      "Tensor num_allowed_token_ids, "
      "Tensor allowed_token_ids, "
      "Tensor num_logit_bias, "
      "Tensor bias_token_ids, "
      "Tensor bias, "
      "Tensor pos, "
      "Tensor min_lens, "
      "Tensor num_stop_token_ids, "
      "Tensor stop_token_ids"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_bias_kernel", &vllm_bias_kernel_impl);
}
