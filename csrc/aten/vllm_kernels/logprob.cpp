// SPDX-License-Identifier: Apache-2.0
//
// C++ kernels for vllm/v1/worker/gpu/sample/logprob.py:
//   _topk_log_softmax_kernel  — log-softmax for specific token IDs
//   _ranks_kernel             — rank of sampled token in the logit distribution

#include "common.h"

namespace {

// ---------------------------------------------------------------------------
// _topk_log_softmax_kernel
// output[req, k] = logits[req, topk_ids[req, k]] - log(sum_exp(logits[req]))
// ---------------------------------------------------------------------------
template <typename scalar_t>
static void vllm_topk_log_softmax_kernel_typed(
    float* out_ptr,
    const scalar_t* logits_ptr,
    int64_t batch,
    int64_t logits_stride,
    const int64_t* ids_ptr,
    int64_t topk_ids_stride,
    int64_t topk,
    int64_t vocab_size) {

  for (int64_t req = 0; req < batch; req++) {
    const scalar_t* row = logits_ptr + req * logits_stride;
    const int64_t* ids_row = ids_ptr + req * topk_ids_stride;
    float* out_row = out_ptr + req * topk;

    // log-sum-exp with max subtraction for numerical stability
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

    double se = 0.0;
    if constexpr (std::is_same_v<scalar_t, float>) {
      for (int64_t i = 0; i < vocab_size; i++) {
        se += std::exp((double)(row[i] - max_val));
      }
    } else {
      for (int64_t i = 0; i < vocab_size; i++) {
        se += std::exp((double)(static_cast<float>(row[i]) - max_val));
      }
    }
    float lse = max_val + (float)std::log(se);

    for (int64_t k = 0; k < topk; k++) {
      int64_t tid = ids_row[k];
      if (tid < 0) tid = 0;
      if (tid >= vocab_size) tid = vocab_size - 1;
      if constexpr (std::is_same_v<scalar_t, float>) {
        out_row[k] = row[tid] - lse;
      } else {
        out_row[k] = static_cast<float>(row[tid]) - lse;
      }
    }
  }
}

void vllm_topk_log_softmax_kernel_impl(
    at::Tensor& output,            // [batch, topk], float32  (output)
    const at::Tensor& logits,      // [batch, vocab_size], any float
    const at::Tensor& topk_ids,    // [batch, topk], int64
    int64_t topk,
    int64_t vocab_size) {

  VLLM_MCPU_CHECK_DIM(logits, 2, "logits");
  VLLM_MCPU_CHECK_FLOAT(logits, "logits");
  VLLM_MCPU_CHECK_DIM(output, 2, "output");
  VLLM_MCPU_CHECK_DTYPE(output, at::kFloat, "output");
  VLLM_MCPU_CHECK_DIM(topk_ids, 2, "topk_ids");
  VLLM_MCPU_CHECK_DTYPE(topk_ids, at::kLong, "topk_ids");

  int64_t batch = logits.size(0);
  int64_t logits_stride = logits.stride(0);
  int64_t topk_ids_stride = topk_ids.stride(0);
  const int64_t* ids_ptr = topk_ids.data_ptr<int64_t>();
  float* out_ptr = output.data_ptr<float>();

  VLLM_MCPU_DISPATCH_FLOAT(logits, "vllm_topk_log_softmax_kernel", {
    vllm_topk_log_softmax_kernel_typed<scalar_t>(
        out_ptr, logits.data_ptr<scalar_t>(),
        batch, logits_stride, ids_ptr, topk_ids_stride, topk, vocab_size);
  });
}

// ---------------------------------------------------------------------------
// _ranks_kernel
// output[req] = number of logits >= logits[req, token_ids[req]]
// ---------------------------------------------------------------------------
template <typename scalar_t>
static void vllm_ranks_kernel_typed(
    int64_t* out_ptr,
    const scalar_t* logits_ptr,
    int64_t batch,
    int64_t logits_stride,
    const int64_t* ids_ptr,
    int64_t vocab_size) {

  for (int64_t req = 0; req < batch; req++) {
    const scalar_t* row = logits_ptr + req * logits_stride;
    int64_t tid = ids_ptr[req];
    int64_t rank = 0;
    if constexpr (std::is_same_v<scalar_t, float>) {
      float x = row[tid];
      for (int64_t i = 0; i < vocab_size; i++) {
        if (row[i] >= x) rank++;
      }
    } else {
      float x = static_cast<float>(row[tid]);
      for (int64_t i = 0; i < vocab_size; i++) {
        if (static_cast<float>(row[i]) >= x) rank++;
      }
    }
    out_ptr[req] = rank;
  }
}

void vllm_ranks_kernel_impl(
    at::Tensor& output,          // [batch], int64
    const at::Tensor& logits,    // [batch, vocab_size], any float
    const at::Tensor& token_ids, // [batch], int64
    int64_t vocab_size) {

  VLLM_MCPU_CHECK_DIM(logits, 2, "logits");
  VLLM_MCPU_CHECK_FLOAT(logits, "logits");
  VLLM_MCPU_CHECK_DIM(output, 1, "output");
  VLLM_MCPU_CHECK_DTYPE(output, at::kLong, "output");
  VLLM_MCPU_CHECK_DIM(token_ids, 1, "token_ids");
  VLLM_MCPU_CHECK_DTYPE(token_ids, at::kLong, "token_ids");

  int64_t batch = logits.size(0);
  int64_t logits_stride = logits.stride(0);
  const int64_t* ids_ptr = token_ids.data_ptr<int64_t>();
  int64_t* out_ptr = output.data_ptr<int64_t>();

  VLLM_MCPU_DISPATCH_FLOAT(logits, "vllm_ranks_kernel", {
    vllm_ranks_kernel_typed<scalar_t>(
        out_ptr, logits.data_ptr<scalar_t>(),
        batch, logits_stride, ids_ptr, vocab_size);
  });
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_topk_log_softmax_kernel("
      "Tensor(a!) output, "
      "Tensor logits, "
      "Tensor topk_ids, "
      "int topk, "
      "int vocab_size"
      ") -> ()");
  m.def(
      "vllm_ranks_kernel("
      "Tensor(a!) output, "
      "Tensor logits, "
      "Tensor token_ids, "
      "int vocab_size"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_topk_log_softmax_kernel", &vllm_topk_log_softmax_kernel_impl);
  m.impl("vllm_ranks_kernel", &vllm_ranks_kernel_impl);
}
