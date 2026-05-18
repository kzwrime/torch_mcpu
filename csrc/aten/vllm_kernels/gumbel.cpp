// SPDX-License-Identifier: Apache-2.0
//
// C++ kernels for vllm/v1/worker/gpu/sample/gumbel.py:
//   _temperature_kernel — divide logits by per-request temperature

#include "common.h"

namespace {

// ---------------------------------------------------------------------------
// _temperature_kernel
// logits[tok] /= temperature[expanded_idx_mapping[tok]]  (skip if temp==0 or 1)
// ---------------------------------------------------------------------------
template <typename scalar_t>
static void vllm_temperature_kernel_typed(
    scalar_t* logits_ptr,
    int64_t num_tokens,
    int64_t logits_stride,
    const int32_t* idx_ptr,
    const float* temp_ptr,
    int64_t vocab_size) {
  for (int64_t tok = 0; tok < num_tokens; tok++) {
    int32_t req = idx_ptr[tok];
    float temp = temp_ptr[req];
    if (temp == 0.0f || temp == 1.0f)
      continue;

    scalar_t* row = logits_ptr + tok * logits_stride;
    if constexpr (std::is_same_v<scalar_t, float>) {
      for (int64_t i = 0; i < vocab_size; i++) {
        row[i] = row[i] / temp;
      }
    } else {
      for (int64_t i = 0; i < vocab_size; i++) {
        row[i] = static_cast<scalar_t>(static_cast<float>(row[i]) / temp);
      }
    }
  }
}

void vllm_temperature_kernel_impl(
    at::Tensor& logits, // [num_tokens, vocab_size]
    const at::Tensor& expanded_idx_mapping, // [num_tokens], int32
    const at::Tensor& temperature, // [max_num_reqs], float32
    int64_t vocab_size) {
  VLLM_MCPU_CHECK_DIM(logits, 2, "logits");
  VLLM_MCPU_CHECK_FLOAT(logits, "logits");
  VLLM_MCPU_CHECK_DIM(expanded_idx_mapping, 1, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(expanded_idx_mapping, at::kInt, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DIM(temperature, 1, "temperature");
  VLLM_MCPU_CHECK_DTYPE(temperature, at::kFloat, "temperature");

  int64_t num_tokens = logits.size(0);
  int64_t logits_stride = logits.stride(0);
  const int32_t* idx_ptr = expanded_idx_mapping.data_ptr<int32_t>();
  const float* temp_ptr = temperature.data_ptr<float>();

  VLLM_MCPU_DISPATCH_FLOAT(logits, "vllm_temperature_kernel", {
    vllm_temperature_kernel_typed<scalar_t>(
        logits.data_ptr<scalar_t>(),
        num_tokens,
        logits_stride,
        idx_ptr,
        temp_ptr,
        vocab_size);
  });
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_temperature_kernel("
      "Tensor(a!) logits, "
      "Tensor expanded_idx_mapping, "
      "Tensor temperature, "
      "int vocab_size"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_temperature_kernel", &vllm_temperature_kernel_impl);
}
