// SPDX-License-Identifier: Apache-2.0
//
// C++ kernel for vllm/v1/worker/gpu/structured_outputs.py::_apply_grammar_bitmask_kernel

#include "common.h"

namespace {

template <typename scalar_t>
static void vllm_apply_grammar_bitmask_typed(
    scalar_t* logits_ptr,
    int64_t num_masks,
    int64_t logits_stride,
    const int32_t* idx_ptr,
    const int32_t* bitmask_ptr,
    int64_t bitmask_stride,
    int64_t vocab_size) {

  const scalar_t neg_inf = static_cast<scalar_t>(-std::numeric_limits<float>::infinity());
  int64_t num_words = (vocab_size + 31) / 32;

  for (int64_t mask_idx = 0; mask_idx < num_masks; mask_idx++) {
    int64_t row = (int64_t)idx_ptr[mask_idx];
    scalar_t* row_ptr = logits_ptr + row * logits_stride;
    const int32_t* mask_row = bitmask_ptr + mask_idx * bitmask_stride;

    for (int64_t w = 0; w < num_words; w++) {
      uint32_t packed = (uint32_t)mask_row[w];
      int64_t base = w * 32;
      int lim = (int)((base + 32 <= vocab_size) ? 32 : vocab_size - base);
      for (int b = 0; b < lim; b++) {
        if ((packed >> b) & 1u) row_ptr[base + b] = neg_inf;
      }
    }
  }
}

void vllm_apply_grammar_bitmask_impl(
    at::Tensor& logits,
    const at::Tensor& logits_indices,
    const at::Tensor& bitmask,
    int64_t vocab_size) {

  VLLM_MCPU_CHECK_DIM(logits, 2, "logits");
  VLLM_MCPU_CHECK_FLOAT(logits, "logits");
  VLLM_MCPU_CHECK_DIM(logits_indices, 1, "logits_indices");
  VLLM_MCPU_CHECK_DTYPE(logits_indices, at::kInt, "logits_indices");
  VLLM_MCPU_CHECK_DIM(bitmask, 2, "bitmask");
  VLLM_MCPU_CHECK_DTYPE(bitmask, at::kInt, "bitmask");

  int64_t num_masks = logits_indices.size(0);
  if (num_masks == 0) return;

  int64_t logits_stride = logits.stride(0);
  const int32_t* idx_ptr = logits_indices.data_ptr<int32_t>();
  const int32_t* bitmask_ptr = bitmask.data_ptr<int32_t>();
  int64_t bitmask_stride = bitmask.stride(0);

  VLLM_MCPU_DISPATCH_FLOAT(logits, "vllm_apply_grammar_bitmask", {
    vllm_apply_grammar_bitmask_typed<scalar_t>(
        logits.data_ptr<scalar_t>(), num_masks, logits_stride,
        idx_ptr, bitmask_ptr, bitmask_stride, vocab_size);
  });
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_apply_grammar_bitmask("
      "Tensor(a!) logits, Tensor logits_indices, Tensor bitmask, int vocab_size"
      ") -> ()");
}
TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_apply_grammar_bitmask", &vllm_apply_grammar_bitmask_impl);
}
