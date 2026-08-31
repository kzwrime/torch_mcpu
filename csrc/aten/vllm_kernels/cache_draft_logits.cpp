// SPDX-License-Identifier: Apache-2.0

#include "common.h"

#include <limits>

namespace {

void vllm_dflash2_cache_draft_logits_impl(
    at::Tensor& draft_logits,
    at::Tensor& cached_candidate_ids,
    const at::Tensor& candidate_ids,
    const at::Tensor& scores,
    const at::Tensor& req_state,
    int64_t num_sample,
    int64_t num_steps,
    int64_t top_k) {
  VLLM_MCPU_CHECK_DIM(draft_logits, 3, "draft_logits");
  VLLM_MCPU_CHECK_DIM(cached_candidate_ids, 3, "cached_candidate_ids");
  VLLM_MCPU_CHECK_DIM(candidate_ids, 3, "candidate_ids");
  VLLM_MCPU_CHECK_DIM(scores, 3, "scores");
  VLLM_MCPU_CHECK_DIM(req_state, 1, "req_state");
  VLLM_MCPU_CHECK_DTYPE(draft_logits, at::kFloat, "draft_logits");
  VLLM_MCPU_CHECK_DTYPE(scores, at::kFloat, "scores");
  VLLM_MCPU_CHECK_DTYPE(
      cached_candidate_ids, at::kLong, "cached_candidate_ids");
  VLLM_MCPU_CHECK_DTYPE(candidate_ids, at::kLong, "candidate_ids");
  VLLM_MCPU_CHECK_DTYPE(req_state, at::kInt, "req_state");
  VLLM_MCPU_CHECK(
      num_sample >= 0 && num_steps > 0 && top_k > 0,
      "invalid DFlash2 cache dimensions");
  VLLM_MCPU_CHECK(
      num_sample <= candidate_ids.size(0) * num_steps &&
          num_sample <= scores.size(0) * num_steps &&
          req_state.numel() >= num_sample,
      "DFlash2 cache input capacity is too small");
  VLLM_MCPU_CHECK(
      candidate_ids.size(1) == num_steps && candidate_ids.size(2) == top_k &&
          scores.size(1) == num_steps && scores.size(2) == top_k,
      "DFlash2 candidate/score shape mismatch");
  VLLM_MCPU_CHECK(
      cached_candidate_ids.size(1) == num_steps &&
          cached_candidate_ids.size(2) == top_k,
      "DFlash2 cached candidate shape mismatch");

  at::mcpu::launch_timed_kernel(
      "mcpu::vllm_dflash2_cache_draft_logits",
      [draft_logits,
       cached_candidate_ids,
       candidate_ids,
       scores,
       req_state,
       num_sample,
       num_steps,
       top_k](at::mcpu::kernel_timing::Event* timing_event) mutable {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::vllm_dflash2_cache_draft_logits", timing_event);
        at::mcpu::KernelMemoryGuard guard(
            draft_logits,
            cached_candidate_ids,
            candidate_ids,
            scores,
            req_state);
        auto* logits_ptr = draft_logits.data_ptr<float>();
        auto* cached_ptr = cached_candidate_ids.data_ptr<int64_t>();
        const auto* candidate_ptr = candidate_ids.const_data_ptr<int64_t>();
        const auto* score_ptr = scores.const_data_ptr<float>();
        const auto* req_state_ptr = req_state.const_data_ptr<int32_t>();
#pragma omp parallel for schedule(static)
        for (int64_t flat = 0; flat < num_sample; ++flat) {
          const int64_t state = req_state_ptr[flat];
          if (state < 0) {
            continue;
          }
          const int64_t step = flat % num_steps;
          const int64_t source_req = flat / num_steps;
          const int64_t cache_base = state * cached_candidate_ids.stride(0) +
              step * cached_candidate_ids.stride(1);
          const int64_t logits_base =
              state * draft_logits.stride(0) + step * draft_logits.stride(1);
          const int64_t candidate_base = source_req * candidate_ids.stride(0) +
              step * candidate_ids.stride(1);
          const int64_t score_base =
              source_req * scores.stride(0) + step * scores.stride(1);
          for (int64_t i = 0; i < top_k; ++i) {
            const int64_t old_id =
                cached_ptr[cache_base + i * cached_candidate_ids.stride(2)];
            TORCH_INTERNAL_ASSERT(old_id >= 0 && old_id < draft_logits.size(2));
            logits_ptr[logits_base + old_id * draft_logits.stride(2)] =
                -std::numeric_limits<float>::infinity();
          }
          for (int64_t i = 0; i < top_k; ++i) {
            const int64_t token_id =
                candidate_ptr[candidate_base + i * candidate_ids.stride(2)];
            TORCH_INTERNAL_ASSERT(
                token_id >= 0 && token_id < draft_logits.size(2));
            logits_ptr[logits_base + token_id * draft_logits.stride(2)] =
                score_ptr[score_base + i * scores.stride(2)];
            cached_ptr[cache_base + i * cached_candidate_ids.stride(2)] =
                token_id;
          }
        }
      });
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_dflash2_cache_draft_logits("
      "Tensor(a!) draft_logits, Tensor(b!) cached_candidate_ids, "
      "Tensor candidate_ids, Tensor scores, Tensor req_state, int num_sample, "
      "int num_steps, int top_k) -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl(
      "vllm_dflash2_cache_draft_logits", &vllm_dflash2_cache_draft_logits_impl);
}
