// SPDX-License-Identifier: Apache-2.0

#include "common.h"

#include <limits>

namespace {

void vllm_dflash2_selector_walk_impl(
    const at::Tensor& scores,
    const at::Tensor& candidate_ids,
    const at::Tensor& sample_positions,
    const at::Tensor& req_state,
    const at::Tensor& temperature,
    const at::Tensor& seeds,
    at::Tensor& tokens,
    at::Tensor& realized_scores,
    int64_t num_reqs,
    int64_t num_steps,
    int64_t top_k,
    bool sample_probabilistic,
    bool use_fp64) {
  TORCH_CHECK(
      !sample_probabilistic,
      "probabilistic DFlash2 selector requires candidate-id-keyed "
      "Philox Gumbel sampling and is not implemented on MCPU");
  (void)use_fp64;
  VLLM_MCPU_CHECK_DIM(scores, 4, "scores");
  VLLM_MCPU_CHECK_DIM(candidate_ids, 3, "candidate_ids");
  VLLM_MCPU_CHECK_DIM(tokens, 2, "tokens");
  VLLM_MCPU_CHECK_DIM(realized_scores, 3, "realized_scores");
  VLLM_MCPU_CHECK_DIM(req_state, 1, "req_state");
  VLLM_MCPU_CHECK_DIM(sample_positions, 1, "sample_positions");
  VLLM_MCPU_CHECK_DIM(temperature, 1, "temperature");
  VLLM_MCPU_CHECK_DIM(seeds, 1, "seeds");
  VLLM_MCPU_CHECK_DTYPE(scores, at::kFloat, "scores");
  VLLM_MCPU_CHECK_DTYPE(realized_scores, at::kFloat, "realized_scores");
  VLLM_MCPU_CHECK_DTYPE(candidate_ids, at::kLong, "candidate_ids");
  VLLM_MCPU_CHECK_DTYPE(tokens, at::kLong, "tokens");
  VLLM_MCPU_CHECK_DTYPE(req_state, at::kInt, "req_state");
  VLLM_MCPU_CHECK_DTYPE(sample_positions, at::kLong, "sample_positions");
  VLLM_MCPU_CHECK_DTYPE(temperature, at::kFloat, "temperature");
  VLLM_MCPU_CHECK_DTYPE(seeds, at::kLong, "seeds");
  VLLM_MCPU_CHECK(
      num_reqs >= 0 && num_steps > 0 && top_k > 0,
      "invalid DFlash2 selector dimensions");
  VLLM_MCPU_CHECK(
      scores.size(0) >= num_reqs && scores.size(1) == num_steps &&
          scores.size(2) == top_k && scores.size(3) == top_k,
      "scores shape mismatch");
  VLLM_MCPU_CHECK(
      candidate_ids.size(0) >= num_reqs && candidate_ids.size(1) == num_steps &&
          candidate_ids.size(2) == top_k,
      "candidate_ids shape mismatch");
  VLLM_MCPU_CHECK(
      tokens.size(0) >= num_reqs && tokens.size(1) >= num_steps,
      "tokens shape mismatch");
  VLLM_MCPU_CHECK(
      realized_scores.size(0) >= num_reqs &&
          realized_scores.size(1) == num_steps &&
          realized_scores.size(2) == top_k,
      "realized_scores shape mismatch");
  VLLM_MCPU_CHECK(
      req_state.numel() >= num_reqs * num_steps,
      "req_state capacity is too small");

  at::mcpu::launch_timed_kernel(
      "mcpu::vllm_dflash2_selector_walk",
      [scores,
       candidate_ids,
       sample_positions,
       req_state,
       temperature,
       seeds,
       tokens,
       realized_scores,
       num_reqs,
       num_steps,
       top_k](at::mcpu::kernel_timing::Event* timing_event) mutable {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::vllm_dflash2_selector_walk", timing_event);
        at::mcpu::KernelMemoryGuard guard(
            scores,
            candidate_ids,
            sample_positions,
            req_state,
            temperature,
            seeds,
            tokens,
            realized_scores);
        const auto* score_ptr = scores.const_data_ptr<float>();
        const auto* candidate_ptr = candidate_ids.const_data_ptr<int64_t>();
        const auto* req_state_ptr = req_state.const_data_ptr<int32_t>();
        auto* token_ptr = tokens.data_ptr<int64_t>();
        auto* realized_ptr = realized_scores.data_ptr<float>();
#pragma omp parallel for schedule(static)
        for (int64_t req = 0; req < num_reqs; ++req) {
          if (req_state_ptr[req * num_steps] < 0) {
            continue;
          }
          int64_t previous = 0;
          for (int64_t step = 0; step < num_steps; ++step) {
            const int64_t score_base = req * scores.stride(0) +
                step * scores.stride(1) + previous * scores.stride(2);
            const int64_t candidate_base =
                req * candidate_ids.stride(0) + step * candidate_ids.stride(1);
            const int64_t realized_base = req * realized_scores.stride(0) +
                step * realized_scores.stride(1);
            int64_t selected = 0;
            float selected_score = score_ptr[score_base];
            for (int64_t candidate = 0; candidate < top_k; ++candidate) {
              const float score =
                  score_ptr[score_base + candidate * scores.stride(3)];
              realized_ptr
                  [realized_base + candidate * realized_scores.stride(2)] =
                      score;
              if (score > selected_score) {
                selected_score = score;
                selected = candidate;
              }
            }
            token_ptr[req * tokens.stride(0) + step * tokens.stride(1)] =
                candidate_ptr
                    [candidate_base + selected * candidate_ids.stride(2)];
            previous = selected;
          }
        }
      });
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_dflash2_selector_walk("
      "Tensor scores, Tensor candidate_ids, Tensor sample_positions, "
      "Tensor req_state, Tensor temperature, Tensor seeds, Tensor(a!) tokens, "
      "Tensor(b!) realized_scores, int num_reqs, int num_steps, int top_k, "
      "bool sample_probabilistic, bool use_fp64) -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_dflash2_selector_walk", &vllm_dflash2_selector_walk_impl);
}
