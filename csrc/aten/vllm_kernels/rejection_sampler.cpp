// SPDX-License-Identifier: Apache-2.0
//
// C++ kernel for vllm/v1/sample/rejection_sampler.py::expand_kernel

#include "common.h"
#include "runtime/McpuKernelLaunch.h"

#include <algorithm>

namespace {

template <typename scalar_t>
void launch_vllm_rejection_block_stats(
    at::Tensor& target_local_argmax,
    at::Tensor& target_local_max,
    at::Tensor& target_local_sumexp,
    at::Tensor& draft_local_max,
    at::Tensor& draft_local_sumexp,
    const at::Tensor& target_logits,
    const std::optional<at::Tensor>& draft_logits,
    const at::Tensor& expanded_idx_mapping,
    const at::Tensor& expanded_local_pos,
    const at::Tensor& temperature,
    int64_t vocab_size,
    int64_t num_speculative_steps,
    int64_t block_size) {
  auto* target_argmax = target_local_argmax.data_ptr<int64_t>();
  auto* target_max = target_local_max.data_ptr<float>();
  auto* target_sumexp = target_local_sumexp.data_ptr<float>();
  auto* draft_max = draft_local_max.data_ptr<float>();
  auto* draft_sumexp = draft_local_sumexp.data_ptr<float>();
  const auto* target = target_logits.data_ptr<scalar_t>();
  const auto* draft =
      draft_logits ? draft_logits->data_ptr<scalar_t>() : nullptr;
  const auto* req_mapping = expanded_idx_mapping.data_ptr<int32_t>();
  const auto* local_pos = expanded_local_pos.data_ptr<int32_t>();
  const auto* temp = temperature.data_ptr<float>();

  const int64_t num_logits = target_logits.size(0);
  const int64_t num_blocks = target_local_max.size(1);
  const int64_t target_stride = target_logits.stride(0);
  const int64_t draft_stride_0 = draft_logits ? draft_logits->stride(0) : 0;
  const int64_t draft_stride_1 = draft_logits ? draft_logits->stride(1) : 0;
  const int64_t target_argmax_stride = target_local_argmax.stride(0);
  const int64_t target_max_stride = target_local_max.stride(0);
  const int64_t target_sumexp_stride = target_local_sumexp.stride(0);
  const int64_t draft_max_stride = draft_local_max.stride(0);
  const int64_t draft_sumexp_stride = draft_local_sumexp.stride(0);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::vllm_rejection_compute_block_stats",
      ([
        target_argmax,
        target_max,
        target_sumexp,
        draft_max,
        draft_sumexp,
        target,
        draft,
        req_mapping,
        local_pos,
        temp,
        num_logits,
        num_blocks,
        vocab_size,
        num_speculative_steps,
        block_size,
        target_stride,
        draft_stride_0,
        draft_stride_1,
        target_argmax_stride,
        target_max_stride,
        target_sumexp_stride,
        draft_max_stride,
        draft_sumexp_stride
      ]),
      {
        at::mcpu::KernelPointerMemoryGuard guard(
            {target_argmax,
             target_max,
             target_sumexp,
             draft_max,
             draft_sumexp,
             target,
             draft,
             req_mapping,
             local_pos,
             temp});
        const int64_t num_programs = num_logits * num_blocks;
#pragma omp parallel for schedule(static)
        for (int64_t program = 0; program < num_programs; ++program) {
          const int64_t logit_idx = program / num_blocks;
          const int64_t block_idx = program % num_blocks;
          const int32_t draft_step_idx = local_pos[logit_idx];
          // Match Triton exactly: bonus-token output cells are not written.
          if (draft_step_idx >= num_speculative_steps)
            continue;
          const int32_t req_state_idx = req_mapping[logit_idx];
          const bool greedy = temp[req_state_idx] == 0.0f;
          const int64_t begin = block_idx * block_size;
          const int64_t end = std::min(begin + block_size, vocab_size);
          float local_target_max = -std::numeric_limits<float>::infinity();
          int64_t local_target_argmax = 0;
          for (int64_t token = begin; token < end; ++token) {
            const float value =
                static_cast<float>(target[logit_idx * target_stride + token]);
            if (value > local_target_max) {
              local_target_max = value;
              local_target_argmax = token - begin;
            }
          }
          target_max[logit_idx * target_max_stride + block_idx] =
              local_target_max;
          if (greedy) {
            target_argmax[logit_idx * target_argmax_stride + block_idx] =
                begin + local_target_argmax;
            continue;
          }

          float local_target_sumexp = 0.0f;
          if (local_target_max > -std::numeric_limits<float>::infinity()) {
            for (int64_t token = begin; token < end; ++token) {
              const float value =
                  static_cast<float>(target[logit_idx * target_stride + token]);
              local_target_sumexp += std::exp(value - local_target_max);
            }
          }
          target_sumexp[logit_idx * target_sumexp_stride + block_idx] =
              local_target_sumexp;

          if (!draft)
            continue;
          float local_draft_max = -std::numeric_limits<float>::infinity();
          const int64_t draft_base =
              static_cast<int64_t>(req_state_idx) * draft_stride_0 +
              static_cast<int64_t>(draft_step_idx) * draft_stride_1;
          for (int64_t token = begin; token < end; ++token) {
            local_draft_max = std::max(
                local_draft_max, static_cast<float>(draft[draft_base + token]));
          }
          float local_draft_sumexp = 0.0f;
          if (local_draft_max > -std::numeric_limits<float>::infinity()) {
            for (int64_t token = begin; token < end; ++token) {
              local_draft_sumexp += std::exp(
                  static_cast<float>(draft[draft_base + token]) -
                  local_draft_max);
            }
          }
          draft_max[logit_idx * draft_max_stride + block_idx] = local_draft_max;
          draft_sumexp[logit_idx * draft_sumexp_stride + block_idx] =
              local_draft_sumexp;
        }
      });
}

void vllm_rejection_compute_block_stats_impl(
    at::Tensor& target_local_argmax,
    at::Tensor& target_local_max,
    at::Tensor& target_local_sumexp,
    at::Tensor& draft_local_max,
    at::Tensor& draft_local_sumexp,
    const at::Tensor& target_logits,
    const std::optional<at::Tensor>& draft_logits,
    const at::Tensor& expanded_idx_mapping,
    const at::Tensor& expanded_local_pos,
    const at::Tensor& temperature,
    int64_t vocab_size,
    int64_t num_speculative_steps,
    int64_t block_size) {
  VLLM_MCPU_CHECK_DIM(target_local_argmax, 2, "target_local_argmax");
  VLLM_MCPU_CHECK_DIM(target_local_max, 2, "target_local_max");
  VLLM_MCPU_CHECK_DIM(target_local_sumexp, 2, "target_local_sumexp");
  VLLM_MCPU_CHECK_DIM(draft_local_max, 2, "draft_local_max");
  VLLM_MCPU_CHECK_DIM(draft_local_sumexp, 2, "draft_local_sumexp");
  VLLM_MCPU_CHECK_DIM(target_logits, 2, "target_logits");
  VLLM_MCPU_CHECK_DIM(expanded_idx_mapping, 1, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DIM(expanded_local_pos, 1, "expanded_local_pos");
  VLLM_MCPU_CHECK_DIM(temperature, 1, "temperature");
  VLLM_MCPU_CHECK_DTYPE(target_local_argmax, at::kLong, "target_local_argmax");
  VLLM_MCPU_CHECK_DTYPE(target_local_max, at::kFloat, "target_local_max");
  VLLM_MCPU_CHECK_DTYPE(target_local_sumexp, at::kFloat, "target_local_sumexp");
  VLLM_MCPU_CHECK_DTYPE(draft_local_max, at::kFloat, "draft_local_max");
  VLLM_MCPU_CHECK_DTYPE(draft_local_sumexp, at::kFloat, "draft_local_sumexp");
  VLLM_MCPU_CHECK_FLOAT(target_logits, "target_logits");
  VLLM_MCPU_CHECK_DTYPE(expanded_idx_mapping, at::kInt, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(expanded_local_pos, at::kInt, "expanded_local_pos");
  VLLM_MCPU_CHECK_DTYPE(temperature, at::kFloat, "temperature");
  VLLM_MCPU_CHECK(vocab_size > 0, "vocab_size must be positive");
  VLLM_MCPU_CHECK(
      vocab_size <= target_logits.size(1),
      "vocab_size exceeds target logits width");
  VLLM_MCPU_CHECK(
      num_speculative_steps >= 0, "num_speculative_steps must be non-negative");
  VLLM_MCPU_CHECK(block_size == 8192, "BLOCK_SIZE must be 8192");
  const int64_t num_logits = target_logits.size(0);
  const int64_t num_blocks = (vocab_size + block_size - 1) / block_size;
  const auto has_expected_shape = [num_logits,
                                   num_blocks](const at::Tensor& tensor) {
    return tensor.size(0) == num_logits && tensor.size(1) == num_blocks;
  };
  VLLM_MCPU_CHECK(
      has_expected_shape(target_local_argmax) &&
          has_expected_shape(target_local_max) &&
          has_expected_shape(target_local_sumexp) &&
          has_expected_shape(draft_local_max) &&
          has_expected_shape(draft_local_sumexp),
      "block-stat outputs must have shape [num_logits, ceil(vocab_size / 8192)]");
  VLLM_MCPU_CHECK(
      expanded_idx_mapping.numel() == num_logits &&
          expanded_local_pos.numel() == num_logits,
      "expanded mappings must have num_logits elements");
  VLLM_MCPU_CHECK(
      target_logits.stride(1) == 1, "target logits columns must be contiguous");
  VLLM_MCPU_CHECK(
      target_local_argmax.stride(1) == 1 && target_local_max.stride(1) == 1 &&
          target_local_sumexp.stride(1) == 1 &&
          draft_local_max.stride(1) == 1 && draft_local_sumexp.stride(1) == 1,
      "block-stat output columns must be contiguous");
  VLLM_MCPU_CHECK(
      expanded_idx_mapping.is_contiguous() &&
          expanded_local_pos.is_contiguous() && temperature.is_contiguous(),
      "mapping and temperature tensors must be contiguous");
  if (draft_logits) {
    VLLM_MCPU_CHECK_DIM(*draft_logits, 3, "draft_logits");
    VLLM_MCPU_CHECK(
        draft_logits->scalar_type() == target_logits.scalar_type(),
        "draft and target logits must have the same dtype");
    VLLM_MCPU_CHECK(
        draft_logits->size(1) >= num_speculative_steps &&
            draft_logits->size(2) >= vocab_size,
        "draft logits shape is too small");
    VLLM_MCPU_CHECK(
        draft_logits->stride(2) == 1,
        "draft logits columns must be contiguous");
  }

  VLLM_MCPU_DISPATCH_FLOAT(
      target_logits, "vllm_rejection_compute_block_stats", {
        launch_vllm_rejection_block_stats<scalar_t>(
            target_local_argmax,
            target_local_max,
            target_local_sumexp,
            draft_local_max,
            draft_local_sumexp,
            target_logits,
            draft_logits,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            vocab_size,
            num_speculative_steps,
            block_size);
      });
}

struct PhiloxResult {
  uint32_t c0;
  uint32_t c1;
  uint32_t c2;
  uint32_t c3;
};

inline uint32_t mul_hi_u32(uint32_t lhs, uint32_t rhs) {
  return static_cast<uint32_t>(
      (static_cast<uint64_t>(lhs) * static_cast<uint64_t>(rhs)) >> 32);
}

inline PhiloxResult triton_philox(uint64_t seed, uint64_t offset) {
  uint32_t c0 = static_cast<uint32_t>(offset);
  uint32_t c1 = static_cast<uint32_t>(offset >> 32);
  uint32_t c2 = 0;
  uint32_t c3 = 0;
  uint32_t k0 = static_cast<uint32_t>(seed);
  uint32_t k1 = static_cast<uint32_t>(seed >> 32);
  for (int round = 0; round < 10; ++round) {
    const uint32_t old_c0 = c0;
    const uint32_t old_c2 = c2;
    c0 = mul_hi_u32(0xCD9E8D57U, old_c2) ^ c1 ^ k0;
    c2 = mul_hi_u32(0xD2511F53U, old_c0) ^ c3 ^ k1;
    c1 = 0xCD9E8D57U * old_c2;
    c3 = 0xD2511F53U * old_c0;
    k0 += 0x9E3779B9U;
    k1 += 0xBB67AE85U;
  }
  return {c0, c1, c2, c3};
}

inline float triton_rand32(uint64_t seed, uint64_t offset) {
  const int32_t random = static_cast<int32_t>(triton_philox(seed, offset).c0);
  const uint32_t magnitude = random < 0 ? static_cast<uint32_t>(~random)
                                        : static_cast<uint32_t>(random);
  return std::max(
      static_cast<float>(magnitude) * 4.6566127342e-10f, 4.6566127342e-10f);
}

inline float block_global_lse(
    const float* local_max,
    int64_t max_stride,
    const float* local_sumexp,
    int64_t sumexp_stride,
    int64_t logit_idx,
    int64_t num_blocks) {
  float global_max = -std::numeric_limits<float>::infinity();
  for (int64_t block = 0; block < num_blocks; ++block) {
    global_max =
        std::max(global_max, local_max[logit_idx * max_stride + block]);
  }
  float scaled_sum = 0.0f;
  for (int64_t block = 0; block < num_blocks; ++block) {
    const float block_max = local_max[logit_idx * max_stride + block];
    scaled_sum += local_sumexp[logit_idx * sumexp_stride + block] *
        std::exp(block_max - global_max);
  }
  return global_max + std::log(scaled_sum);
}

template <typename scalar_t>
void launch_vllm_rejection(
    at::Tensor& sampled,
    at::Tensor& rejected_steps,
    at::Tensor& target_rejected_lse,
    at::Tensor& draft_rejected_lse,
    const at::Tensor& target_logits,
    const at::Tensor& target_local_argmax,
    const at::Tensor& target_local_max,
    const at::Tensor& target_local_sumexp,
    const at::Tensor& draft_sampled,
    const std::optional<at::Tensor>& draft_logits,
    const at::Tensor& draft_local_max,
    const at::Tensor& draft_local_sumexp,
    const at::Tensor& cu_num_logits,
    const at::Tensor& idx_mapping,
    const at::Tensor& temperature,
    const at::Tensor& seed,
    const at::Tensor& pos,
    int64_t vocab_num_blocks) {
  auto* sampled_ptr = sampled.data_ptr<int64_t>();
  auto* rejected_ptr = rejected_steps.data_ptr<int32_t>();
  auto* target_rejected_ptr = target_rejected_lse.data_ptr<float>();
  auto* draft_rejected_ptr = draft_rejected_lse.data_ptr<float>();
  const auto* target_ptr = target_logits.data_ptr<scalar_t>();
  const auto* target_argmax_ptr = target_local_argmax.data_ptr<int64_t>();
  const auto* target_max_ptr = target_local_max.data_ptr<float>();
  const auto* target_sumexp_ptr = target_local_sumexp.data_ptr<float>();
  const auto* draft_sampled_ptr = draft_sampled.data_ptr<int32_t>();
  const auto* draft_ptr =
      draft_logits ? draft_logits->data_ptr<scalar_t>() : nullptr;
  const auto* draft_max_ptr = draft_local_max.data_ptr<float>();
  const auto* draft_sumexp_ptr = draft_local_sumexp.data_ptr<float>();
  const auto* cu_ptr = cu_num_logits.data_ptr<int32_t>();
  const auto* mapping_ptr = idx_mapping.data_ptr<int32_t>();
  const auto* temp_ptr = temperature.data_ptr<float>();
  const auto* seed_ptr = seed.data_ptr<int64_t>();
  const auto* pos_ptr = pos.data_ptr<int64_t>();
  const int64_t num_reqs = idx_mapping.numel();
  const int64_t sampled_stride = sampled.stride(0);
  const int64_t target_stride = target_logits.stride(0);
  const int64_t target_argmax_stride = target_local_argmax.stride(0);
  const int64_t target_max_stride = target_local_max.stride(0);
  const int64_t target_sumexp_stride = target_local_sumexp.stride(0);
  const int64_t draft_stride_0 = draft_logits ? draft_logits->stride(0) : 0;
  const int64_t draft_stride_1 = draft_logits ? draft_logits->stride(1) : 0;
  const int64_t draft_max_stride = draft_local_max.stride(0);
  const int64_t draft_sumexp_stride = draft_local_sumexp.stride(0);

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::vllm_rejection", ([=]), {
    at::mcpu::KernelPointerMemoryGuard guard(
        {sampled_ptr,
         rejected_ptr,
         target_rejected_ptr,
         draft_rejected_ptr,
         target_ptr,
         target_argmax_ptr,
         target_max_ptr,
         target_sumexp_ptr,
         draft_sampled_ptr,
         draft_ptr,
         draft_max_ptr,
         draft_sumexp_ptr,
         cu_ptr,
         mapping_ptr,
         temp_ptr,
         seed_ptr,
         pos_ptr});
#pragma omp parallel for schedule(static)
    for (int64_t req = 0; req < num_reqs; ++req) {
      const int64_t req_state = mapping_ptr[req];
      const int64_t start = cu_ptr[req];
      const int64_t end = cu_ptr[req + 1];
      const float temp = temp_ptr[req_state];
      bool accepted = true;
      int32_t rejected_step = 0;
      float target_lse = 0.0f;
      float draft_lse = 0.0f;
      for (int64_t step = 0; step < end - start - 1; ++step) {
        if (!accepted)
          break;
        const int64_t logit_idx = start + step;
        int64_t draft_token = draft_sampled_ptr[logit_idx + 1];
        if (temp == 0.0f) {
          int64_t best_block = 0;
          float best_value = -std::numeric_limits<float>::infinity();
          for (int64_t block = 0; block < vocab_num_blocks; ++block) {
            const float value =
                target_max_ptr[logit_idx * target_max_stride + block];
            if (value > best_value) {
              best_value = value;
              best_block = block;
            }
          }
          const int64_t target_argmax =
              target_argmax_ptr[logit_idx * target_argmax_stride + best_block];
          accepted = target_argmax == draft_token;
          sampled_ptr[req * sampled_stride + step] =
              accepted ? draft_token : target_argmax;
        } else {
          const bool valid_draft = draft_token >= 0;
          draft_token = std::max<int64_t>(0, draft_token);
          const float target_logit = static_cast<float>(
              target_ptr[logit_idx * target_stride + draft_token]);
          target_lse = block_global_lse(
              target_max_ptr,
              target_max_stride,
              target_sumexp_ptr,
              target_sumexp_stride,
              logit_idx,
              vocab_num_blocks);
          float draft_log_prob = 0.0f;
          if (draft_ptr) {
            const float draft_logit =
                static_cast<float>(draft_ptr
                                       [req_state * draft_stride_0 +
                                        step * draft_stride_1 + draft_token]);
            draft_lse = block_global_lse(
                draft_max_ptr,
                draft_max_stride,
                draft_sumexp_ptr,
                draft_sumexp_stride,
                logit_idx,
                vocab_num_blocks);
            draft_log_prob = draft_logit - draft_lse;
          }
          const float u = triton_rand32(
              static_cast<uint64_t>(seed_ptr[req_state]),
              static_cast<uint64_t>(pos_ptr[logit_idx]));
          accepted = valid_draft &&
              target_logit - target_lse > std::log(u) + draft_log_prob;
          sampled_ptr[req * sampled_stride + step] = draft_token;
        }
        rejected_step += accepted ? 1 : 0;
      }
      rejected_ptr[req] = rejected_step;
      target_rejected_ptr[req] = target_lse;
      draft_rejected_ptr[req] = draft_lse;
    }
  });
}

void vllm_rejection_impl(
    at::Tensor& sampled,
    at::Tensor& rejected_steps,
    at::Tensor& target_rejected_lse,
    at::Tensor& draft_rejected_lse,
    const at::Tensor& target_logits,
    const at::Tensor& target_local_argmax,
    const at::Tensor& target_local_max,
    const at::Tensor& target_local_sumexp,
    const at::Tensor& draft_sampled,
    const std::optional<at::Tensor>& draft_logits,
    const at::Tensor& draft_local_max,
    const at::Tensor& draft_local_sumexp,
    const at::Tensor& cu_num_logits,
    const at::Tensor& idx_mapping,
    const at::Tensor& temperature,
    const at::Tensor& seed,
    const at::Tensor& pos,
    int64_t vocab_num_blocks) {
  VLLM_MCPU_CHECK_DIM(sampled, 2, "sampled");
  VLLM_MCPU_CHECK_DIM(rejected_steps, 1, "rejected_steps");
  VLLM_MCPU_CHECK_DIM(target_logits, 2, "target_logits");
  VLLM_MCPU_CHECK_DTYPE(sampled, at::kLong, "sampled");
  VLLM_MCPU_CHECK_DTYPE(rejected_steps, at::kInt, "rejected_steps");
  VLLM_MCPU_CHECK_DTYPE(target_rejected_lse, at::kFloat, "target_rejected_lse");
  VLLM_MCPU_CHECK_DTYPE(draft_rejected_lse, at::kFloat, "draft_rejected_lse");
  VLLM_MCPU_CHECK_DTYPE(target_local_argmax, at::kLong, "target_local_argmax");
  VLLM_MCPU_CHECK_DTYPE(target_local_max, at::kFloat, "target_local_max");
  VLLM_MCPU_CHECK_DTYPE(target_local_sumexp, at::kFloat, "target_local_sumexp");
  VLLM_MCPU_CHECK_DTYPE(draft_sampled, at::kInt, "draft_sampled");
  VLLM_MCPU_CHECK_DTYPE(draft_local_max, at::kFloat, "draft_local_max");
  VLLM_MCPU_CHECK_DTYPE(draft_local_sumexp, at::kFloat, "draft_local_sumexp");
  VLLM_MCPU_CHECK_DTYPE(cu_num_logits, at::kInt, "cu_num_logits");
  VLLM_MCPU_CHECK_DTYPE(idx_mapping, at::kInt, "idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(temperature, at::kFloat, "temperature");
  VLLM_MCPU_CHECK_DTYPE(seed, at::kLong, "seed");
  VLLM_MCPU_CHECK_DTYPE(pos, at::kLong, "pos");
  VLLM_MCPU_CHECK_FLOAT(target_logits, "target_logits");
  const int64_t num_reqs = idx_mapping.numel();
  const int64_t num_logits = target_logits.size(0);
  VLLM_MCPU_CHECK(vocab_num_blocks > 0, "vocab_num_blocks must be positive");
  VLLM_MCPU_CHECK(
      sampled.size(0) == num_reqs && rejected_steps.numel() == num_reqs &&
          target_rejected_lse.numel() == num_reqs &&
          draft_rejected_lse.numel() == num_reqs &&
          cu_num_logits.numel() == num_reqs + 1,
      "rejection request dimensions mismatch");
  VLLM_MCPU_CHECK(
      draft_sampled.numel() == num_logits && pos.numel() == num_logits,
      "rejection token dimensions mismatch");
  VLLM_MCPU_CHECK(
      target_local_argmax.sizes() == target_local_max.sizes() &&
          target_local_max.sizes() == target_local_sumexp.sizes() &&
          draft_local_max.sizes() == target_local_max.sizes() &&
          draft_local_sumexp.sizes() == target_local_max.sizes() &&
          target_local_max.size(0) == num_logits &&
          target_local_max.size(1) == vocab_num_blocks,
      "rejection block-stat shapes mismatch");
  if (draft_logits) {
    VLLM_MCPU_CHECK(
        draft_logits->scalar_type() == target_logits.scalar_type(),
        "draft and target logits must have the same dtype");
  }
  VLLM_MCPU_DISPATCH_FLOAT(target_logits, "vllm_rejection", {
    launch_vllm_rejection<scalar_t>(
        sampled,
        rejected_steps,
        target_rejected_lse,
        draft_rejected_lse,
        target_logits,
        target_local_argmax,
        target_local_max,
        target_local_sumexp,
        draft_sampled,
        draft_logits,
        draft_local_max,
        draft_local_sumexp,
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        vocab_num_blocks);
  });
}

inline double triton_rand64(uint64_t seed, uint64_t offset) {
  const auto random = triton_philox(seed, offset);
  const uint64_t bits = (static_cast<uint64_t>(random.c1) << 32) | random.c0;
  return std::max(
      static_cast<double>(bits) * 5.421010862427522170037e-20,
      std::numeric_limits<double>::min());
}

template <typename scalar_t, typename out_t>
void launch_vllm_rejection_resample(
    at::Tensor& local_argmax,
    at::Tensor& local_max,
    const at::Tensor& target_logits,
    const at::Tensor& target_rejected_lse,
    const std::optional<at::Tensor>& draft_logits,
    const at::Tensor& draft_rejected_lse,
    const at::Tensor& rejected_step,
    const at::Tensor& cu_num_logits,
    const at::Tensor& expanded_idx_mapping,
    const at::Tensor& draft_sampled,
    const at::Tensor& temperature,
    const at::Tensor& seed,
    const at::Tensor& pos,
    int64_t vocab_size,
    int64_t block_size) {
  auto* argmax_ptr = local_argmax.data_ptr<int64_t>();
  auto* max_ptr = local_max.data_ptr<out_t>();
  const auto* target_ptr = target_logits.data_ptr<scalar_t>();
  const auto* target_lse_ptr = target_rejected_lse.data_ptr<float>();
  const auto* draft_ptr =
      draft_logits ? draft_logits->data_ptr<scalar_t>() : nullptr;
  const auto* draft_lse_ptr = draft_rejected_lse.data_ptr<float>();
  const auto* rejected_ptr = rejected_step.data_ptr<int32_t>();
  const auto* cu_ptr = cu_num_logits.data_ptr<int32_t>();
  const auto* mapping_ptr = expanded_idx_mapping.data_ptr<int32_t>();
  const auto* draft_sampled_ptr = draft_sampled.data_ptr<int32_t>();
  const auto* temp_ptr = temperature.data_ptr<float>();
  const auto* seed_ptr = seed.data_ptr<int64_t>();
  const auto* pos_ptr = pos.data_ptr<int64_t>();
  const int64_t num_reqs = rejected_step.numel();
  const int64_t num_blocks = local_argmax.size(1);
  const int64_t argmax_stride = local_argmax.stride(0);
  const int64_t max_stride = local_max.stride(0);
  const int64_t target_stride = target_logits.stride(0);
  const int64_t draft_stride_0 = draft_logits ? draft_logits->stride(0) : 0;
  const int64_t draft_stride_1 = draft_logits ? draft_logits->stride(1) : 0;

  MCPU_LAUNCH_TIMED_KERNEL("mcpu::vllm_rejection_resample", ([=]), {
    at::mcpu::KernelPointerMemoryGuard guard(
        {argmax_ptr,
         max_ptr,
         target_ptr,
         target_lse_ptr,
         draft_ptr,
         draft_lse_ptr,
         rejected_ptr,
         cu_ptr,
         mapping_ptr,
         draft_sampled_ptr,
         temp_ptr,
         seed_ptr,
         pos_ptr});
    const int64_t programs = num_reqs * num_blocks;
#pragma omp parallel for schedule(static)
    for (int64_t program = 0; program < programs; ++program) {
      const int64_t req = program / num_blocks;
      const int64_t block_idx = program % num_blocks;
      const int64_t resample_idx = rejected_ptr[req];
      const int64_t start = cu_ptr[req];
      const int64_t end = cu_ptr[req + 1];
      const int64_t token_idx = start + resample_idx;
      const int64_t req_state = mapping_ptr[token_idx];
      const float temp = temp_ptr[req_state];
      const bool is_bonus = token_idx == end - 1;
      if (temp == 0.0f && !is_bonus)
        continue;
      const int64_t begin = block_idx * block_size;
      const int64_t block_end = std::min(begin + block_size, vocab_size);
      out_t best_value = -std::numeric_limits<out_t>::infinity();
      int64_t best_offset = 0;
      const int64_t rejected_token =
          is_bonus ? -1 : draft_sampled_ptr[token_idx + 1];
      const int32_t gumbel_seed_i32 =
          static_cast<int32_t>(triton_philox(
                                   static_cast<uint64_t>(seed_ptr[req_state]),
                                   static_cast<uint64_t>(pos_ptr[token_idx]))
                                   .c0);
      const uint64_t gumbel_seed =
          static_cast<uint64_t>(static_cast<int64_t>(gumbel_seed_i32));
      for (int64_t token = begin; token < block_end; ++token) {
        const float target_logit =
            static_cast<float>(target_ptr[token_idx * target_stride + token]);
        float residual = target_logit;
        if (!is_bonus && draft_ptr) {
          const float target_log_prob = target_logit - target_lse_ptr[req];
          const float draft_log_prob =
              static_cast<float>(draft_ptr
                                     [req_state * draft_stride_0 +
                                      resample_idx * draft_stride_1 + token]) -
              draft_lse_ptr[req];
          const float ratio = std::exp(draft_log_prob - target_log_prob);
          residual = ratio < 1.0f ? target_log_prob + std::log1p(-ratio)
                                  : -std::numeric_limits<float>::infinity();
        } else if (!is_bonus && token == rejected_token) {
          residual = -std::numeric_limits<float>::infinity();
        }
        out_t value = static_cast<out_t>(residual);
        if (temp != 0.0f) {
          out_t u;
          if constexpr (std::is_same_v<out_t, double>) {
            u = triton_rand64(gumbel_seed, static_cast<uint64_t>(token));
          } else {
            u = triton_rand32(gumbel_seed, static_cast<uint64_t>(token));
          }
          value += -std::log(-std::log1p(-u));
        }
        if (value > best_value) {
          best_value = value;
          best_offset = token - begin;
        }
      }
      argmax_ptr[req * argmax_stride + block_idx] = begin + best_offset;
      max_ptr[req * max_stride + block_idx] = best_value;
    }
  });
}

void vllm_rejection_resample_impl(
    at::Tensor& local_argmax,
    at::Tensor& local_max,
    const at::Tensor& target_logits,
    const at::Tensor& target_rejected_lse,
    const std::optional<at::Tensor>& draft_logits,
    const at::Tensor& draft_rejected_lse,
    const at::Tensor& rejected_step,
    const at::Tensor& cu_num_logits,
    const at::Tensor& expanded_idx_mapping,
    const at::Tensor& draft_sampled,
    const at::Tensor& temperature,
    const at::Tensor& seed,
    const at::Tensor& pos,
    int64_t vocab_size,
    int64_t block_size,
    bool use_fp64) {
  VLLM_MCPU_CHECK_DIM(local_argmax, 2, "local_argmax");
  VLLM_MCPU_CHECK_DIM(local_max, 2, "local_max");
  VLLM_MCPU_CHECK_DTYPE(local_argmax, at::kLong, "local_argmax");
  VLLM_MCPU_CHECK(
      local_max.scalar_type() == (use_fp64 ? at::kDouble : at::kFloat),
      "local_max dtype does not match USE_FP64");
  VLLM_MCPU_CHECK_DTYPE(target_rejected_lse, at::kFloat, "target_rejected_lse");
  VLLM_MCPU_CHECK_DTYPE(draft_rejected_lse, at::kFloat, "draft_rejected_lse");
  VLLM_MCPU_CHECK_DTYPE(rejected_step, at::kInt, "rejected_step");
  VLLM_MCPU_CHECK_DTYPE(cu_num_logits, at::kInt, "cu_num_logits");
  VLLM_MCPU_CHECK_DTYPE(expanded_idx_mapping, at::kInt, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(draft_sampled, at::kInt, "draft_sampled");
  VLLM_MCPU_CHECK_DTYPE(temperature, at::kFloat, "temperature");
  VLLM_MCPU_CHECK_DTYPE(seed, at::kLong, "seed");
  VLLM_MCPU_CHECK_DTYPE(pos, at::kLong, "pos");
  VLLM_MCPU_CHECK_FLOAT(target_logits, "target_logits");
  VLLM_MCPU_CHECK(block_size == 1024, "resample BLOCK_SIZE must be 1024");
  const int64_t num_reqs = rejected_step.numel();
  const int64_t num_blocks = (vocab_size + block_size - 1) / block_size;
  VLLM_MCPU_CHECK(
      local_argmax.size(0) == num_reqs && local_argmax.size(1) == num_blocks &&
          local_max.sizes() == local_argmax.sizes(),
      "resample output shape mismatch");
  VLLM_MCPU_CHECK(
      cu_num_logits.numel() == num_reqs + 1 &&
          target_rejected_lse.numel() == num_reqs &&
          draft_rejected_lse.numel() == num_reqs,
      "resample request dimensions mismatch");
  if (draft_logits) {
    VLLM_MCPU_CHECK(
        draft_logits->scalar_type() == target_logits.scalar_type(),
        "draft and target logits must have the same dtype");
  }
  VLLM_MCPU_DISPATCH_FLOAT(target_logits, "vllm_rejection_resample", {
    if (use_fp64) {
      launch_vllm_rejection_resample<scalar_t, double>(
          local_argmax,
          local_max,
          target_logits,
          target_rejected_lse,
          draft_logits,
          draft_rejected_lse,
          rejected_step,
          cu_num_logits,
          expanded_idx_mapping,
          draft_sampled,
          temperature,
          seed,
          pos,
          vocab_size,
          block_size);
    } else {
      launch_vllm_rejection_resample<scalar_t, float>(
          local_argmax,
          local_max,
          target_logits,
          target_rejected_lse,
          draft_logits,
          draft_rejected_lse,
          rejected_step,
          cu_num_logits,
          expanded_idx_mapping,
          draft_sampled,
          temperature,
          seed,
          pos,
          vocab_size,
          block_size);
    }
  });
}

template <typename max_t>
void launch_vllm_rejection_insert(
    at::Tensor& sampled,
    at::Tensor& num_sampled,
    const at::Tensor& local_argmax,
    const at::Tensor& local_max,
    const at::Tensor& cu_num_logits,
    const at::Tensor& expanded_idx_mapping,
    const at::Tensor& temperature,
    int64_t resample_num_blocks) {
  auto* sampled_ptr = sampled.data_ptr<int64_t>();
  auto* num_sampled_ptr = num_sampled.data_ptr<int32_t>();
  const auto* argmax_ptr = local_argmax.data_ptr<int64_t>();
  const auto* max_ptr = local_max.data_ptr<max_t>();
  const auto* cu_ptr = cu_num_logits.data_ptr<int32_t>();
  const auto* mapping_ptr = expanded_idx_mapping.data_ptr<int32_t>();
  const auto* temp_ptr = temperature.data_ptr<float>();
  const int64_t num_reqs = num_sampled.numel();
  const int64_t sampled_stride = sampled.stride(0);
  const int64_t argmax_stride = local_argmax.stride(0);
  const int64_t max_stride = local_max.stride(0);
  MCPU_LAUNCH_TIMED_KERNEL("mcpu::vllm_rejection_insert", ([=]), {
    at::mcpu::KernelPointerMemoryGuard guard(
        {sampled_ptr,
         num_sampled_ptr,
         argmax_ptr,
         max_ptr,
         cu_ptr,
         mapping_ptr,
         temp_ptr});
#pragma omp parallel for schedule(static)
    for (int64_t req = 0; req < num_reqs; ++req) {
      const int32_t count = num_sampled_ptr[req];
      const int64_t token_idx = cu_ptr[req] + count;
      const int64_t req_state = mapping_ptr[token_idx];
      num_sampled_ptr[req] = count + 1;
      const bool is_bonus = token_idx == cu_ptr[req + 1] - 1;
      if (temp_ptr[req_state] == 0.0f && !is_bonus)
        continue;
      int64_t best_block = 0;
      max_t best_value = -std::numeric_limits<max_t>::infinity();
      for (int64_t block = 0; block < resample_num_blocks; ++block) {
        const max_t value = max_ptr[req * max_stride + block];
        if (value > best_value) {
          best_value = value;
          best_block = block;
        }
      }
      sampled_ptr[req * sampled_stride + count] =
          argmax_ptr[req * argmax_stride + best_block];
    }
  });
}

void vllm_rejection_insert_impl(
    at::Tensor& sampled,
    at::Tensor& num_sampled,
    const at::Tensor& local_argmax,
    const at::Tensor& local_max,
    const at::Tensor& cu_num_logits,
    const at::Tensor& expanded_idx_mapping,
    const at::Tensor& temperature,
    int64_t resample_num_blocks) {
  VLLM_MCPU_CHECK_DIM(sampled, 2, "sampled");
  VLLM_MCPU_CHECK_DIM(num_sampled, 1, "num_sampled");
  VLLM_MCPU_CHECK_DTYPE(sampled, at::kLong, "sampled");
  VLLM_MCPU_CHECK_DTYPE(num_sampled, at::kInt, "num_sampled");
  VLLM_MCPU_CHECK_DTYPE(local_argmax, at::kLong, "local_argmax");
  VLLM_MCPU_CHECK(
      local_max.scalar_type() == at::kFloat ||
          local_max.scalar_type() == at::kDouble,
      "local_max must be float32 or float64");
  VLLM_MCPU_CHECK_DTYPE(cu_num_logits, at::kInt, "cu_num_logits");
  VLLM_MCPU_CHECK_DTYPE(expanded_idx_mapping, at::kInt, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(temperature, at::kFloat, "temperature");
  VLLM_MCPU_CHECK(
      sampled.size(0) == num_sampled.numel() &&
          local_argmax.size(0) == num_sampled.numel() &&
          local_max.sizes() == local_argmax.sizes() &&
          local_argmax.size(1) == resample_num_blocks &&
          cu_num_logits.numel() == num_sampled.numel() + 1,
      "insert-resampled dimensions mismatch");
  if (local_max.scalar_type() == at::kDouble) {
    launch_vllm_rejection_insert<double>(
        sampled,
        num_sampled,
        local_argmax,
        local_max,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        resample_num_blocks);
  } else {
    launch_vllm_rejection_insert<float>(
        sampled,
        num_sampled,
        local_argmax,
        local_max,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        resample_num_blocks);
  }
}

void vllm_rejection_flatten_impl(
    at::Tensor& flat_sampled,
    const at::Tensor& sampled,
    const at::Tensor& num_sampled,
    const at::Tensor& cu_num_logits) {
  VLLM_MCPU_CHECK_DIM(flat_sampled, 1, "flat_sampled");
  VLLM_MCPU_CHECK_DIM(sampled, 2, "sampled");
  VLLM_MCPU_CHECK_DIM(num_sampled, 1, "num_sampled");
  VLLM_MCPU_CHECK_DIM(cu_num_logits, 1, "cu_num_logits");
  VLLM_MCPU_CHECK_DTYPE(flat_sampled, at::kLong, "flat_sampled");
  VLLM_MCPU_CHECK_DTYPE(sampled, at::kLong, "sampled");
  VLLM_MCPU_CHECK_DTYPE(num_sampled, at::kInt, "num_sampled");
  VLLM_MCPU_CHECK_DTYPE(cu_num_logits, at::kInt, "cu_num_logits");
  const int64_t num_reqs = num_sampled.numel();
  VLLM_MCPU_CHECK(
      sampled.size(0) == num_reqs && cu_num_logits.numel() == num_reqs + 1,
      "flatten-sampled request dimensions mismatch");
  auto* flat_ptr = flat_sampled.data_ptr<int64_t>();
  const auto* sampled_ptr = sampled.data_ptr<int64_t>();
  const auto* count_ptr = num_sampled.data_ptr<int32_t>();
  const auto* cu_ptr = cu_num_logits.data_ptr<int32_t>();
  const int64_t sampled_stride = sampled.stride(0);
  const int64_t sampled_width = sampled.size(1);
  const int64_t flat_size = flat_sampled.numel();
  MCPU_LAUNCH_TIMED_KERNEL("mcpu::vllm_rejection_flatten", ([=]), {
    at::mcpu::KernelPointerMemoryGuard guard(
        {flat_ptr, sampled_ptr, count_ptr, cu_ptr});
#pragma omp parallel for schedule(static)
    for (int64_t req = 0; req < num_reqs; ++req) {
      const int64_t start = cu_ptr[req];
      const int64_t count = count_ptr[req];
      // Structural shapes are checked synchronously; dynamic counts follow
      // Triton's trusted input contract.
      if (start < 0 || count < 0 || count > sampled_width ||
          start + count > flat_size)
        continue;
      for (int64_t i = 0; i < count; ++i) {
        flat_ptr[start + i] = sampled_ptr[req * sampled_stride + i];
      }
    }
  });
}

void vllm_rejection_greedy_impl(
    at::Tensor& output,
    const at::Tensor& cu_tokens,
    const at::Tensor& draft_ids,
    const at::Tensor& target_argmax,
    const at::Tensor& bonus_ids,
    const std::optional<at::Tensor>& is_greedy,
    int64_t max_spec_len) {
  VLLM_MCPU_CHECK_DIM(output, 2, "output");
  VLLM_MCPU_CHECK_DIM(cu_tokens, 1, "cu_tokens");
  VLLM_MCPU_CHECK_DIM(draft_ids, 1, "draft_ids");
  VLLM_MCPU_CHECK_DIM(target_argmax, 1, "target_argmax");
  VLLM_MCPU_CHECK_DIM(bonus_ids, 2, "bonus_ids");
  VLLM_MCPU_CHECK_DTYPE(output, at::kInt, "output");
  VLLM_MCPU_CHECK_DTYPE(cu_tokens, at::kInt, "cu_tokens");
  VLLM_MCPU_CHECK_DTYPE(draft_ids, at::kInt, "draft_ids");
  VLLM_MCPU_CHECK_DTYPE(target_argmax, at::kLong, "target_argmax");
  VLLM_MCPU_CHECK_DTYPE(bonus_ids, at::kInt, "bonus_ids");
  VLLM_MCPU_CHECK(output.is_contiguous(), "output must be contiguous");
  VLLM_MCPU_CHECK(cu_tokens.is_contiguous(), "cu_tokens must be contiguous");
  VLLM_MCPU_CHECK(draft_ids.is_contiguous(), "draft_ids must be contiguous");
  VLLM_MCPU_CHECK(
      target_argmax.is_contiguous(), "target_argmax must be contiguous");
  VLLM_MCPU_CHECK(bonus_ids.is_contiguous(), "bonus_ids must be contiguous");
  VLLM_MCPU_CHECK(
      output.size(0) == cu_tokens.numel(), "greedy batch size mismatch");
  VLLM_MCPU_CHECK(
      output.size(1) == max_spec_len + 1,
      "output width must be max_spec_len + 1");
  VLLM_MCPU_CHECK(
      target_argmax.numel() == draft_ids.numel(),
      "target_argmax and draft_ids size mismatch");
  VLLM_MCPU_CHECK(
      bonus_ids.size(0) == cu_tokens.numel() && bonus_ids.size(1) == 1,
      "bonus_ids must have shape [batch_size, 1]");
  if (is_greedy) {
    VLLM_MCPU_CHECK_DIM(*is_greedy, 1, "is_greedy");
    VLLM_MCPU_CHECK_DTYPE(*is_greedy, at::kBool, "is_greedy");
    VLLM_MCPU_CHECK(
        is_greedy->numel() == cu_tokens.numel(),
        "is_greedy batch size mismatch");
  }
  auto* out = output.data_ptr<int32_t>();
  const auto* cu = cu_tokens.data_ptr<int32_t>();
  const auto* draft = draft_ids.data_ptr<int32_t>();
  const auto* target = target_argmax.data_ptr<int64_t>();
  const auto* bonus = bonus_ids.data_ptr<int32_t>();
  const bool* greedy = is_greedy ? is_greedy->data_ptr<bool>() : nullptr;
  const int64_t batch = cu_tokens.numel();
  const int64_t stride = output.stride(0);
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::vllm_rejection_greedy",
      ([ out, cu, draft, target, bonus, greedy, batch, stride, max_spec_len ]),
      {
        at::mcpu::KernelPointerMemoryGuard guard(
            {out, cu, draft, target, bonus, greedy});
        for (int64_t r = 0; r < batch; ++r) {
          if (greedy && !greedy[r])
            continue;
          int32_t start = r ? cu[r - 1] : 0;
          int32_t end = cu[r];
          bool rejected = false;
          for (int32_t pos = 0; pos < end - start; ++pos) {
            int32_t id = static_cast<int32_t>(target[start + pos]);
            out[r * stride + pos] = id;
            if (draft[start + pos] != id) {
              rejected = true;
              break;
            }
          }
          if (!rejected)
            out[r * stride + end - start] = bonus[r];
        }
      });
}

template <typename inv_t>
void launch_recovered(
    at::Tensor& output,
    const at::Tensor& cu_tokens,
    const at::Tensor& draft_ids,
    const std::optional<at::Tensor>& draft_probs,
    const at::Tensor& target_probs,
    const at::Tensor& inv_q,
    int64_t vocab_size) {
  auto* out = output.data_ptr<int32_t>();
  const auto* cu = cu_tokens.data_ptr<int32_t>();
  const auto* draft = draft_ids.data_ptr<int32_t>();
  const float* dp = draft_probs ? draft_probs->data_ptr<float>() : nullptr;
  const auto* tp = target_probs.data_ptr<float>();
  const auto* iq = inv_q.data_ptr<inv_t>();
  int64_t batch = cu_tokens.numel();
  int64_t ps = target_probs.stride(0);
  int64_t ds = draft_probs ? draft_probs->stride(0) : 0;
  int64_t qs = inv_q.stride(0);
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::vllm_sample_recovered",
      ([ out, cu, draft, dp, tp, iq, batch, ps, ds, qs, vocab_size ]),
      {
        at::mcpu::KernelPointerMemoryGuard guard({out, cu, draft, dp, tp, iq});
        for (int64_t r = 0; r < batch; ++r) {
          int32_t start = r ? cu[r - 1] : 0;
          for (int32_t t = start; t < cu[r]; ++t) {
            int32_t best = 0;
            double best_val = -1.0;
            for (int64_t v = 0; v < vocab_size; ++v) {
              float prob = tp[t * ps + v];
              prob = dp ? std::max(prob - dp[t * ds + v], 0.0f)
                        : (v == draft[t] ? 0.0f : prob);
              double val = prob * static_cast<double>(iq[r * qs + v]);
              if (val > best_val) {
                best_val = val;
                best = v;
              }
            }
            out[t] = best;
          }
        }
      });
}

void vllm_sample_recovered_impl(
    at::Tensor& output,
    const at::Tensor& cu_tokens,
    const at::Tensor& draft_ids,
    const std::optional<at::Tensor>& draft_probs,
    const at::Tensor& target_probs,
    const at::Tensor& inv_q,
    int64_t vocab_size) {
  VLLM_MCPU_CHECK_DIM(output, 1, "output");
  VLLM_MCPU_CHECK_DIM(cu_tokens, 1, "cu_tokens");
  VLLM_MCPU_CHECK_DIM(draft_ids, 1, "draft_ids");
  VLLM_MCPU_CHECK_DIM(target_probs, 2, "target_probs");
  VLLM_MCPU_CHECK_DIM(inv_q, 2, "inv_q");
  VLLM_MCPU_CHECK_DTYPE(output, at::kInt, "output");
  VLLM_MCPU_CHECK_DTYPE(cu_tokens, at::kInt, "cu_tokens");
  VLLM_MCPU_CHECK_DTYPE(draft_ids, at::kInt, "draft_ids");
  VLLM_MCPU_CHECK_DTYPE(target_probs, at::kFloat, "target_probs");
  VLLM_MCPU_CHECK(
      inv_q.scalar_type() == at::kFloat || inv_q.scalar_type() == at::kDouble,
      "inv_q must be float32 or float64");
  VLLM_MCPU_CHECK(
      output.numel() == draft_ids.numel(),
      "output and draft_ids size mismatch");
  VLLM_MCPU_CHECK(
      target_probs.size(0) == draft_ids.numel(),
      "target_probs token count mismatch");
  VLLM_MCPU_CHECK(
      target_probs.size(1) == vocab_size, "target_probs vocab size mismatch");
  VLLM_MCPU_CHECK(
      inv_q.size(0) == cu_tokens.numel() && inv_q.size(1) == vocab_size,
      "inv_q must have shape [batch_size, vocab_size]");
  if (draft_probs) {
    VLLM_MCPU_CHECK_DIM(*draft_probs, 2, "draft_probs");
    VLLM_MCPU_CHECK_DTYPE(*draft_probs, at::kFloat, "draft_probs");
    VLLM_MCPU_CHECK(
        draft_probs->sizes() == target_probs.sizes(),
        "draft_probs and target_probs shape mismatch");
  }
  if (inv_q.scalar_type() == at::kDouble)
    launch_recovered<double>(
        output,
        cu_tokens,
        draft_ids,
        draft_probs,
        target_probs,
        inv_q,
        vocab_size);
  else
    launch_recovered<float>(
        output,
        cu_tokens,
        draft_ids,
        draft_probs,
        target_probs,
        inv_q,
        vocab_size);
}

void vllm_rejection_random_impl(
    at::Tensor& output,
    const at::Tensor& cu_tokens,
    const at::Tensor& draft_ids,
    const std::optional<at::Tensor>& draft_probs,
    const at::Tensor& target_probs,
    const at::Tensor& bonus_ids,
    const at::Tensor& recovered_ids,
    const at::Tensor& uniform_probs,
    const std::optional<at::Tensor>& is_greedy,
    int64_t max_spec_len,
    int64_t vocab_size) {
  VLLM_MCPU_CHECK_DIM(output, 2, "output");
  VLLM_MCPU_CHECK_DIM(cu_tokens, 1, "cu_tokens");
  VLLM_MCPU_CHECK_DIM(draft_ids, 1, "draft_ids");
  VLLM_MCPU_CHECK_DIM(target_probs, 2, "target_probs");
  VLLM_MCPU_CHECK_DIM(bonus_ids, 2, "bonus_ids");
  VLLM_MCPU_CHECK_DIM(recovered_ids, 1, "recovered_ids");
  VLLM_MCPU_CHECK_DIM(uniform_probs, 1, "uniform_probs");
  VLLM_MCPU_CHECK_DTYPE(output, at::kInt, "output");
  VLLM_MCPU_CHECK_DTYPE(cu_tokens, at::kInt, "cu_tokens");
  VLLM_MCPU_CHECK_DTYPE(draft_ids, at::kInt, "draft_ids");
  VLLM_MCPU_CHECK_DTYPE(target_probs, at::kFloat, "target_probs");
  VLLM_MCPU_CHECK_DTYPE(bonus_ids, at::kInt, "bonus_ids");
  VLLM_MCPU_CHECK_DTYPE(recovered_ids, at::kInt, "recovered_ids");
  VLLM_MCPU_CHECK_DTYPE(uniform_probs, at::kDouble, "uniform_probs");
  VLLM_MCPU_CHECK(
      output.size(0) == cu_tokens.numel() && output.size(1) == max_spec_len + 1,
      "random output shape mismatch");
  VLLM_MCPU_CHECK(
      target_probs.size(0) == draft_ids.numel() &&
          target_probs.size(1) == vocab_size,
      "target_probs shape mismatch");
  VLLM_MCPU_CHECK(
      recovered_ids.numel() == draft_ids.numel() &&
          uniform_probs.numel() == draft_ids.numel(),
      "per-token input size mismatch");
  VLLM_MCPU_CHECK(
      bonus_ids.size(0) == cu_tokens.numel() && bonus_ids.size(1) == 1,
      "bonus_ids must have shape [batch_size, 1]");
  if (draft_probs) {
    VLLM_MCPU_CHECK_DIM(*draft_probs, 2, "draft_probs");
    VLLM_MCPU_CHECK_DTYPE(*draft_probs, at::kFloat, "draft_probs");
    VLLM_MCPU_CHECK(
        draft_probs->sizes() == target_probs.sizes(),
        "draft_probs and target_probs shape mismatch");
  }
  if (is_greedy) {
    VLLM_MCPU_CHECK_DIM(*is_greedy, 1, "is_greedy");
    VLLM_MCPU_CHECK_DTYPE(*is_greedy, at::kBool, "is_greedy");
    VLLM_MCPU_CHECK(
        is_greedy->numel() == cu_tokens.numel(),
        "is_greedy batch size mismatch");
  }
  auto* out = output.data_ptr<int32_t>();
  const auto* cu = cu_tokens.data_ptr<int32_t>();
  const auto* draft = draft_ids.data_ptr<int32_t>();
  const float* dp = draft_probs ? draft_probs->data_ptr<float>() : nullptr;
  const auto* tp = target_probs.data_ptr<float>();
  const auto* bonus = bonus_ids.data_ptr<int32_t>();
  const auto* recovered = recovered_ids.data_ptr<int32_t>();
  const auto* uniform = uniform_probs.data_ptr<double>();
  const bool* greedy = is_greedy ? is_greedy->data_ptr<bool>() : nullptr;
  int64_t batch = cu_tokens.numel(), os = output.stride(0);
  int64_t ps = target_probs.stride(0),
          ds = draft_probs ? draft_probs->stride(0) : 0;
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::vllm_rejection_random",
      ([
        out,
        cu,
        draft,
        dp,
        tp,
        bonus,
        recovered,
        uniform,
        greedy,
        batch,
        os,
        ps,
        ds,
        max_spec_len,
        vocab_size
      ]),
      {
        at::mcpu::KernelPointerMemoryGuard guard(
            {out, cu, draft, dp, tp, bonus, recovered, uniform, greedy});
        for (int64_t r = 0; r < batch; ++r) {
          if (greedy && greedy[r])
            continue;
          int32_t start = r ? cu[r - 1] : 0, end = cu[r];
          bool rejected = false;
          for (int32_t pos = 0; pos < end - start; ++pos) {
            int32_t t = start + pos, id = draft[t];
            bool accepted = false;
            if (id >= 0 && id < vocab_size) {
              float q = dp ? dp[t * ds + id] : 1.0f;
              accepted = q > 0 && tp[t * ps + id] / q >= uniform[t];
            }
            out[r * os + pos] = accepted ? id : recovered[t];
            if (!accepted) {
              rejected = true;
              break;
            }
          }
          if (!rejected)
            out[r * os + end - start] = bonus[r];
        }
      });
}

template <typename scalar_t>
void vllm_rejection_sampler_expand_typed(
    scalar_t* output_ptr,
    const scalar_t* input_ptr,
    const int32_t* cu_num_tokens_ptr,
    int64_t batch_size,
    scalar_t replace_from,
    scalar_t replace_to,
    int64_t max_num_tokens) {
  for (int64_t req_idx = 0; req_idx < batch_size; ++req_idx) {
    const int32_t start_idx = req_idx == 0 ? 0 : cu_num_tokens_ptr[req_idx - 1];
    const int32_t end_idx = cu_num_tokens_ptr[req_idx];
    const int32_t write_end =
        std::min<int64_t>(end_idx, start_idx + max_num_tokens);
    scalar_t value = input_ptr[req_idx];
    if (value == replace_from) {
      value = replace_to;
    }
    for (int32_t token_idx = start_idx; token_idx < write_end; ++token_idx) {
      output_ptr[token_idx] = value;
    }
  }
}

template <typename scalar_t>
void launch_vllm_rejection_sampler_expand(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& cu_num_tokens,
    const at::Scalar& replace_from,
    const at::Scalar& replace_to,
    int64_t max_num_tokens) {
  scalar_t* output_ptr = output.data_ptr<scalar_t>();
  const scalar_t* input_ptr = input.data_ptr<scalar_t>();
  const int32_t* cu_num_tokens_ptr = cu_num_tokens.data_ptr<int32_t>();
  const int64_t batch_size = input.numel();
  const scalar_t replace_from_value = replace_from.to<scalar_t>();
  const scalar_t replace_to_value = replace_to.to<scalar_t>();

  at::mcpu::launch_timed_kernel(
      "mcpu::vllm_rejection_sampler_expand",
      [output_ptr,
       input_ptr,
       cu_num_tokens_ptr,
       batch_size,
       replace_from_value,
       replace_to_value,
       max_num_tokens](at::mcpu::kernel_timing::Event* timing_event) mutable {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::vllm_rejection_sampler_expand", timing_event);
        at::mcpu::KernelPointerMemoryGuard guard(
            {output_ptr, input_ptr, cu_num_tokens_ptr});
        vllm_rejection_sampler_expand_typed<scalar_t>(
            output_ptr,
            input_ptr,
            cu_num_tokens_ptr,
            batch_size,
            replace_from_value,
            replace_to_value,
            max_num_tokens);
      });
}

void vllm_rejection_sampler_expand_impl(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& cu_num_tokens,
    const at::Scalar& replace_from,
    const at::Scalar& replace_to,
    int64_t max_num_tokens) {
  VLLM_MCPU_CHECK_DIM(output, 1, "output");
  VLLM_MCPU_CHECK_DIM(input, 1, "input");
  VLLM_MCPU_CHECK_DIM(cu_num_tokens, 1, "cu_num_tokens");
  VLLM_MCPU_CHECK(output.is_contiguous(), "output must be contiguous");
  VLLM_MCPU_CHECK(input.is_contiguous(), "input must be contiguous");
  VLLM_MCPU_CHECK(
      cu_num_tokens.is_contiguous(), "cu_num_tokens must be contiguous");
  VLLM_MCPU_CHECK(
      output.scalar_type() == input.scalar_type(),
      "output and input must have the same dtype");
  VLLM_MCPU_CHECK_DTYPE(cu_num_tokens, at::kInt, "cu_num_tokens");
  VLLM_MCPU_CHECK(
      cu_num_tokens.numel() == input.numel(),
      "cu_num_tokens and input must have the same number of elements");
  VLLM_MCPU_CHECK(max_num_tokens > 0, "max_num_tokens must be positive");

  switch (input.scalar_type()) {
    case at::kFloat:
      launch_vllm_rejection_sampler_expand<float>(
          output,
          input,
          cu_num_tokens,
          replace_from,
          replace_to,
          max_num_tokens);
      break;
    case at::kHalf:
      launch_vllm_rejection_sampler_expand<at::Half>(
          output,
          input,
          cu_num_tokens,
          replace_from,
          replace_to,
          max_num_tokens);
      break;
    case at::kBFloat16:
      launch_vllm_rejection_sampler_expand<at::BFloat16>(
          output,
          input,
          cu_num_tokens,
          replace_from,
          replace_to,
          max_num_tokens);
      break;
    case at::kInt:
      launch_vllm_rejection_sampler_expand<int32_t>(
          output,
          input,
          cu_num_tokens,
          replace_from,
          replace_to,
          max_num_tokens);
      break;
    case at::kLong:
      launch_vllm_rejection_sampler_expand<int64_t>(
          output,
          input,
          cu_num_tokens,
          replace_from,
          replace_to,
          max_num_tokens);
      break;
    default:
      TORCH_CHECK(
          false,
          "vllm_rejection_sampler_expand: unsupported dtype ",
          input.scalar_type());
  }
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_rejection_compute_block_stats("
      "Tensor(a!) target_local_argmax, "
      "Tensor(b!) target_local_max, "
      "Tensor(c!) target_local_sumexp, "
      "Tensor(d!) draft_local_max, "
      "Tensor(e!) draft_local_sumexp, "
      "Tensor target_logits, "
      "Tensor? draft_logits, "
      "Tensor expanded_idx_mapping, "
      "Tensor expanded_local_pos, "
      "Tensor temperature, "
      "int vocab_size, "
      "int num_speculative_steps, "
      "int block_size"
      ") -> ()");
  m.def(
      "vllm_rejection("
      "Tensor(a!) sampled, Tensor(b!) rejected_steps, "
      "Tensor(c!) target_rejected_lse, Tensor(d!) draft_rejected_lse, "
      "Tensor target_logits, Tensor target_local_argmax, "
      "Tensor target_local_max, Tensor target_local_sumexp, "
      "Tensor draft_sampled, Tensor? draft_logits, "
      "Tensor draft_local_max, Tensor draft_local_sumexp, "
      "Tensor cu_num_logits, Tensor idx_mapping, Tensor temperature, "
      "Tensor seed, Tensor pos, int vocab_num_blocks"
      ") -> ()");
  m.def(
      "vllm_rejection_resample("
      "Tensor(a!) local_argmax, Tensor(b!) local_max, "
      "Tensor target_logits, Tensor target_rejected_lse, "
      "Tensor? draft_logits, Tensor draft_rejected_lse, "
      "Tensor rejected_step, Tensor cu_num_logits, "
      "Tensor expanded_idx_mapping, Tensor draft_sampled, "
      "Tensor temperature, Tensor seed, Tensor pos, "
      "int vocab_size, int block_size, bool use_fp64"
      ") -> ()");
  m.def(
      "vllm_rejection_insert("
      "Tensor(a!) sampled, Tensor(b!) num_sampled, "
      "Tensor local_argmax, Tensor local_max, Tensor cu_num_logits, "
      "Tensor expanded_idx_mapping, Tensor temperature, "
      "int resample_num_blocks"
      ") -> ()");
  m.def(
      "vllm_rejection_flatten("
      "Tensor(a!) flat_sampled, Tensor sampled, Tensor num_sampled, "
      "Tensor cu_num_logits"
      ") -> ()");
  m.def(
      "vllm_rejection_greedy(Tensor(a!) output, Tensor cu_tokens, Tensor draft_ids, Tensor target_argmax, Tensor bonus_ids, Tensor? is_greedy, int max_spec_len) -> ()");
  m.def(
      "vllm_sample_recovered(Tensor(a!) output, Tensor cu_tokens, Tensor draft_ids, Tensor? draft_probs, Tensor target_probs, Tensor inv_q, int vocab_size) -> ()");
  m.def(
      "vllm_rejection_random(Tensor(a!) output, Tensor cu_tokens, Tensor draft_ids, Tensor? draft_probs, Tensor target_probs, Tensor bonus_ids, Tensor recovered_ids, Tensor uniform_probs, Tensor? is_greedy, int max_spec_len, int vocab_size) -> ()");
  m.def(
      "vllm_rejection_sampler_expand("
      "Tensor(a!) output, "
      "Tensor input, "
      "Tensor cu_num_tokens, "
      "Scalar replace_from, "
      "Scalar replace_to, "
      "int max_num_tokens"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl(
      "vllm_rejection_compute_block_stats",
      &vllm_rejection_compute_block_stats_impl);
  m.impl("vllm_rejection", &vllm_rejection_impl);
  m.impl("vllm_rejection_resample", &vllm_rejection_resample_impl);
  m.impl("vllm_rejection_insert", &vllm_rejection_insert_impl);
  m.impl("vllm_rejection_flatten", &vllm_rejection_flatten_impl);
  m.impl("vllm_rejection_greedy", &vllm_rejection_greedy_impl);
  m.impl("vllm_sample_recovered", &vllm_sample_recovered_impl);
  m.impl("vllm_rejection_random", &vllm_rejection_random_impl);
  m.impl("vllm_rejection_sampler_expand", &vllm_rejection_sampler_expand_impl);
}
