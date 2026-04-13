// SPDX-License-Identifier: Apache-2.0
//
// C++ kernels for vllm/v1/worker/gpu/sample/gumbel.py:
//   _temperature_kernel   — divide logits by per-request temperature
//   _gumbel_sample_kernel — Gumbel-max sampling (returns argmax)

#include <random>
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

// ---------------------------------------------------------------------------
// _gumbel_sample_kernel  (returns sampled token IDs, shape [num_tokens])
//
// RNG: mt19937_64 seeded with (seed[req] ^ pos[tok]) & 0x7fff...
// ---------------------------------------------------------------------------
template <typename scalar_t>
static void vllm_gumbel_sample_typed(
    scalar_t* logits_ptr,
    int64_t* result_ptr,
    float* proc_ptr, // nullable; always float32
    int64_t num_tokens,
    int64_t logits_stride,
    int64_t proc_stride,
    const int32_t* idx_ptr,
    const float* temp_ptr,
    const int64_t* seed_ptr,
    const int64_t* pos_ptr,
    int64_t vocab_size,
    bool apply_temperature) {
  const double tiny = std::numeric_limits<double>::min();

  for (int64_t tok = 0; tok < num_tokens; tok++) {
    int32_t req = idx_ptr[tok];
    float temp = temp_ptr[req];
    scalar_t* row = logits_ptr + tok * logits_stride;

    // Apply temperature in-place if requested
    if (apply_temperature && temp != 0.0f && temp != 1.0f) {
      if constexpr (std::is_same_v<scalar_t, float>) {
        for (int64_t i = 0; i < vocab_size; i++)
          row[i] = row[i] / temp;
      } else {
        for (int64_t i = 0; i < vocab_size; i++) {
          row[i] = static_cast<scalar_t>(static_cast<float>(row[i]) / temp);
        }
      }
    }

    // Copy to processed_logits_out[req, :] (always float32)
    if (proc_ptr != nullptr) {
      float* proc_row = proc_ptr + (int64_t)req * proc_stride;
      if constexpr (std::is_same_v<scalar_t, float>) {
        for (int64_t i = 0; i < vocab_size; i++)
          proc_row[i] = row[i];
      } else {
        for (int64_t i = 0; i < vocab_size; i++) {
          proc_row[i] = static_cast<float>(row[i]);
        }
      }
    }

    bool stochastic = (temp != 0.0f);
    double best_val = -std::numeric_limits<double>::infinity();
    int64_t best_idx = 0;

    if (stochastic) {
      int64_t req_seed = seed_ptr[req];
      int64_t token_pos = pos_ptr[tok];
      uint64_t combined =
          (uint64_t)((req_seed ^ token_pos) & (int64_t)0x7FFFFFFFFFFFFFFF);
      std::mt19937_64 rng(combined);
      std::uniform_real_distribution<double> dist(0.0, 1.0);

      for (int64_t i = 0; i < vocab_size; i++) {
        double u = dist(rng);
        if (u < tiny)
          u = tiny;
        double gumbel = -std::log(-std::log(u));
        double logit_f;
        if constexpr (std::is_same_v<scalar_t, float>) {
          logit_f = (double)row[i];
        } else {
          logit_f = (double)static_cast<float>(row[i]);
        }
        double val = logit_f + gumbel;
        if (val > best_val) {
          best_val = val;
          best_idx = i;
        }
      }
    } else {
      // Greedy: argmax
      for (int64_t i = 0; i < vocab_size; i++) {
        double v;
        if constexpr (std::is_same_v<scalar_t, float>) {
          v = (double)row[i];
        } else {
          v = (double)static_cast<float>(row[i]);
        }
        if (v > best_val) {
          best_val = v;
          best_idx = i;
        }
      }
    }
    result_ptr[tok] = best_idx;
  }
}

at::Tensor vllm_gumbel_sample_impl(
    at::Tensor& logits, // [num_tokens, vocab_size]
    const at::Tensor& expanded_idx_mapping, // [num_tokens], int32
    const at::Tensor& temperature, // [max_num_reqs], float32
    const at::Tensor& seed, // [max_num_reqs], int64
    const at::Tensor& pos, // [num_tokens], int64
    int64_t vocab_size,
    const std::optional<at::Tensor>& processed_logits_out,
    bool apply_temperature) {
  VLLM_MCPU_CHECK_DIM(logits, 2, "logits");
  VLLM_MCPU_CHECK_FLOAT(logits, "logits");
  VLLM_MCPU_CHECK_DIM(expanded_idx_mapping, 1, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(expanded_idx_mapping, at::kInt, "expanded_idx_mapping");
  VLLM_MCPU_CHECK_DIM(temperature, 1, "temperature");
  VLLM_MCPU_CHECK_DTYPE(temperature, at::kFloat, "temperature");
  VLLM_MCPU_CHECK_DIM(seed, 1, "seed");
  VLLM_MCPU_CHECK_DTYPE(seed, at::kLong, "seed");
  VLLM_MCPU_CHECK_DIM(pos, 1, "pos");
  VLLM_MCPU_CHECK_DTYPE(pos, at::kLong, "pos");

  int64_t num_tokens = logits.size(0);
  int64_t logits_stride = logits.stride(0);
  const int32_t* idx_ptr = expanded_idx_mapping.data_ptr<int32_t>();
  const float* temp_ptr = temperature.data_ptr<float>();
  const int64_t* seed_ptr = seed.data_ptr<int64_t>();
  const int64_t* pos_ptr = pos.data_ptr<int64_t>();

  float* proc_ptr = nullptr;
  int64_t proc_stride = 0;
  if (processed_logits_out.has_value() && processed_logits_out->defined()) {
    proc_ptr = processed_logits_out->data_ptr<float>();
    proc_stride = processed_logits_out->stride(0);
  }

  at::Tensor result = at::zeros(
      {num_tokens},
      at::TensorOptions().dtype(at::kLong).device(logits.device()));
  int64_t* result_ptr = result.data_ptr<int64_t>();

  VLLM_MCPU_DISPATCH_FLOAT(logits, "vllm_gumbel_sample", {
    vllm_gumbel_sample_typed<scalar_t>(
        logits.data_ptr<scalar_t>(),
        result_ptr,
        proc_ptr,
        num_tokens,
        logits_stride,
        proc_stride,
        idx_ptr,
        temp_ptr,
        seed_ptr,
        pos_ptr,
        vocab_size,
        apply_temperature);
  });
  return result;
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
  m.def(
      "vllm_gumbel_sample("
      "Tensor(a!) logits, "
      "Tensor expanded_idx_mapping, "
      "Tensor temperature, "
      "Tensor seed, "
      "Tensor pos, "
      "int vocab_size, "
      "Tensor? processed_logits_out, "
      "bool apply_temperature"
      ") -> Tensor");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_temperature_kernel", &vllm_temperature_kernel_impl);
  m.impl("vllm_gumbel_sample", &vllm_gumbel_sample_impl);
}
