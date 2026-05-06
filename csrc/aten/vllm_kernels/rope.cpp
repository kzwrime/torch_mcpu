// SPDX-License-Identifier: Apache-2.0
//
// C++ kernel for vllm/v1/worker/gpu/mm/rope.py::_prepare_rope_positions_kernel

#include "common.h"

namespace {

template <
    typename idx_t,
    typename query_loc_t,
    typename prefill_len_t,
    typename computed_t>
void prepare_rope_positions_kernel_typed(
    int64_t* __restrict__ positions_ptr,
    int64_t positions_stride,
    const int32_t* __restrict__ prefill_positions_ptr,
    int64_t prefill_positions_stride0,
    int64_t prefill_positions_stride1,
    const int32_t* __restrict__ prefill_delta_ptr,
    const idx_t* __restrict__ idx_mapping_ptr,
    const query_loc_t* __restrict__ query_start_loc_ptr,
    const prefill_len_t* __restrict__ prefill_lens_ptr,
    const computed_t* __restrict__ num_computed_tokens_ptr,
    int64_t num_reqs,
    int64_t num_dims) {
#pragma omp parallel for
  for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
    const int64_t req_state_idx =
        static_cast<int64_t>(idx_mapping_ptr[batch_idx]);
    const int64_t prefill_len =
        static_cast<int64_t>(prefill_lens_ptr[req_state_idx]);
    const int64_t num_computed =
        static_cast<int64_t>(num_computed_tokens_ptr[req_state_idx]);
    const bool is_prefill = num_computed < prefill_len;

    const int64_t query_start =
        static_cast<int64_t>(query_start_loc_ptr[batch_idx]);
    const int64_t query_end =
        static_cast<int64_t>(query_start_loc_ptr[batch_idx + 1]);
    const int64_t query_len = query_end - query_start;
    if (query_len <= 0) {
      continue;
    }

    const int64_t delta =
        static_cast<int64_t>(prefill_delta_ptr[req_state_idx]);
    for (int64_t dim = 0; dim < num_dims; ++dim) {
      int64_t* __restrict__ out =
          positions_ptr + dim * positions_stride + query_start;
      if (is_prefill) {
        const int32_t* __restrict__ src = prefill_positions_ptr +
            req_state_idx * prefill_positions_stride0 +
            dim * prefill_positions_stride1 + num_computed;
        for (int64_t token = 0; token < query_len; ++token) {
          out[token] = static_cast<int64_t>(src[token]);
        }
      } else {
        for (int64_t token = 0; token < query_len; ++token) {
          out[token] = num_computed + token + delta;
        }
      }
    }
  }
}

#define VLLM_MCPU_DISPATCH_INT_TENSOR(TYPE, TENSOR, NAME, ...) \
  do {                                                         \
    switch ((TENSOR).scalar_type()) {                          \
      case at::kInt: {                                         \
        using TYPE = int32_t;                                  \
        __VA_ARGS__;                                           \
        break;                                                 \
      }                                                        \
      case at::kLong: {                                        \
        using TYPE = int64_t;                                  \
        __VA_ARGS__;                                           \
        break;                                                 \
      }                                                        \
      default:                                                 \
        TORCH_CHECK(                                           \
            false,                                             \
            NAME ": expected int32 or int64 tensor, got ",     \
            (TENSOR).scalar_type());                           \
    }                                                          \
  } while (0)

void prepare_rope_positions_kernel_impl(
    at::Tensor& positions,
    int64_t positions_stride,
    const at::Tensor& prefill_positions,
    int64_t prefill_positions_stride0,
    int64_t prefill_positions_stride1,
    const at::Tensor& prefill_delta,
    const at::Tensor& idx_mapping,
    const at::Tensor& query_start_loc,
    const at::Tensor& prefill_lens,
    const at::Tensor& num_computed_tokens,
    int64_t num_dims) {
  VLLM_MCPU_CHECK_DIM(positions, 2, "positions");
  VLLM_MCPU_CHECK_DIM(prefill_positions, 2, "prefill_positions");
  VLLM_MCPU_CHECK_DIM(prefill_delta, 1, "prefill_delta");
  VLLM_MCPU_CHECK_DIM(idx_mapping, 1, "idx_mapping");
  VLLM_MCPU_CHECK_DIM(query_start_loc, 1, "query_start_loc");
  VLLM_MCPU_CHECK_DIM(prefill_lens, 1, "prefill_lens");
  VLLM_MCPU_CHECK_DIM(num_computed_tokens, 1, "num_computed_tokens");
  VLLM_MCPU_CHECK_DTYPE(positions, at::kLong, "positions");
  VLLM_MCPU_CHECK_DTYPE(prefill_positions, at::kInt, "prefill_positions");
  VLLM_MCPU_CHECK_DTYPE(prefill_delta, at::kInt, "prefill_delta");
  VLLM_MCPU_CHECK(num_dims > 0, "num_dims must be positive");
  VLLM_MCPU_CHECK(
      positions.size(0) >= num_dims, "positions must cover num_dims rows");

  const int64_t num_reqs = idx_mapping.size(0);
  int64_t* __restrict__ positions_ptr = positions.data_ptr<int64_t>();
  const int32_t* __restrict__ prefill_positions_ptr =
      prefill_positions.data_ptr<int32_t>();
  const int32_t* __restrict__ prefill_delta_ptr =
      prefill_delta.data_ptr<int32_t>();

  VLLM_MCPU_DISPATCH_INT_TENSOR(idx_t, idx_mapping, "idx_mapping", {
    VLLM_MCPU_DISPATCH_INT_TENSOR(
        query_loc_t, query_start_loc, "query_start_loc", {
          VLLM_MCPU_DISPATCH_INT_TENSOR(
              prefill_len_t, prefill_lens, "prefill_lens", {
                VLLM_MCPU_DISPATCH_INT_TENSOR(
                    computed_t, num_computed_tokens, "num_computed_tokens", {
                      prepare_rope_positions_kernel_typed<
                          idx_t,
                          query_loc_t,
                          prefill_len_t,
                          computed_t>(
                          positions_ptr,
                          positions_stride,
                          prefill_positions_ptr,
                          prefill_positions_stride0,
                          prefill_positions_stride1,
                          prefill_delta_ptr,
                          idx_mapping.data_ptr<idx_t>(),
                          query_start_loc.data_ptr<query_loc_t>(),
                          prefill_lens.data_ptr<prefill_len_t>(),
                          num_computed_tokens.data_ptr<computed_t>(),
                          num_reqs,
                          num_dims);
                    });
              });
        });
  });
}

#undef VLLM_MCPU_DISPATCH_INT_TENSOR

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "prepare_rope_positions_kernel_impl("
      "Tensor(a!) positions, "
      "SymInt positions_stride, "
      "Tensor prefill_positions, "
      "SymInt prefill_positions_stride0, "
      "SymInt prefill_positions_stride1, "
      "Tensor prefill_delta, "
      "Tensor idx_mapping, "
      "Tensor query_start_loc, "
      "Tensor prefill_lens, "
      "Tensor num_computed_tokens, "
      "SymInt num_dims"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl(
      "prepare_rope_positions_kernel_impl",
      &prepare_rope_positions_kernel_impl);
}
