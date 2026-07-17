// SPDX-License-Identifier: Apache-2.0
//
// C++ kernels for vllm/v1/worker/gpu/model_states/mamba_hybrid.py.

#include "common.h"

#include <algorithm>

namespace {

void vllm_scatter_num_accepted_impl(
    const at::Tensor& idx_mapping,
    const at::Tensor& num_sampled,
    at::Tensor& num_accepted) {
  VLLM_MCPU_CHECK_DIM(idx_mapping, 1, "idx_mapping");
  VLLM_MCPU_CHECK_DIM(num_sampled, 1, "num_sampled");
  VLLM_MCPU_CHECK_DIM(num_accepted, 1, "num_accepted");
  VLLM_MCPU_CHECK_DTYPE(idx_mapping, at::kInt, "idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(num_sampled, at::kInt, "num_sampled");
  VLLM_MCPU_CHECK_DTYPE(num_accepted, at::kInt, "num_accepted");
  VLLM_MCPU_CHECK(
      idx_mapping.numel() == num_sampled.numel(),
      "idx_mapping and num_sampled must have equal lengths");

  const int64_t num_reqs = idx_mapping.numel();
  const int32_t* idx_ptr = idx_mapping.data_ptr<int32_t>();
  const int32_t* sampled_ptr = num_sampled.data_ptr<int32_t>();
  int32_t* accepted_ptr = num_accepted.data_ptr<int32_t>();
  const int64_t max_num_reqs = num_accepted.numel();

  at::mcpu::launch_timed_kernel(
      "mcpu::vllm_scatter_num_accepted",
      [idx_ptr, sampled_ptr, accepted_ptr, num_reqs, max_num_reqs](
          at::mcpu::kernel_timing::Event* timing_event) mutable {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::vllm_scatter_num_accepted", timing_event);
        at::mcpu::KernelPointerMemoryGuard guard(
            {idx_ptr, sampled_ptr, accepted_ptr});
        for (int64_t row = 0; row < num_reqs; ++row) {
          const int32_t request_index = idx_ptr[row];
          // Match Triton: -1 is a filtered pipeline-parallel row and must
          // not write the destination.
          if (request_index < 0) {
            continue;
          }
          if (request_index >= max_num_reqs) {
            continue;
          }
          accepted_ptr[request_index] = std::max(sampled_ptr[row], int32_t{1});
        }
      });
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_scatter_num_accepted("
      "Tensor idx_mapping, "
      "Tensor num_sampled, "
      "Tensor(a!) num_accepted"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_scatter_num_accepted", &vllm_scatter_num_accepted_impl);
}
