// SPDX-License-Identifier: Apache-2.0
//
// Common utilities for vllm_kernels C++ operators.

#pragma once
#include <ATen/ATen.h>
#include <torch/library.h>
#include <limits>
#include <cstdint>
#include <cmath>
#include <type_traits>

namespace vllm_mcpu {

// Set to false to skip input validation checks (e.g. for benchmarking).
// Exposed to Python via torch.ops.mcpu.set_check_inputs(bool).
extern bool CHECK_INPUTS;

} // namespace vllm_mcpu

// Input validation macro — no-op when CHECK_INPUTS is false.
#define VLLM_MCPU_CHECK(cond, ...) \
    do { if (vllm_mcpu::CHECK_INPUTS) { TORCH_CHECK(cond, ##__VA_ARGS__); } } while (0)

// Check that a tensor has the expected floating-point dtype (float32/float16/bfloat16).
#define VLLM_MCPU_CHECK_FLOAT(t, name) \
    VLLM_MCPU_CHECK( \
        (t).scalar_type() == at::kFloat || \
        (t).scalar_type() == at::kHalf  || \
        (t).scalar_type() == at::kBFloat16, \
        name " must be float32, float16, or bfloat16, got ", (t).scalar_type())

// Check that a tensor has the expected scalar type.
// NOTE: parameter is named 'expected_st' to avoid shadowing tensor's .dtype() method.
#define VLLM_MCPU_CHECK_DTYPE(t, expected_st, name) \
    VLLM_MCPU_CHECK((t).scalar_type() == (expected_st), \
        name " must be " #expected_st ", got ", (t).scalar_type())

// Check tensor dimensions.
#define VLLM_MCPU_CHECK_DIM(t, d, name) \
    VLLM_MCPU_CHECK((t).dim() == (d), name " must be " #d "D, got ", (t).dim(), "D")

// Unified float-type dispatch over float32 / float16 / bfloat16.
//
// Usage:
//   VLLM_MCPU_DISPATCH_FLOAT(tensor, "kernel_name", [&] {
//       my_kernel_typed<scalar_t>(tensor.data_ptr<scalar_t>(), ...);
//   });
//
// Inside the lambda, `scalar_t` is one of: float, at::Half, at::BFloat16.
// Only a single runtime branch is taken; the lambda body is instantiated
// three times at compile time (one per type).
#define VLLM_MCPU_DISPATCH_FLOAT(tensor, name, ...)                      \
  do {                                                                    \
    switch ((tensor).scalar_type()) {                                     \
      case at::kBFloat16: {                                               \
        using scalar_t = at::BFloat16;                                    \
        __VA_ARGS__;                                                      \
        break;                                                            \
      }                                                                   \
      case at::kHalf: {                                                   \
        using scalar_t = at::Half;                                        \
        __VA_ARGS__;                                                      \
        break;                                                            \
      }                                                                   \
      case at::kFloat: {                                                  \
        using scalar_t = float;                                           \
        __VA_ARGS__;                                                      \
        break;                                                            \
      }                                                                   \
      default:                                                            \
        TORCH_CHECK(false, name ": unsupported dtype ",                   \
                    (tensor).scalar_type(),                               \
                    " — expected float32, float16, or bfloat16");         \
    }                                                                     \
  } while (0)
