// SPDX-License-Identifier: Apache-2.0
//
// Runtime configuration for vllm_kernels operators.

#include "common.h"
#include <torch/library.h>

namespace vllm_mcpu {

bool CHECK_INPUTS = true;

} // namespace vllm_mcpu

// Expose setter to Python: torch.ops.mcpu.set_check_inputs(True/False)
namespace {

void set_check_inputs_impl(bool enabled) {
  vllm_mcpu::CHECK_INPUTS = enabled;
}

bool get_check_inputs_impl() {
  return vllm_mcpu::CHECK_INPUTS;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def("set_check_inputs(bool enabled) -> ()");
  m.def("get_check_inputs() -> bool");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("set_check_inputs", &set_check_inputs_impl);
  m.impl("get_check_inputs", &get_check_inputs_impl);
}
