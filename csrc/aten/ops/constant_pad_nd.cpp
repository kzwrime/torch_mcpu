#include "Common.h"
#include "runtime/McpuKernelLaunch.h"

#include <ATen/ops/constant_pad_nd.h>
#include <torch/library.h>

#include <utility>
#include <vector>

namespace at::mcpu {
namespace {

at::Tensor constant_pad_nd(
    const at::Tensor& self,
    at::IntArrayRef pad,
    const at::Scalar& value) {
  // Run only shape/stride inference on the submission thread.  The actual
  // padding must remain ordered with the surrounding MCPU stream work.
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::constant_pad_nd(meta_self, pad, value);
  auto out = ops::empty_mcpu_from_meta(meta_out, self.options());

  auto self_spec = ops::make_cpu_view_spec(self);
  auto out_spec = ops::make_cpu_view_spec(out);
  std::vector<int64_t> pad_values(pad.begin(), pad.end());

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::constant_pad_nd",
      ([
        self_spec = std::move(self_spec),
        out_spec = std::move(out_spec),
        pad_values = std::move(pad_values),
        value
      ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data, out_spec.data});
        auto cpu_self = ops::cpu_view_from_spec(self_spec);
        auto cpu_out = ops::cpu_view_from_spec(out_spec);
        at::constant_pad_nd_out(cpu_out, cpu_self, pad_values, value);
      });
  return out;
}

at::Tensor& constant_pad_nd_out(
    const at::Tensor& self,
    at::IntArrayRef pad,
    const at::Scalar& value,
    at::Tensor& out) {
  auto meta_self = ops::to_meta_tensor(self);
  auto meta_out = at::constant_pad_nd(meta_self, pad, value);
  ops::check_out_sizes("aten::constant_pad_nd.out", out, meta_out);

  auto self_spec = ops::make_cpu_view_spec(self);
  auto out_spec = ops::make_cpu_view_spec(out);
  std::vector<int64_t> pad_values(pad.begin(), pad.end());

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::constant_pad_nd.out",
      ([
        self_spec = std::move(self_spec),
        out_spec = std::move(out_spec),
        pad_values = std::move(pad_values),
        value
      ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data, out_spec.data});
        auto cpu_self = ops::cpu_view_from_spec(self_spec);
        auto cpu_out = ops::cpu_view_from_spec(out_spec);
        at::constant_pad_nd_out(cpu_out, cpu_self, pad_values, value);
      });
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("constant_pad_nd", &constant_pad_nd);
  m.impl("constant_pad_nd.out", &constant_pad_nd_out);
}

} // namespace at::mcpu
