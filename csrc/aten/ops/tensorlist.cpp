#include "Common.h"

#include <ATen/ops/cat.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

at::Tensor& cat_out(
    const at::ITensorListRef& tensors,
    int64_t dim,
    at::Tensor& out) {
  std::vector<at::Tensor> tensor_vec;
  tensor_vec.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    tensor_vec.push_back(tensor);
  }

  auto meta_tensors = ops::to_meta_tensors(tensor_vec);
  auto meta_out = at::empty({0}, out.options().device(c10::DeviceType::Meta));
  at::cat_out(meta_out, at::ITensorListRef(meta_tensors), dim);
  ops::check_out_sizes("aten::cat.out", out, meta_out);

  auto cpu_tensors = ops::to_cpu_tensors_if_needed(tensor_vec);
  at::native::mcpu::MemoryGuard guard(out, c10::IValue(tensor_vec));
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  at::cat_out(cpu_out, at::ITensorListRef(cpu_tensors), dim);
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("cat.out", &cat_out);
}

} // namespace at::mcpu
