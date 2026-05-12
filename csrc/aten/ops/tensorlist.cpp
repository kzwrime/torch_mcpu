#include "Common.h"

#include <ATen/ops/cat.h>
#include <torch/library.h>

namespace at::mcpu {
namespace {

std::vector<int64_t> cat_sizes(
    const std::vector<at::Tensor>& tensors,
    int64_t dim) {
  TORCH_CHECK(
      !tensors.empty(), "aten::cat.out: expected a non-empty tensor list");

  auto is_skippable_empty = [](const at::Tensor& tensor) {
    return tensor.dim() == 1 && tensor.numel() == 0;
  };

  auto first_nonempty = tensors.end();
  for (auto it = tensors.begin(); it != tensors.end(); ++it) {
    if (!is_skippable_empty(*it)) {
      first_nonempty = it;
      break;
    }
  }

  if (first_nonempty == tensors.end()) {
    return {0};
  }

  const auto ndim = first_nonempty->dim();
  auto wrapped_dim = at::maybe_wrap_dim(dim, ndim);
  std::vector<int64_t> result(
      first_nonempty->sizes().begin(), first_nonempty->sizes().end());
  result[wrapped_dim] = 0;

  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    if (is_skippable_empty(tensor)) {
      continue;
    }
    TORCH_CHECK(
        tensor.dim() == ndim,
        "aten::cat.out: expected all tensors to have the same number of dimensions");
    for (const auto d : c10::irange(ndim)) {
      if (d == static_cast<size_t>(wrapped_dim)) {
        continue;
      }
      TORCH_CHECK(
          tensor.size(d) == result[d],
          "aten::cat.out: expected tensor sizes to match except in dimension ",
          wrapped_dim);
    }
    result[wrapped_dim] += tensor.size(wrapped_dim);
  }
  return result;
}

at::Tensor& cat_out(
    const at::ITensorListRef& tensors,
    int64_t dim,
    at::Tensor& out) {
  std::vector<at::Tensor> tensor_vec;
  tensor_vec.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    tensor_vec.push_back(tensor);
  }

  auto expected_sizes = cat_sizes(tensor_vec, dim);
  ops::check_out_sizes("aten::cat.out", out, expected_sizes);

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
