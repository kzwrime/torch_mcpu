// SPDX-License-Identifier: Apache-2.0
//
// C++ replacements for vllm/v1/worker/gpu/buffer_utils.py Triton kernels.

#include <cstdint>
#include "common.h"

namespace {

template <typename scalar_t>
void apply_write_single_typed(
    scalar_t* output_ptr,
    int64_t output_stride,
    const int32_t* write_indices_ptr,
    const int32_t* write_starts_ptr,
    const scalar_t* write_contents_ptr,
    const int32_t* write_cu_lens_ptr,
    int64_t num_writes) {
  for (int64_t write_idx = 0; write_idx < num_writes; ++write_idx) {
    const int32_t cu_start =
        write_idx == 0 ? 0 : write_cu_lens_ptr[write_idx - 1];
    const int32_t cu_end = write_cu_lens_ptr[write_idx];
    scalar_t* row = output_ptr +
        (int64_t)write_indices_ptr[write_idx] * output_stride +
        write_starts_ptr[write_idx];
    for (int32_t i = cu_start; i < cu_end; ++i) {
      row[i - cu_start] = write_contents_ptr[i];
    }
  }
}

void vllm_apply_write_single_impl(
    at::Tensor& output,
    int64_t output_stride,
    const at::Tensor& write_indices,
    const at::Tensor& write_starts,
    const at::Tensor& write_contents,
    const at::Tensor& write_cu_lens) {
  VLLM_MCPU_CHECK_DTYPE(write_indices, at::kInt, "write_indices");
  VLLM_MCPU_CHECK_DTYPE(write_starts, at::kInt, "write_starts");
  VLLM_MCPU_CHECK_DTYPE(write_cu_lens, at::kInt, "write_cu_lens");
  VLLM_MCPU_CHECK(
      output.scalar_type() == write_contents.scalar_type(),
      "output and write_contents must have matching dtypes");

  const int64_t num_writes = write_indices.numel();
  VLLM_MCPU_CHECK(
      write_starts.numel() >= num_writes && write_cu_lens.numel() >= num_writes,
      "write metadata must cover num_writes");
  const int32_t* indices_ptr = write_indices.data_ptr<int32_t>();
  const int32_t* starts_ptr = write_starts.data_ptr<int32_t>();
  const int32_t* cu_lens_ptr = write_cu_lens.data_ptr<int32_t>();

#define DISPATCH_APPLY_WRITE_SINGLE(cpp_type, scalar_type)              \
  case scalar_type: {                                                   \
    cpp_type* output_ptr = output.data_ptr<cpp_type>();                 \
    const cpp_type* contents_ptr = write_contents.data_ptr<cpp_type>(); \
    at::mcpu::launch_timed_kernel(                                      \
        "mcpu::vllm_apply_write_single",                                \
        [output_ptr,                                                    \
         output_stride,                                                 \
         indices_ptr,                                                   \
         starts_ptr,                                                    \
         contents_ptr,                                                  \
         cu_lens_ptr,                                                   \
         num_writes](at::mcpu::kernel_timing::Event* timing_event) {    \
          MCPU_KERNEL_TIMING_SCOPE_EVENT(                               \
              "mcpu::vllm_apply_write_single", timing_event);           \
          at::mcpu::KernelPointerMemoryGuard guard(                     \
              {output_ptr,                                              \
               indices_ptr,                                             \
               starts_ptr,                                              \
               contents_ptr,                                            \
               cu_lens_ptr});                                           \
          apply_write_single_typed<cpp_type>(                           \
              output_ptr,                                               \
              output_stride,                                            \
              indices_ptr,                                              \
              starts_ptr,                                               \
              contents_ptr,                                             \
              cu_lens_ptr,                                              \
              num_writes);                                              \
        });                                                             \
    break;                                                              \
  }

  switch (output.scalar_type()) {
    DISPATCH_APPLY_WRITE_SINGLE(int32_t, at::kInt)
    DISPATCH_APPLY_WRITE_SINGLE(int64_t, at::kLong)
    DISPATCH_APPLY_WRITE_SINGLE(float, at::kFloat)
    default:
      TORCH_CHECK(
          false,
          "vllm_apply_write_single: unsupported dtype ",
          output.scalar_type());
  }
#undef DISPATCH_APPLY_WRITE_SINGLE
}

void vllm_apply_write_multi_impl(
    const at::Tensor& output_ptrs,
    const at::Tensor& output_strides,
    const at::Tensor& write_indices,
    const at::Tensor& write_starts,
    const at::Tensor& write_contents,
    const at::Tensor& write_cu_lens,
    const at::Tensor& write_group_ids) {
  VLLM_MCPU_CHECK_DTYPE(output_ptrs, at::kUInt64, "output_ptrs");
  VLLM_MCPU_CHECK_DTYPE(output_strides, at::kLong, "output_strides");
  VLLM_MCPU_CHECK_DTYPE(write_indices, at::kInt, "write_indices");
  VLLM_MCPU_CHECK_DTYPE(write_starts, at::kInt, "write_starts");
  VLLM_MCPU_CHECK_DTYPE(write_contents, at::kInt, "write_contents");
  VLLM_MCPU_CHECK_DTYPE(write_cu_lens, at::kInt, "write_cu_lens");
  VLLM_MCPU_CHECK_DTYPE(write_group_ids, at::kInt, "write_group_ids");

  const int64_t num_writes = write_indices.numel();
  const uint64_t* output_ptrs_ptr = output_ptrs.data_ptr<uint64_t>();
  const int64_t* output_strides_ptr = output_strides.data_ptr<int64_t>();
  const int32_t* indices_ptr = write_indices.data_ptr<int32_t>();
  const int32_t* starts_ptr = write_starts.data_ptr<int32_t>();
  const int32_t* contents_ptr = write_contents.data_ptr<int32_t>();
  const int32_t* cu_lens_ptr = write_cu_lens.data_ptr<int32_t>();
  const int32_t* group_ids_ptr = write_group_ids.data_ptr<int32_t>();

  at::mcpu::launch_timed_kernel(
      "mcpu::vllm_apply_write_multi",
      [output_ptrs_ptr,
       output_strides_ptr,
       indices_ptr,
       starts_ptr,
       contents_ptr,
       cu_lens_ptr,
       group_ids_ptr,
       num_writes](at::mcpu::kernel_timing::Event* timing_event) {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::vllm_apply_write_multi", timing_event);
        at::mcpu::KernelAllMemoryGuard guard;
        for (int64_t write_idx = 0; write_idx < num_writes; ++write_idx) {
          const int32_t cu_start =
              write_idx == 0 ? 0 : cu_lens_ptr[write_idx - 1];
          const int32_t cu_end = cu_lens_ptr[write_idx];
          const int32_t group_id = group_ids_ptr[write_idx];
          int32_t* row = reinterpret_cast<int32_t*>(output_ptrs_ptr[group_id]);
          row +=
              (int64_t)indices_ptr[write_idx] * output_strides_ptr[group_id] +
              starts_ptr[write_idx];
          for (int32_t i = cu_start; i < cu_end; ++i) {
            row[i - cu_start] = contents_ptr[i];
          }
        }
      });
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_apply_write_single("
      "Tensor(a!) output, "
      "int output_stride, "
      "Tensor write_indices, "
      "Tensor write_starts, "
      "Tensor write_contents, "
      "Tensor write_cu_lens"
      ") -> ()");
  m.def(
      "vllm_apply_write_multi("
      "Tensor output_ptrs, "
      "Tensor output_strides, "
      "Tensor write_indices, "
      "Tensor write_starts, "
      "Tensor write_contents, "
      "Tensor write_cu_lens, "
      "Tensor write_group_ids"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_apply_write_single", &vllm_apply_write_single_impl);
  m.impl("vllm_apply_write_multi", &vllm_apply_write_multi_impl);
}
