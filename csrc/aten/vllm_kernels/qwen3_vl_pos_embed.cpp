// SPDX-License-Identifier: Apache-2.0
//
// XCPU implementation of Qwen3-VL's fused bilinear position-embedding
// interpolation kernel.

#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

template <typename scalar_t>
void qwen3_vl_pos_embed_kernel_typed(
    const scalar_t* embed_ptr,
    scalar_t* output_ptr,
    int64_t height,
    int64_t width,
    double h_scale,
    double w_scale,
    int64_t num_grid,
    int64_t merge_size,
    int64_t hidden_dim,
    int64_t block_d,
    int64_t total_out) {
  // BLOCK_D is part of the Triton ABI. The CPU implementation iterates over
  // the hidden dimension directly, but the adapter validates this value.
  (void)block_d;

  const int64_t total_spatial = height * width;
  const float h_scale_f = static_cast<float>(h_scale);
  const float w_scale_f = static_cast<float>(w_scale);

#pragma omp parallel for
  for (int64_t pid = 0; pid < total_out; ++pid) {
    const int64_t spatial_idx = pid % total_spatial;

    // Match the Triton spatial-merge order exactly. Temporal repetition is
    // represented by pid values sharing the same spatial_idx.
    const int64_t num_blocks_w = width / merge_size;
    const int64_t block_idx = spatial_idx / (merge_size * merge_size);
    const int64_t local_idx = spatial_idx % (merge_size * merge_size);
    const int64_t block_row = block_idx / num_blocks_w;
    const int64_t block_col = block_idx % num_blocks_w;
    const int64_t local_row = local_idx / merge_size;
    const int64_t local_col = local_idx % merge_size;
    const int64_t row = block_row * merge_size + local_row;
    const int64_t col = block_col * merge_size + local_col;

    const float h_frac = static_cast<float>(row) * h_scale_f;
    const float w_frac = static_cast<float>(col) * w_scale_f;
    const int64_t h_floor = static_cast<int64_t>(std::floor(h_frac));
    const int64_t w_floor = static_cast<int64_t>(std::floor(w_frac));
    const int64_t h_ceil = std::min(h_floor + 1, num_grid - 1);
    const int64_t w_ceil = std::min(w_floor + 1, num_grid - 1);

    const float dh = h_frac - static_cast<float>(h_floor);
    const float dw = w_frac - static_cast<float>(w_floor);
    const float w11 = dh * dw;
    const float w10 = dh - w11;
    const float w01 = dw - w11;
    const float w00 = 1.0f - dh - w01;

    const int64_t off00 = (h_floor * num_grid + w_floor) * hidden_dim;
    const int64_t off01 = (h_floor * num_grid + w_ceil) * hidden_dim;
    const int64_t off10 = (h_ceil * num_grid + w_floor) * hidden_dim;
    const int64_t off11 = (h_ceil * num_grid + w_ceil) * hidden_dim;
    scalar_t* out = output_ptr + pid * hidden_dim;

    for (int64_t d = 0; d < hidden_dim; ++d) {
      const float value = w00 * static_cast<float>(embed_ptr[off00 + d]) +
          w01 * static_cast<float>(embed_ptr[off01 + d]) +
          w10 * static_cast<float>(embed_ptr[off10 + d]) +
          w11 * static_cast<float>(embed_ptr[off11 + d]);
      out[d] = static_cast<scalar_t>(value);
    }
  }
}

template <typename scalar_t>
void launch_qwen3_vl_pos_embed(
    const at::Tensor& embed,
    at::Tensor& output,
    int64_t height,
    int64_t width,
    double h_scale,
    double w_scale,
    int64_t num_grid,
    int64_t merge_size,
    int64_t hidden_dim,
    int64_t block_d) {
  const scalar_t* embed_ptr = embed.data_ptr<scalar_t>();
  scalar_t* output_ptr = output.data_ptr<scalar_t>();
  const int64_t total_out = output.size(0);

  at::mcpu::launch_timed_kernel(
      "mcpu::vllm_qwen3_vl_bilinear_pos_embed",
      [embed_ptr,
       output_ptr,
       height,
       width,
       h_scale,
       w_scale,
       num_grid,
       merge_size,
       hidden_dim,
       block_d,
       total_out](at::mcpu::kernel_timing::Event* timing_event) {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::vllm_qwen3_vl_bilinear_pos_embed", timing_event);
        at::mcpu::KernelAllMemoryGuard guard;
        qwen3_vl_pos_embed_kernel_typed<scalar_t>(
            embed_ptr,
            output_ptr,
            height,
            width,
            h_scale,
            w_scale,
            num_grid,
            merge_size,
            hidden_dim,
            block_d,
            total_out);
      });
}

void vllm_qwen3_vl_bilinear_pos_embed_impl(
    const at::Tensor& embed,
    at::Tensor& output,
    int64_t height,
    int64_t width,
    double h_scale,
    double w_scale,
    int64_t num_grid,
    int64_t merge_size,
    int64_t hidden_dim,
    int64_t block_d) {
  VLLM_MCPU_CHECK(
      embed.device().type() == c10::DeviceType::PrivateUse1,
      "embed must be an mcpu tensor");
  VLLM_MCPU_CHECK(
      output.device().type() == c10::DeviceType::PrivateUse1,
      "output must be an mcpu tensor");
  VLLM_MCPU_CHECK(embed.device() == output.device(), "device mismatch");
  VLLM_MCPU_CHECK_DIM(embed, 2, "embed");
  VLLM_MCPU_CHECK_DIM(output, 2, "output");
  VLLM_MCPU_CHECK(embed.is_contiguous(), "embed must be contiguous");
  VLLM_MCPU_CHECK(output.is_contiguous(), "output must be contiguous");
  VLLM_MCPU_CHECK(
      embed.scalar_type() == at::kFloat || embed.scalar_type() == at::kBFloat16,
      "embed must be float32 or bfloat16");
  VLLM_MCPU_CHECK(
      output.scalar_type() == embed.scalar_type(),
      "embed and output dtypes must match");
  VLLM_MCPU_CHECK(height > 0, "height must be positive");
  VLLM_MCPU_CHECK(width > 0, "width must be positive");
  VLLM_MCPU_CHECK(num_grid > 0, "num_grid must be positive");
  VLLM_MCPU_CHECK(merge_size > 0, "merge_size must be positive");
  VLLM_MCPU_CHECK(hidden_dim > 0, "hidden_dim must be positive");
  VLLM_MCPU_CHECK(block_d > 0, "block_d must be positive");
  VLLM_MCPU_CHECK(
      height % merge_size == 0, "height must be divisible by merge_size");
  VLLM_MCPU_CHECK(
      width % merge_size == 0, "width must be divisible by merge_size");
  VLLM_MCPU_CHECK(
      embed.size(0) == num_grid * num_grid,
      "embed row count must equal num_grid squared");
  VLLM_MCPU_CHECK(
      embed.size(1) == hidden_dim, "embed hidden dimension mismatch");
  VLLM_MCPU_CHECK(
      output.size(1) == hidden_dim, "output hidden dimension mismatch");
  VLLM_MCPU_CHECK(
      output.size(0) > 0 && output.size(0) % (height * width) == 0,
      "output row count must be a positive multiple of height*width");
  VLLM_MCPU_CHECK(std::isfinite(h_scale), "h_scale must be finite");
  VLLM_MCPU_CHECK(std::isfinite(w_scale), "w_scale must be finite");
  const double expected_h_scale =
      height > 1 ? static_cast<double>(num_grid - 1) / (height - 1) : 0.0;
  const double expected_w_scale =
      width > 1 ? static_cast<double>(num_grid - 1) / (width - 1) : 0.0;
  VLLM_MCPU_CHECK(
      std::abs(h_scale - expected_h_scale) <= 1e-6,
      "h_scale does not match height and num_grid");
  VLLM_MCPU_CHECK(
      std::abs(w_scale - expected_w_scale) <= 1e-6,
      "w_scale does not match width and num_grid");
  int64_t expected_block_d = 1;
  while (expected_block_d < hidden_dim) {
    expected_block_d <<= 1;
  }
  VLLM_MCPU_CHECK(
      block_d == expected_block_d,
      "block_d must be the next power of two of hidden_dim");

  if (embed.scalar_type() == at::kFloat) {
    launch_qwen3_vl_pos_embed<float>(
        embed,
        output,
        height,
        width,
        h_scale,
        w_scale,
        num_grid,
        merge_size,
        hidden_dim,
        block_d);
  } else {
    launch_qwen3_vl_pos_embed<at::BFloat16>(
        embed,
        output,
        height,
        width,
        h_scale,
        w_scale,
        num_grid,
        merge_size,
        hidden_dim,
        block_d);
  }
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_qwen3_vl_bilinear_pos_embed("
      "Tensor embed, "
      "Tensor(a!) output, "
      "int height, "
      "int width, "
      "float h_scale, "
      "float w_scale, "
      "int num_grid, "
      "int merge_size, "
      "int hidden_dim, "
      "int block_d"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl(
      "vllm_qwen3_vl_bilinear_pos_embed",
      &vllm_qwen3_vl_bilinear_pos_embed_impl);
}
