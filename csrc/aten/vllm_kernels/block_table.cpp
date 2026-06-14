// SPDX-License-Identifier: Apache-2.0
//
// C++ kernels for vllm/v1/worker/gpu/block_table.py:
//   gather_block_tables
//   compute_slot_mappings

#include <ATen/ATen.h>
#include <ATen/core/List.h>
#include <torch/library.h>
#include <cstdint>
#include <cstring>
#include "common.h"

namespace {

void check_block_table_list(
    at::TensorList block_tables,
    int64_t num_groups,
    const char* name) {
  VLLM_MCPU_CHECK(
      block_tables.size() == static_cast<size_t>(num_groups),
      name,
      " size must match num_groups, got ",
      block_tables.size(),
      " vs ",
      num_groups);
}

void vllm_gather_block_tables_impl(
    const at::Tensor& idx_mapping, // [num_reqs], int32
    at::TensorList
        src_block_tables, // num_groups x [max_num_reqs, max_num_blocks]
    const at::Tensor& num_blocks, // [num_groups, max_num_reqs], int32
    int64_t num_reqs_padded,
    at::TensorList
        dst_block_tables) { // num_groups x [max_num_reqs, max_num_blocks]
  VLLM_MCPU_CHECK_DTYPE(idx_mapping, at::kInt, "idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(num_blocks, at::kInt, "num_blocks");
  VLLM_MCPU_CHECK_DIM(idx_mapping, 1, "idx_mapping");
  VLLM_MCPU_CHECK_DIM(num_blocks, 2, "num_blocks");
  VLLM_MCPU_CHECK(
      idx_mapping.is_contiguous(), "idx_mapping must be contiguous");
  VLLM_MCPU_CHECK(num_blocks.is_contiguous(), "num_blocks must be contiguous");

  const int64_t num_groups = num_blocks.size(0);
  const int64_t max_num_reqs = num_blocks.size(1);
  const int64_t num_reqs = idx_mapping.size(0);

  VLLM_MCPU_CHECK(
      0 <= num_reqs_padded && num_reqs <= num_reqs_padded,
      "num_reqs_padded must satisfy num_reqs <= num_reqs_padded");

  check_block_table_list(src_block_tables, num_groups, "src_block_tables");
  check_block_table_list(dst_block_tables, num_groups, "dst_block_tables");

  const int32_t* idx_ptr = idx_mapping.data_ptr<int32_t>();
  const int32_t* num_blocks_ptr = num_blocks.data_ptr<int32_t>();
  const int64_t num_blocks_stride = num_blocks.stride(0);

  for (int64_t group_id = 0; group_id < num_groups; ++group_id) {
    const at::Tensor& src = src_block_tables[group_id];
    const at::Tensor& dst = dst_block_tables[group_id];

    VLLM_MCPU_CHECK_DTYPE(src, at::kInt, "src_block_table");
    VLLM_MCPU_CHECK_DTYPE(dst, at::kInt, "dst_block_table");
    VLLM_MCPU_CHECK_DIM(src, 2, "src_block_table");
    VLLM_MCPU_CHECK_DIM(dst, 2, "dst_block_table");
    VLLM_MCPU_CHECK(src.is_contiguous(), "src_block_table must be contiguous");
    VLLM_MCPU_CHECK(dst.is_contiguous(), "dst_block_table must be contiguous");
    VLLM_MCPU_CHECK(
        src.size(0) == max_num_reqs,
        "src_block_table rows must match num_blocks second dim");
    VLLM_MCPU_CHECK(
        dst.size(0) >= num_reqs_padded,
        "dst_block_table rows must cover padded batch size");
    VLLM_MCPU_CHECK(
        src.size(1) == dst.size(1), "src/dst block table widths must match");

    const int64_t max_num_blocks = dst.size(1);
    const int64_t src_stride0 = src.stride(0);
    const int64_t dst_stride0 = dst.stride(0);

    const int32_t* src_ptr = src.data_ptr<int32_t>();
    int32_t* dst_ptr = dst.data_ptr<int32_t>();
    const int32_t* group_num_blocks_ptr =
        num_blocks_ptr + group_id * num_blocks_stride;

    at::mcpu::launch_timed_kernel(
        "mcpu::vllm_gather_block_tables",
        [idx_ptr,
         group_num_blocks_ptr,
         src_ptr,
         dst_ptr,
         num_reqs,
         num_reqs_padded,
         max_num_reqs,
         max_num_blocks,
         src_stride0,
         dst_stride0](at::mcpu::kernel_timing::Event* timing_event) mutable {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::vllm_gather_block_tables", timing_event);
          at::mcpu::KernelPointerMemoryGuard guard(
              {idx_ptr, group_num_blocks_ptr, src_ptr, dst_ptr});
          for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
            const int32_t req_idx = idx_ptr[batch_idx];
            VLLM_MCPU_CHECK(
                0 <= req_idx && req_idx < max_num_reqs,
                "idx_mapping contains out-of-range request index");

      int32_t n = group_num_blocks_ptr[req_idx];
      if (n > max_num_blocks) {
        n = max_num_blocks;
      }
      VLLM_MCPU_CHECK(
          0 <= n && n <= max_num_blocks,
          "num_blocks contains out-of-range block count");

            const int32_t* src_row =
                src_ptr + static_cast<int64_t>(req_idx) * src_stride0;
            int32_t* dst_row = dst_ptr + batch_idx * dst_stride0;

            int64_t j = 0;
            for (; j < n; ++j) {
              dst_row[j] = src_row[j];
            }
            for (; j < max_num_blocks; ++j) {
              dst_row[j] = 0;
            }
          }

          for (int64_t batch_idx = num_reqs; batch_idx < num_reqs_padded;
               ++batch_idx) {
            int32_t* dst_row = dst_ptr + batch_idx * dst_stride0;
            for (int64_t j = 0; j < max_num_blocks; ++j) {
              dst_row[j] = 0;
            }
          }
        });
  }
}

void vllm_compute_slot_mappings_impl(
    const at::Tensor& idx_mapping, // [num_reqs], int32
    const at::Tensor& query_start_loc, // [num_reqs + 1], int32
    const at::Tensor& positions, // [num_tokens], int64
    at::TensorList block_tables, // num_groups x [max_num_reqs, max_num_blocks]
    const at::Tensor& block_sizes, // [num_groups], int32
    at::Tensor& slot_mappings, // [num_groups, max_num_tokens], int64
    int64_t cp_size,
    int64_t cp_rank,
    int64_t cp_interleave,
    int64_t pad_id) {
  VLLM_MCPU_CHECK_DTYPE(idx_mapping, at::kInt, "idx_mapping");
  VLLM_MCPU_CHECK_DTYPE(query_start_loc, at::kInt, "query_start_loc");
  VLLM_MCPU_CHECK_DTYPE(positions, at::kLong, "positions");
  VLLM_MCPU_CHECK_DTYPE(block_sizes, at::kInt, "block_sizes");
  VLLM_MCPU_CHECK_DTYPE(slot_mappings, at::kLong, "slot_mappings");
  VLLM_MCPU_CHECK_DIM(idx_mapping, 1, "idx_mapping");
  VLLM_MCPU_CHECK_DIM(query_start_loc, 1, "query_start_loc");
  VLLM_MCPU_CHECK_DIM(positions, 1, "positions");
  VLLM_MCPU_CHECK_DIM(block_sizes, 1, "block_sizes");
  VLLM_MCPU_CHECK_DIM(slot_mappings, 2, "slot_mappings");
  VLLM_MCPU_CHECK(
      idx_mapping.is_contiguous(), "idx_mapping must be contiguous");
  VLLM_MCPU_CHECK(
      query_start_loc.is_contiguous(), "query_start_loc must be contiguous");
  VLLM_MCPU_CHECK(positions.is_contiguous(), "positions must be contiguous");
  VLLM_MCPU_CHECK(
      block_sizes.is_contiguous(), "block_sizes must be contiguous");
  VLLM_MCPU_CHECK(
      slot_mappings.is_contiguous(), "slot_mappings must be contiguous");

  const int64_t num_reqs = idx_mapping.size(0);
  const int64_t num_groups = block_sizes.size(0);
  const int64_t max_num_tokens = slot_mappings.size(1);

  VLLM_MCPU_CHECK(
      query_start_loc.size(0) == num_reqs + 1,
      "query_start_loc must have num_reqs + 1 elements");
  VLLM_MCPU_CHECK(
      slot_mappings.size(0) == num_groups,
      "slot_mappings first dim must match block_sizes");
  VLLM_MCPU_CHECK(cp_size >= 1, "cp_size must be >= 1");
  VLLM_MCPU_CHECK(
      cp_rank >= 0 && cp_rank < cp_size, "cp_rank must be in [0, cp_size)");
  VLLM_MCPU_CHECK(cp_interleave >= 1, "cp_interleave must be >= 1");

  check_block_table_list(block_tables, num_groups, "block_tables");

  const int32_t* idx_ptr = idx_mapping.data_ptr<int32_t>();
  const int32_t* qsl_ptr = query_start_loc.data_ptr<int32_t>();
  const int64_t* pos_ptr = positions.data_ptr<int64_t>();
  const int32_t* block_sizes_ptr = block_sizes.data_ptr<int32_t>();
  int64_t* slot_ptr = slot_mappings.data_ptr<int64_t>();
  const int64_t slot_stride0 = slot_mappings.stride(0);
  const int64_t positions_numel = positions.size(0);

  for (int64_t group_id = 0; group_id < num_groups; ++group_id) {
    const at::Tensor& block_table = block_tables[group_id];
    VLLM_MCPU_CHECK_DTYPE(block_table, at::kInt, "block_table");
    VLLM_MCPU_CHECK_DIM(block_table, 2, "block_table");
    VLLM_MCPU_CHECK(
        block_table.is_contiguous(), "block_table must be contiguous");

    const int64_t max_num_reqs = block_table.size(0);
    const int64_t max_num_blocks = block_table.size(1);
    const int64_t block_table_stride0 = block_table.stride(0);
    const int32_t* block_table_ptr = block_table.data_ptr<int32_t>();
    int64_t* slot_row_ptr = slot_ptr + group_id * slot_stride0;
    const int32_t* group_block_size_ptr = block_sizes_ptr + group_id;

    at::mcpu::launch_timed_kernel(
        "mcpu::vllm_compute_slot_mappings",
        [idx_ptr,
         qsl_ptr,
         pos_ptr,
         group_block_size_ptr,
         block_table_ptr,
         slot_row_ptr,
         num_reqs,
         max_num_tokens,
         positions_numel,
         max_num_reqs,
         max_num_blocks,
         block_table_stride0,
         cp_size,
         cp_rank,
         cp_interleave,
         pad_id](at::mcpu::kernel_timing::Event* timing_event) mutable {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::vllm_compute_slot_mappings", timing_event);
          at::mcpu::KernelPointerMemoryGuard guard(
              {idx_ptr,
               qsl_ptr,
               pos_ptr,
               group_block_size_ptr,
               block_table_ptr,
               slot_row_ptr});

          const int32_t actual_num_tokens = qsl_ptr[num_reqs];
          VLLM_MCPU_CHECK(
              0 <= actual_num_tokens && actual_num_tokens <= max_num_tokens,
              "query_start_loc[num_reqs] must be within slot_mappings width");
          VLLM_MCPU_CHECK(
              positions_numel >= actual_num_tokens,
              "positions must cover all actual tokens");

          const int32_t block_size = *group_block_size_ptr;
          VLLM_MCPU_CHECK(block_size > 0, "block_size must be > 0");

          for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
            const int32_t req_idx = idx_ptr[batch_idx];
            VLLM_MCPU_CHECK(
                0 <= req_idx && req_idx < max_num_reqs,
                "idx_mapping contains out-of-range request index");

            const int32_t start = qsl_ptr[batch_idx];
            const int32_t end = qsl_ptr[batch_idx + 1];
            VLLM_MCPU_CHECK(
                0 <= start && start <= end && end <= actual_num_tokens,
                "query_start_loc must be non-decreasing and within bounds");

            const int32_t* req_block_table_ptr = block_table_ptr +
                static_cast<int64_t>(req_idx) * block_table_stride0;
            for (int32_t token_idx = start; token_idx < end; ++token_idx) {
              const int64_t position = pos_ptr[token_idx];
              VLLM_MCPU_CHECK(position >= 0, "positions must be non-negative");

              const int64_t global_block_size =
                  static_cast<int64_t>(block_size) * cp_size;
              const int64_t block_idx = position / global_block_size;
              const int64_t block_off = position % global_block_size;
              VLLM_MCPU_CHECK(
                  0 <= block_idx && block_idx < max_num_blocks,
                  "position maps to out-of-range block index");

              const int64_t block_number = req_block_table_ptr[block_idx];
              if (cp_size == 1) {
                slot_row_ptr[token_idx] =
                    block_number * static_cast<int64_t>(block_size) +
                    block_off;
                continue;
              }

              const bool is_local =
                  ((block_off / cp_interleave) % cp_size) == cp_rank;
              const int64_t rounds = block_off / (cp_interleave * cp_size);
              const int64_t remainder = block_off % cp_interleave;
              const int64_t local_off = rounds * cp_interleave + remainder;
              slot_row_ptr[token_idx] = is_local
                  ? block_number * static_cast<int64_t>(block_size) + local_off
                  : pad_id;
            }
          }

          for (int64_t token_idx = actual_num_tokens;
               token_idx < max_num_tokens;
               ++token_idx) {
            slot_row_ptr[token_idx] = pad_id;
          }
        });
  }
}

void compute_slot_mapping_kernel_impl(
    const at::Tensor& query_start_loc, // [num_reqs + 1], int32
    const at::Tensor& positions, // [num_tokens], int64
    const at::Tensor& block_table, // [max_num_reqs, max_num_blocks], int32
    at::Tensor& slot_mapping, // [max_num_tokens], int64
    int64_t block_size) {
  const int32_t req_num = query_start_loc.size(0) - 1;
  const int64_t block_table_stride = block_table.stride(0);

  const int32_t* __restrict__ qsl_ptr = query_start_loc.data_ptr<int32_t>();
  const int64_t* __restrict__ pos_ptr = positions.data_ptr<int64_t>();
  const int32_t* __restrict__ block_table_ptr = block_table.data_ptr<int32_t>();
  int64_t* __restrict__ slot_ptr = slot_mapping.data_ptr<int64_t>();

  at::mcpu::launch_timed_kernel(
      "mcpu::compute_slot_mapping_kernel_impl",
      [req_num,
       block_table_stride,
       qsl_ptr,
       pos_ptr,
       block_table_ptr,
       slot_ptr,
       block_size](at::mcpu::kernel_timing::Event* timing_event) mutable {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::compute_slot_mapping_kernel_impl", timing_event);
        at::mcpu::KernelPointerMemoryGuard guard(
            {qsl_ptr, pos_ptr, block_table_ptr, slot_ptr});
#pragma omp parallel for
        for (int32_t req_idx = 0; req_idx < req_num; ++req_idx) {
          int32_t start = qsl_ptr[req_idx];
          int32_t end = qsl_ptr[req_idx + 1];
          int32_t token_num = end - start;

          const int64_t* __restrict__ curr_position_ptr = pos_ptr + start;
          int64_t* __restrict__ curr_slot_mapping_ptr = slot_ptr + start;
          const int32_t* __restrict__ curr_block_table_ptr =
              block_table_ptr + req_idx * block_table_stride;

          for (int32_t token_idx = 0; token_idx < token_num; ++token_idx) {
            int64_t position = curr_position_ptr[token_idx];
            int64_t block_number = curr_block_table_ptr[position / block_size];
            curr_slot_mapping_ptr[token_idx] =
                block_number * block_size + position % block_size;
          }
        }
      });
}

void zero_kv_blocks_kernel_impl(
    const at::Tensor& seg_addrs, // [n_segs], uint64 byte addresses
    const at::Tensor& block_ids, // [n_blocks], int64
    int64_t n_blocks,
    int64_t n_segs,
    int64_t page_size_el) {
  VLLM_MCPU_CHECK_DTYPE(seg_addrs, at::kUInt64, "seg_addrs");
  VLLM_MCPU_CHECK_DTYPE(block_ids, at::kLong, "block_ids");
  VLLM_MCPU_CHECK_DIM(seg_addrs, 1, "seg_addrs");
  VLLM_MCPU_CHECK_DIM(block_ids, 1, "block_ids");
  VLLM_MCPU_CHECK(seg_addrs.is_contiguous(), "seg_addrs must be contiguous");
  VLLM_MCPU_CHECK(block_ids.is_contiguous(), "block_ids must be contiguous");
  VLLM_MCPU_CHECK(n_blocks >= 0, "n_blocks must be non-negative");
  VLLM_MCPU_CHECK(n_segs >= 0, "n_segs must be non-negative");
  VLLM_MCPU_CHECK(page_size_el > 0, "page_size_el must be positive");
  VLLM_MCPU_CHECK(seg_addrs.size(0) >= n_segs, "seg_addrs must cover n_segs");
  VLLM_MCPU_CHECK(
      block_ids.size(0) >= n_blocks, "block_ids must cover n_blocks");

  const uint64_t* __restrict__ seg_addrs_ptr = seg_addrs.data_ptr<uint64_t>();
  const int64_t* __restrict__ block_ids_ptr = block_ids.data_ptr<int64_t>();

  at::mcpu::launch_timed_kernel(
      "mcpu::zero_kv_blocks_kernel_impl",
      [seg_addrs_ptr,
       block_ids_ptr,
       n_blocks,
       n_segs,
       page_size_el](at::mcpu::kernel_timing::Event* timing_event) mutable {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::zero_kv_blocks_kernel_impl", timing_event);
        at::mcpu::KernelAllMemoryGuard guard;
#pragma omp parallel for collapse(2)
        for (int64_t block_index = 0; block_index < n_blocks; ++block_index) {
          for (int64_t seg_index = 0; seg_index < n_segs; ++seg_index) {
            const int64_t block_id = block_ids_ptr[block_index];
            int32_t* __restrict__ ptr =
                reinterpret_cast<int32_t*>(seg_addrs_ptr[seg_index]);
            std::memset(
                ptr + block_id * page_size_el,
                0,
                page_size_el * sizeof(int32_t));
          }
        }
      });
}

} // namespace

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "vllm_gather_block_tables("
      "Tensor idx_mapping, "
      "Tensor[] src_block_tables, "
      "Tensor num_blocks, "
      "int num_reqs_padded, "
      "Tensor[] dst_block_tables"
      ") -> ()");
  m.def(
      "vllm_compute_slot_mappings("
      "Tensor idx_mapping, "
      "Tensor query_start_loc, "
      "Tensor positions, "
      "Tensor[] block_tables, "
      "Tensor block_sizes, "
      "Tensor(a!) slot_mappings, "
      "int cp_size, "
      "int cp_rank, "
      "int cp_interleave, "
      "int pad_id"
      ") -> ()");
  m.def(
      "compute_slot_mapping_kernel_impl("
      "Tensor query_start_loc, "
      "Tensor positions, "
      "Tensor block_table, "
      "Tensor(a3!) slot_mapping, "
      "SymInt block_size"
      ") -> ()");
  m.def(
      "zero_kv_blocks_kernel_impl("
      "Tensor seg_addrs, "
      "Tensor block_ids, "
      "SymInt n_blocks, "
      "SymInt n_segs, "
      "SymInt page_size_el"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("vllm_gather_block_tables", &vllm_gather_block_tables_impl);
  m.impl("vllm_compute_slot_mappings", &vllm_compute_slot_mappings_impl);
  m.impl("compute_slot_mapping_kernel_impl", &compute_slot_mapping_kernel_impl);
  m.impl("zero_kv_blocks_kernel_impl", &zero_kv_blocks_kernel_impl);
}
