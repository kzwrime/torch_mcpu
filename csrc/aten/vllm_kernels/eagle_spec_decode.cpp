#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <cstring>

namespace {

int64_t load_integral(const at::Tensor& tensor, int64_t index) {
  switch (tensor.scalar_type()) {
    case at::kInt:
      return tensor.const_data_ptr<int32_t>()[index];
    case at::kLong:
      return tensor.const_data_ptr<int64_t>()[index];
    default:
      TORCH_CHECK(false, tensor.scalar_type(), " is not an integral dtype");
  }
  return 0;
}

void check_1d_integral_capacity(
    const at::Tensor& tensor,
    int64_t min_numel,
    const char* name) {
  TORCH_CHECK(tensor.dim() == 1, name, " must be a 1D tensor");
  TORCH_CHECK(
      tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong,
      name,
      " must be int32 or int64");
  TORCH_CHECK(
      tensor.numel() >= min_numel,
      name,
      " has insufficient capacity: expected at least ",
      min_numel,
      " elements, got ",
      tensor.numel());
}

void store_integral(const at::Tensor& tensor, int64_t index, int64_t value) {
  switch (tensor.scalar_type()) {
    case at::kInt:
      tensor.data_ptr<int32_t>()[index] = static_cast<int32_t>(value);
      return;
    case at::kLong:
      tensor.data_ptr<int64_t>()[index] = value;
      return;
    default:
      TORCH_CHECK(false, tensor.scalar_type(), " is not an integral dtype");
  }
}

void copy_hidden_row(
    const at::Tensor& dst,
    int64_t dst_row,
    int64_t dst_stride,
    const at::Tensor& src,
    int64_t src_row,
    int64_t src_stride,
    int64_t hidden_size) {
  TORCH_CHECK(
      dst.scalar_type() == src.scalar_type(), "hidden-state dtype mismatch");
  const int64_t elem_size = dst.element_size();
  auto* dst_ptr =
      static_cast<char*>(dst.data_ptr()) + dst_row * dst_stride * elem_size;
  const auto* src_ptr = static_cast<const char*>(src.const_data_ptr()) +
      src_row * src_stride * elem_size;
  std::memcpy(dst_ptr, src_ptr, hidden_size * elem_size);
}

} // namespace

void prepare_eagle_inputs_kernel_impl(
    at::Tensor& last_token_indices,
    at::Tensor& eagle_input_ids,
    at::Tensor& eagle_positions,
    const at::Tensor& target_input_ids,
    const at::Tensor& target_positions,
    const at::Tensor& idx_mapping,
    const at::Tensor& last_sampled,
    const at::Tensor& next_prefill_tokens,
    const at::Tensor& num_sampled,
    const at::Tensor& num_rejected,
    const at::Tensor& query_start_loc,
    int64_t num_reqs) {
  // Original: vllm/v1/worker/gpu/spec_decode/eagle/speculator.py
  // _prepare_eagle_inputs_kernel.
  TORCH_CHECK(num_reqs >= 0, "num_reqs must be non-negative");
  check_1d_integral_capacity(
      last_token_indices, num_reqs, "last_token_indices");
  TORCH_CHECK(
      last_token_indices.scalar_type() == at::kLong,
      "last_token_indices must be int64");
  check_1d_integral_capacity(idx_mapping, num_reqs, "idx_mapping");
  check_1d_integral_capacity(num_sampled, num_reqs, "num_sampled");
  check_1d_integral_capacity(num_rejected, num_reqs, "num_rejected");
  check_1d_integral_capacity(query_start_loc, num_reqs + 1, "query_start_loc");

#pragma omp parallel for
  for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
    const int64_t req_state_idx = load_integral(idx_mapping, batch_idx);
    const int64_t query_start = load_integral(query_start_loc, batch_idx);
    const int64_t query_end = load_integral(query_start_loc, batch_idx + 1);
    int64_t query_len =
        query_end - query_start - load_integral(num_rejected, batch_idx);

    const int64_t next_token = load_integral(num_sampled, batch_idx) > 0
        ? load_integral(last_sampled, req_state_idx)
        : load_integral(next_prefill_tokens, req_state_idx);

    for (int64_t i = 1; i < query_len; ++i) {
      store_integral(
          eagle_input_ids,
          query_start + i - 1,
          load_integral(target_input_ids, query_start + i));
    }

    const int64_t last_token_index = query_start + query_len - 1;
    last_token_indices.data_ptr<int64_t>()[batch_idx] = last_token_index;
    store_integral(eagle_input_ids, last_token_index, next_token);

    for (int64_t i = 0; i < query_len; ++i) {
      store_integral(
          eagle_positions,
          query_start + i,
          load_integral(target_positions, query_start + i));
    }
  }
}

void prepare_eagle_decode_kernel_impl(
    const at::Tensor& draft_tokens,
    const at::Tensor& output_hidden_states,
    int64_t output_hidden_states_stride,
    const at::Tensor& last_token_indices,
    const at::Tensor& target_seq_lens,
    const at::Tensor& num_rejected,
    at::Tensor& input_ids,
    at::Tensor& positions,
    at::Tensor& input_hidden_states,
    int64_t input_hidden_states_stride,
    at::Tensor& query_start_loc,
    at::Tensor& seq_lens,
    int64_t hidden_size,
    int64_t max_model_len,
    int64_t max_num_reqs,
    int64_t num_reqs) {
  // Original: vllm/v1/worker/gpu/spec_decode/eagle/speculator.py
  // _prepare_eagle_docode_kernel.
  TORCH_CHECK(num_reqs >= 0, "num_reqs must be non-negative");
  TORCH_CHECK(max_num_reqs >= num_reqs, "max_num_reqs must be >= num_reqs");
  TORCH_CHECK(max_model_len > 0, "max_model_len must be positive");
  TORCH_CHECK(hidden_size >= 0, "hidden_size must be non-negative");
  check_1d_integral_capacity(draft_tokens, num_reqs, "draft_tokens");
  check_1d_integral_capacity(
      last_token_indices, num_reqs, "last_token_indices");
  TORCH_CHECK(
      last_token_indices.scalar_type() == at::kLong,
      "last_token_indices must be int64");
  check_1d_integral_capacity(target_seq_lens, num_reqs, "target_seq_lens");
  check_1d_integral_capacity(num_rejected, num_reqs, "num_rejected");
  check_1d_integral_capacity(input_ids, num_reqs, "input_ids");
  check_1d_integral_capacity(positions, num_reqs, "positions");
  check_1d_integral_capacity(
      query_start_loc, max_num_reqs + 1, "query_start_loc");
  check_1d_integral_capacity(seq_lens, max_num_reqs, "seq_lens");
  TORCH_CHECK(
      output_hidden_states.dim() == 2, "output_hidden_states must be 2D");
  TORCH_CHECK(input_hidden_states.dim() == 2, "input_hidden_states must be 2D");
  TORCH_CHECK(
      output_hidden_states_stride >= hidden_size,
      "output_hidden_states_stride must cover hidden_size");
  TORCH_CHECK(
      input_hidden_states_stride >= hidden_size,
      "input_hidden_states_stride must cover hidden_size");
  TORCH_CHECK(
      input_hidden_states.size(0) >= num_reqs,
      "input_hidden_states has insufficient rows");
  const int64_t draft_tokens_stride = draft_tokens.stride(0);
  const int64_t* last_token_indices_ptr =
      last_token_indices.const_data_ptr<int64_t>();

#pragma omp parallel for
  for (int64_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    store_integral(
        input_ids,
        req_idx,
        load_integral(draft_tokens, req_idx * draft_tokens_stride));

    const int64_t src_idx = last_token_indices_ptr[req_idx];
    copy_hidden_row(
        input_hidden_states,
        req_idx,
        input_hidden_states_stride,
        output_hidden_states,
        src_idx,
        output_hidden_states_stride,
        hidden_size);

    const int64_t position =
        std::min(load_integral(positions, req_idx) + 1, max_model_len - 1);
    store_integral(positions, req_idx, position);

    const int64_t seq_len = std::min(
        load_integral(target_seq_lens, req_idx) -
            load_integral(num_rejected, req_idx) + 1,
        max_model_len);
    store_integral(seq_lens, req_idx, seq_len);
  }

#pragma omp parallel for
  for (int64_t i = 0; i < max_num_reqs + 1; ++i) {
    store_integral(query_start_loc, i, std::min(i, num_reqs));
  }
#pragma omp parallel for
  for (int64_t i = num_reqs; i < max_num_reqs; ++i) {
    store_integral(seq_lens, i, 0);
  }
}

void update_eagle_inputs_kernel_impl(
    at::Tensor& input_ids,
    at::Tensor& positions,
    at::Tensor& input_hidden_states,
    int64_t input_hidden_states_stride,
    at::Tensor& seq_lens,
    int64_t max_model_len,
    const at::Tensor& draft_tokens,
    const at::Tensor& output_hidden_states,
    int64_t output_hidden_states_stride,
    int64_t hidden_size) {
  // Original: vllm/v1/worker/gpu/spec_decode/eagle/speculator.py
  // _update_eagle_inputs_kernel.
  const int64_t num_reqs = draft_tokens.numel();
  const int64_t draft_tokens_stride = draft_tokens.stride(0);

#pragma omp parallel for
  for (int64_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    store_integral(
        input_ids,
        req_idx,
        load_integral(draft_tokens, req_idx * draft_tokens_stride));
    copy_hidden_row(
        input_hidden_states,
        req_idx,
        input_hidden_states_stride,
        output_hidden_states,
        req_idx,
        output_hidden_states_stride,
        hidden_size);

    store_integral(
        positions,
        req_idx,
        std::min(load_integral(positions, req_idx) + 1, max_model_len - 1));
    store_integral(
        seq_lens,
        req_idx,
        std::min(load_integral(seq_lens, req_idx) + 1, max_model_len));
  }
}

void strict_rejection_sample_kernel_impl(
    at::Tensor& sampled,
    int64_t sampled_stride,
    at::Tensor& num_sampled,
    const at::Tensor& target_sampled,
    const at::Tensor& draft_sampled,
    const at::Tensor& cu_num_logits) {
  // Original: vllm/v1/worker/gpu/spec_decode/rejection_sampler.py
  // _strict_rejection_sample_kernel.
  const int64_t num_reqs = cu_num_logits.numel() - 1;

#pragma omp parallel for
  for (int64_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    const int64_t start_idx = load_integral(cu_num_logits, req_idx);
    const int64_t end_idx = load_integral(cu_num_logits, req_idx + 1);
    const int64_t num_tokens = end_idx - start_idx;

    int64_t accepted = 0;
    bool rejected = false;
    for (int64_t i = 0; i < num_tokens - 1; ++i) {
      if (!rejected) {
        const int64_t target = load_integral(target_sampled, start_idx + i);
        const int64_t draft = load_integral(draft_sampled, start_idx + i + 1);
        store_integral(sampled, req_idx * sampled_stride + i, target);
        ++accepted;
        if (target != draft) {
          rejected = true;
        }
      }
    }
    if (!rejected) {
      const int64_t target =
          load_integral(target_sampled, start_idx + num_tokens - 1);
      store_integral(
          sampled, req_idx * sampled_stride + num_tokens - 1, target);
      ++accepted;
    }
    num_sampled.data_ptr<int32_t>()[req_idx] = static_cast<int32_t>(accepted);
  }
}

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "prepare_eagle_inputs_kernel_impl("
      "Tensor(a!) last_token_indices, Tensor(a!) eagle_input_ids, "
      "Tensor(a!) eagle_positions, Tensor target_input_ids, "
      "Tensor target_positions, Tensor idx_mapping, Tensor last_sampled, "
      "Tensor next_prefill_tokens, Tensor num_sampled, Tensor num_rejected, "
      "Tensor query_start_loc, int num_reqs) -> ()");
  m.def(
      "prepare_eagle_decode_kernel_impl("
      "Tensor draft_tokens, Tensor output_hidden_states, "
      "int output_hidden_states_stride, Tensor last_token_indices, "
      "Tensor target_seq_lens, Tensor num_rejected, Tensor(a!) input_ids, "
      "Tensor(a!) positions, Tensor(a!) input_hidden_states, "
      "int input_hidden_states_stride, Tensor(a!) query_start_loc, "
      "Tensor(a!) seq_lens, int hidden_size, int max_model_len, "
      "int max_num_reqs, int num_reqs) -> ()");
  m.def(
      "update_eagle_inputs_kernel_impl("
      "Tensor(a!) input_ids, Tensor(a!) positions, "
      "Tensor(a!) input_hidden_states, int input_hidden_states_stride, "
      "Tensor(a!) seq_lens, int max_model_len, Tensor draft_tokens, "
      "Tensor output_hidden_states, int output_hidden_states_stride, "
      "int hidden_size) -> ()");
  m.def(
      "strict_rejection_sample_kernel_impl("
      "Tensor(a!) sampled, int sampled_stride, Tensor(a!) num_sampled, "
      "Tensor target_sampled, Tensor draft_sampled, Tensor cu_num_logits) -> ()");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("prepare_eagle_inputs_kernel_impl", &prepare_eagle_inputs_kernel_impl);
  m.impl("prepare_eagle_decode_kernel_impl", &prepare_eagle_decode_kernel_impl);
  m.impl("update_eagle_inputs_kernel_impl", &update_eagle_inputs_kernel_impl);
  m.impl(
      "strict_rejection_sample_kernel_impl",
      &strict_rejection_sample_kernel_impl);
}
