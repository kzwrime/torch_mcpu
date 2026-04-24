#include "Extra.h"

#include "runtime/OpenRegException.h"
#include "runtime/OpenRegStream.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace at::native::mcpu {

namespace {

void check_stream_test_tensor(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1);
  TORCH_CHECK(
      tensor.scalar_type() == at::kLong,
      "stream test ops only support int64 tensors");
  TORCH_CHECK(
      tensor.is_contiguous(), "stream test ops require contiguous input");
  TORCH_CHECK(tensor.numel() > 0, "stream test ops require non-empty input");
}

void sleep_for_ms(int64_t sleep_ms) {
  TORCH_CHECK(sleep_ms >= 0, "sleep_ms must be non-negative");
  std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
}

} // namespace

at::Tensor quantize_per_tensor(
    const at::Tensor& self,
    double scale,
    int64_t zero_point,
    at::ScalarType dtype) {
  return at::native::quantize_per_tensor(self, scale, zero_point, dtype);
}

int64_t _fused_sdp_choice(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  auto backend = sdp::SDPBackend::overrideable;
  return static_cast<int64_t>(backend);
}

void quantize_tensor_per_tensor_affine_stub(
    const at::Tensor& rtensor,
    at::Tensor& qtensor,
    double scale,
    int64_t zero_point) {}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>
_scaled_dot_product_fused_attention_overrideable(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t head_dim_v = value.size(3);
  const int64_t max_seqlen_q = query.size(2);
  const int64_t max_seqlen_kv = key.size(2);

  auto opts = query.options();
  auto output =
      at::empty({batch_size, num_heads, max_seqlen_q, head_dim_v}, opts);
  auto logsumexp =
      at::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
  auto debug_attn_mask = at::empty(
      {batch_size, num_heads, max_seqlen_q, max_seqlen_kv},
      opts.dtype(at::kFloat));
  auto philox_seed = at::empty({}, at::dtype(at::kLong));
  auto philox_offset = at::empty({}, at::dtype(at::kLong));

  return std::make_tuple(
      output,
      logsumexp,
      at::Tensor(),
      at::Tensor(),
      max_seqlen_q,
      max_seqlen_kv,
      philox_seed,
      philox_offset,
      debug_attn_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_fused_attention_overrideable_backward(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    std::array<bool, 4> grad_input_mask,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    std::optional<double> scale) {
  return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
      at::empty_like(query),
      at::empty_like(key),
      at::empty_like(value),
      at::empty_like(attn_bias));
}

namespace {
struct CustomAutogradFnReturnsSelf
    : public torch::autograd::Function<CustomAutogradFnReturnsSelf> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self) {
    return self;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    return {grad_output[0] * 0.5};
  }
};

struct CustomAutogradFnAliasing
    : public torch::autograd::Function<CustomAutogradFnAliasing> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self) {
    return self.view_symint(self.sym_sizes());
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    return {grad_output[0] * 0.5};
  }
};
} // namespace

at::Tensor custom_autograd_fn_returns_self(at::Tensor x) {
  return CustomAutogradFnReturnsSelf::apply(x);
}

at::Tensor custom_autograd_fn_aliasing(at::Tensor x) {
  return CustomAutogradFnAliasing::apply(x);
}

/*
 This implementation is only used to test stub registration, so not all
 capabilities are fully supported.

 Current Limitations:
 - dtype: Float only
 - input tensor: must be contiguous layout
*/
// LITERALINCLUDE START: STUB ABS
void abs_kernel(at::TensorIteratorBase& iter) {
  TORCH_CHECK(iter.ntensors() == 2, "Abs kernel expects 2 tensors");
  TORCH_CHECK(
      iter.common_dtype() == at::ScalarType::Float,
      "Abs kernel only supports float type");

  auto& output_tensor = iter.tensor(0);
  auto& input_tensor = iter.tensor(1);

  TORCH_CHECK(
      input_tensor.sizes() == output_tensor.sizes(),
      "Input and output tensor sizes must match.");

  auto abs_loop = [](float* out_ptr, const float* in_ptr, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
      out_ptr[i] = std::abs(in_ptr[i]);
    }
  };

  MemoryGuard guard(input_tensor, output_tensor);

  if (iter.is_contiguous()) {
    abs_loop(
        static_cast<float*>(iter.data_ptr(0)),
        static_cast<float*>(iter.data_ptr(1)),
        iter.numel());
  } else {
    TORCH_CHECK(
        input_tensor.is_contiguous(), "Input tensor must be contiguous.")

    auto output = at::empty(
        input_tensor.sizes(),
        input_tensor.options().memory_format(
            input_tensor.suggest_memory_format()));

    MemoryGuard guard(output);

    abs_loop(
        static_cast<float*>(output.data_ptr()),
        static_cast<float*>(iter.data_ptr(1)),
        iter.numel());

    output_tensor.copy_(output);
  }
}
// LITERALINCLUDE END: STUB ABS

at::Tensor& abs_out(const at::Tensor& self, at::Tensor& out) {
  return at::native::abs_out(self, out);
}

at::Tensor custom_abs(at::Tensor x) {
  return at::abs(x);
}

at::Tensor& stream_sleep_fill_(at::Tensor& x, int64_t value, int64_t sleep_ms) {
  check_stream_test_tensor(x);
  auto stream = c10::mcpu::getCurrentMcpuStream(x.device().index());

  MCPU_CHECK(openreg::addTaskToStream(stream, [x, value, sleep_ms]() mutable {
    sleep_for_ms(sleep_ms);
    MemoryGuard guard(x);
    auto* data = x.data_ptr<int64_t>();
    std::fill_n(data, x.numel(), value);
  }));

  return x;
}

at::Tensor& stream_sleep_copy_(
    at::Tensor& dst,
    const at::Tensor& src,
    int64_t sleep_ms) {
  check_stream_test_tensor(dst);
  check_stream_test_tensor(src);
  TORCH_CHECK(
      dst.sizes() == src.sizes(),
      "stream_sleep_copy_ expects src/dst with identical shapes");

  auto stream = c10::mcpu::getCurrentMcpuStream(dst.device().index());
  MCPU_CHECK(openreg::addTaskToStream(stream, [dst, src, sleep_ms]() mutable {
    sleep_for_ms(sleep_ms);
    MemoryGuard guard(dst, src);
    std::copy_n(src.data_ptr<int64_t>(), src.numel(), dst.data_ptr<int64_t>());
  }));

  return dst;
}

int64_t first_element_int(const at::Tensor& x) {
  check_stream_test_tensor(x);
  MemoryGuard guard(x);
  return x.data_ptr<int64_t>()[0];
}

} // namespace at::native::mcpu
