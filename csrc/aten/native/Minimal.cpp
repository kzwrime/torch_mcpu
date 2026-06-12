#include "Minimal.h"
#if TORCH_MCPU_ENABLE_CPU_FALLBACK
#include "MCPUFallback.h"
#endif
#include "runtime/McpuKernelLaunch.h"

#include <ATen/WrapDimUtils.h>

#if TORCH_MCPU_ENABLE_CPU_FALLBACK
#include <unordered_set>
#endif

namespace at::native::mcpu {

namespace {

bool is_mcpu_tensor(const at::Tensor& tensor) {
  return tensor.device().type() == c10::DeviceType::PrivateUse1;
}

bool is_host_device_copy(const at::Tensor& self, const at::Tensor& dst) {
  return (self.is_cpu() && is_mcpu_tensor(dst)) ||
      (is_mcpu_tensor(self) && dst.is_cpu());
}

at::Tensor make_mcpu_tensor_cpu_view(const at::Tensor& tensor) {
  return at::from_blob(
      tensor.data_ptr(),
      tensor.sizes(),
      tensor.strides(),
      tensor.options().device(at::kCPU));
}

void launch_async_host_device_copy(
    const at::Tensor& self,
    const at::Tensor& dst) {
  const at::Tensor& mcpu_tensor = self.is_cpu() ? dst : self;
  auto stream = c10::mcpu::getCurrentMcpuStream(mcpu_tensor.device().index());

  at::mcpu::launch_timed_kernel_on_stream(
      stream,
      "mcpu::_copy_from.host_device",
      [self, dst](at::mcpu::kernel_timing::Event* timing_event) mutable {
        MCPU_KERNEL_TIMING_SCOPE_EVENT(
            "mcpu::_copy_from.host_device", timing_event);

        if (self.is_cpu()) {
          at::mcpu::KernelMemoryGuard guard(dst);
          at::Tensor dst_as_cpu = make_mcpu_tensor_cpu_view(dst);
          at::native::copy_(const_cast<at::Tensor&>(dst_as_cpu), self, true);
        } else {
          at::mcpu::KernelMemoryGuard guard(self);
          at::Tensor self_as_cpu = make_mcpu_tensor_cpu_view(self);
          at::native::copy_(const_cast<at::Tensor&>(dst), self_as_cpu, true);
        }
      });
}

} // namespace

// LITERALINCLUDE START: EMPTY.MEMORY_FORMAT IMPL
at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_generic(
      size, allocator, pu1_dks, dtype, memory_format_opt);
}
// LITERALINCLUDE END: EMPTY.MEMORY_FORMAT IMPL

at::Tensor empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_strided_generic(
      size, stride, allocator, pu1_dks, dtype);
}

at::Tensor as_strided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset) {
  MemoryGuard guard(self);

  return at::cpu::as_strided_symint(self, size, stride, storage_offset);
}

const at::Tensor& resize_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::resize_(
      self, C10_AS_INTARRAYREF_SLOW(size), memory_format);
}

at::Tensor _reshape_alias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  return at::native::_reshape_alias(
      self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
}

at::Tensor _copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  TORCH_CHECK(self.defined(), "Source tensor (self) is not defined.");
  TORCH_CHECK(dst.defined(), "Destination tensor (dst) is not defined.");

  if (is_mcpu_tensor(self) && is_mcpu_tensor(dst) &&
      self.device() == dst.device()) {
    at::mcpu::launch_timed_kernel(
        "mcpu::_copy_from.same_device",
        [self, dst, non_blocking](
            at::mcpu::kernel_timing::Event* timing_event) mutable {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::_copy_from.same_device", timing_event);
          at::mcpu::KernelMemoryGuard guard(self, dst);
          at::Tensor dst_as_cpu = at::from_blob(
              dst.data_ptr(),
              dst.sizes(),
              dst.strides(),
              dst.options().device(at::kCPU));
          const at::Tensor self_as_cpu = at::from_blob(
              self.data_ptr(),
              self.sizes(),
              self.strides(),
              self.options().device(at::kCPU));
          at::native::copy_(
              const_cast<at::Tensor&>(dst_as_cpu), self_as_cpu, non_blocking);
        });
    return dst;
  }

  if (non_blocking && is_host_device_copy(self, dst)) {
    launch_async_host_device_copy(self, dst);
    return dst;
  }

  synchronize_if_mcpu(self);
  synchronize_if_mcpu(dst);
  MemoryGuard guard(self, dst);

  if (self.device() == dst.device()) {
    at::Tensor dst_as_cpu = at::from_blob(
        dst.data_ptr(),
        dst.sizes(),
        dst.strides(),
        dst.options().device(at::kCPU));
    const at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));

    at::native::copy_(
        const_cast<at::Tensor&>(dst_as_cpu), self_as_cpu, non_blocking);

  } else {
    if (self.is_cpu()) {
      at::Tensor dst_as_cpu = make_mcpu_tensor_cpu_view(dst);

      at::native::copy_(
          const_cast<at::Tensor&>(dst_as_cpu), self, non_blocking);

    } else {
      at::Tensor self_as_cpu = make_mcpu_tensor_cpu_view(self);

      at::native::copy_(
          const_cast<at::Tensor&>(dst), self_as_cpu, non_blocking);
    }
  }

  return dst;
}

at::Tensor _copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  at::native::resize_(dst, self.sizes(), std::nullopt);
  return at::native::copy_(const_cast<at::Tensor&>(dst), self, false);
}

at::Scalar _local_scalar_dense(const at::Tensor& self) {
  synchronize_if_mcpu(self);
  MemoryGuard guard(self);
  return at::native::_local_scalar_dense_cpu(self);
}

at::Tensor& set_source_Tensor_(at::Tensor& self, const at::Tensor& source) {
  return at::native::set_tensor_(self, source);
}

at::Tensor& set_source_Storage_(at::Tensor& self, at::Storage source) {
  return at::native::set_(self, source);
}

at::Tensor& set_source_Storage_storage_offset_(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  return at::cpu::set_(result, storage, storage_offset, size, stride);
}

at::Tensor view(const at::Tensor& self, c10::SymIntArrayRef size) {
  MemoryGuard guard(self);
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

at::Tensor unfold(
    const at::Tensor& self,
    int64_t dimension,
    int64_t size,
    int64_t step) {
  const auto ndim = self.dim();
  const auto wrapped_dim =
      at::maybe_wrap_dim(dimension, ndim, /*wrap_scalar=*/true);

  auto sizes = self.sizes().vec();
  auto strides = self.strides().vec();
  const int64_t max_size = ndim == 0 ? 1 : sizes[wrapped_dim];
  TORCH_CHECK(size >= 0, "size is ", size, " but must be >= 0");
  TORCH_CHECK(
      size <= max_size,
      "maximum size for tensor at dimension ",
      wrapped_dim,
      " is ",
      max_size,
      " but size is ",
      size);
  TORCH_CHECK(step > 0, "step is ", step, " but must be > 0");

  sizes.push_back(size);
  strides.push_back(ndim == 0 ? 1 : strides[wrapped_dim]);
  if (wrapped_dim < ndim) {
    sizes[wrapped_dim] = (sizes[wrapped_dim] - size) / step + 1;
    strides[wrapped_dim] *= step;
  }

  return at::native::mcpu::as_strided(
      self,
      c10::fromIntArrayRefSlow(sizes),
      c10::fromIntArrayRefSlow(strides),
      std::nullopt);
}

#if TORCH_MCPU_ENABLE_CPU_FALLBACK
// LITERALINCLUDE START: FALLBACK IMPL
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  static const std::unordered_set<c10::OperatorName> cpu_fallback_blocklist = {
      c10::OperatorName("aten::abs", ""),
      c10::OperatorName("aten::abs", "out"),
  };

  const auto& op_name = op.schema().operator_name();
  if (cpu_fallback_blocklist.count(op_name)) {
    TORCH_CHECK(
        false, "Operator '", op_name, "' is not implemented for device mcpu.");
  } else {
    // Call our custom CPU fallback implementation instead of PyTorch's
    at::native::mcpu::custom::cpu_fallback(op, stack);
  }
}
// LITERALINCLUDE END: FALLBACK IMPL
#endif

} // namespace at::native::mcpu
