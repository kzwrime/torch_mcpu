#include <runtime/McpuKernelLaunch.h>

#include "OpenRegException.h"
#include "OpenRegStream.h"

#include <include/openreg.h>

namespace at::mcpu::detail {

bool enter_kernel_task() {
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  bool previous = openreg::isInKernelTask();
  openreg::setInKernelTask(true);
  return previous;
#else
  return true;
#endif
}

void exit_kernel_task(bool previous) noexcept {
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  openreg::setInKernelTask(previous);
#else
  (void)previous;
#endif
}

void protect_memory(void* ptr) noexcept {
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  (void)orMemoryProtect(ptr);
#else
  (void)ptr;
#endif
}

void unprotect_tensor_memory(
    const at::TensorBase& tensor,
    std::unordered_set<void*>& unprotected_pointers) {
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  TORCH_CHECK(
      openreg::isInKernelTask(),
      "mcpu tensor memory may only be accessed from a launched kernel task");
  if (!tensor.defined() || !tensor.has_storage() || tensor.numel() == 0) {
    return;
  }

  void* ptr = tensor.data_ptr();
  orPointerAttributes attr;
  if (orPointerGetAttributes(&attr, ptr) != orSuccess ||
      attr.type != orMemoryTypeDevice) {
    return;
  }

  auto [_, inserted] = unprotected_pointers.insert(attr.pointer);
  if (inserted) {
    MCPU_CHECK(orMemoryUnprotect(attr.pointer));
  }
#else
  (void)tensor;
  (void)unprotected_pointers;
#endif
}

void launch_kernel_task(
    const at::Tensor& stream_tensor,
    std::function<void()> task) {
  TORCH_CHECK(
      stream_tensor.defined() &&
          stream_tensor.device().type() == c10::DeviceType::PrivateUse1,
      "launch_kernel expects an mcpu tensor to select the stream");
#if TORCH_MCPU_ENABLE_ASYNC_LAUNCH
  auto stream = c10::mcpu::getCurrentMcpuStream(stream_tensor.device().index());
  MCPU_CHECK(orLaunchKernel(stream, std::move(task)));
#else
  task();
#endif
}

} // namespace at::mcpu::detail
