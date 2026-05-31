#pragma once

#include "OpenRegException.h"
#include "OpenRegStream.h"

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <include/openreg.h>

#include <functional>

#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
#include <unordered_set>
#endif

namespace at::mcpu {

class KernelTaskScope {
 public:
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  KernelTaskScope() : previous_(openreg::isInKernelTask()) {
    openreg::setInKernelTask(true);
  }

  ~KernelTaskScope() {
    openreg::setInKernelTask(previous_);
  }
#else
  KernelTaskScope() = default;
  ~KernelTaskScope() = default;
#endif

  KernelTaskScope(const KernelTaskScope&) = delete;
  KernelTaskScope& operator=(const KernelTaskScope&) = delete;

 private:
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  bool previous_;
#endif
};

class KernelMemoryGuard {
 public:
  template <typename... Args>
  explicit KernelMemoryGuard(const Args&... args) {
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
    TORCH_CHECK(
        openreg::isInKernelTask(),
        "mcpu tensor memory may only be accessed from a launched kernel task");
    (find_and_unprotect_tensors(args), ...);
#endif
  }

  ~KernelMemoryGuard() noexcept {
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
    for (void* ptr : unprotected_pointers_) {
      orMemoryProtect(ptr);
    }
#endif
  }

  KernelMemoryGuard(const KernelMemoryGuard&) = delete;
  KernelMemoryGuard& operator=(const KernelMemoryGuard&) = delete;

 private:
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  template <typename T>
  void find_and_unprotect_tensors(const T& item) {
    if constexpr (std::is_base_of_v<at::TensorBase, T>) {
      unprotect_if_needed(item);
    } else if constexpr (std::is_same_v<T, c10::IValue>) {
      if (item.isTensor()) {
        unprotect_if_needed(item.toTensor());
      } else if (item.isTensorList()) {
        for (const at::Tensor& tensor : item.toTensorList()) {
          unprotect_if_needed(tensor);
        }
      } else if (item.isList()) {
        for (const c10::IValue& element : item.toListRef()) {
          find_and_unprotect_tensors(element);
        }
      } else if (item.isGenericDict()) {
        for (const auto& entry : item.toGenericDict()) {
          find_and_unprotect_tensors(entry.key());
          find_and_unprotect_tensors(entry.value());
        }
      }
    }
  }

  void unprotect_if_needed(const at::TensorBase& tensor) {
    if (!tensor.defined() || !tensor.has_storage() || tensor.numel() == 0) {
      return;
    }

    void* ptr = tensor.data_ptr();
    orPointerAttributes attr;
    if (orPointerGetAttributes(&attr, ptr) != orSuccess ||
        attr.type != orMemoryTypeDevice) {
      return;
    }

    auto [_, inserted] = unprotected_pointers_.insert(attr.pointer);
    if (inserted) {
      MCPU_CHECK(orMemoryUnprotect(attr.pointer));
    }
  }

  std::unordered_set<void*> unprotected_pointers_;
#endif
};

template <typename Func>
inline void launch_kernel(const at::Tensor& stream_tensor, Func&& func) {
  TORCH_CHECK(
      stream_tensor.defined() &&
          stream_tensor.device().type() == c10::DeviceType::PrivateUse1,
      "launch_kernel expects an mcpu tensor to select the stream");
#if TORCH_MCPU_ENABLE_ASYNC_LAUNCH
  auto stream = c10::mcpu::getCurrentMcpuStream(stream_tensor.device().index());
  auto task = [func = std::forward<Func>(func)]() mutable {
    KernelTaskScope kernel_task;
    std::invoke(func);
  };
  MCPU_CHECK(orLaunchKernel(stream, std::move(task)));
#else
  KernelTaskScope kernel_task;
  std::forward<Func>(func)();
#endif
}

} // namespace at::mcpu
