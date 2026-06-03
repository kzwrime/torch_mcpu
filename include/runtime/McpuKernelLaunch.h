#pragma once

#include <ATen/ATen.h>
#include <ATen/ThreadLocalState.h>
#include <ATen/core/ivalue.h>
#include <ATen/record_function.h>

#include <functional>
#include <type_traits>
#include <unordered_set>
#include <utility>

#ifdef _WIN32
#define MCPU_KERNEL_LAUNCH_EXPORT __declspec(dllexport)
#else
#define MCPU_KERNEL_LAUNCH_EXPORT __attribute__((visibility("default")))
#endif

namespace at::mcpu::detail {

MCPU_KERNEL_LAUNCH_EXPORT bool enter_kernel_task();
MCPU_KERNEL_LAUNCH_EXPORT void exit_kernel_task(bool previous) noexcept;
MCPU_KERNEL_LAUNCH_EXPORT void protect_memory(void* ptr) noexcept;
MCPU_KERNEL_LAUNCH_EXPORT void unprotect_tensor_memory(
    const at::TensorBase& tensor,
    std::unordered_set<void*>& unprotected_pointers);
MCPU_KERNEL_LAUNCH_EXPORT void launch_kernel_task(
    const at::Tensor& stream_tensor,
    std::function<void()> task);

} // namespace at::mcpu::detail

namespace at::mcpu {

class KernelTaskScope {
 public:
  KernelTaskScope() : previous_(detail::enter_kernel_task()) {}

  ~KernelTaskScope() {
    detail::exit_kernel_task(previous_);
  }

  KernelTaskScope(const KernelTaskScope&) = delete;
  KernelTaskScope& operator=(const KernelTaskScope&) = delete;

 private:
  bool previous_;
};

class KernelMemoryGuard {
 public:
  template <typename... Args>
  explicit KernelMemoryGuard(const Args&... args) {
    (find_and_unprotect_tensors(args), ...);
  }

  ~KernelMemoryGuard() noexcept {
    for (void* ptr : unprotected_pointers_) {
      detail::protect_memory(ptr);
    }
  }

  KernelMemoryGuard(const KernelMemoryGuard&) = delete;
  KernelMemoryGuard& operator=(const KernelMemoryGuard&) = delete;

 private:
  template <typename T>
  void find_and_unprotect_tensors(const T& item) {
    if constexpr (std::is_base_of_v<at::TensorBase, T>) {
      detail::unprotect_tensor_memory(item, unprotected_pointers_);
    } else if constexpr (std::is_same_v<T, c10::IValue>) {
      if (item.isTensor()) {
        detail::unprotect_tensor_memory(item.toTensor(), unprotected_pointers_);
      } else if (item.isTensorList()) {
        for (const at::Tensor& tensor : item.toTensorList()) {
          detail::unprotect_tensor_memory(tensor, unprotected_pointers_);
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

  std::unordered_set<void*> unprotected_pointers_;
};

template <typename Func>
inline void launch_kernel(
    const at::Tensor& stream_tensor,
    const char* record_name,
    Func&& func) {
  auto thread_local_state = at::ThreadLocalState();
  detail::launch_kernel_task(
      stream_tensor,
      [record_name,
       thread_local_state = std::move(thread_local_state),
       func = std::forward<Func>(func)]() mutable {
        at::ThreadLocalStateGuard thread_local_state_guard(thread_local_state);
        // RECORD_USER_SCOPE(record_name);
        KernelTaskScope kernel_task;
        std::invoke(func);
      });
}

template <typename Func>
inline void launch_kernel(const at::Tensor& stream_tensor, Func&& func) {
  launch_kernel(stream_tensor, "mcpu::kernel_task", std::forward<Func>(func));
}

} // namespace at::mcpu
