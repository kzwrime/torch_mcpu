#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>

#if __has_include("../openreg/openreg.h")
#include "../openreg/openreg.h"
#else
#include <include/openreg.h>
#endif

#include "runtime/OpenRegStream.h"

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
MCPU_KERNEL_LAUNCH_EXPORT orStream_t
get_kernel_launch_stream(const at::Tensor& stream_tensor);
[[deprecated("use launch_kernel or launch_kernel_on_stream instead")]]
MCPU_KERNEL_LAUNCH_EXPORT void launch_kernel_task(
    const at::Tensor& stream_tensor,
    std::function<void()> task);

} // namespace at::mcpu::detail

namespace at::mcpu {

class KernelTaskScope {
 public:
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  KernelTaskScope() : previous_(detail::enter_kernel_task()) {}

  ~KernelTaskScope() {
    detail::exit_kernel_task(previous_);
  }
#else
  KernelTaskScope() noexcept = default;
  ~KernelTaskScope() noexcept = default;
#endif

  KernelTaskScope(const KernelTaskScope&) = delete;
  KernelTaskScope& operator=(const KernelTaskScope&) = delete;

 private:
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  bool previous_;
#endif
};

#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
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
#else
class KernelMemoryGuard {
 public:
  template <typename... Args>
  explicit KernelMemoryGuard(const Args&...) noexcept {}

  KernelMemoryGuard(const KernelMemoryGuard&) = delete;
  KernelMemoryGuard& operator=(const KernelMemoryGuard&) = delete;
};
#endif

template <typename Func>
inline void launch_kernel_on_stream(
    orStream_t stream,
    const char* record_name,
    Func&& func) {
  (void)record_name;
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  auto status =
      orLaunchKernel(stream, [func = std::forward<Func>(func)]() mutable {
        KernelTaskScope kernel_task;
        std::invoke(func);
      });
#else
  auto status = orLaunchKernel(stream, std::forward<Func>(func));
#endif
  TORCH_CHECK(status == orSuccess, "orLaunchKernel failed");
}

template <typename Func>
inline void launch_kernel_on_stream(orStream_t stream, Func&& func) {
  launch_kernel_on_stream(
      stream, "mcpu::kernel_task", std::forward<Func>(func));
}

template <
    typename Func,
    typename Arg0,
    typename... Args,
    typename = std::enable_if_t<!std::is_same_v<std::decay_t<Func>, const char*>>>
inline void launch_kernel_on_stream(
    orStream_t stream,
    Func&& func,
    Arg0&& arg0,
    Args&&... args) {
  auto status = orLaunchKernel(
      stream,
      std::forward<Func>(func),
      std::forward<Arg0>(arg0),
      std::forward<Args>(args)...);
  TORCH_CHECK(status == orSuccess, "orLaunchKernel failed");
}

template <typename Func>
inline void launch_kernel(
    Func&& func,
    orStream_t stream = c10::mcpu::getCurrentMcpuStream()) {
  launch_kernel_on_stream(
      stream, "mcpu::kernel_task", std::forward<Func>(func));
}

template <
    typename Func,
    typename Arg0,
    typename... Args,
    typename = std::enable_if_t<
        !std::is_base_of_v<at::TensorBase, std::decay_t<Func>> &&
        ((sizeof...(Args) > 0) ||
         !std::is_same_v<std::decay_t<Arg0>, orStream_t>)>>
inline void launch_kernel(Func&& func, Arg0&& arg0, Args&&... args) {
  auto status = orLaunchKernel(
      c10::mcpu::getCurrentMcpuStream(),
      std::forward<Func>(func),
      std::forward<Arg0>(arg0),
      std::forward<Args>(args)...);
  TORCH_CHECK(status == orSuccess, "orLaunchKernel failed");
}

template <typename Func>
inline void launch_kernel(
    const char* record_name,
    Func&& func,
    orStream_t stream = c10::mcpu::getCurrentMcpuStream()) {
  launch_kernel_on_stream(stream, record_name, std::forward<Func>(func));
}

template <typename Func>
inline void launch_kernel(
    const at::Tensor& stream_tensor,
    const char* record_name,
    Func&& func) {
  launch_kernel_on_stream(
      detail::get_kernel_launch_stream(stream_tensor),
      record_name,
      std::forward<Func>(func));
}

template <typename Func>
inline void launch_kernel(const at::Tensor& stream_tensor, Func&& func) {
  launch_kernel(stream_tensor, "mcpu::kernel_task", std::forward<Func>(func));
}

} // namespace at::mcpu
