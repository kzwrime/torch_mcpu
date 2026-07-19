#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>

#if __has_include("../openreg/openreg.h")
#include "../openreg/openreg.h"
#else
#include <openreg.h>
#endif

#include "runtime/McpuKernelTiming.h"
#include "runtime/OpenRegStream.h"

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <type_traits>
#include <unordered_set>
#include <utility>

#ifdef _WIN32
#define MCPU_KERNEL_LAUNCH_EXPORT __declspec(dllexport)
#else
#define MCPU_KERNEL_LAUNCH_EXPORT __attribute__((visibility("default")))
#endif

#define MCPU_LAUNCH_UNPAREN(...) __VA_ARGS__
#define MCPU_LAUNCH_TIMED_KERNEL(name, captures, ...)               \
  ::at::mcpu::launch_timed_kernel(                                  \
      name,                                                         \
      MCPU_LAUNCH_UNPAREN captures(                                 \
          ::at::mcpu::kernel_timing::Event* timing_event) mutable { \
        MCPU_KERNEL_TIMING_SCOPE_EVENT(name, timing_event);         \
        __VA_ARGS__                                                 \
      })

namespace at::mcpu::detail {

MCPU_KERNEL_LAUNCH_EXPORT bool enter_kernel_task();
MCPU_KERNEL_LAUNCH_EXPORT void exit_kernel_task(bool previous) noexcept;
MCPU_KERNEL_LAUNCH_EXPORT void protect_memory(void* ptr) noexcept;
MCPU_KERNEL_LAUNCH_EXPORT void unprotect_memory(
    const void* ptr,
    std::unordered_set<void*>& unprotected_pointers);
MCPU_KERNEL_LAUNCH_EXPORT void unprotect_tensor_memory(
    const at::TensorBase& tensor,
    std::unordered_set<void*>& unprotected_pointers);
MCPU_KERNEL_LAUNCH_EXPORT void unprotect_all_device_memory(
    std::unordered_set<void*>& unprotected_pointers);
MCPU_KERNEL_LAUNCH_EXPORT orStream_t
get_kernel_launch_stream(const at::Tensor& stream_tensor);

#if TORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH
inline void synchronize_kernel_launch_stream(orStream_t stream) {
  TORCH_CHECK(
      orStreamSynchronize(stream) == orSuccess,
      "orStreamSynchronize failed after kernel launch");
}
#endif

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

class KernelPointerMemoryGuard {
 public:
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  explicit KernelPointerMemoryGuard(std::initializer_list<const void*> ptrs) {
    for (const void* ptr : ptrs) {
      detail::unprotect_memory(ptr, unprotected_pointers_);
    }
  }

  explicit KernelPointerMemoryGuard(c10::ArrayRef<const void*> ptrs) {
    for (const void* ptr : ptrs) {
      detail::unprotect_memory(ptr, unprotected_pointers_);
    }
  }

  ~KernelPointerMemoryGuard() noexcept {
    for (void* ptr : unprotected_pointers_) {
      detail::protect_memory(ptr);
    }
  }
#else
  explicit KernelPointerMemoryGuard(
      std::initializer_list<const void*>) noexcept {}
  explicit KernelPointerMemoryGuard(c10::ArrayRef<const void*>) noexcept {}
  ~KernelPointerMemoryGuard() noexcept = default;
#endif

  KernelPointerMemoryGuard(const KernelPointerMemoryGuard&) = delete;
  KernelPointerMemoryGuard& operator=(const KernelPointerMemoryGuard&) = delete;

 private:
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  std::unordered_set<void*> unprotected_pointers_;
#endif
};

class KernelAllMemoryGuard {
 public:
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  KernelAllMemoryGuard() {
    detail::unprotect_all_device_memory(unprotected_pointers_);
  }

  ~KernelAllMemoryGuard() noexcept {
    for (void* ptr : unprotected_pointers_) {
      detail::protect_memory(ptr);
    }
  }
#else
  KernelAllMemoryGuard() noexcept = default;
  ~KernelAllMemoryGuard() noexcept = default;
#endif

  KernelAllMemoryGuard(const KernelAllMemoryGuard&) = delete;
  KernelAllMemoryGuard& operator=(const KernelAllMemoryGuard&) = delete;

 private:
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  std::unordered_set<void*> unprotected_pointers_;
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
inline void launch_kernel_on_stream(orStream_t stream, Func&& func) {
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
#if TORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH
  detail::synchronize_kernel_launch_stream(stream);
#endif
}

template <typename Func>
[[deprecated(
    "record_name is ignored for ordinary launch; use launch_kernel_on_stream(stream, lambda) or launch_timed_kernel_on_stream(stream, name, lambda(Event*))")]]
inline void launch_kernel_on_stream(
    orStream_t stream,
    const char* record_name,
    Func&& func) {
  (void)record_name;
  launch_kernel_on_stream(stream, std::forward<Func>(func));
}

template <typename Func>
inline void launch_timed_kernel_on_stream(
    orStream_t stream,
    const char* record_name,
    Func&& func) {
  auto* event_slot = kernel_timing::reserve_event_slot(
      record_name,
      static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(stream)));
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
  auto status = orLaunchKernel(
      stream, [event_slot, func = std::forward<Func>(func)]() mutable {
        KernelTaskScope kernel_task;
        std::invoke(func, event_slot);
      });
#else
  auto status = orLaunchKernel(stream, std::forward<Func>(func), event_slot);
#endif
  TORCH_CHECK(status == orSuccess, "orLaunchKernel failed");
#if TORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH
  detail::synchronize_kernel_launch_stream(stream);
#endif
}

template <typename Func>
inline void launch_kernel(
    Func&& func,
    orStream_t stream = c10::mcpu::getCurrentMcpuStream()) {
  launch_kernel_on_stream(stream, std::forward<Func>(func));
}

template <typename Func>
inline void launch_timed_kernel(const char* record_name, Func&& func) {
  launch_timed_kernel_on_stream(
      c10::mcpu::getCurrentMcpuStream(), record_name, std::forward<Func>(func));
}

template <typename Func>
[[deprecated(
    "record_name is ignored for ordinary launch; use launch_kernel(lambda) or launch_timed_kernel(name, lambda(Event*))")]]
inline void launch_kernel(
    const char* record_name,
    Func&& func,
    orStream_t stream = c10::mcpu::getCurrentMcpuStream()) {
  launch_kernel_on_stream(stream, record_name, std::forward<Func>(func));
}

template <typename Func>
[[deprecated(
    "record_name is ignored for ordinary launch; use launch_kernel(stream_tensor, lambda) or launch_timed_kernel_on_stream(stream, name, lambda(Event*))")]]
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
  launch_kernel_on_stream(
      detail::get_kernel_launch_stream(stream_tensor),
      std::forward<Func>(func));
}

} // namespace at::mcpu
