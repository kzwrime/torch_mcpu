#ifndef OPENREG_H
#error "Don`t include openreg.inl directly, include openreg.h instead."
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <new>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

namespace openreg {

#if defined(__x86_64__) || defined(__i386__)
inline void cpu_relax() {
  __builtin_ia32_pause();
}
#elif defined(__aarch64__)
inline void cpu_relax() {
  asm volatile("yield" ::: "memory");
}
#else
inline void cpu_relax() {
  std::this_thread::yield();
}
#endif

struct QueueSlot {
  using RunFn = void (*)(void*);
  using DestroyFn = void (*)(void*);

  RunFn run = nullptr;
  DestroyFn destroy = nullptr;
  alignas(std::max_align_t) unsigned char storage[256];
};

template <typename Func, typename... Args>
struct TupleTask {
  std::tuple<Func, Args...> payload;

  template <typename F, typename... A>
  explicit TupleTask(F&& func, A&&... args)
      : payload(std::forward<F>(func), std::forward<A>(args)...) {}

  void operator()() {
    invoke(std::make_index_sequence<sizeof...(Args)>{});
  }

  template <std::size_t... I>
  void invoke(std::index_sequence<I...>) {
    std::invoke(
        std::get<0>(payload), std::get<I + 1>(payload)...);
  }
};

template <typename Func>
struct StoredCallable {
  using RawFunc = typename std::remove_reference<Func>::type;
  using type = typename std::conditional<
      std::is_function<RawFunc>::value,
      Func,
      typename std::decay<Func>::type>::type;
};

template <typename Task>
inline void runTask(void* ptr) {
  (*static_cast<Task*>(ptr))();
}

template <typename Task>
inline void destroyTask(void* ptr) {
  static_cast<Task*>(ptr)->~Task();
}

#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
OPENREG_EXPORT bool isInKernelTask();
OPENREG_EXPORT void setInKernelTask(bool enabled);
#else
inline bool isInKernelTask() {
  return true;
}
#endif

} // namespace openreg

struct orStream {
  static constexpr std::uint64_t kQueueCapacity = 16384;
  static constexpr std::uint64_t kQueueMask = kQueueCapacity - 1;

  openreg::QueueSlot slots[kQueueCapacity];
  alignas(64) std::atomic<std::uint64_t> head{0};
  alignas(64) std::atomic<std::uint64_t> tail{0};
  alignas(64) std::atomic<std::uint64_t> completed{0};
  alignas(64) std::atomic<bool> stop{false};
  std::thread worker;
  int device_index = -1;

  OPENREG_EXPORT orStream();
  OPENREG_EXPORT ~orStream();
  OPENREG_EXPORT void workerLoop();
  OPENREG_EXPORT void synchronize();
  OPENREG_EXPORT bool idle() const;

  template <typename Func, typename... Args>
  inline orError_t launch(Func&& func, Args&&... args) {
    using Task = openreg::TupleTask<
        typename openreg::StoredCallable<Func>::type,
        typename std::decay<Args>::type...>;

    static_assert(
        sizeof(Task) <= sizeof(openreg::QueueSlot::storage),
        "openreg queued task is too large for the inline tuple slot");
    static_assert(
        alignof(Task) <= alignof(std::max_align_t),
        "openreg queued task alignment exceeds the slot alignment");

    auto index = tail.load(std::memory_order_relaxed);
    while (index - head.load(std::memory_order_acquire) >= kQueueCapacity) {
      openreg::cpu_relax();
    }

    auto& slot = slots[index & kQueueMask];
    new (slot.storage) Task(std::forward<Func>(func), std::forward<Args>(args)...);
    slot.run = &openreg::runTask<Task>;
    slot.destroy = &openreg::destroyTask<Task>;
    tail.store(index + 1, std::memory_order_release);
    return orSuccess;
  }
};

namespace openreg {

template <typename Func>
inline orError_t addTaskToStream(orStream* stream, Func&& task) {
  if (!stream) {
    return orErrorUnknown;
  }
  return stream->launch(std::forward<Func>(task));
}

} // namespace openreg

template <typename Func, typename... Args>
OPENREG_EXPORT inline orError_t orLaunchKernel(
    orStream* stream,
    Func&& kernel_func,
    Args&&... args) {
  if (!stream) {
    return orErrorUnknown;
  }
  return stream->launch(
      std::forward<Func>(kernel_func), std::forward<Args>(args)...);
}
