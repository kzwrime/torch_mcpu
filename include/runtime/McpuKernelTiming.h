#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <c10/macros/Macros.h>
#include <c10/util/ApproximateClock.h>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

#ifdef _WIN32
#define MCPU_KERNEL_TIMING_EXPORT __declspec(dllexport)
#else
#define MCPU_KERNEL_TIMING_EXPORT __attribute__((visibility("default")))
#endif

#define MCPU_KERNEL_TIMING_CONCAT_INNER(a, b) a##b
#define MCPU_KERNEL_TIMING_CONCAT(a, b) MCPU_KERNEL_TIMING_CONCAT_INNER(a, b)
#define MCPU_KERNEL_TIMING_SCOPE(name)                                      \
  ::at::mcpu::kernel_timing::Scope MCPU_KERNEL_TIMING_CONCAT(               \
      _mcpu_kernel_timing_scope_, __COUNTER__)(name)
#define MCPU_KERNEL_TIMING_SCOPE_EVENT(name, event)                         \
  ::at::mcpu::kernel_timing::EventScope MCPU_KERNEL_TIMING_CONCAT(          \
      _mcpu_kernel_timing_scope_, __COUNTER__)(name, event)

namespace at::mcpu::kernel_timing {

struct Event {
  const char* name;
  std::uint64_t begin_tsc;
  std::uint64_t end_tsc;
};

struct ThreadSnapshot {
  const char* role;
  std::vector<Event> events;
};

constexpr std::size_t kMaxEventsPerThread = 262144;
constexpr const char* kWorkerRole = "worker";

struct ThreadBuffer {
  const char* role{kWorkerRole};
  std::array<Event, kMaxEventsPerThread> events{};
  std::size_t count{0};
};

MCPU_KERNEL_TIMING_EXPORT extern std::atomic<bool> g_enabled;
MCPU_KERNEL_TIMING_EXPORT void set_enabled(bool value);
MCPU_KERNEL_TIMING_EXPORT void reset();
MCPU_KERNEL_TIMING_EXPORT Event* reserve_event_slot(const char* name);
MCPU_KERNEL_TIMING_EXPORT std::size_t event_count();
MCPU_KERNEL_TIMING_EXPORT double elapsed_us_between(
    std::size_t begin,
    std::size_t end);
MCPU_KERNEL_TIMING_EXPORT std::vector<ThreadSnapshot> snapshot();

inline thread_local Event* current_event_slot = nullptr;

inline bool enabled() {
  return g_enabled.load(std::memory_order_relaxed);
}

inline std::uint64_t read_tsc() {
#if TORCH_MCPU_KERNEL_TIMING_USE_TSC && \
    (defined(__x86_64__) || defined(__i386__))
  unsigned int aux = 0;
  return static_cast<std::uint64_t>(__rdtscp(&aux));
#else
  return static_cast<std::uint64_t>(c10::getTime(/*allow_monotonic=*/true));
#endif
}

class CurrentEventSlotGuard {
 public:
  explicit CurrentEventSlotGuard(Event* slot) noexcept
      : previous_(current_event_slot) {
    current_event_slot = slot;
  }

  ~CurrentEventSlotGuard() noexcept {
    current_event_slot = previous_;
  }

  CurrentEventSlotGuard(const CurrentEventSlotGuard&) = delete;
  CurrentEventSlotGuard& operator=(const CurrentEventSlotGuard&) = delete;

 private:
  Event* previous_;
};

class Scope {
 public:
  explicit Scope(const char* name) noexcept
      : event_(current_event_slot), begin_tsc_(0) {
    (void)name;
    if (C10_LIKELY(event_ != nullptr)) {
      begin_tsc_ = read_tsc();
      event_->name = name;
    }
  }

  ~Scope() noexcept {
    if (C10_LIKELY(event_ != nullptr)) {
      event_->begin_tsc = begin_tsc_;
      event_->end_tsc = read_tsc();
    }
  }

  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

 private:
  Event* event_;
  std::uint64_t begin_tsc_;
};

class EventScope {
 public:
  EventScope(const char* name, Event* event) noexcept
      : event_(event), begin_tsc_(0) {
    if (C10_LIKELY(event_ != nullptr)) {
      begin_tsc_ = read_tsc();
      event_->name = name;
    }
  }

  ~EventScope() noexcept {
    if (C10_LIKELY(event_ != nullptr)) {
      event_->begin_tsc = begin_tsc_;
      event_->end_tsc = read_tsc();
    }
  }

  EventScope(const EventScope&) = delete;
  EventScope& operator=(const EventScope&) = delete;

 private:
  Event* event_;
  std::uint64_t begin_tsc_;
};

} // namespace at::mcpu::kernel_timing
