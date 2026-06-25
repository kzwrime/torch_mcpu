#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <c10/macros/Macros.h>
#include <c10/util/ApproximateClock.h>

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
  std::uint64_t stream;
  std::uint64_t begin_time;
  std::uint64_t end_time;
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
MCPU_KERNEL_TIMING_EXPORT Event* reserve_event_slot(
    const char* name,
    std::uint64_t stream = 0);
MCPU_KERNEL_TIMING_EXPORT std::size_t event_count();
MCPU_KERNEL_TIMING_EXPORT double elapsed_us_between(
    std::size_t begin,
    std::size_t end);
MCPU_KERNEL_TIMING_EXPORT std::vector<ThreadSnapshot> snapshot();

inline thread_local Event* current_event_slot = nullptr;

inline bool enabled() {
  return g_enabled.load(std::memory_order_relaxed);
}

inline std::uint64_t read_timer() {
  return static_cast<std::uint64_t>(c10::getTime(/*allow_monotonic=*/true));
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
      : event_(current_event_slot), begin_time_(0) {
    (void)name;
    if (C10_LIKELY(event_ != nullptr)) {
      begin_time_ = read_timer();
      event_->name = name;
    }
  }

  ~Scope() noexcept {
    if (C10_LIKELY(event_ != nullptr)) {
      event_->begin_time = begin_time_;
      event_->end_time = read_timer();
    }
  }

  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

 private:
  Event* event_;
  std::uint64_t begin_time_;
};

class EventScope {
 public:
  EventScope(const char* name, Event* event) noexcept
      : event_(event), begin_time_(0) {
    if (C10_LIKELY(event_ != nullptr)) {
      begin_time_ = read_timer();
      event_->name = name;
    }
  }

  ~EventScope() noexcept {
    if (C10_LIKELY(event_ != nullptr)) {
      event_->begin_time = begin_time_;
      event_->end_time = read_timer();
    }
  }

  EventScope(const EventScope&) = delete;
  EventScope& operator=(const EventScope&) = delete;

 private:
  Event* event_;
  std::uint64_t begin_time_;
};

} // namespace at::mcpu::kernel_timing
