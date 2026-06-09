#include <runtime/McpuKernelTiming.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <thread>

#include <c10/util/ApproximateClock.h>

namespace at::mcpu::kernel_timing {
namespace {

ThreadBuffer g_worker_buffer{};

#if TORCH_MCPU_KERNEL_TIMING_USE_TSC && \
    (defined(__x86_64__) || defined(__i386__))
double timer_ticks_per_ns() {
  static const double value = []() {
    const auto start_tsc = read_tsc();
    const auto start_ns = c10::getTime(/*allow_monotonic=*/true);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const auto end_tsc = read_tsc();
    const auto end_ns = c10::getTime(/*allow_monotonic=*/true);
    const auto elapsed_ns = std::max<std::uint64_t>(1, end_ns - start_ns);
    const auto elapsed_tsc = end_tsc > start_tsc ? end_tsc - start_tsc : 0;
    return std::max(0.001, static_cast<double>(elapsed_tsc) / elapsed_ns);
  }();
  return value;
}
#endif

} // namespace

std::atomic<bool> g_enabled{false};

void set_enabled(bool value) {
  g_enabled.store(value, std::memory_order_relaxed);
}

void reset() {
  g_worker_buffer.count = 0;
}

Event* reserve_event_slot(const char* name) {
  if (!enabled()) {
    return nullptr;
  }
  const std::size_t index = g_worker_buffer.count++;
  if (C10_UNLIKELY(index >= kMaxEventsPerThread)) {
    g_worker_buffer.count = kMaxEventsPerThread;
    return nullptr;
  }
  Event& event = g_worker_buffer.events[index];
  event = {name, 0, 0};
  return &event;
}

std::size_t event_count() {
  return std::min(g_worker_buffer.count, kMaxEventsPerThread);
}

double elapsed_us_between(std::size_t begin, std::size_t end) {
  const std::size_t count = event_count();
  begin = std::min(begin, count);
  end = std::min(end, count);
  if (end <= begin) {
    return 0.0;
  }

  std::uint64_t elapsed_tsc = 0;
  for (std::size_t i = begin; i < end; ++i) {
    const Event& event = g_worker_buffer.events[i];
    if (event.end_tsc > event.begin_tsc) {
      elapsed_tsc += event.end_tsc - event.begin_tsc;
    }
  }
#if TORCH_MCPU_KERNEL_TIMING_USE_TSC && \
    (defined(__x86_64__) || defined(__i386__))
  return static_cast<double>(elapsed_tsc) / timer_ticks_per_ns() / 1000.0;
#else
  return static_cast<double>(elapsed_tsc) / 1000.0;
#endif
}

std::vector<ThreadSnapshot> snapshot() {
  std::vector<ThreadSnapshot> out;
  out.reserve(1);
  const std::size_t event_count =
      std::min(g_worker_buffer.count, kMaxEventsPerThread);
  ThreadSnapshot snapshot;
  snapshot.role = g_worker_buffer.role;
  snapshot.events.assign(
      g_worker_buffer.events.begin(), g_worker_buffer.events.begin() + event_count);
  out.push_back(std::move(snapshot));
  return out;
}

} // namespace at::mcpu::kernel_timing
