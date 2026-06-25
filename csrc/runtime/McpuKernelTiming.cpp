#include <runtime/McpuKernelTiming.h>

#include <algorithm>
#include <array>

#include <c10/util/ApproximateClock.h>

namespace at::mcpu::kernel_timing {
namespace {

ThreadBuffer g_worker_buffer{};

} // namespace

std::atomic<bool> g_enabled{false};

void set_enabled(bool value) {
  g_enabled.store(value, std::memory_order_relaxed);
}

void reset() {
  g_worker_buffer.count = 0;
}

Event* reserve_event_slot(const char* name, std::uint64_t stream) {
  if (!enabled()) {
    return nullptr;
  }
  const std::size_t index = g_worker_buffer.count++;
  if (C10_UNLIKELY(index >= kMaxEventsPerThread)) {
    g_worker_buffer.count = kMaxEventsPerThread;
    return nullptr;
  }
  Event& event = g_worker_buffer.events[index];
  event = {name, stream, 0, 0};
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

  std::uint64_t elapsed_time = 0;
  for (std::size_t i = begin; i < end; ++i) {
    const Event& event = g_worker_buffer.events[i];
    if (event.end_time > event.begin_time) {
      elapsed_time += event.end_time - event.begin_time;
    }
  }
  return static_cast<double>(elapsed_time) / 1000.0;
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
