#include <runtime/McpuOpTiming.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>

#include <c10/util/ApproximateClock.h>

namespace at::mcpu::op_timing {
namespace {

constexpr std::size_t kMaxRecordsPerThread = 262144;
constexpr const char* kUnknownRole = "unknown";

struct ThreadBuffer {
  const char* role{kUnknownRole};
  std::array<Record, kMaxRecordsPerThread> records{};
  std::size_t count{0};
  bool registered{false};
};

std::atomic<bool> g_enabled{false};
std::mutex g_buffers_mutex;
std::vector<ThreadBuffer*> g_buffers;

thread_local ThreadBuffer tls_buffer;

std::uint64_t now_ns() {
  return static_cast<std::uint64_t>(c10::getTime(/*allow_monotonic=*/true));
}

void register_buffer(ThreadBuffer& buffer, const char* role) {
  if (buffer.registered) {
    if (buffer.role == nullptr || buffer.role == kUnknownRole) {
      buffer.role = role;
    }
    return;
  }
  buffer.role = role;
  std::lock_guard<std::mutex> lock(g_buffers_mutex);
  if (!buffer.registered) {
    g_buffers.push_back(&buffer);
    buffer.registered = true;
  }
}

} // namespace

bool enabled() {
  return g_enabled.load(std::memory_order_relaxed);
}

void set_enabled(bool value) {
  g_enabled.store(value, std::memory_order_relaxed);
}

void reset() {
  std::lock_guard<std::mutex> lock(g_buffers_mutex);
  for (ThreadBuffer* buffer : g_buffers) {
    buffer->count = 0;
    buffer->role = kUnknownRole;
  }
}

void record(const char* role, const char* op, const char* phase) {
  if (!enabled()) {
    return;
  }
  const std::uint64_t t_ns = now_ns();
  register_buffer(tls_buffer, role);
  const std::size_t index = tls_buffer.count++;
  if (index >= kMaxRecordsPerThread) {
    tls_buffer.count = kMaxRecordsPerThread;
    return;
  }
  tls_buffer.records[index] = {op, phase, t_ns};
}

std::vector<ThreadSnapshot> snapshot() {
  std::lock_guard<std::mutex> lock(g_buffers_mutex);
  std::vector<ThreadSnapshot> out;
  out.reserve(g_buffers.size());
  for (ThreadBuffer* buffer : g_buffers) {
    const std::size_t count = std::min(buffer->count, kMaxRecordsPerThread);
    ThreadSnapshot snapshot;
    snapshot.role = buffer->role;
    snapshot.records.assign(buffer->records.begin(), buffer->records.begin() + count);
    out.push_back(std::move(snapshot));
  }
  return out;
}

} // namespace at::mcpu::op_timing
