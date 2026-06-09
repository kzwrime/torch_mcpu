#include <sstream>
#include <cmath>

#include <c10/core/DeviceGuard.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/irange.h>
#include <torch/csrc/profiler/stubs/base.h>
#include <torch/csrc/profiler/util.h>

#include <include/openreg.h>

#include "runtime/McpuKernelTiming.h"
#include "runtime/OpenRegFunctions.h"
#include "runtime/OpenRegStream.h"

namespace torch::profiler::impl {
namespace {

static void mcpuCheck(orError_t result, const char* file, int line) {
  if (result != orSuccess) {
    std::stringstream ss;
    ss << file << ":" << line << ": ";
    if (result == orErrorNotReady) {
      ss << "Mcpu operation not ready";
    } else {
      ss << "Mcpu error: " << result;
    }
    TORCH_CHECK(false, ss.str());
  }
}
#define TORCH_MCPU_CHECK(result) mcpuCheck(result, __FILE__, __LINE__);

struct McpuProfilerEvent {
  c10::mcpu::McpuStream stream;
  std::size_t timing_count{0};

  explicit McpuProfilerEvent(c10::mcpu::McpuStream recorded_stream)
      : stream(recorded_stream),
        timing_count(at::mcpu::kernel_timing::event_count()) {}
};

struct McpuMethods : public ProfilerStubs {
  void record(
      c10::DeviceIndex* device,
      ProfilerVoidEventStub* event,
      int64_t* cpu_ns) const override {
    auto stream = c10::mcpu::getCurrentMcpuStream();

    // Get current device if requested
    if (device) {
      *device = c10::mcpu::current_device();
    }

    // Record CPU timestamp if requested
    if (cpu_ns) {
      *cpu_ns = c10::getTime();
    }

    at::mcpu::kernel_timing::set_enabled(true);
    *event = std::make_shared<McpuProfilerEvent>(stream);
  }

  float elapsed(
      const ProfilerVoidEventStub* event_,
      const ProfilerVoidEventStub* event2_) const override {
    auto event =
        reinterpret_cast<const std::shared_ptr<McpuProfilerEvent>*>(event_);
    auto event2 =
        reinterpret_cast<const std::shared_ptr<McpuProfilerEvent>*>(event2_);

    // Check if events are valid
    if (!event || !(*event) || !event2 || !(*event2)) {
      return 0.0f;
    }

    (*event2)->stream.synchronize();
    const double elapsed_us = at::mcpu::kernel_timing::elapsed_us_between(
        (*event)->timing_count, (*event2)->timing_count);
    return static_cast<float>(std::floor(elapsed_us + 0.5));
  }

  void mark(const char* name) const override {
    // Mcpu doesn't have built-in annotation support like NVTX
    // This is a no-op for KINETO_PRIVATEUSE1_FALLBACK mode
    // PRIVATEUSE1 mode will use Mcpu defined `enter()` and `exit()` instead
  }

  void rangePush(const char* name) const override {
    // Mcpu doesn't have built-in annotation support like NVTX
    // This is a no-op for KINETO_PRIVATEUSE1_FALLBACK mode
    // PRIVATEUSE1 mode will use Mcpu defined `enter()` and `exit()` instead
  }

  void rangePop() const override {
    // Mcpu doesn't have built-in annotation support like NVTX
    // This is a no-op for KINETO_PRIVATEUSE1_FALLBACK mode
    // PRIVATEUSE1 mode will use Mcpu defined `enter()` and `exit()` instead
  }

  void onEachDevice(std::function<void(int)> op) const override {
    c10::DeviceGuard device_guard(c10::DeviceType::PrivateUse1);
    int device_count = c10::mcpu::device_count();
    for (const auto i : c10::irange(device_count)) {
      device_guard.set_index(i);
      op(i);
    }
  }

  void synchronize() const override {
    TORCH_MCPU_CHECK(orDeviceSynchronize());
  }

  bool enabled() const override {
    return true;
  }
};

struct RegisterMcpuMethods {
  RegisterMcpuMethods() {
    static McpuMethods methods;
    registerPrivateUse1Methods(&methods);
  }
};
RegisterMcpuMethods reg;

} // namespace
} // namespace torch::profiler::impl
