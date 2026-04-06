#include "OpenRegGenerator.h"

// Default, global generators, one per device.
static std::vector<at::Generator> default_generators;

namespace c10::mcpu {

// LITERALINCLUDE MCPU GET DEFAULT GENERATOR IMPL
const at::Generator& getDefaultMcpuGenerator(c10::DeviceIndex device_index) {
  static bool flag [[maybe_unused]] = []() {
    auto deivce_nums = device_count();
    default_generators.resize(deivce_nums);
    for (auto i = 0; i < deivce_nums; i++) {
      default_generators[i] = at::make_generator<McpuGeneratorImpl>(i);
      default_generators[i].seed();
    }
    return true;
  }();

  c10::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < device_count());
  }
  return default_generators[idx];
}
// LITERALINCLUDE MCPU GET DEFAULT GENERATOR IMPL

} // namespace c10::mcpu
