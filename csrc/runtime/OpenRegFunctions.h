#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

#include <include/Macros.h>

#include <limits>

namespace c10::mcpu {

MCPU_EXPORT DeviceIndex device_count() noexcept;
MCPU_EXPORT DeviceIndex current_device();
MCPU_EXPORT void set_device(DeviceIndex device);
MCPU_EXPORT DeviceIndex maybe_exchange_device(DeviceIndex to_device);

MCPU_EXPORT DeviceIndex ExchangeDevice(DeviceIndex device);

MCPU_EXPORT void getStreamPriorityRange(int* least_priority, int* greatest_priority);

static inline void check_device_index(int64_t device) {
  TORCH_CHECK(device >= 0 && device < c10::mcpu::device_count(),
              "The device index is out of range. It must be in [0, ",
              static_cast<int>(c10::mcpu::device_count()),
              "), but got ",
              static_cast<int>(device),
              ".");
}

} // namespace c10::mcpu
