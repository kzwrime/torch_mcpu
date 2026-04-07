#include "OpenRegDeviceAllocator.h"
#include "OpenRegFunctions.h"

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

using namespace c10::CachingAllocator;

namespace c10::mcpu {

constexpr size_t kAggregate = static_cast<size_t>(StatType::AGGREGATE);


DeviceMemoryAllocator::DeviceMemoryAllocator() {}

void* DeviceMemoryAllocator::malloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  void* data = nullptr;
  auto ret = orMalloc(&data, nbytes);

  TORCH_CHECK(
      ret == orSuccess && data != nullptr,
      "Failed to allocate ",
      nbytes,
      " bytes on mcpu device. ",
      "Allocated: ",
      stats_.allocated_bytes[0].current,
      " bytes, ",
      "Reserved: ",
      stats_.reserved_bytes[0].current,
      " bytes");

  // Track allocation size for proper deallocation statistics
  allocation_sizes_[data] = nbytes;

  // Update statistics
  stats_.allocated_bytes[kAggregate].increase(nbytes);
  stats_.reserved_bytes[kAggregate].increase(nbytes);
  stats_.num_device_alloc++;

  return data;
}

void DeviceMemoryAllocator::free(void* ptr) {
  if (!ptr) {
    return;
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  auto ret = orFree(ptr);

  if (ret == orSuccess) {
    auto it = allocation_sizes_.find(ptr);
    if (it != allocation_sizes_.end()) {
      size_t nbytes = it->second;

      stats_.allocated_bytes[kAggregate].decrease(nbytes);
      stats_.reserved_bytes[kAggregate].decrease(nbytes);
      stats_.num_device_free++;

      allocation_sizes_.erase(it);
    } else {
      TORCH_WARN(
          "Successfully freed Mcpu memory pointer ",
          ptr,
          " that was not tracked by the allocator. "
          "Statistics may be inaccurate.");
    }
  } else {
    // orFree failed
    auto it = allocation_sizes_.find(ptr);
    if (it != allocation_sizes_.end()) {
      TORCH_WARN(
          "orFree failed for tracked pointer ",
          ptr,
          " with size ",
          it->second,
          " bytes. ",
          "Return code: ",
          ret,
          ". Keeping tracking record - this may indicate a double-free or invalid pointer.");
    } else {
      TORCH_WARN(
          "orFree failed for untracked pointer ",
          ptr,
          ". Return code: ",
          ret,
          ". This likely indicates a double-free or invalid pointer.");
    }
  }
}

c10::CachingDeviceAllocator::DeviceStats DeviceMemoryAllocator::getStats() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return stats_;
}

void DeviceMemoryAllocator::resetAccumulatedStats() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  // Reset accumulated statistics for all StatTypes
  for (const auto stat_type :
       c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
    stats_.allocated_bytes[stat_type].reset_accumulated();
    stats_.reserved_bytes[stat_type].reset_accumulated();
    stats_.active_bytes[stat_type].reset_accumulated();
    stats_.inactive_split_bytes[stat_type].reset_accumulated();
    stats_.requested_bytes[stat_type].reset_accumulated();
  }

  stats_.num_alloc_retries = 0;
  stats_.num_ooms = 0;
  stats_.num_sync_all_streams = 0;
  stats_.num_device_alloc = 0;
  stats_.num_device_free = 0;
}

void DeviceMemoryAllocator::resetPeakStats() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  // Reset peak statistics for all StatTypes
  for (const auto stat_type :
       c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
    stats_.allocated_bytes[stat_type].reset_peak();
    stats_.reserved_bytes[stat_type].reset_peak();
    stats_.active_bytes[stat_type].reset_peak();
    stats_.inactive_split_bytes[stat_type].reset_peak();
    stats_.requested_bytes[stat_type].reset_peak();
  }

  stats_.oversize_allocations.reset_peak();
  stats_.oversize_segments.reset_peak();
}

namespace {

McpuDeviceAllocator g_allocator;

void deleteMcpuMemory(void* ptr) {
  g_allocator.freeMemory(ptr);
}

}

McpuDeviceAllocator::McpuDeviceAllocator() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  // Single device allocator (no loop needed)
  device_allocator_ = std::make_unique<DeviceMemoryAllocator>();
}


at::DataPtr McpuDeviceAllocator::allocate(size_t nbytes) {
  // Always use device 0 for single-device configuration
  constexpr int device_index = 0;
  auto curr_device =
      c10::Device(c10::DeviceType::PrivateUse1, device_index);

  void* data = nullptr;
  if (nbytes > 0) {
    // Allocate memory via device allocator
    data = device_allocator_->malloc(nbytes);
  }

  return {data, data, &deleteMcpuMemory, curr_device};
}

at::DeleterFnPtr McpuDeviceAllocator::raw_deleter() const {
  return &deleteMcpuMemory;
}

void McpuDeviceAllocator::copy_data(
    void* dest,
    const void* src,
    std::size_t count) const {
  auto ret = orMemcpy(dest, src, count, orMemcpyDeviceToDevice);
  TORCH_CHECK(
      ret == orSuccess, "Failed to copy ", count, " bytes on mcpu device");
}

bool McpuDeviceAllocator::initialized() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return device_allocator_ != nullptr;
}

void McpuDeviceAllocator::freeMemory(void* ptr) {
  if (!ptr) {
    return;
  }

  // Directly free via device allocator
  // No need to track which device owns the pointer (single device)
  device_allocator_->free(ptr);
}

c10::CachingDeviceAllocator::DeviceStats McpuDeviceAllocator::
getDeviceStats(c10::DeviceIndex device) {
  // Ignore device index for single-device configuration
  return device_allocator_->getStats();
}

void McpuDeviceAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  // Ignore device index for single-device configuration
  device_allocator_->resetAccumulatedStats();
}

void McpuDeviceAllocator::resetPeakStats(c10::DeviceIndex device) {
  // Ignore device index for single-device configuration
  device_allocator_->resetPeakStats();
}

void McpuDeviceAllocator::emptyCache(MempoolId_t mempool_id) {
  // Mcpu doesn't implement caching yet
  // TODO: When caching is implemented, release all free blocks here
}

void McpuDeviceAllocator::recordStream(
    const DataPtr& ptr,
    c10::Stream stream) {
  // Mcpu doesn't track stream usage yet
  // TODO: When stream support is added, track which streams are using this pointer
}
// ============ Global Registration ============

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_allocator);

} // namespace c10::mcpu
