// Ported from pytorch/c10/xpu/XPUCachingAllocator.h
// Changes from XPU source:
//  [1]  namespace c10::xpu::XPUCachingAllocator → c10::mcpu
//  [2]  XPUAllocator               → McpuDeviceAllocator
//  [3]  XPUStream                  → McpuStream
//  [4]  C10_XPU_API                → MCPU_EXPORT
//  [5]  #include <c10/xpu/XPUStream.h> → OpenRegStream.h
//  [6]  REMOVED: enablePeerAccess (no peer-access API in openreg)
//  [7]  REMOVED: createOrIncrefPool / beginAllocateToPool / endAllocateToPool /
//               releasePool / getPoolUseCount  (PrivatePool / graph-capture removed)
//  [8]  REMOVED: namespace c10::xpu MemPool class (PrivatePool removed)
#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/core/CachingDeviceAllocator.h>
#include "OpenRegStream.h"   // [5] provides McpuStream
#include <include/Macros.h>  // [4] provides MCPU_EXPORT

namespace c10::mcpu { // [1]

class McpuDeviceAllocator : public DeviceAllocator { // [2]
 public:
  virtual void init(c10::DeviceIndex device_count) = 0;
  virtual void* raw_alloc(size_t nbytes) = 0;
  virtual void raw_delete(void* ptr) = 0;
};

MCPU_EXPORT extern std::atomic<McpuDeviceAllocator*> allocator; // [2][4]

inline McpuDeviceAllocator* get() { // [2]
  return allocator.load();
}

inline void init(c10::DeviceIndex device_count) {
  get()->init(device_count);
}

// Non-inline so torch_bindings can link against them without openreg.h
MCPU_EXPORT void emptyCache(MempoolId_t mempool_id = {0, 0});
MCPU_EXPORT void resetPeakStats(DeviceIndex device);
MCPU_EXPORT void resetAccumulatedStats(DeviceIndex device);
MCPU_EXPORT c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
    DeviceIndex device);

inline void* raw_alloc(size_t size) {
  return get()->raw_alloc(size);
}

inline void raw_delete(void* ptr) {
  get()->raw_delete(ptr);
}

inline void recordStream(const DataPtr& dataPtr, McpuStream stream) { // [3]
  get()->recordStream(dataPtr, stream.unwrap());
}

// [6] REMOVED: enablePeerAccess — openreg has no peer-access API
MCPU_EXPORT double getMemoryFraction(DeviceIndex device); // [4]
MCPU_EXPORT void setMemoryFraction(double fraction, DeviceIndex device); // [4]

// [7] REMOVED: createOrIncrefPool, beginAllocateToPool, endAllocateToPool,
//              releasePool, getPoolUseCount  — PrivatePool / graph-capture removed

} // namespace c10::mcpu // [1]

// [8] REMOVED: namespace c10::xpu { struct MemPool { ... }; }
//     MemPool depended on PrivatePool infrastructure which has been removed.
