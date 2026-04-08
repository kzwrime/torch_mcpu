// Ported from pytorch/aten/src/ATen/xpu/CachingHostAllocator.cpp
// Changes from XPU source:
//  [1]  namespace at::xpu → c10::mcpu (anonymous sub-namespace)
//  [2]  XPUStream / XPUEvent → McpuStream / McpuEvent
//  [3]  sycl::aligned_alloc_host → orMallocHost
//  [4]  sycl::free → orFreeHost
//  [5]  REGISTER_HOST_ALLOCATOR(at::kXPU, ...) → REGISTER_HOST_ALLOCATOR(at::kPrivateUse1, ...)
#include "OpenRegHostAllocator.h"
#include <include/openreg.h>

namespace c10::mcpu { // [1]
namespace {

constexpr size_t kHostAlignment = 512;

using Block = at::HostBlock<McpuStream>; // [2]

struct McpuCachingHostAllocatorImpl
    : public at::CachingHostAllocatorImpl<McpuStream, McpuEvent> { // [2]
  /* These following functions are runtime-related. */
  void allocate_host_memory(size_t size, void** ptr) override {
    MCPU_CHECK(orMallocHost(ptr, size)); // [3] orMallocHost replaces sycl::aligned_alloc_host
  }

  void free_block(Block* block) override {
    MCPU_CHECK(orFreeHost(block->ptr_)); // [4] orFreeHost replaces sycl::free
  }

  void record_stream(
      std::optional<std::vector<McpuEvent>>& events, // [2]
      McpuStream stream) override { // [2]
    McpuEvent event(/* enable_timing= */ false); // [2]
    event.record(stream);
    events->push_back(std::move(event));
  }

  bool query_event(McpuEvent& event) override { // [2]
    return event.query();
  }
};

DECLARE_HOST_ALLOCATOR(
    McpuCachingHostAllocator,
    McpuCachingHostAllocatorImpl,
    raw_local_deleter,
    caching_host_allocator)

REGISTER_HOST_ALLOCATOR(at::kPrivateUse1, &caching_host_allocator); // [5]

} // anonymous namespace
} // namespace c10::mcpu
