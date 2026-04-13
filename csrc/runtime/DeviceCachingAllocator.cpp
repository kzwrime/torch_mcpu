// Ported from pytorch/c10/xpu/XPUCachingAllocator.cpp
// Changes from XPU source:
//  [1]  sycl::queue*                 → orStream_t
//  [2]  McpuStream                   → McpuStream
//  [3]  sycl::aligned_alloc_device   → orMalloc
//  [4]  sycl::free                   → orFree
//  [5]  sycl::event barrier          → McpuEvent::record
//  [6]  event.wait()                 → McpuEvent::synchronize()
//  [7]  event query                  → McpuEvent::query()
//  [8]  xpu::syncStreamsOnDevice     → orDeviceSynchronize()
//  [9]  c10::DeviceType::PrivateUse1 → DeviceType::PrivateUse1
// [10]  device memory query          → DEVICE_TOTAL_MEM macro (32 GiB default)
// [11]  REMOVED: ExpandableSegment — openreg has no virtual memory API
// [12]  REMOVED: PrivatePool / graph-capture infrastructure
#include "DeviceCachingAllocator.h"
#include <c10/core/impl/GPUTrace.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <include/openreg.h>
#include "OpenRegEvent.h"
#include "OpenRegFunctions.h"
#include "OpenRegStream.h"

#include <cstring>
#include <deque>
#include <mutex>
#include <set>
#include <vector>

// [MCPU-10] No device memory query API in openreg. DEVICE_TOTAL_MEM is a
// compile-time constant for the simulated total device memory. Free memory is
// estimated as: DEVICE_TOTAL_MEM - stats.reserved_bytes[AGGREGATE].current
// Override at compile time with -DDEVICE_TOTAL_MEM=<bytes>.
#ifndef DEVICE_TOTAL_MEM
#define DEVICE_TOTAL_MEM (32ULL * 1024 * 1024 * 1024) // 32 GiB default
#endif

namespace c10::mcpu { // [1]

using namespace c10::CachingAllocator;
using namespace c10::CachingDeviceAllocator;

// newly allocated memory with 512-byte alignment.
constexpr size_t kDeviceAlignment = 512;

namespace {
using stream_set = ska::flat_hash_set<McpuStream>; // [2]

struct Block;
typedef bool (*Comparison)(const Block*, const Block*);
bool BlockComparatorSize(const Block* a, const Block* b);
bool BlockComparatorAddress(const Block* a, const Block* b);

// [MCPU-12-REMOVED] struct PrivatePool forward declaration (graph-capture
// removed)

struct BlockPool {
  // [MCPU-12] Removed PrivatePool* parameter from constructor
  // [MCPU-11] Removed unmapped set (was used by ExpandableSegment)
  // [MCPU-12] Removed owner_PrivatePool field (was used by graph capture)
  BlockPool(bool small) : blocks(BlockComparatorSize), is_small(small) {}

  std::set<Block*, Comparison> blocks;
  const bool is_small;
};

// [MCPU-11-REMOVED] struct ExpandableSegment forward declaration

struct Block {
  DeviceIndex device;
  orStream_t stream{nullptr}; // [1] orStream_t replaces sycl::queue*
  stream_set stream_uses; // streams on which the block was used
  size_t size; // block size in bytes
  size_t requested_size; // memory originally requested
  BlockPool* pool{nullptr}; // owning memory pool
  void* ptr{nullptr}; // memory address
  bool allocated{false}; // in-use flag
  // [MCPU-11-REMOVED] bool mapped — was used by ExpandableSegment
  Block* prev{nullptr}; // prev block if split from a larger allocation
  Block* next{nullptr}; // next block if split from a larger allocation
  int event_count{0}; // number of outstanding events
  // [MCPU-11-REMOVED] ExpandableSegment* expandable_segment — ExpandableSegment
  // removed

  Block(
      DeviceIndex device,
      orStream_t stream, // [1] orStream_t replaces sycl::queue*
      size_t size,
      BlockPool* pool,
      void* ptr)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr) {}

  // constructor for search key
  Block(DeviceIndex device, orStream_t stream, size_t size) // [1]
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        requested_size(0) {}

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }

  // Inserts this block between two existing blocks with [before, this, after].
  void splice(Block* before, Block* after) {
    if (before) {
      TORCH_INTERNAL_ASSERT(before->next == after);
      before->next = this;
    }
    prev = before;
    if (after) {
      TORCH_INTERNAL_ASSERT(after->prev == before);
      after->prev = this;
    }
    next = after;
  }
};

bool BlockComparatorSize(const Block* a, const Block* b) {
  if (a->stream != b->stream) {
    return reinterpret_cast<uintptr_t>(a->stream) <
        reinterpret_cast<uintptr_t>(b->stream);
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return reinterpret_cast<uintptr_t>(a->ptr) <
      reinterpret_cast<uintptr_t>(b->ptr);
}

bool BlockComparatorAddress(const Block* a, const Block* b) {
  if (a->stream != b->stream) {
    return reinterpret_cast<uintptr_t>(a->stream) <
        reinterpret_cast<uintptr_t>(b->stream);
  }
  return reinterpret_cast<uintptr_t>(a->ptr) <
      reinterpret_cast<uintptr_t>(b->ptr);
}

// [MCPU-11-REMOVED] struct SegmentRange — only used by ExpandableSegment
// [MCPU-11-REMOVED] struct ExpandableSegment — openreg has no virtual memory
// API
//   (XPU equivalent: sycl::ext::oneapi::experimental::reserve_virtual_mem /
//    physical_mem / map / unmap)
struct AllocParams {
  AllocParams(
      DeviceIndex device,
      size_t size,
      orStream_t stream, // [1]
      BlockPool* pool,
      size_t alloc_size)
      : search_key(device, stream, size), // [1]
        pool(pool),
        alloc_size(alloc_size),
        block(nullptr) {}

  DeviceIndex device() const {
    return search_key.device;
  }

  orStream_t stream() const { // [1] replaces sycl::queue* queue()
    return search_key.stream;
  }

  size_t size() const {
    return search_key.size;
  }

  Block search_key;
  BlockPool* pool;
  size_t alloc_size;
  Block* block;
  StatTypes stat_types = {};
};

// [MCPU-12-REMOVED] struct PrivatePool — graph-capture infrastructure removed
// [MCPU-12-REMOVED] struct MempoolIdHash — only used by PrivatePool maps

// [MCPU-3/4] allocPrimitive/deletePrimitive replaced by inline orMalloc/orFree
// in alloc_block/release_block.  The PrivatePool custom-allocator branch is
// also removed as PrivatePool itself is removed.

} // anonymous namespace

class DeviceCachingAllocator {
 private:
  mutable std::recursive_mutex mutex;
  DeviceStats stats;
  BlockPool large_blocks; // unallocated cached blocks larger than 1 MB
  BlockPool small_blocks; // unallocated cached blocks 1 MB or smaller
  ska::flat_hash_set<Block*> active_blocks; // allocated or in use by a stream
  // [MCPU-2/5] McpuEvent replaces sycl::event; renamed mcpu_events →
  // mcpu_events
  ska::flat_hash_map<McpuStream, std::deque<std::pair<McpuEvent, Block*>>>
      mcpu_events;
  DeviceIndex device_index;
  size_t allowed_memory_maximum = 0;
  bool set_fraction = false;
  // [MCPU-11-REMOVED] std::vector<ExpandableSegment*> expandable_segments
  // [MCPU-11-REMOVED] std::vector<c10::DeviceIndex> devices_with_peer_access
  // [MCPU-12-REMOVED] std::vector<...> captures_underway
  // [MCPU-12-REMOVED] ska::flat_hash_map<...> graph_pools
  // [MCPU-12-REMOVED] ska::flat_hash_map<...> graph_pools_freeable

  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    // [MCPU-11] Removed: dst->mapped != src->mapped guard (ExpandableSegment)
    if (!src || src->allocated || src->event_count > 0 ||
        !src->stream_uses.empty()) {
      return 0;
    }

    TORCH_INTERNAL_ASSERT(dst->is_split() && src->is_split());
    if (dst->prev == src) { // [src dst]
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else { // [dst src]
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }
    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    // [MCPU-11] Removed mapped/unmapped split; only pool.blocks exists now
    auto erased = pool.blocks.erase(src);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    delete src;

    return subsumed_size;
  }

  void free_block(Block* block) {
    TORCH_INTERNAL_ASSERT(
        !block->allocated && block->event_count == 0 &&
        block->stream_uses.empty());

    size_t original_block_size = block->size;
    size_t requested_size = block->requested_size;
    auto& pool = *block->pool;
    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      try_merge_blocks(block, merge_candidate, pool);
    }

    active_blocks.erase(block);
    bool inserted = pool.blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

    StatTypes stat_types = get_stat_types_for_pool(pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.active_bytes[stat_type].decrease(original_block_size);
      stats.requested_bytes[stat_type].decrease(requested_size);
    });
  }

  // [MCPU-7] Replaced sycl event query with McpuEvent::query()
  void process_events() {
    for (auto it = mcpu_events.begin(); it != mcpu_events.end();) {
      while (!it->second.empty()) {
        auto& e = it->second.front();
        auto& event = e.first;
        auto* block = e.second;
        if (!event.query()) { // [7] McpuEvent::query() replaces sycl event
                              // status
          break;
        }
        block->event_count--;
        if (block->event_count == 0) {
          free_block(block);
        }
        it->second.pop_front();
      }

      if (it->second.empty()) {
        it = mcpu_events.erase(it);
      } else {
        it++;
      }
    }
  }

  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  // [MCPU-12] Removed captures_underway graph-capture routing
  // [MCPU-1]  Removed sycl::queue* parameter (not needed without graph capture)
  BlockPool& get_pool(size_t size) {
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  // [MCPU-11-REMOVED] find_expandable_block — ExpandableSegment removed
  // [MCPU-11-REMOVED] map_block           — ExpandableSegment removed
  // [MCPU-11-REMOVED] try_allocate_expandable_block — ExpandableSegment removed

  // [MCPU-11] Removed expandable_segment best-fit search branch
  bool get_free_block(AllocParams& p) {
    BlockPool& pool = *p.pool;
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() ||
        (*it)->stream != p.stream()) { // [1] p.stream()
      return false;
    }
    p.block = *it;
    pool.blocks.erase(it);
    return true;
  }

  // [MCPU-11] Removed expandable_segments branch
  // [MCPU-12] Removed owner_PrivatePool allocation_count tracking
  bool alloc_block(AllocParams& p, bool isRetry) {
    auto size = p.alloc_size;
    void* ptr = nullptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }
    if (set_fraction &&
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current +
                size >
            allowed_memory_maximum) {
      return false;
    }
    MCPU_CHECK(orMalloc(&ptr, size)); // [1] orMalloc replaces allocPrimitive
    if (!ptr) {
      return false;
    }
    p.block =
        new Block(p.device(), p.stream(), size, p.pool, ptr); // [1] p.stream()
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      stats.reserved_bytes[stat_type].increase(size);
    });
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    return true;
  }

  // [MCPU-12] Removed PrivatePool* pool filter parameter
  void synchronize_and_free_events() {
    for (auto& xe : mcpu_events) {
      for (auto& e : xe.second) {
        auto& event = e.first;
        auto* block = e.second;
        event.synchronize(); // [1] event.wait() → event.synchronize()
        block->event_count--;
        if (block->event_count == 0) {
          free_block(block);
        }
      }
    }
    mcpu_events.clear();
  }

  // [MCPU-11-REMOVED] release_expandable_segment — ExpandableSegment removed

  // [MCPU-12] Removed owner_PrivatePool allocation_count tracking
  void release_block(Block* block) {
    /*
     * Note [Safe to Free Blocks on BlockPool]
     *
     * Callers must ensure that all accesses to the block have been completed
     * before invoking orFree.  We do a device-level synchronization before
     * freeing these blocks to guarantee that all kernels have finished.
     */
    auto* pool = block->pool;
    MCPU_CHECK(orFree(block->ptr)); // [1] orFree replaces deletePrimitive
    pool->blocks.erase(block);

    StatTypes stat_types = get_stat_types_for_pool(*pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.reserved_bytes[stat_type].decrease(block->size);
    });

    delete block;
  }

  // [MCPU-11-REMOVED] unmap_block — ExpandableSegment removed

  // [MCPU-11] Removed to_unmap/expandable_segment branch
  void release_blocks(BlockPool& pool) {
    // Frees all non-split blocks in the given pool.
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (!block->prev && !block->next) {
        release_block(block);
      }
    }
  }

  // [MCPU-12] Removed captures_underway/graph_pools_freeable PrivatePool paths
  bool release_cached_blocks(MempoolId_t mempool_id) {
    if (mempool_id.first == 0 && mempool_id.second == 0) {
      synchronize_and_free_events();
      // See Note [Safe to Free Blocks on BlockPool]
      MCPU_CHECK(orDeviceSynchronize()); // [1] orDeviceSynchronize replaces
                                         // c10::xpu::syncStreamsOnDevice
      release_blocks(large_blocks);
      release_blocks(small_blocks);
    }
    return true;
  }

  // [MCPU-11] Removed use_expandable_segments() branch
  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small) {
      return remaining >= kMinBlockSize;
    } else {
      return remaining > kSmallSize;
    }
  }

  StatTypes get_stat_types_for_pool(const BlockPool& pool) {
    StatTypes stat_types = {};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(
        pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
    return stat_types;
  }

  Block* alloc_found_block(
      AllocParams params,
      size_t orig_size,
      bool split_remainder) {
    auto size = params.size();
    auto device = params.device();
    BlockPool* pool = params.pool;
    orStream_t stream =
        params.stream(); // [1] params.stream() replaces params.queue()

    TORCH_INTERNAL_ASSERT(
        params.block != nullptr && params.block->ptr != nullptr);
    Block* block = params.block;
    Block* remaining = nullptr;

    if (split_remainder) {
      remaining = block;

      block = new Block(device, stream, size, pool, block->ptr); // [1]
      // [MCPU-11] Removed: block->expandable_segment =
      // remaining->expandable_segment
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      bool inserted = pool->blocks.insert(remaining).second;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);
    }

    block->allocated = true;
    block->requested_size = orig_size;
    bool inserted = active_blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted)

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      stats.allocated_bytes[stat_type].increase(block->size);
      stats.active_bytes[stat_type].increase(block->size);
      stats.requested_bytes[stat_type].increase(block->requested_size);
    });

    c10::reportMemoryUsageToProfiler(
        block->ptr,
        static_cast<int64_t>(block->size),
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::PrivateUse1, device));

    return block;
  }

  // [MCPU-8] McpuEvent::record replaces sycl submit_barrier
  void insert_events(Block* block) {
    stream_set streams(std::move(block->stream_uses));
    TORCH_INTERNAL_ASSERT(block->stream_uses.empty());
    for (auto& stream : streams) {
      block->event_count++;
      McpuEvent event(/* enable_timing= */ false);
      event.record(stream); // [8] record on stream
      mcpu_events[stream].emplace_back(std::move(event), block);
    }
  }

  // [MCPU-12-REMOVED] create_or_incref_pool — PrivatePool removed
  // [MCPU-12-REMOVED] get_private_pool       — PrivatePool removed

 public:
  DeviceCachingAllocator(DeviceIndex device_index)
      : large_blocks(/* small */ false),
        small_blocks(/* small */ true),
        device_index(device_index) {}

  // [MCPU-1] sycl::queue& → McpuStream; [MCPU-12] removed captures_underway
  // check
  Block* malloc(DeviceIndex device, size_t orig_size, McpuStream stream) {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    process_events(); // [MCPU-12] removed captures_underway.empty() guard
    size_t size = round_size(orig_size);
    auto& pool = get_pool(size); // [MCPU-12] removed queue param
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream.stream(), &pool, alloc_size); // [1]
    params.stat_types = get_stat_types_for_pool(pool);

    // First, try to get a block from the existing pool.
    bool block_found = get_free_block(params);
    // Can't reuse an existing block, try to get a new one.
    if (!block_found) {
      block_found = alloc_block(params, false) ||
          (release_cached_blocks({0, 0}) && alloc_block(params, true));
    }
    if (!block_found) {
      // [MCPU-9] DEVICE_TOTAL_MEM replaces sycl device memory query API
      const size_t device_total = DEVICE_TOTAL_MEM;
      const size_t device_free = device_total -
          stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      std::string allowed_info;
      if (set_fraction) {
        allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
      }

      auto allocated_bytes =
          stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto reserved_bytes =
          stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;

      c10::reportOutOfMemoryToProfiler(
          static_cast<int64_t>(size),
          allocated_bytes,
          reserved_bytes,
          c10::Device(c10::DeviceType::PrivateUse1, device));

      TORCH_CHECK_WITH(
          OutOfMemoryError,
          false,
          "MCPU out of memory. Tried to allocate ", // [1] XPU → MCPU
          format_size(alloc_size),
          ". Device ",
          static_cast<int>(device),
          " has a total capacity of ",
          format_size(device_total),
          " of which ",
          format_size(device_free),
          " is free. ",
          allowed_info,
          "Of the allocated memory ",
          format_size(allocated_bytes),
          " is allocated by PyTorch, and ",
          format_size(reserved_bytes - allocated_bytes),
          " is reserved by PyTorch but unallocated.",
          " Please use `empty_cache` to release all unoccupied cached memory.");
    }
    bool split_remainder = should_split(params.block, params.size());
    return alloc_found_block(std::move(params), orig_size, split_remainder);
  }

  void free(Block* block) {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    block->allocated = false;

    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.allocated_bytes[stat_type].decrease(block->size);
    });

    if (!block->stream_uses.empty()) {
      insert_events(block);
    } else {
      free_block(block);
    }

    c10::reportMemoryUsageToProfiler(
        orig_block_ptr,
        -static_cast<int64_t>(orig_block_size),
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::PrivateUse1, block->device));
  }

  void recordStream(Block* block, McpuStream stream) {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    if (stream.stream() ==
        block->stream) { // [1] stream.stream() replaces stream.queue()
      return;
    }
    block->stream_uses.insert(stream);
  }

  void emptyCache(MempoolId_t mempool_id) {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    release_cached_blocks(mempool_id);
  }

  DeviceStats getStats() {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    return stats;
  }

  void resetAccumulatedStats() {
    std::scoped_lock<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      stats.allocated_bytes[statType].reset_accumulated();
      stats.reserved_bytes[statType].reset_accumulated();
      stats.active_bytes[statType].reset_accumulated();
      stats.requested_bytes[statType].reset_accumulated();
    }
    stats.num_alloc_retries = 0;
  }

  void resetPeakStats() {
    std::scoped_lock<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      stats.allocated_bytes[statType].reset_peak();
      stats.reserved_bytes[statType].reset_peak();
      stats.active_bytes[statType].reset_peak();
      stats.requested_bytes[statType].reset_peak();
    }
  }

  // [MCPU-9] DEVICE_TOTAL_MEM replaces sycl device memory query API
  std::pair<size_t, size_t> getMemoryInfo() {
    const size_t total = DEVICE_TOTAL_MEM;
    const size_t free = total -
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current;
    return {free, total};
  }

  double getMemoryFraction() {
    if (!set_fraction) {
      return 1.0;
    }
    return static_cast<double>(allowed_memory_maximum) /
        static_cast<double>(DEVICE_TOTAL_MEM);
  }

  void setMemoryFraction(double fraction) {
    allowed_memory_maximum =
        static_cast<size_t>(fraction * static_cast<double>(DEVICE_TOTAL_MEM));
    set_fraction = true;
  }

  // [MCPU-12-REMOVED] createOrIncrefPool — PrivatePool removed
  // [MCPU-12-REMOVED] getPoolUseCount    — PrivatePool removed
  // [MCPU-12-REMOVED] beginAllocateToPool — PrivatePool/captures_underway
  // removed [MCPU-12-REMOVED] endAllocateToPool   —
  // PrivatePool/captures_underway removed [MCPU-12-REMOVED] releasePool —
  // PrivatePool removed
};

static void local_raw_delete(void* ptr);

class NativeCachingAllocator : public McpuDeviceAllocator {
 private:
  alignas(hardware_destructive_interference_size) std::mutex mutex;
  ska::flat_hash_map<void*, Block*> allocated_blocks;

  void add_allocated_block(Block* block) {
    std::lock_guard<std::mutex> lock(mutex);
    allocated_blocks[block->ptr] = block;
  }

  Block* get_allocated_block(void* ptr, bool remove = false) {
    std::scoped_lock<std::mutex> lock(mutex);
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    Block* block = it->second;
    if (remove) {
      allocated_blocks.erase(it);
    }
    return block;
  }

  void assertValidDevice(DeviceIndex device) {
    const auto device_num = device_allocators.size();
    TORCH_CHECK(
        0 <= device && device < static_cast<int64_t>(device_num),
        "Invalid device argument ",
        device,
        ": did you call init?");
  }

 public:
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocators;

  void init(DeviceIndex device_count) override {
    const auto size = static_cast<DeviceIndex>(device_allocators.size());
    if (size < device_count) {
      device_allocators.resize(device_count);
      for (const auto i : c10::irange(size, device_count)) {
        device_allocators[i] = std::make_unique<DeviceCachingAllocator>(i);
      }
    }
  }

  bool initialized() override {
    return !device_allocators.empty();
  }

  // [MCPU-1] sycl::queue& → McpuStream
  void malloc(
      void** devPtr,
      DeviceIndex device,
      size_t size,
      McpuStream stream) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocators.size(),
        "Allocator not initialized for device ",
        static_cast<int16_t>(device),
        ": did you call init?");
    Block* block =
        device_allocators[device]->malloc(device, size, stream); // [1]
    add_allocated_block(block);
    *devPtr = block->ptr;
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_allocation(
          c10::DeviceType::PrivateUse1, reinterpret_cast<uintptr_t>(*devPtr));
    }
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Block* block = get_allocated_block(ptr, /* remove */ true);
    TORCH_CHECK(block, "invalid device pointer: ", ptr);
    device_allocators[block->device]->free(block);
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_deallocation(
          c10::DeviceType::PrivateUse1,
          reinterpret_cast<uintptr_t>(block->ptr));
    }
  }

  void emptyCache(MempoolId_t mempool_id) override {
    for (auto& da : device_allocators) {
      da->emptyCache(mempool_id);
    }
  }

  void recordStream(const DataPtr& ptr, c10::Stream stream) override {
    if (!ptr.get()) {
      return;
    }
    if (ptr.get_deleter() != &local_raw_delete) {
      return;
    }

    Block* block = get_allocated_block(ptr.get());
    TORCH_CHECK(block, "No allocated block can be found.");
    McpuStream mcpu_stream{
        stream}; // [1] c10::McpuStream → McpuStream (in c10::mcpu ns)
    device_allocators[block->device]->recordStream(block, mcpu_stream);
  }

  DataPtr allocate(size_t size) override {
    auto device = c10::mcpu::current_device();
    void* r = nullptr;
    if (size != 0) {
      this->malloc(&r, device, size, getCurrentMcpuStream(device));
    }
    return {
        r, r, &local_raw_delete, Device(c10::DeviceType::PrivateUse1, device)};
  }

  DeleterFnPtr raw_deleter() const override {
    return &local_raw_delete;
  }

  void* raw_alloc(size_t size) override {
    if (size == 0) {
      return nullptr;
    }
    auto device = c10::mcpu::current_device();
    void* r = nullptr;
    malloc(&r, device, size, getCurrentMcpuStream(device));
    return r;
  }

  // [MCPU-1-REMOVED] raw_alloc_with_stream(XPUStream) — XPUStream type removed

  void raw_delete(void* ptr) override {
    this->free(ptr);
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    std::memcpy(dest, src, count); // [1] SYCL queue memcpy → std::memcpy
  }

  DeviceStats getDeviceStats(DeviceIndex device) override {
    assertValidDevice(device);
    return device_allocators[device]->getStats();
  }

  void resetPeakStats(DeviceIndex device) override {
    assertValidDevice(device);
    device_allocators[device]->resetPeakStats();
  }

  void resetAccumulatedStats(DeviceIndex device) override {
    assertValidDevice(device);
    device_allocators[device]->resetAccumulatedStats();
  }

  // [MCPU-6-REMOVED] enablePeerAccess — no peer-access API in openreg

  std::pair<size_t, size_t> getMemoryInfo(DeviceIndex device) override {
    assertValidDevice(device);
    return device_allocators[device]->getMemoryInfo();
  }

  double getMemoryFraction(DeviceIndex device) {
    assertValidDevice(device);
    return device_allocators[device]->getMemoryFraction();
  }

  void setMemoryFraction(double fraction, DeviceIndex device) {
    assertValidDevice(device);
    TORCH_CHECK_VALUE(
        0 < fraction && fraction <= 1,
        "invalid fraction:",
        fraction,
        ". Please set within (0, 1].");
    device_allocators[device]->setMemoryFraction(fraction);
  }

  // [MCPU-12-REMOVED] createOrIncrefPool — PrivatePool removed
  // [MCPU-12-REMOVED] beginAllocateToPool — PrivatePool/captures_underway
  // removed [MCPU-12-REMOVED] endAllocateToPool   — PrivatePool removed
  // [MCPU-12-REMOVED] releasePool         — PrivatePool removed
  // [MCPU-12-REMOVED] getPoolUseCount     — PrivatePool removed
};

static NativeCachingAllocator native_allocator;

void local_raw_delete(void* ptr) {
  native_allocator.free(ptr);
}

// [MCPU-10] REGISTER_ALLOCATOR macro replaces manual
// NativeAllocatorStaticInitializer
std::atomic<McpuDeviceAllocator*> allocator;

struct NativeAllocatorStaticInitializer {
  NativeAllocatorStaticInitializer() {
    allocator.store(&native_allocator);
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &native_allocator, 0);
  }
};

static NativeAllocatorStaticInitializer native_allocator_static_initializer;

// [MCPU-6-REMOVED] enablePeerAccess — no peer-access API in openreg

// Definitions of MCPU_EXPORT functions declared in DeviceCachingAllocator.h.
// Non-inline so torch_bindings can link without pulling in openreg.h.
void emptyCache(MempoolId_t mempool_id) {
  native_allocator.emptyCache(mempool_id);
}

DeviceStats getDeviceStats(DeviceIndex device) {
  return native_allocator.getDeviceStats(device);
}

void resetPeakStats(DeviceIndex device) {
  native_allocator.resetPeakStats(device);
}

void resetAccumulatedStats(DeviceIndex device) {
  native_allocator.resetAccumulatedStats(device);
}

double getMemoryFraction(DeviceIndex device) {
  return native_allocator.getMemoryFraction(device);
}

void setMemoryFraction(double fraction, DeviceIndex device) {
  return native_allocator.setMemoryFraction(fraction, device);
}

// [MCPU-12-REMOVED] createOrIncrefPool — PrivatePool removed
// [MCPU-12-REMOVED] beginAllocateToPool — PrivatePool/captures_underway removed
// [MCPU-12-REMOVED] endAllocateToPool   — PrivatePool removed
// [MCPU-12-REMOVED] releasePool         — PrivatePool removed
// [MCPU-12-REMOVED] getPoolUseCount     — PrivatePool removed

} // namespace c10::mcpu

// [MCPU-12-REMOVED] namespace c10::xpu MemPool implementation — PrivatePool
// removed
