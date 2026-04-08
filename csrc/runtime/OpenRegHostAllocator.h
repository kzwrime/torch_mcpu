// Ported from pytorch/aten/src/ATen/xpu/CachingHostAllocator.h
// Changes from XPU source:
//  [1]  namespace at::xpu → c10::mcpu
//  [2]  XPUStream → McpuStream / XPUEvent → McpuEvent
//  [3]  Removed deprecated getCachingHostAllocator / CachingHostAllocator_recordEvent
//       / CachingHostAllocator_emptyCache / HostAlloc helper wrappers (YAGNI)
#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/Allocator.h>

#include "OpenRegEvent.h"
#include "OpenRegStream.h"

namespace c10::mcpu { // [1]

// [3] Deprecated helper wrappers omitted — use at::getHostAllocator(at::kPrivateUse1) directly.

} // namespace c10::mcpu
