# Architecture

## Memory Model

The central architectural property of MCPU is **unified physical memory**: the host CPU and MCPU share the same DRAM. Any virtual address returned by a normal host `malloc` / `posix_memalign` is directly readable and writable by the MCPU without a DMA copy or page-pinning step.

This is analogous to CUDA pinned (page-locked) memory, but it applies to **all** allocations — there is no unpinned memory from MCPU's perspective.

### Consequences for the Backend

| Concern | CUDA backend | torch_mcpu |
|---|---|---|
| Device allocator | `cudaMalloc` (separate VRAM) | `c10::alloc_cpu` (shared DRAM) |
| Host→Device transfer | `cudaMemcpy` (PCIe DMA) | `memcpy` (cache-coherent DRAM) |
| Pinned memory | Special `cudaHostAlloc` path | All allocations are "pinned" |
| CPU kernel fallback | Expensive H2D+D2H copies | Near-zero overhead (see below) |

---

## Component Overview

```
Python layer
  torch_mcpu/__init__.py      rename_privateuse1_backend("mcpu")
  torch_mcpu/device.py        is_available / current_device / device_count

C++ layer (torch_mcpu._C)
  Allocator.cpp               MCPUAllocator  ──► c10::alloc_cpu / c10::free_cpu
  DeviceGuard.cpp             C10_REGISTER_GUARD_IMPL
  Hooks.cpp                   MCPUHooksInterface ──► RegisterPrivateUse1HooksInterface
  Ops.cpp                     TORCH_LIBRARY_IMPL(aten, PrivateUse1)
  Fallback.cpp                TORCH_LIBRARY_IMPL(_, PrivateUse1) → cpu_fallback
  Module.cpp                  PYBIND11_MODULE + static init
```

---

## Allocator Design

`MCPUAllocator` is a thin wrapper around the CPU allocator:

```cpp
DataPtr MCPUAllocator::allocate(size_t nbytes) {
    void* ptr = c10::alloc_cpu(nbytes);   // posix_memalign, CPU-accessible
    return {ptr, ptr, &Delete,
            at::Device(at::DeviceType::PrivateUse1, 0)};
}
```

The `DataPtr` is tagged as `PrivateUse1`, so PyTorch's dispatch system routes operations through the MCPU kernels. The underlying pointer is an ordinary host address, accessible from both CPU and MCPU.

In Phase 2, the allocator tracks live allocations in a thread-safe set, enabling `MCPUHooksInterface::isPinnedPtr()` to correctly identify MCPU-allocated memory.

---

## Operator Dispatch Flow

```
torch.add(mcpu_a, mcpu_b)
        │
        ▼ DispatchKey: PrivateUse1
  ┌─────────────────────────────────┐
  │  TORCH_LIBRARY_IMPL             │
  │  (aten, PrivateUse1)            │
  │                                 │
  │  add → not registered           │
  │        ↓                        │
  │  TORCH_LIBRARY_IMPL             │
  │  (_, PrivateUse1) fallback      │
  │        ↓                        │
  │  at::native::cpu_fallback():    │
  │    1. _to_cpu(mcpu_a)           │  ← Phase 2: zero-copy via from_blob
  │    2. _to_cpu(mcpu_b)           │  ← Phase 2: zero-copy via from_blob
  │    3. CPU add kernel            │
  │    4. output.to("mcpu")         │  ← _copy_from (memcpy, no DMA)
  └─────────────────────────────────┘
```

### Registered Core Operators (Phase 1)

These are always dispatched natively, bypassing the fallback:

| Operator | Notes |
|---|---|
| `aten::empty.memory_format` | Allocates via MCPUAllocator |
| `aten::empty_strided` | Allocates via MCPUAllocator |
| `aten::as_strided` | Zero-copy view, delegates to `at::cpu::as_strided` |
| `aten::view` | Zero-copy reshape, delegates to `at::native::view` |
| `aten::_copy_from` | In-place copy using `at::from_blob` + `at::native::copy_` |
| `aten::_copy_from_and_resize` | Resize then copy |
| `aten::_local_scalar_dense` | Read scalar value from MCPU tensor |
| `aten::set_.source_Storage` | Assign storage |
| `aten::set_.source_Storage_storage_offset` | Assign with offset/stride |
| `aten::set_.source_Tensor` | Share storage with another tensor |
| `aten::set_` | Reset to empty |
| `aten::resize_` | In-place resize |
| `aten::_reshape_alias` | Reshape alias |

### `_copy_from` Implementation

Because MCPU data pointers are valid CPU addresses, copies never require
a device-level transfer. The implementation reinterprets both source and
destination as CPU tensors via `at::from_blob`, then delegates to
`at::native::copy_`:

```cpp
// MCPU → CPU (or MCPU → MCPU): reinterpret as CPU, use native copy
auto src_cpu = at::from_blob(self.data_ptr(), self.sizes(), self.strides(),
                              self.options().device(at::kCPU));
at::native::copy_(dst_cpu, src_cpu, non_blocking);
```

No device handle, no DMA descriptor, no special allocator flag — just
a cache-coherent `memcpy` between two host virtual addresses.

---

## PrivateUse1 Hooks Interface

`MCPUHooksInterface` extends `at::PrivateUse1HooksInterface` and is registered
via `at::RegisterPrivateUse1HooksInterface`. It provides PyTorch's global
context with device metadata and the pinned memory allocator.

### Phase 1 (minimal)

```
isBuilt()              → true
isAvailable()          → true
deviceCount()          → 1
getCurrentDevice()     → 0
hasPrimaryContext()    → true
isPinnedPtr()          → false   (tracking not yet enabled)
getPinnedMemoryAllocator() → TORCH_CHECK_NOT_IMPLEMENTED
```

### Phase 2 (pinned memory)

```
isPinnedPtr(ptr)           → look up ptr in MCPUAllocator's live-set
getPinnedMemoryAllocator() → &g_mcpu_allocator
```

With Phase 2 in place:

- `tensor.is_pinned()` returns `true` for MCPU tensors
- `tensor.pin_memory()` early-exits (no copy) for MCPU tensors
- DataLoader `pin_memory=True` works efficiently with MCPU tensors
- CPU fallback's `_to_cpu()` step can be registered to return a zero-copy
  `from_blob` alias instead of copying data

---

## CPU Fallback: Zero-Copy Path (Phase 2)

The standard `at::native::cpu_fallback` converts inputs to CPU, runs the
op, and converts outputs back. With the shared-memory model, both directions
can be zero-copy:

**Input direction (MCPU → CPU):** Register `aten::_to_cpu` for `PrivateUse1`
to return `at::from_blob(data_ptr, ...)` — a CPU tensor aliasing the same
memory, with no data movement.

**Output direction (CPU → MCPU):** `_copy_from` already uses `from_blob` +
`at::native::copy_` internally, which reduces to a `memcpy`. For inference
workloads where outputs are never mutated in-place, this copy can be
eliminated by having `_copy_from` detect that src and dst are in the same
physical memory region and return early.

For large-model inference the dominant fallback ops (e.g., `aten::softmax`,
`aten::layer_norm`) are non-mutating, so the zero-copy input path alone
eliminates most data-movement overhead.

---

## Scope: Inference Only

This backend targets large-model inference. The following are explicitly
**out of scope**:

- Autograd (`AutogradPrivateUse1` dispatch key)
- `torch.autograd.Function` custom formulas
- `torch.library.register_autograd`
- `register_fake` / FakeTensor abstract kernels (only needed for `torch.compile` + training)

The CPU fallback handles all unimplemented operators correctly for forward
passes. Calling `.backward()` on an MCPU tensor will raise a `RuntimeError`.
