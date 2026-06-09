# MCPU stream memory lifetime and pointer-only kernels

This note explains how MCPU should reason about tensor lifetime when kernels
are launched asynchronously and task bodies only receive raw pointers and scalar
arguments.

## CUDA model

CUDA kernels receive raw pointers. They do not keep Python `Tensor` objects, C++
`Tensor` handles, or `Storage` objects alive. PyTorch makes this safe through
the CUDA caching allocator.

The important CUDA rules are:

- Each allocated block has an allocation stream.
- If a block is freed after use only on its allocation stream, it can return to
  the allocator cache immediately. Same-stream ordering prevents the block from
  being reused before earlier kernels on that stream finish.
- If a block is used on a different stream, that use must be recorded with
  `recordStream`. When the tensor is freed, the allocator records events on the
  recorded streams and delays block reuse until those events complete.
- `recordStream` is therefore a reuse-safety mechanism, not a tensor lifetime
  mechanism. Tensor objects may be destroyed; the allocator must keep the memory
  block unavailable for unsafe reuse until stream work is complete.

In upstream PyTorch CUDA this is implemented in
`../pytorch/c10/cuda/CUDACachingAllocator.cpp`:

- `recordStream(Block*, CUDAStream)` ignores same-stream use and records
  cross-stream use in `block->stream_uses`.
- `free(Block*)` inserts CUDA events instead of immediately returning blocks
  with pending stream uses to the free pool.
- `process_events()` moves a block back to the reusable pool only after all
  outstanding events have completed.

## MCPU model

MCPU follows the same ownership model in
`csrc/runtime/DeviceCachingAllocator.cpp`:

- `Block::stream` stores the allocation stream.
- `Block::stream_uses` stores extra streams that used the block.
- `NativeCachingAllocator::recordStream` maps a `DataPtr` to the owning block
  and records a `McpuStream`.
- `DeviceCachingAllocator::free` inserts `McpuEvent`s for blocks with pending
  `stream_uses`.
- `process_events()` returns a block to the reusable pool only after those
  events have completed.

This means a pointer-only MCPU kernel can use CUDA-style arguments:

```cpp
kernel(ptr0, ptr1, scalar0, scalar1);
```

but the same stream rule still applies. If all participating tensors are
allocated and used on the same MCPU stream, allocator ordering is enough. If a
tensor is used on a different stream from its allocation stream and may die
before that stream completes, the use must be recorded.

## Page protection is separate

`TORCH_MCPU_ENABLE_MEMORY_PROTECTION` controls page-level access protection for
MCPU device allocations. It catches direct CPU access outside launched kernel
tasks.

This is separate from stream lifetime:

- `KernelPointerMemoryGuard` unprotects the allocation backing a raw pointer
  while a launched task runs.
- It does not keep a `Tensor` object alive.
- It does not decide when an allocator block can be reused.
- With `TORCH_MCPU_ENABLE_MEMORY_PROTECTION=OFF`, it compiles to an empty
  object so disabled-protection benchmark paths do not pay a guard cost.

Pointer-only kernels therefore need both properties:

- page protection: unprotect raw pointer allocations inside the launched task;
- lifetime/reuse safety: rely on same-stream allocation/use or record
  cross-stream use before the tensor can be destroyed.

## Current project changes

The pointer-only kernel migration introduces `KernelPointerMemoryGuard` in
`include/runtime/McpuKernelLaunch.h`.

It accepts raw pointers:

```cpp
at::mcpu::KernelPointerMemoryGuard guard({out_ptr, input_ptr});
```

When page protection is enabled, each pointer is mapped to its owning OpenReg
allocation with `orPointerGetAttributes`, deduplicated, unprotected before the
kernel body, and protected again at task exit. When page protection is disabled,
the guard is an empty RAII object.

The vLLM-style kernels should prefer timed launch APIs:

```cpp
at::mcpu::launch_timed_kernel_on_stream(
    at::mcpu::detail::get_kernel_launch_stream(out),
    "mcpu::my_kernel",
    [out_ptr, input_ptr, n](at::mcpu::kernel_timing::Event* timing_event) {
      MCPU_KERNEL_TIMING_SCOPE_EVENT("mcpu::my_kernel", timing_event);
      at::mcpu::KernelPointerMemoryGuard guard({out_ptr, input_ptr});
      my_kernel(out_ptr, input_ptr, n);
    });
```

The task body captures pointers and scalar launch parameters, not tensors. Tensor
storage safety must be provided by same-stream ordering or allocator
`recordStream`, matching CUDA.

The benchmark-only pointer paths keep the disabled-protection hot path as direct
`orLaunchKernel(function, args...)` where the extra lambda would affect the
measurement.

## Validation in small tests

Build both configurations:

```bash
TORCH_MCPU_ENABLE_MEMORY_PROTECTION=OFF python -m pip install --no-build-isolation .
TORCH_MCPU_ENABLE_MEMORY_PROTECTION=ON python -m pip install --no-build-isolation .
```

Run pointer-only smoke tests:

```bash
python example/03_mlp_sigmoid/run_pointer_gap.py \
  --tasks 8 --warmup-tasks 2 --pre-layer-sleep-ms 1 \
  --mode all --kernel for-loop --work-items 8
```

Run the MLP stream timing smoke test:

```bash
python example/03_mlp_sigmoid/run.py \
  --warmup-iters 1 --profile-iters 2 \
  --pre-layer-sleep-ms 1 \
  --tokens 1 --model-dim 16 --hidden-dim 16 \
  --execution-mode preallocated
```

Expected results:

- `OFF` build: pointer benchmarks should preserve the direct low-overhead raw
  launch path.
- `ON` build: pointer kernels should not fault due to page protection.
- MLP correctness should pass `reference_check`.

## Validation in vLLM-scale workloads

For a real workload such as vLLM, verify three things separately.

### Correctness and synchronization

Run with page protection enabled first. It is slower, but it catches direct CPU
access outside launched tasks:

```bash
TORCH_MCPU_ENABLE_MEMORY_PROTECTION=ON python -m pip install --no-build-isolation .
```

Then run a representative vLLM decode/prefill workload and compare outputs
against CPU or a known-good MCPU build:

- fixed seed;
- fixed prompt batch;
- identical sampling parameters;
- compare selected logits, token ids, and final text.

Add explicit `torch.mcpu.synchronize()` at workload boundaries during
debugging. If adding synchronization changes correctness, the workload has a
missing stream dependency.

For cross-stream experiments, intentionally allocate tensors on stream A, use
them on stream B, destroy Python references before B synchronizes, and stress
the allocator with new allocations. Correct behavior requires recording the
cross-stream use. Without it, failures may show up only as silent corruption in
`OFF` mode.

### Reuse and memory pressure

Allocator reuse should be delayed only for blocks with outstanding cross-stream
events. To check this in a large workload:

- monitor `torch.mcpu.memory_allocated()` and `torch.mcpu.memory_reserved()` if
  available through the Python API;
- compare steady-state reserved memory between same-stream and cross-stream
  runs;
- call `torch.mcpu.empty_cache()` only at coarse boundaries, not inside the hot
  path, and observe whether reserved memory drops;
- track whether reserved memory grows without returning to a steady state after
  streams synchronize.

Healthy behavior:

- same-stream workloads should reuse blocks without event accumulation;
- cross-stream workloads may reserve more memory temporarily;
- after stream synchronization and allocator event processing, reusable memory
  should stabilize.

Warning signs:

- correctness changes when adding `synchronize()`;
- reserved memory grows monotonically with a fixed batch shape;
- many short-lived tensors on side streams require `recordStream` but are never
  recorded;
- page protection `ON` fails while `OFF` silently produces different outputs.

### Timing overhead

Use MCPU kernel timing to separate launch overhead from body time:

```python
torch.mcpu.set_kernel_timing_enabled(True)
# run workload
torch.mcpu.synchronize()
events = torch.mcpu.get_kernel_timing()
```

For pointer-only kernels, check that:

- task names appear from `launch_timed_kernel_on_stream`;
- body time does not include Python submission stalls;
- page protection `OFF` benchmark paths do not regress because of extra guard
  work;
- page protection `ON` overhead is visible only in guarded pointer access paths,
  not in disabled-protection builds.

## Developer checklist

When adding a pointer-only kernel:

1. Take `data_ptr` and scalar launch parameters before enqueueing.
2. Select the stream from an MCPU tensor.
3. Use `launch_timed_kernel_on_stream` as the default launch API.
4. Capture only pointers and scalar launch parameters in the task body.
5. Add `KernelPointerMemoryGuard` inside the task body for every MCPU pointer
   the kernel dereferences.
6. Preserve direct `orLaunchKernel(function, args...)` only for benchmark paths
   where disabled page protection must have no extra overhead.
7. If the kernel may use tensors on a stream other than their allocation stream,
   make sure that use is recorded through the allocator before those tensors can
   be destroyed.

