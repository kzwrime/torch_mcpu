# MCPU kernel launch migration guide

This document describes the migration rule for `torch_mcpu`.

For allocator lifetime and stream-reuse rules behind pointer-only kernels, see
[`mcpu_stream_memory_lifetime.md`](mcpu_stream_memory_lifetime.md).

## Contents

- [Runtime model](#runtime-model)
- [Operator structure](#operator-structure)
- [Migration checkpoints](#migration-checkpoints)
- [Launch macro](#launch-macro)
- [Pointer-capture pattern](#pointer-capture-pattern)
- [Tensor-capture pattern](#tensor-capture-pattern)
- [External torch extensions](#external-torch-extensions)
  - [`torch_xcpu` migration example](#torch_xcpu-migration-example)

## Runtime model

MCPU device memory is page protected by default. A tensor allocation is readable
and writable only while a kernel task is running on an MCPU stream. Direct CPU
access to MCPU storage is intentionally invalid because it bypasses the launch
queue.

Build-time mode:

- `-DTORCH_MCPU_ENABLE_MEMORY_PROTECTION=OFF`: device allocations use ordinary
  aligned allocation, `orMemoryProtect/Unprotect` are no-ops, and
  `KernelMemoryGuard` / `KernelPointerMemoryGuard` compile to empty RAII
  objects.

`MCPU_LAUNCH_TIMED_KERNEL` is the default launch interface for migrated kernel
work. It enqueues memory access on the current MCPU stream, reserves a timing
event, and opens a timing scope with the same name.

## Operator structure

Split every operator into metadata work and memory work.

- Metadata work stays outside the launch: input checks, dtype checks, dtype
  dispatch, output tensor allocation, resizing, sizes/strides, scalar argument
  normalization, and raw `data_ptr` extraction.
- Memory work goes inside the launch: dereferencing pointers, reads, writes,
  CPU fallback execution over CPU views, and copies.
- If Python needs a scalar or CPU tensor value immediately, synchronize the
  relevant stream before reading.

## Migration checkpoints

When migrating an operator or external extension to the asynchronous launch
path, check the following before considering the migration complete:

- Python-side tests that inspect results immediately must add the necessary
  `torch.accelerator.synchronize()` calls after launching MCPU work. The
  operator implementation should stay asynchronous; the test owns the
  synchronization needed for deterministic reads.
- For every launched function, inspect its callers up the call stack and verify
  that pointer-like or array-like arguments are not backed by stack-allocated
  storage that can go out of scope before the worker thread runs the task. This
  includes raw pointers, pointer arrays, temporary `std::array`/C arrays, and
  small metadata buffers passed through wrapper functions.
- Prefer pointer capture for hand-written kernels. Use tensor capture when the
  launched task must call ATen operators or helper APIs that need tensor views.
- Prefer `KernelPointerMemoryGuard` whenever the pointer set is known before
  launch. Use `KernelAllMemoryGuard` only when the kernel discovers indirect
  MCPU allocation addresses from metadata inside the launched task.
- If the captured launch arguments exceed the stream inline slot limit, put
  copied scalar/pointer metadata in a heap args struct and move-capture a
  `std::unique_ptr` to it. Borrowed tensor `data_ptr`s inside that struct must
  not be freed.

## Launch Macro

Use `MCPU_LAUNCH_TIMED_KERNEL` from `runtime/McpuKernelLaunch.h`:

```cpp
MCPU_LAUNCH_TIMED_KERNEL("mcpu::my_kernel", ([captures]), {
  // memory work
});
```

The single name is used both as the launch record name and as the timing scope
name. Wrap the capture list in parentheses, for example `([out_ptr, n])`, so
commas in the capture list are parsed as part of one macro argument. Wrap the
kernel body in `{ ... }` for readability.

For oversized argument lists, capture one owning args object instead of many
values: `([args = std::move(args)])`.

## Pointer-Capture Pattern

Use this by default for hand-written kernels, including kernels under
`csrc/aten/vllm_kernels`. Extract raw pointers and scalar launch parameters
before enqueueing, then capture only those values in the task.

```cpp
int64_t n = out.numel();
const float* in_ptr = input.data_ptr<float>();

VLLM_MCPU_DISPATCH_FLOAT(out, "my_pointer_kernel", {
  scalar_t* out_ptr = out.data_ptr<scalar_t>();
  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::my_pointer_kernel",
      ([out_ptr, in_ptr, n]),
      {
        at::mcpu::KernelPointerMemoryGuard guard({out_ptr, in_ptr});
        my_pointer_kernel<scalar_t>(out_ptr, in_ptr, n);
      });
});
```

The raw pointers are passed to the implementation just like CUDA kernels receive
device pointers. Dtype dispatch stays outside the launched task. The launched
task should not perform pointer lookup unless it must call an ATen operator.

If the full pointer set is not known until after reading MCPU metadata inside
the task, keep the pointer-capture shape but use `KernelAllMemoryGuard` instead
of `KernelPointerMemoryGuard`:

```cpp
MCPU_LAUNCH_TIMED_KERNEL("mcpu::my_indirect_kernel", ([metadata_ptr, n]), {
  at::mcpu::KernelAllMemoryGuard guard;
  my_indirect_kernel(metadata_ptr, n);
});
```

Prefer `KernelPointerMemoryGuard` when the pointer set is statically available.
`KernelAllMemoryGuard` is intended for kernels that need indirect pointers such
as address tables.

## Tensor-Capture Pattern

Use tensor capture when the task calls ATen operators or helpers that construct
CPU views from MCPU tensors. Capturing tensors keeps their storage alive and
`KernelMemoryGuard` unlocks the backing allocations for the duration of the
task.

```cpp
MCPU_LAUNCH_TIMED_KERNEL(
    "mcpu::aten::some_op.out",
    ([out, input, alpha]),
    {
      at::mcpu::KernelMemoryGuard guard(out, input);
      auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
      auto cpu_input = ops::get_cpu_tensor_view_if_needed(input);
      at::some_cpu_op_out(cpu_out, cpu_input, alpha);
    });
```

This pattern is appropriate for fallback-style ATen operators. If the kernel is
pure hand-written pointer code, prefer the pointer-capture pattern instead.

## External Torch Extensions

External projects registering MCPU kernels should follow the same rules:

- include `runtime/McpuKernelLaunch.h`;
- link against `libtorch_mcpu.so` so stream, timing, and memory-guard symbols
  resolve normally;
- extract pointers, shape, and stride metadata before enqueueing;
- enqueue memory work with `MCPU_LAUNCH_TIMED_KERNEL`;
- use pointer capture by default with `KernelPointerMemoryGuard`;
- use tensor capture only when the task must call ATen operators or tensor-view
  helpers;
- keep Python tests from reading protected storage directly. If a test needs
  results, synchronize explicitly and copy with `tensor.to("cpu")`.

### `torch_xcpu` Migration Example

`torch_xcpu` is an important external-project example for the `torch_mcpu`
launch and memory-protection model. Its custom ops are registered outside
`torch_mcpu`, but their kernels still run against MCPU allocations and must obey
the same launch boundary.

For hand-written pointer kernels in `torch_xcpu_impl`, use the pointer-capture
pattern shown above:

- compile with `TORCH_MCPU_ENABLE_MEMORY_PROTECTION=1` so
  `KernelPointerMemoryGuard` is active. External projects should prefer reading
  `torch_mcpu.get_compile_flags()` from the installed package and appending
  those flags to their extension build, so memory-protection and timing settings
  stay aligned with the installed runtime;
- pass only raw pointers and scalar metadata into the launched task;
- guard all dereferenced MCPU pointers with `KernelPointerMemoryGuard`;
- keep tensor capture for code paths that call ATen operators.

This keeps external kernel code close to the in-tree
`torch_mcpu/csrc/aten/vllm_kernels` style while avoiding direct CPU access to
MCPU storage outside the stream task.
