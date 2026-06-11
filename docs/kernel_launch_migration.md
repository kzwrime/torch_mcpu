# MCPU kernel launch migration guide

This document describes the migration rule for `torch_mcpu`.

For allocator lifetime and stream-reuse rules behind pointer-only kernels, see
[`mcpu_stream_memory_lifetime.md`](mcpu_stream_memory_lifetime.md).

## Contents

- [Runtime model](#runtime-model)
- [Operator structure](#operator-structure)
- [Fallback-style ATen operators](#fallback-style-aten-operators)
- [Hand-written kernels](#hand-written-kernels)
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
  `KernelMemoryGuard` compiles to an empty RAII object.

`launch_kernel` always uses the stream worker queue. The old synchronous
compile-time mode has been removed so the launch path has one implementation.
There is intentionally no environment-variable fallback for memory protection;
it is a compile-time choice so the disabled path has no runtime branch cost.

## Operator structure

Split every operator into metadata work and memory work.

- Metadata work stays outside launch: input checks, dtype checks, dtype
  dispatch, output tensor allocation, resizing, sizes/strides, scalar argument
  normalization, and raw `data_ptr` extraction for hand-written kernels.
- Memory work goes inside launch: dereferencing pointers, reads, writes, CPU
  fallback execution over CPU views, and copies.
- If Python needs a scalar or CPU tensor value immediately, synchronize the
  relevant stream before reading.

## Fallback-style ATen operators

For fallback-style code under `torch_mcpu/csrc/aten/ops`, the temporary accepted
pattern is:

```cpp
at::mcpu::launch_kernel(out, [out, input]() mutable {
  at::mcpu::KernelMemoryGuard guard(out, input);
  auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
  auto cpu_input = ops::get_cpu_tensor_view_if_needed(input);
  at::some_cpu_op_out(cpu_out, cpu_input);
});
```

This still uses CPU kernels internally, but the actual memory access happens
from a launched stream task. Tests that inspect results immediately should
synchronize from Python after the operator call; the ATen operator should remain
asynchronous.

## Hand-written kernels

For hand-written kernels such as `torch_mcpu/csrc/aten/vllm_kernels`, follow the
CUDA shape: collect pointers and launch parameters before launch, and keep the
task body minimal.

```cpp
int64_t n = out.numel();
const float* in_ptr = input.data_ptr<float>();

VLLM_MCPU_DISPATCH_FLOAT(out, "my_kernel", {
  scalar_t* out_ptr = out.data_ptr<scalar_t>();
  at::mcpu::launch_kernel(out, [out, input, out_ptr, in_ptr, n]() mutable {
    at::mcpu::KernelMemoryGuard guard(out, input);
    my_kernel<scalar_t>(out_ptr, in_ptr, n);
  });
});
```

The lambda keeps tensors alive and unlocks their backing storage. The raw
pointers are passed to the kernel just like CUDA kernels receive device
pointers. Dtype dispatch is outside the launch task; the task should not contain
dispatch or pointer lookup unless a fallback CPU operator requires it.

For pointer-only kernels that intentionally pass only raw pointers and scalar
launch parameters into the task body, use `KernelPointerMemoryGuard` inside the
launched task. The task body should capture pointers and scalar launch
parameters rather than tensors:

```cpp
scalar_t* out_ptr = out.data_ptr<scalar_t>();
const float* scale_ptr = scale.data_ptr<float>();

at::mcpu::launch_timed_kernel(
    "mcpu::my_pointer_kernel",
    [out_ptr, scale_ptr, n](at::mcpu::kernel_timing::Event* timing_event) {
      MCPU_KERNEL_TIMING_SCOPE_EVENT("mcpu::my_pointer_kernel", timing_event);
      at::mcpu::KernelPointerMemoryGuard guard({out_ptr, scale_ptr});
      my_pointer_kernel(out_ptr, scale_ptr, n);
    });
```

When `TORCH_MCPU_ENABLE_MEMORY_PROTECTION=OFF`, `KernelPointerMemoryGuard`
compiles to an empty object. Hot benchmark paths should keep direct
`orLaunchKernel(function, args...)` calls behind the disabled-protection branch
if an extra lambda would affect the measurement.

For kernels with indirect memory access where the full pointer set is not known
until after reading MCPU metadata inside the task, use `KernelAllMemoryGuard`.
It snapshots the current MCPU allocator's active device allocations and
unprotects them for the duration of the launched task:

```cpp
at::mcpu::launch_timed_kernel(
    "mcpu::my_indirect_kernel",
    [metadata_ptr, n](at::mcpu::kernel_timing::Event* timing_event) {
      MCPU_KERNEL_TIMING_SCOPE_EVENT("mcpu::my_indirect_kernel", timing_event);
      at::mcpu::KernelAllMemoryGuard guard;
      my_indirect_kernel(metadata_ptr, n);
    });
```

Prefer `KernelPointerMemoryGuard` when the pointer set is statically available;
`KernelAllMemoryGuard` is intended for kernels that need indirect pointers such
as address tables. When `TORCH_MCPU_ENABLE_MEMORY_PROTECTION=OFF`,
`KernelAllMemoryGuard` compiles to an empty object with no container member and
no allocator scan.

## External torch extensions

Some operators are registered from external projects and are not located under
`torch_mcpu`. Those extensions should use the same rules:

- use MCPU stream launch for memory work;
- take `data_ptr` and compute launch parameters before enqueueing;
- keep tensors captured by the task so storage lifetime is correct;
- do not create CPU views and access MCPU storage outside a launched task.

### `torch_xcpu` migration example

`torch_xcpu` is an important external-project example for the new `torch_mcpu`
launch and memory-protection model. Its custom ops are registered outside
`torch_mcpu`, but their kernels still run against MCPU allocations and must obey
the same launch boundary.

For hand-written pointer kernels in `torch_xcpu_impl`, the intended pattern is
the pointer-only form shown above:

- compile the implementation with `TORCH_MCPU_ENABLE_MEMORY_PROTECTION=1` so
  `KernelPointerMemoryGuard` is active. External projects should prefer
  reading `torch_mcpu.get_compile_flags()` from the installed package and
  appending those flags to their extension build, so memory-protection and
  timing settings stay aligned with the installed runtime;
- include the installed `torch_mcpu` runtime headers needed for
  `runtime/McpuKernelLaunch.h`;
- link the extension against `libtorch_mcpu.so` so `launch_timed_kernel`,
  stream, timing, and memory-guard symbols resolve normally;
- extract pointers, shape, and stride metadata before enqueueing;
- enqueue memory work with `at::mcpu::launch_timed_kernel`;
- use `KernelPointerMemoryGuard` inside the launched task before dereferencing
  raw pointers;
- keep Python tests from reading protected storage directly. If a test needs
  results, synchronize explicitly and copy with `tensor.to("cpu")` rather than
  using a CPU view helper.

This keeps the external kernel code close to the in-tree
`torch_mcpu/csrc/aten/vllm_kernels` style while avoiding direct CPU access to
MCPU storage outside the stream task.
