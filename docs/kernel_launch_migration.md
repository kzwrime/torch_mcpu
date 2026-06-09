# MCPU kernel launch migration guide

This document describes the migration rule for `torch_mcpu`.

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

## External torch extensions

Some operators are registered from external projects and are not located under
`torch_mcpu`. Those extensions should use the same rules:

- use MCPU stream launch for memory work;
- take `data_ptr` and compute launch parameters before enqueueing;
- keep tensors captured by the task so storage lifetime is correct;
- do not create CPU views and access MCPU storage outside a launched task.
