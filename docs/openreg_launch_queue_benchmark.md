# OpenReg launch queue benchmark and optimization notes

This document summarizes the task-queue benchmark work for the MCPU/OpenReg
kernel launch path. The goal is to measure and reduce the overhead of submitting
many tiny kernels to one stream. The target hot path is one producer thread and
one worker thread per stream.

## Goal

The workload has many kernels launched sequentially. The benchmark intentionally
uses about 500 very small tasks so the queueing overhead is visible instead of
being hidden by kernel body work.

## Important measurement rules

- Memory protection is disabled for queue-overhead measurements:
  `-DTORCH_MCPU_ENABLE_MEMORY_PROTECTION=0`.
- Producer and worker must be pinned to different high-numbered cores.
  Pinning the whole process to only one CPU while asking the worker to use a
  different CPU is invalid because the process affinity mask can reject the
  worker affinity.
- Do not run multiple busy-spin benchmarks in parallel. The queue workers spin
  and will pollute each other's measurements.
- The benchmark commands below used CPU 76 for the producer and CPU 78 for the
  worker on an 80-CPU machine. Use topology-appropriate high-numbered cores on
  other machines.

Example:

```bash
./benchmarks/task_queue_dispatch/run_benchmarks.sh --build-only

taskset -c 76,78 env TORCH_MCPU_STREAM_WORKER_CORE=78 \
  benchmarks/task_queue_dispatch/build/openreg_queue_overhead \
  --tasks 500 --iterations 30000 --warmup 3000 --producer-core 76
```

## Benchmark files

- `benchmarks/task_queue_dispatch/kernel_dispatch_patterns.cpp`
  compares dispatch representations with kernel-like argument shapes:
  direct, `std::function`, tuple direct call, tuple with `std::invoke`, and a
  typed descriptor ring.
- `benchmarks/task_queue_dispatch/openreg_queue_overhead.cpp`
  links against the real OpenReg stream implementation and measures
  `orLaunchKernel` with both ordinary function arguments and lambda launches.
- `benchmarks/task_queue_dispatch/stream_progression_benchmark.cpp`
  starts from the tuple-with-invoke idea and progressively adds production-like
  costs: larger slots, fixed-capacity arrays, head-based synchronization,
  cached producer state, and an API wrapper.
- `benchmarks/task_queue_dispatch/run_benchmarks.sh`
  compiles all standalone benchmarks and supports `--run progression`.

## Baseline findings

The old OpenReg stream path was heavy:

```text
orLaunchKernel
  -> lambda/std::function wrapping
  -> openreg::addTaskToStream
  -> mutex
  -> std::queue<std::function<void()>>
  -> condition_variable notify/wake
  -> worker pops and invokes task
```

The high-cost parts are not tuple unpacking or `std::invoke`. They are dynamic
type erasure, mutex/condition-variable queueing, and extra stream layers on the
hot launch path.

## Best standalone dispatch shape

The best standalone design is an SPSC inline tuple ring:

- producer constructs the callable and arguments directly inside a ring slot;
- worker invokes the stored tuple with `std::invoke`;
- ordinary functions, lambdas, member function pointers, and member functions
  share one path;
- no `std::function`, no heap allocation, no mutex, no condition variable.

Representative fixed-core result:

```text
mode                 task_ns
direct                  ~4-8
tuple_direct_call      ~20-30
tuple_with_invoke      ~20-30
```

The `tuple_with_invoke` result is important because real call sites include
member function pointers, for example `memory.cpp` calls
`orLaunchKernel(stream, &MemoryManager::memcpy, &mm, ...)`. Direct
`callable(...)` is not general enough; tuple execution must use `std::invoke`.

## Progression benchmark

`stream_progression_benchmark.cpp` tests the hypothesis that production OpenReg
is slower because of slot size, more realistic API shape, and synchronization
strategy.

Recent serial fixed-core run:

```text
taskset -c 76,78 benchmarks/task_queue_dispatch/build/stream_progression_benchmark \
  --tasks 500 --iterations 30000 --warmup 3000 \
  --producer-core 76 --worker-core 78

mode                    batch_ns    task_ns    extra_ns
direct                    1626.5        3.3        0.0
vector_slot_128          15621.7       31.2       28.0
vector_slot_256          15542.4       31.1       27.8
fixed_head_256           14319.7       28.6       25.4
vector_cached_256        18017.4       36.0       32.8
fixed_cached_256         25584.0       51.2       47.9
fixed_cached_256_api     21155.3       42.3       39.1
```

Conclusions from this benchmark:

- Increasing slot storage from 128 bytes to 256 bytes is not consistently the
  main bottleneck. It is worth keeping 256 bytes for real capture/argument
  headroom.
- A fixed-capacity slot array is not inherently slower than a vector-backed
  benchmark ring.
- The cached-completed producer scheme is not always faster. On this workload,
  a simple head-based ring is faster.
- A thin API wrapper with null/status checks is not the dominant cost.
- The largest remaining production gap must be checked against the exact
  OpenReg hot path, not guessed from slot size alone.

## Current OpenReg design

The current OpenReg stream implementation has been simplified to:

```text
orLaunchKernel(stream, func, args...)
  -> stream->launch(func, args...)
  -> placement-new TupleTask<Func, Args...> into SPSC ring slot
  -> release-store tail

worker thread
  -> acquire-load tail
  -> run slot through std::invoke
  -> destroy tuple payload
  -> release-store head and completed
```

The code lives in:

- `third_party/openreg/include/openreg.inl`
  defines `QueueSlot`, `TupleTask`, `orStream`, and inline `orLaunchKernel`.
- `third_party/openreg/csrc/stream.cpp`
  owns worker lifecycle, high-core worker pinning, synchronization, and the
  remaining event/device cold paths.

The stream now assumes the hot case is one producer per stream. This matches the
current optimization target and avoids MPSC complexity on the launch path.

## Head-based ring decision

Two SPSC capacity schemes were tested:

1. Cached-completed producer state.
   The producer keeps a local tail and reloads `completed` only when it believes
   the ring is full.
2. Head-based ring.
   The producer reads the worker's `head` for capacity and the worker updates
   `head` after destroying the task.

The head-based scheme is currently preferred because it is faster in the real
OpenReg benchmark and simpler to reason about.

The worker updates `head` after `slot.destroy(...)`, not before running the task.
Updating before task execution can be faster in a microbenchmark, but it allows
the producer to reuse a slot before the worker has finished reading it when the
ring wraps. The current version keeps the safe ordering.

## Callable storage decision

`TupleTask` stores a tuple of the callable plus arguments and runs it with
`std::invoke`.

For ordinary function symbols, `StoredCallable` preserves a function reference
instead of immediately decaying to a function pointer. This gives the compiler a
better chance to keep a direct call shape for:

```cpp
orLaunchKernel(stream, lightweight_task, payload_ptr);
```

For lambdas and member function pointers, the code still uses decayed storage
and `std::invoke`, which is required for correctness.

## Current OpenReg benchmark result

Serial fixed-core run, memory protection disabled:

```text
taskset -c 76,78 env TORCH_MCPU_STREAM_WORKER_CORE=78 \
  benchmarks/task_queue_dispatch/build/openreg_queue_overhead \
  --tasks 500 --iterations 30000 --warmup 3000 --producer-core 76

mode        batch_ns    per_task_ns
direct        1224.6            2.4
func_args    27671.0           55.3
lambda       16341.2           32.7
```

The lambda path is the most relevant path for many migrated ATen/vLLM kernels
because `McpuKernelLaunch.h` wraps kernel bodies in a lambda that adds
`KernelTaskScope` and then calls `orLaunchKernel`.

Current interpretation:

- Lambda launch is close to the standalone `tuple_with_invoke` range.
- Function-argument launch remains slower, likely due to callable representation
  and invocation shape around ordinary function pointers/references.
- The previous cached-completed stream version was around 58-66 ns/task; the
  head-based stream is materially faster, especially for lambda launches.

## Validation

After the current OpenReg queue changes:

```text
cmake --build build --target ortests -j 8
43 passed

python -m pytest tests/test_streams.py -q
15 passed, 1 skipped
```

Earlier targeted stream/event tests also passed:

```text
build/third_party/openreg/ortests --gtest_filter='StreamTest.*:EventTest.*'
21 passed, 1 disabled
```

## Remaining questions

- Whether real ATen/vLLM launches are mostly lambda launches or direct
  function-argument launches after all migration work is complete.
- Whether `func_args` can be pushed closer to lambda performance with a
  specialized function-reference task representation.
- Whether stream worker spinning should be configurable for production,
  because it gives low latency but consumes a CPU.
- Whether a future MPSC design is needed. For now, the hot target is one
  producer per stream; adding MPSC should be measured separately and should not
  slow the SPSC path.
