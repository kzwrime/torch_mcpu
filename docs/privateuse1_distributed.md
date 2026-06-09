# Supporting `torch.distributed` on a `PrivateUse1` backend

`PrivateUse1` device integration and `torch.distributed` integration are two
separate layers. The in-tree example under
`test/cpp_extensions/open_registration_extension/torch_openreg` shows how to
rename `PrivateUse1`, register device hooks, streams, allocators, and kernels.
It does not register a c10d communication backend. To make tensors on a
`PrivateUse1` device work with `torch.distributed`, add a c10d `Backend` (or a
legacy `ProcessGroup`) implementation and register it with
`torch.distributed.Backend.register_backend`.

## How c10d routes `PrivateUse1` tensors

The relevant pieces in the current source tree are:

- `torch/distributed/distributed_c10d.py` owns Python backend registration.
  `Backend.register_backend(name, creator, devices=...)` records the creator,
  the backend name, and the device types it supports.
- `_new_process_group_helper()` builds a generic `ProcessGroup`, calls the
  registered creator for each device/backend pair, and attaches the returned
  backend with `pg._register_backend(torch.device(device), ..., backend)`.
- `torch/csrc/distributed/c10d/ProcessGroup.hpp` stores the mapping from
  `c10::DeviceType` to the concrete c10d backend. For `PrivateUse1`, this means
  the generic process group must have a backend registered for
  `c10::DeviceType::PrivateUse1`.
- `torch/csrc/distributed/c10d/Ops.cpp` already registers c10d dispatcher
  kernels for `PrivateUse1`:

  ```cpp
  REGISTER_C10D_OP1(FUNC, PrivateUse1)
  ```

  Those kernels call
  `process_group->getBackend(c10::DeviceType::PrivateUse1)->allreduce(...)`,
  `broadcast(...)`, and so on. No extra dispatcher registration is needed for
  the standard c10d collective ops unless the operation itself is new.

The main requirement is therefore: when the user initializes distributed with a
backend/device mapping that includes the renamed `PrivateUse1` device type, the
registered creator must return an object that implements the c10d collectives.

## Pick the user-visible device name

If the backend has not been renamed, the user-visible device type is
`privateuseone`. Most real backends call:

```python
torch.utils.rename_privateuse1_backend("foo")
torch.utils.generate_methods_for_privateuse1_backend()
```

After that, the user-visible device type is `foo`, and `torch.device("foo")`
maps to `c10::DeviceType::PrivateUse1`. Use that same string in
`Backend.register_backend(..., devices=[...])`.

For example, OpenReg renames the device to `openreg`, so a distributed backend
for it should register with `devices=["openreg"]`, not with
`devices=["privateuseone"]`.

## Register the c10d backend from Python

The minimal Python-side registration is:

```python
import torch
import torch.distributed as dist

from . import _C  # pybind extension exposing create_process_group_foo

device_type = torch._C._get_privateuse1_backend_name()

dist.Backend.register_backend(
    "foo",
    _C.create_process_group_foo,
    devices=[device_type],
)
```

Users can then choose the backend explicitly:

```python
dist.init_process_group(
    backend="foo",
    rank=rank,
    world_size=world_size,
    init_method="env://",
    device_id=torch.device(f"{device_type}:{local_rank}"),
)
```

or as an explicit multi-device mapping:

```python
dist.init_process_group(
    backend=f"cpu:gloo,{device_type}:foo",
    rank=rank,
    world_size=world_size,
)
```

If `devices=[device_type]` is provided, `Backend.register_backend` also fills
`Backend.default_device_backend_map` for that device type if no default exists.
That allows `init_process_group(device_id=torch.device(f"{device_type}:0"))` to
infer `"foo"`.

## Creator signatures

There are two supported creator signatures.

The simple API receives the store, rank, size, and timeout:

```python
def create_process_group_foo(store, rank, world_size, timeout):
    ...
```

Use this from C++ with a pybind function like:

```cpp
c10::intrusive_ptr<c10d::Backend> createProcessGroupFoo(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout);
```

The extended API receives a `DistributedBackendOptions` object plus an optional
backend-specific options object:

```python
dist.Backend.register_backend(
    "foo",
    _C.create_process_group_foo,
    extended_api=True,
    devices=[device_type],
)
```

The C++ creator can then consume fields such as `store`, `group_rank`,
`group_size`, `timeout`, `group_id`, and `global_ranks_in_group`.

Use the extended API if the backend needs group metadata or a custom options
object. The built-in fake backend in
`torch/testing/_internal/distributed/fake_pg.py` is a compact Python example.

## Implement a `c10d::Backend` subclass

For a new accelerator backend, prefer subclassing `c10d::Backend`. This fits the
current `ProcessGroup` design, where one generic `ProcessGroup` can route
different device types to different backend objects.

Skeleton:

```cpp
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d {

class WorkFoo final : public Work {
 public:
  WorkFoo(int rank, OpType opType) : Work(rank, opType) {}

  bool isCompleted() override {
    // Query the backend event/request.
    return true;
  }

  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    // Block until the operation completes or timeout expires.
    // Throw on communication errors.
    return true;
  }

  void synchronize() override {
    // If completion happens on a device stream, make the current stream wait.
  }

  void blockCurrentStream() override {
    // Optional but important for async collectives and graph/capture support.
  }
};

class ProcessGroupFoo final : public Backend {
 public:
  class Options final : public Backend::Options {
   public:
    explicit Options(std::chrono::milliseconds timeout = kBackendDefaultTimeout)
        : Backend::Options("foo", timeout) {}
  };

  ProcessGroupFoo(
      c10::intrusive_ptr<Store> store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options)
      : Backend(rank, size),
        store_(std::move(store)),
        options_(std::move(options)) {}

  const std::string getBackendName() const override {
    return "foo";
  }

  c10::intrusive_ptr<Backend::Options> getBackendOptions() override {
    return options_;
  }

  void setTimeout(std::chrono::milliseconds timeout) override {
    options_->timeout = timeout;
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    for (const auto& tensor : tensors) {
      TORCH_CHECK(
          tensor.device().is_privateuseone(),
          "ProcessGroupFoo expected PrivateUse1 tensors, got ",
          tensor.device());
    }

    // Enqueue the backend allreduce on the appropriate Foo device/stream.
    return c10::make_intrusive<WorkFoo>(getRank(), OpType::ALLREDUCE);
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override {
    return c10::make_intrusive<WorkFoo>(getRank(), OpType::BROADCAST);
  }

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override {
    return c10::make_intrusive<WorkFoo>(getRank(), OpType::BARRIER);
  }

 private:
  c10::intrusive_ptr<Store> store_;
  c10::intrusive_ptr<Options> options_;
};

c10::intrusive_ptr<Backend> createProcessGroupFoo(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout) {
  auto options = c10::make_intrusive<ProcessGroupFoo::Options>(
      std::chrono::duration_cast<std::chrono::milliseconds>(timeout));
  return c10::make_intrusive<ProcessGroupFoo>(store, rank, size, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create_process_group_foo", &createProcessGroupFoo);
}

} // namespace c10d
```

The minimal in-tree c10d extension example is
`test/cpp_extensions/cpp_c10d_extension.hpp`. It registers a backend named
`test`, but it uses the older style of returning a `ProcessGroup` subclass
directly. That still works because `_new_process_group_helper()` detects a
returned `ProcessGroup` and uses it as the process group, but a `Backend`
subclass is usually a better match for a `PrivateUse1` accelerator backend.

## What the backend must implement

Implement the collectives required by the features you want to support. The base
class in `torch/csrc/distributed/c10d/Backend.hpp` throws a clear error for
methods that are not overridden.

Common methods for DDP/FSDP training include:

- `allreduce`
- `allgather` and `_allgather_base`
- `reduce_scatter` and `_reduce_scatter_base`
- `broadcast`
- `barrier`

Also consider:

- `allreduce_coalesced` for DDP buckets and communication hooks
- `send`, `recv`, and `recvAnysource` if point-to-point APIs are needed
- `alltoall` and `alltoall_base` if tensor-parallel workloads need them
- `getFuture` or `getFutureResult` on `Work` if Python comm hooks or async
  chaining need futures
- `supportsCoalescing`, `startCoalescing`, and `endCoalescing` if the backend
  can batch operations
- `supportsSplitting`, `split`, and related methods if communicator split is
  needed for fast subgroup creation

## Stream and lifetime semantics

The c10d Python APIs are asynchronous when `async_op=True` and still rely on
correct device-side ordering when `async_op=False`. A `PrivateUse1` backend must
therefore define how collective completion interacts with the device runtime:

- retain input/output tensors or underlying storage until the communication
  request has completed;
- record or wait on the correct device streams/events before reading from input
  tensors and before exposing output tensors to later device work;
- implement `Work::wait()` to surface backend errors and respect timeouts;
- implement `Work::synchronize()` and preferably `Work::blockCurrentStream()`
  so polling users and graph/capture paths can correctly order later work;
- use `Store` only for rendezvous/control metadata, not for high-volume tensor
  payloads.

Look at `ProcessGroupNCCL` and `ProcessGroupGloo` for production patterns, but
keep dependencies out of generic c10d headers. The existing `torch_openreg`
stream/event code is useful for the device-runtime side; the c10d backend must
connect those runtime primitives to `Work`.

### A CUDA-like async model for `PrivateUse1`

The CUDA/NCCL backend is the best reference for an accelerator backend with
real device streams. The important pattern is:

1. User kernels normally run on the current user stream.
2. Communication is enqueued on an internal communication stream.
3. Before communication reads input tensors, the communication stream waits for
   the current user stream.
4. When communication is enqueued, an end event is recorded on the communication
   stream.
5. `Work::wait()` and `Work::synchronize()` make the current user stream wait on
   that end event, so later user work observes the collective result.
6. Tensor storage is kept alive until the communication stream is done using it.

In CUDA, `ProcessGroupNCCL.cpp` does step 3 with a helper equivalent to:

```cpp
ncclEvent.record(at::cuda::getCurrentCUDAStream(device.index()));
ncclEvent.block(ncclStream);
```

and implements `WorkNCCL::synchronize()` by making the current CUDA stream wait
on the NCCL end event:

```cpp
auto currentStream = at::cuda::getCurrentCUDAStream(device_.index());
ncclEndEvent_->block(currentStream);
```

`WorkNCCL::wait()` first calls `synchronize()`. It only blocks the CPU thread
for blocking waits, timeouts, barriers, or error handling. This distinction is
important: for normal async collectives, `wait()` establishes device-side
ordering without unnecessarily synchronizing the whole device.

For a `PrivateUse1` backend, implement the same state machine with the backend's
stream/event types. In OpenReg terms, the shape is:

```cpp
class WorkFoo final : public c10d::Work {
 public:
  WorkFoo(
      int rank,
      c10d::OpType op_type,
      c10::Device device,
      c10::openreg::OpenRegEvent end_event,
      std::vector<at::Tensor> held_tensors)
      : Work(rank, op_type),
        device_(device),
        end_event_(std::move(end_event)),
        held_tensors_(std::move(held_tensors)) {}

  bool isCompleted() override {
    return end_event_.query();
  }

  void synchronize() override {
    auto current = c10::openreg::getCurrentOpenRegStream(device_.index());
    end_event_.block(current);
    held_tensors_.clear();
  }

  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    synchronize();
    if (timeout != kNoTimeout) {
      // Poll isCompleted(), enforce timeout, and surface backend errors.
    }
    return true;
  }

  void blockCurrentStream() override {
    synchronize();
  }

 private:
  c10::Device device_;
  c10::openreg::OpenRegEvent end_event_;
  std::vector<at::Tensor> held_tensors_;
};
```

The collective method should synchronize the internal communication stream with
the caller's current stream before enqueueing communication:

```cpp
auto device = tensors[0].device();
auto user_stream = c10::openreg::getCurrentOpenRegStream(device.index());
auto comm_stream = getFooCommStream(device.index());

c10::openreg::OpenRegEvent ready(/*enable_timing=*/false);
ready.record(user_stream);
ready.block(comm_stream);

{
  // Set or pass comm_stream to the vendor communication library.
  // Enqueue allreduce/broadcast/etc. on comm_stream.
}

c10::openreg::OpenRegEvent done(/*enable_timing=*/false);
done.record(comm_stream);

std::vector<at::Tensor> held_tensors(tensors.begin(), tensors.end());
return c10::make_intrusive<WorkFoo>(
    getRank(),
    c10d::OpType::ALLREDUCE,
    device,
    std::move(done),
    std::move(held_tensors));
```

The exact class names above are OpenReg-specific. A production backend can have
its own `FooStream` and `FooEvent`, but it needs the same operations:

- get the current user stream for a device;
- get or create an internal communication stream, often from a stream pool;
- record an event on one stream;
- make another stream wait for that event;
- query and CPU-synchronize an event for timeout/error paths.

### `async_op=False` versus `async_op=True`

The backend should be explicit about the two modes:

- For `async_op=True`, enqueue communication on an internal communication stream
  and return a `Work` object immediately. The `Work` owns any events, backend
  request handles, and tensor references needed for correctness.
- For `async_op=False`, either enqueue on the current user stream or call
  `work->wait()` before returning. CUDA/NCCL often uses the current stream for
  non-async collectives so the allocator and stream ordering remain simple.

Even when Python does not request async execution, avoid a whole-device
synchronization unless the backend has no stream/event ordering primitive.
Whole-device synchronization is correct but usually too expensive for training.

### Tensor lifetime and allocator safety

Stream ordering is not enough by itself. The allocator also needs to know that
a tensor's storage is used by the communication stream. CUDA handles this with
`CUDACachingAllocator::recordStream(...)`, or by stashing tensor references on
the `Work` object until the communication stream is joined back to the user
stream.

For `PrivateUse1`, choose one of these approaches:

- implement the backend allocator's `recordStream(const DataPtr&, c10::Stream)`
  so storage cannot be reused while the communication stream may still access
  it;
- or keep strong `at::Tensor` references in the `Work` object and release them
  only after `Work::synchronize()` has made the user stream wait on the
  communication end event;
- or use both, matching CUDA/NCCL's more defensive behavior.

The OpenReg test allocator currently has a TODO in
`OpenRegDeviceAllocator::recordStream`; it does not track stream usage yet.
That is acceptable for a test backend, but a real distributed backend must solve
this before supporting asynchronous collectives safely.

### Events and errors

Use an event recorded after the vendor communication enqueue as the completion
source for `Work::isCompleted()`. Use the vendor communication request or
communicator status as the error source. `isCompleted()` should not report
success just because enqueue succeeded; it should indicate that the device-side
communication is complete or that a failure has been detected.

`Work::wait(timeout)` should:

- make the current user stream wait on the communication end event;
- if a timeout is requested, poll event/request completion and throw on timeout;
- check and rethrow backend communication errors on the caller's CPU thread;
- avoid destroying request/event/communicator state before all waiting streams
  have a valid dependency.

If the backend supports timing, record both start and end events on the
communication stream and implement `Work::getDuration()` using backend event
elapsed-time APIs. If not, leave timing unsupported and return the base-class
behavior.

## Testing strategy

Test the registration before testing real communication:

```python
import torch
import torch.distributed as dist
import foo_dist

device_type = torch._C._get_privateuse1_backend_name()

assert dist.is_backend_available("foo")
assert dist.get_default_backend_for_device(torch.device(device_type)) == "foo"
```

Then add focused multi-process tests for one collective at a time. Use the
current distributed test style instead of running the entire suite. Useful
existing tests and examples are:

- `test/distributed/test_c10d_common.py` for `Backend.register_backend`
  behavior and custom backend tests;
- `torch/testing/_internal/distributed/fake_pg.py` for an extended API
  registration example;
- `test/cpp_extensions/cpp_c10d_extension.hpp` for the smallest C++ c10d
  extension registration pattern.

For a real `PrivateUse1` backend, the first end-to-end test should initialize:

```python
dist.init_process_group(
    backend="foo",
    rank=rank,
    world_size=world_size,
    init_method=init_method,
    device_id=torch.device(f"{device_type}:{local_rank}"),
)
```

and run a single `all_reduce` on a tensor allocated on that device. Once that
passes, add tests for `broadcast`, `all_gather_into_tensor`,
`reduce_scatter_tensor`, and the DDP/FSDP path you intend to support.

## Summary

`PrivateUse1` support for `torch.distributed` does not live in the
`torch_openreg` device registration example because c10d has its own extension
point. The `PrivateUse1` c10d dispatcher entries already exist. A backend author
must provide the concrete communication implementation, expose a creator
function through the backend extension module, and call
`torch.distributed.Backend.register_backend` with the renamed `PrivateUse1`
device type in `devices=[...]`.
