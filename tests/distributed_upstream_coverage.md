# MCPU distributed upstream coverage

This backend is a real `ProcessGroup` `PrivateUse1` backend, so validation should
reuse PyTorch c10d tests instead of replacing them with local-only smoke tests.

## Primary upstream files

- `../pytorch/test/distributed/test_c10d_common.py`
  Shared multiprocess c10d fixtures and common ProcessGroup collective test logic.
- `../pytorch/test/distributed/test_c10d_gloo.py`
  Full CPU/Gloo collective, point-to-point, subgroup, timeout, and object collective coverage.
- `../pytorch/test/distributed/test_c10d_nccl.py`
  Dispatcher-based collective coverage and stream-semantics tests. MCPU should reuse the dispatched collective structure, with CUDA-specific assertions adapted to `mcpu` streams.
- `../pytorch/test/distributed/test_dist2.py`
  New ProcessGroup API coverage for `broadcast`, `allgather`, `gather`, `scatter`, `reduce_scatter`, `alltoall`, split, and merge.
- `../pytorch/test/distributed/test_pg_wrapper.py`
  Wrapper shape/device/dtype consistency checks around ProcessGroup calls.
- `../pytorch/test/distributed/test_c10d_functional_native.py`
  Functional collectives: `all_reduce`, `all_gather_into_tensor`, coalesced variants, and wait behavior.
- `../pytorch/test/distributed/test_c10d_object_collectives.py`
  Object collective coverage. These mostly exercise CPU serialization paths, but should remain compatible with a group that has an mcpu backend registered.
- `../pytorch/test/distributed/test_c10d_spawn.py` and `test_c10d_spawn_gloo.py`
  Spawn/init-process-group coverage.

## Required MCPU adapter behavior

- Initialize the process group with CPU/Gloo as the transport backend.
- Create tensors on `torch.device("mcpu")` for tensor collectives.
- Let `torch_mcpu.distributed` register the `PrivateUse1` backend lazily on the ProcessGroup.
- Reuse upstream expected values unchanged; the MCPU backend delegates the actual data movement to the CPU backend through CPU views.
- Replace CUDA stream synchronization checks with `torch.mcpu.current_stream()`, `torch.mcpu.Event`, and `torch.mcpu.synchronize()` checks.

## Minimum local smoke coverage

The local `tests/test_distributed.py` should remain small and only cover MCPU-specific integration:

- lazy `ProcessGroup` mcpu backend registration
- `all_reduce`
- `all_gather`
- `all_gather_into_tensor`
- `gather`
- `broadcast`
- `send` / `recv`
- `isend` / `irecv`
- async work wait preserving mcpu tensor device

The full correctness matrix should come from the upstream files above.
