#!/usr/bin/env python3
"""TP=4 MLP inference profile example for the mcpu backend.

This script simulates the tensor-parallel MLP part of a decoder layer:

    x -> local up projection -> activation -> local down projection -> all_reduce

Each rank owns one local TP shard.  The default shape makes every matmul a
visible (16 x 1024) @ (1024 x 1024) operation, and each rank exports a gzipped
Chrome trace through torch.profiler.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import torch_mcpu  # noqa: F401 - registers and patches the mcpu backend


DEVICE = torch.device("mcpu")


class TPShardMLP(nn.Module):
    def __init__(self, model_dim: int, local_hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(local_hidden_dim, model_dim))
        self.b1 = nn.Parameter(torch.empty(local_hidden_dim))
        self.w2 = nn.Parameter(torch.empty(model_dim, local_hidden_dim))
        self.b2 = nn.Parameter(torch.empty(model_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.w1, mean=0.0, std=0.01)
        nn.init.zeros_(self.b1)
        nn.init.normal_(self.w2, mean=0.0, std=0.01)
        nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.mm(x, self.w1.t()) + self.b1
        # Keep the profiled path on native mcpu kernels. aten::silu currently
        # falls back through CPU in this backend, which would hide stream
        # behavior behind synchronous copies.
        hidden = torch.mul(hidden, hidden)
        return torch.mm(hidden, self.w2.t()) + self.b2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=1024)
    parser.add_argument("--local-hidden-dim", type=int, default=1024)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--profile-iters", type=int, default=4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--skip-serial-check", action="store_true")
    parser.add_argument(
        "--trace-dir",
        default=Path(__file__).resolve().parent / "traces",
    )
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def synchronize_mcpu() -> None:
    if hasattr(torch, "mcpu") and hasattr(torch.mcpu, "synchronize"):
        torch.mcpu.synchronize()


def init_input(tokens: int, model_dim: int) -> torch.Tensor:
    torch.manual_seed(1234)
    return torch.randn(tokens, model_dim, device=DEVICE)


def validate_trace_file(trace_file: Path) -> dict[str, Any]:
    with gzip.open(trace_file, "rt", encoding="utf-8") as f:
        trace = json.load(f)

    events = trace.get("traceEvents", trace) if isinstance(trace, dict) else trace
    if not isinstance(events, list) or not events:
        raise RuntimeError(f"{trace_file} has no trace events")

    names = [event.get("name", "") for event in events if isinstance(event, dict)]
    return {
        "file": str(trace_file),
        "events": len(events),
        "has_mlp_step": any("tp_mlp_profile_step" in name for name in names),
        "has_mm": any("aten::mm" in name or "aten::matmul" in name for name in names),
        "has_all_reduce": any("all_reduce" in name.lower() for name in names),
    }


def run_one_step(
    model: nn.Module,
    x: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    y = model(x)
    dist.all_reduce(y, group=group)
    return y


def serial_reference_from_shards(
    x: torch.Tensor,
    shard_results: list[dict[str, Any]],
) -> torch.Tensor:
    w1 = torch.cat([result["w1"] for result in shard_results], dim=0)
    b1 = torch.cat([result["b1"] for result in shard_results], dim=0)
    w2 = torch.cat([result["w2"] for result in shard_results], dim=1)
    b2 = torch.stack([result["b2"] for result in shard_results]).sum(dim=0)
    hidden = torch.mm(x, w1.t()) + b1
    hidden = torch.mul(hidden, hidden)
    return torch.mm(hidden, w2.t()) + b2


def worker(
    rank: int,
    args: argparse.Namespace,
    init_file: str,
    payload_dir: str,
    queue: mp.Queue,
) -> None:
    trace_root = Path(args.trace_dir).resolve()
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=args.world_size,
        )
        group = dist.new_group(list(range(args.world_size)))

        torch.manual_seed(1000 + rank)
        shard_model = TPShardMLP(args.model_dim, args.local_hidden_dim)
        shard_model = shard_model.to(DEVICE).eval()
        model: nn.Module = shard_model
        if args.compile:
            model = torch.compile(model, options={"cpp_wrapper": False})

        x = init_input(args.tokens, args.model_dim)
        stream = torch.Stream(device=DEVICE)
        rank_trace_dir = trace_root / f"rank{rank}"
        rank_trace_dir.mkdir(parents=True, exist_ok=True)

        with torch.inference_mode(), stream:
            for _ in range(args.warmup_iters):
                y = run_one_step(model, x, group)
            warmup_event = stream.record_event()

        warmup_event.synchronize()
        synchronize_mcpu()

        trace_handler = torch.profiler.tensorboard_trace_handler(
            str(rank_trace_dir),
            use_gzip=True,
        )
        schedule = torch.profiler.schedule(
            wait=0,
            warmup=0,
            active=args.profile_iters,
            repeat=1,
        )
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ]

        stream_mismatches: list[str] = []
        submit_start = time.perf_counter()
        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            on_trace_ready=trace_handler,
            acc_events=True,
        ) as prof:
            with torch.inference_mode(), stream:
                for step in range(args.profile_iters):
                    if torch.accelerator.current_stream() != stream:
                        stream_mismatches.append(f"before_step_{step}")
                    with torch.profiler.record_function(f"tp_mlp_profile_step_{step}"):
                        y = run_one_step(model, x, group)
                    if torch.accelerator.current_stream() != stream:
                        stream_mismatches.append(f"after_step_{step}")
                    prof.step()

        submit_elapsed = time.perf_counter() - submit_start
        sync_start = time.perf_counter()
        stream.synchronize()
        sync_elapsed = time.perf_counter() - sync_start
        total_elapsed = time.perf_counter() - submit_start

        trace_files = sorted(
            rank_trace_dir.glob("*.pt.trace.json.gz"),
            key=lambda path: path.stat().st_mtime,
        )
        if not trace_files:
            raise RuntimeError(f"rank {rank} did not produce a .pt.trace.json.gz file")
        trace_info = validate_trace_file(trace_files[-1])

        profiler_keys = {event.key for event in prof.key_averages()}
        fallback_copy_ops = sorted(
            key for key in profiler_keys if key in {"aten::_to_cpu", "aten::_copy_from"}
        )
        if fallback_copy_ops:
            raise RuntimeError(
                f"rank {rank} saw fallback copy ops in the main profile: {fallback_copy_ops}"
            )
        if stream_mismatches:
            raise RuntimeError(f"rank {rank} left explicit stream: {stream_mismatches}")
        if submit_elapsed >= total_elapsed * 0.75 and sync_elapsed > 0.01:
            raise RuntimeError(
                "main loop appears host-blocked instead of queued on the stream: "
                f"submit={submit_elapsed:.6f}s sync={sync_elapsed:.6f}s total={total_elapsed:.6f}s"
            )

        payload_file = Path(payload_dir) / f"rank{rank}.pt"
        torch.save(
            {
                "x": x.cpu(),
                "y": y.cpu(),
                "w1": shard_model.w1.detach().cpu(),
                "b1": shard_model.b1.detach().cpu(),
                "w2": shard_model.w2.detach().cpu(),
                "b2": shard_model.b2.detach().cpu(),
            },
            payload_file,
        )

        queue.put(
            {
                "rank": rank,
                "device": y.device.type,
                "shape": tuple(y.shape),
                "value": float(y[0, 0].cpu().item()),
                "stream_id": stream.stream_id,
                "submit_elapsed": submit_elapsed,
                "sync_elapsed": sync_elapsed,
                "total_elapsed": total_elapsed,
                "trace": trace_info,
                "table": prof.key_averages().table(sort_by="cpu_time_total", row_limit=12),
                "payload_file": str(payload_file),
            }
        )
    except Exception as exc:
        queue.put({"rank": rank, "error": repr(exc)})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.world_size != 4:
        raise ValueError("this TP example is intended to run with --world-size 4")
    if args.tokens < 16 or args.model_dim < 1024 or args.local_hidden_dim < 1024:
        raise ValueError("use at least --tokens 16 --model-dim 1024 --local-hidden-dim 1024")
    if not torch.mcpu.is_available():
        raise RuntimeError("mcpu backend is not available")

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    fd, init_file = tempfile.mkstemp(prefix="mcpu_mlp_tp4_")
    os.close(fd)
    os.unlink(init_file)
    payload_tmp = tempfile.TemporaryDirectory(prefix="mcpu_mlp_tp4_payload_")

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=worker, args=(rank, args, init_file, payload_tmp.name, queue))
        for rank in range(args.world_size)
    ]

    for proc in procs:
        proc.start()

    results = [queue.get(timeout=600) for _ in range(args.world_size)]

    for proc in procs:
        proc.join(timeout=600)

    failures = [
        f"pid={proc.pid} exitcode={proc.exitcode}" for proc in procs if proc.exitcode != 0
    ]
    errors = [result for result in results if "error" in result]
    if failures or errors:
        raise RuntimeError(f"worker failures={failures} errors={errors}")

    by_rank = {result["rank"]: result for result in results}
    if set(by_rank) != set(range(args.world_size)):
        raise RuntimeError(f"missing rank results: got {sorted(by_rank)}")

    for result in by_rank.values():
        payload = torch.load(result["payload_file"], map_location="cpu", weights_only=True)
        result.update(payload)

    serial_check = None
    if not args.skip_serial_check:
        ordered_results = [by_rank[rank] for rank in range(args.world_size)]
        reference_x = ordered_results[0]["x"]
        for rank, result in enumerate(ordered_results[1:], start=1):
            if not torch.allclose(reference_x, result["x"], rtol=0.0, atol=0.0):
                raise RuntimeError(f"rank {rank} input differs from rank 0")

        reference_y = serial_reference_from_shards(reference_x, ordered_results)
        tp_outputs = [result["y"] for result in ordered_results]
        rank_diffs = [
            float((output - tp_outputs[0]).abs().max().item()) for output in tp_outputs
        ]
        max_rank_diff = max(rank_diffs)
        max_serial_diff = float((tp_outputs[0] - reference_y).abs().max().item())
        if not torch.allclose(tp_outputs[0], reference_y, rtol=args.rtol, atol=args.atol):
            raise RuntimeError(
                "TP all_reduce output does not match serial full-MLP reference: "
                f"max_diff={max_serial_diff:.8e}, rtol={args.rtol}, atol={args.atol}"
            )
        for rank, output in enumerate(tp_outputs[1:], start=1):
            if not torch.allclose(output, tp_outputs[0], rtol=0.0, atol=0.0):
                raise RuntimeError(
                    f"rank {rank} all_reduce output differs from rank 0: "
                    f"max_rank_diff={rank_diffs[rank]:.8e}"
                )
        serial_check = {
            "max_serial_diff": max_serial_diff,
            "max_rank_diff": max_rank_diff,
        }

    print(
        "TP4 MLP profile complete: "
        f"tokens={args.tokens}, model_dim={args.model_dim}, "
        f"local_hidden_dim={args.local_hidden_dim}"
    )
    if serial_check is not None:
        print(
            "serial_check: PASS "
            f"max_serial_diff={serial_check['max_serial_diff']:.8e} "
            f"max_rank_diff={serial_check['max_rank_diff']:.8e} "
            f"rtol={args.rtol} atol={args.atol}"
        )
    for rank in sorted(by_rank):
        result = by_rank[rank]
        trace = result["trace"]
        print(
            f"rank {rank}: device={result['device']} shape={result['shape']} "
            f"stream_id={result['stream_id']} submit={result['submit_elapsed']:.6f}s "
            f"sync={result['sync_elapsed']:.6f}s total={result['total_elapsed']:.6f}s"
        )
        print(
            f"rank {rank}: trace={trace['file']} events={trace['events']} "
            f"has_mlp_step={trace['has_mlp_step']} has_mm={trace['has_mm']} "
            f"has_all_reduce={trace['has_all_reduce']}"
        )
    print(f"trace_dir: {trace_dir.resolve()}")
    print("\nrank 0 profiler table:")
    print(by_rank[0]["table"])
    payload_tmp.cleanup()


if __name__ == "__main__":
    main()
