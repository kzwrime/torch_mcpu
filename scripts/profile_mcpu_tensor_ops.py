#!/usr/bin/env python3
"""Profile a short mcpu tensor-operator loop with torch.profiler."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import torch
import torch_mcpu  # noqa: F401 - registers the mcpu PrivateUse1 backend


TorchProfilerActivity = Literal["CPU", "CUDA", "XPU", "PrivateUse1"]
TorchProfilerActivityMap = {
    "CPU": torch.profiler.ProfilerActivity.CPU,
    "CUDA": torch.profiler.ProfilerActivity.CUDA,
    "XPU": torch.profiler.ProfilerActivity.XPU,
    "PrivateUse1": torch.profiler.ProfilerActivity.PrivateUse1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a 2-iteration mcpu loop containing add_, sub, "
        "cumsum.out, and index.Tensor_out operators."
    )
    parser.add_argument("--torch-profiler-dir", default="./vllm_profile")
    parser.add_argument(
        "--activities",
        nargs="+",
        choices=tuple(TorchProfilerActivityMap),
        default=["CPU", "PrivateUse1"],
    )
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=8)
    return parser.parse_args()


def synchronize_mcpu() -> None:
    if hasattr(torch, "mcpu") and hasattr(torch.mcpu, "synchronize"):
        torch.mcpu.synchronize()


def dump_time_totals(prof: torch.profiler.profile) -> None:
    totals: dict[str, float] = {}
    for evt in prof.key_averages():
        for attr in ("cpu_time_total", "cuda_time_total", "privateuse1_time_total"):
            if hasattr(evt, attr):
                totals[attr] = totals.get(attr, 0.0) + float(getattr(evt, attr))

    for attr, value_us in sorted(totals.items()):
        print(f"{attr}: {value_us / 1000.0:.3f} ms")


def main() -> None:
    args = parse_args()
    if not torch.mcpu.is_available():
        raise RuntimeError("mcpu backend is not available in this PyTorch build")

    profile_dir = Path(args.torch_profiler_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mcpu")
    rows, cols = args.rows, args.cols

    a = torch.ones((rows, cols), device=device, dtype=torch.float32)
    b = torch.arange(rows * cols, device=device, dtype=torch.float32).reshape(rows, cols)
    cumsum_out = torch.empty_like(a)

    row_index = torch.tensor([rows - 1, 0], device=device, dtype=torch.long)
    col_index = torch.tensor([cols - 1, 1], device=device, dtype=torch.long)
    index_out = torch.empty((2,), device=device, dtype=torch.float32)

    profiler_schedule = torch.profiler.schedule(wait=0, warmup=0, active=2, repeat=1)
    trace_handler = torch.profiler.tensorboard_trace_handler(
        str(profile_dir),
        use_gzip=False,
    )

    activities = [TorchProfilerActivityMap[activity] for activity in args.activities]
    with torch.profiler.profile(
        activities=activities,
        schedule=profiler_schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=trace_handler,
    ) as prof:
        for step in range(2):
            with torch.profiler.record_function(f"mcpu_main_loop_step_{step}"):
                a += b
                a = b - 1
                torch.cumsum(a, dim=1, out=cumsum_out)
                torch.ops.aten.index.Tensor_out(
                    cumsum_out,
                    [row_index, col_index],
                    out=index_out,
                )
            synchronize_mcpu()
            prof.step()

    print(f"trace_dir: {profile_dir.resolve()}")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    dump_time_totals(prof)
    print("index_out:", index_out.cpu())


if __name__ == "__main__":
    main()
