#!/usr/bin/env python3
"""Measure worker-side gaps between pointer kernel bodies.

The reported gap is:

    begin_ns[i] - end_ns[i - 1]

where begin/end timestamps are written inside consecutive OpenReg worker tasks.
Use --pre-layer-sleep-ms large enough to cover Python submission time; otherwise
the measured gaps can include producer starvation.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import torch_mcpu  # noqa: F401 - registers and patches the mcpu backend


DEVICE = torch.device("mcpu")
OPENREG_QUEUE_CAPACITY = 16_384
MODE_ID = {"raw": 0, "lambda": 1}
TIMER_ID = {"clock": 0, "tsc": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=5000)
    parser.add_argument(
        "--work-items",
        type=int,
        default=128,
        help="float elements touched by each hand-written for-loop kernel",
    )
    parser.add_argument(
        "--kernel",
        choices=("for-loop", "matmul-128"),
        default="for-loop",
        help="kernel body to run between begin/end timestamps",
    )
    parser.add_argument(
        "--mode",
        choices=("raw", "lambda", "all"),
        default="all",
        help="raw uses orLaunchKernel(func,args); lambda captures pointers in a callable",
    )
    parser.add_argument("--warmup-tasks", type=int, default=64)
    parser.add_argument("--pre-layer-sleep-ms", type=int, default=100)
    parser.add_argument(
        "--timer",
        choices=("tsc", "clock"),
        default="tsc",
        help="timer used by manual begin/end arrays; tsc is lower overhead, clock uses c10::getTime",
    )
    parser.add_argument(
        "--trace-dir", default=Path(__file__).resolve().parent / "traces"
    )
    return parser.parse_args()


def synchronize_mcpu() -> None:
    if hasattr(torch, "mcpu") and hasattr(torch.mcpu, "synchronize"):
        torch.mcpu.synchronize()


def percentile(sorted_values: list[int], pct: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(
        len(sorted_values) - 1,
        max(0, int(round((pct / 100.0) * (len(sorted_values) - 1)))),
    )
    return float(sorted_values[index])


def summarize_ns(values: list[int]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean_ns": 0.0,
            "median_ns": 0.0,
            "min_ns": 0.0,
            "p90_ns": 0.0,
            "p99_ns": 0.0,
            "max_ns": 0.0,
        }
    sorted_values = sorted(values)
    return {
        "count": float(len(values)),
        "mean_ns": statistics.fmean(values),
        "median_ns": statistics.median(values),
        "min_ns": float(sorted_values[0]),
        "p90_ns": percentile(sorted_values, 90.0),
        "p99_ns": percentile(sorted_values, 99.0),
        "max_ns": float(sorted_values[-1]),
    }


def calibrate_tsc_cycles_per_ns() -> float:
    start_tick = int(torch.ops.mcpu.pointer_gap_read_tsc())
    start_ns = time.perf_counter_ns()
    time.sleep(0.05)
    end_tick = int(torch.ops.mcpu.pointer_gap_read_tsc())
    end_ns = time.perf_counter_ns()
    elapsed_ns = max(1, end_ns - start_ns)
    return max(0.001, (end_tick - start_tick) / elapsed_ns)


def run_mode(
    *,
    mode_name: str,
    kernel_name: str,
    tasks: int,
    warmup_tasks: int,
    work_items: int,
    pre_layer_sleep_ms: int,
    timer_name: str,
    cycles_per_ns: float,
    stream: torch.Stream,
) -> dict[str, Any]:
    data_elements = max(1, work_items)
    if kernel_name == "matmul-128":
        data_elements = 128 + 128 * 128 + 128
    data = torch.ones(data_elements, dtype=torch.float32, device=DEVICE)
    begin_ns = torch.empty(tasks, dtype=torch.int64, device=DEVICE)
    end_ns = torch.empty(tasks, dtype=torch.int64, device=DEVICE)
    sleep_marker = torch.empty(1, dtype=torch.int64, device=DEVICE)
    mode_id = MODE_ID[mode_name]
    timer_id = TIMER_ID[timer_name]

    warmup = min(tasks, max(0, warmup_tasks))
    if warmup:
        warmup_begin = torch.empty(warmup, dtype=torch.int64, device=DEVICE)
        warmup_end = torch.empty(warmup, dtype=torch.int64, device=DEVICE)
        with torch.inference_mode(), stream:
            for task in range(warmup):
                if kernel_name == "matmul-128":
                    torch.ops.mcpu.pointer_gap_matmul_128(
                        data, warmup_begin, warmup_end, task, mode_id, timer_id
                    )
                else:
                    torch.ops.mcpu.pointer_gap_for_loop(
                        data,
                        warmup_begin,
                        warmup_end,
                        task,
                        work_items,
                        mode_id,
                        timer_id,
                    )
        stream.synchronize()
        synchronize_mcpu()

    submit_start = time.perf_counter()
    with torch.inference_mode(), stream:
        if pre_layer_sleep_ms > 0:
            torch.ops.mcpu.stream_sleep_fill_(
                sleep_marker, 1, pre_layer_sleep_ms
            )
        for task in range(tasks):
            if kernel_name == "matmul-128":
                torch.ops.mcpu.pointer_gap_matmul_128(
                    data, begin_ns, end_ns, task, mode_id, timer_id
                )
            else:
                torch.ops.mcpu.pointer_gap_for_loop(
                    data, begin_ns, end_ns, task, work_items, mode_id, timer_id
                )
    submit_elapsed_s = time.perf_counter() - submit_start

    sync_start = time.perf_counter()
    stream.synchronize()
    sync_elapsed_s = time.perf_counter() - sync_start
    total_elapsed_s = time.perf_counter() - submit_start

    begin_cpu = begin_ns.cpu().tolist()
    end_cpu = end_ns.cpu().tolist()
    scale = cycles_per_ns if timer_name == "tsc" else 1.0
    body_ns = [
        int(round((int(end) - int(begin)) / scale))
        for begin, end in zip(begin_cpu, end_cpu)
    ]
    gap_ns = [
        int(round((int(begin_cpu[i]) - int(end_cpu[i - 1])) / scale))
        for i in range(1, len(begin_cpu))
    ]

    sleep_covers_submit = (pre_layer_sleep_ms / 1000.0) >= submit_elapsed_s
    return {
        "mode": mode_name,
        "kernel": kernel_name,
        "tasks": tasks,
        "work_items": work_items if kernel_name == "for-loop" else None,
        "matmul_shape": "1x128x128" if kernel_name == "matmul-128" else "",
        "data_elements": data_elements,
        "pre_layer_sleep_ms": pre_layer_sleep_ms,
        "timer": timer_name,
        "cycles_per_ns": cycles_per_ns if timer_name == "tsc" else 0.0,
        "submit_elapsed_s": submit_elapsed_s,
        "sync_elapsed_s": sync_elapsed_s,
        "total_elapsed_s": total_elapsed_s,
        "sleep_covers_submit": sleep_covers_submit,
        "gap_ns": summarize_ns(gap_ns),
        "body_ns": summarize_ns(body_ns),
    }


def print_summary(result: dict[str, Any]) -> None:
    gap = result["gap_ns"]
    body = result["body_ns"]
    print(
        f"mode={result['mode']} tasks={result['tasks']} "
        f"kernel={result['kernel']} "
        f"work_items={result['work_items']} "
        f"matmul_shape={result['matmul_shape']} "
        f"pre_layer_sleep_ms={result['pre_layer_sleep_ms']} "
        f"timer={result['timer']} "
        f"submit={result['submit_elapsed_s']:.6f}s "
        f"sleep_covers_submit={result['sleep_covers_submit']}"
    )
    print(
        "  gap_ns "
        f"mean={gap['mean_ns']:.1f} median={gap['median_ns']:.1f} "
        f"p90={gap['p90_ns']:.1f} p99={gap['p99_ns']:.1f} "
        f"min={gap['min_ns']:.1f} max={gap['max_ns']:.1f}"
    )
    print(
        "  body_ns "
        f"mean={body['mean_ns']:.1f} median={body['median_ns']:.1f} "
        f"p90={body['p90_ns']:.1f} p99={body['p99_ns']:.1f} "
        f"min={body['min_ns']:.1f} max={body['max_ns']:.1f}"
    )


def main() -> None:
    global args
    args = parse_args()
    if not torch.mcpu.is_available():
        raise RuntimeError("mcpu backend is not available")
    if args.tasks <= 1:
        raise RuntimeError("--tasks must be greater than 1")
    if args.work_items < 0:
        raise RuntimeError("--work-items must be non-negative")
    if args.tasks + args.warmup_tasks + 2 >= OPENREG_QUEUE_CAPACITY:
        raise RuntimeError(
            "tasks + warmup_tasks + 2 must be smaller than the OpenReg "
            f"ring capacity ({OPENREG_QUEUE_CAPACITY})"
        )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    trace_dir = Path(args.trace_dir).resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    summary_file = trace_dir / "pointer_gap_summary.json"

    stream = torch.Stream(device=DEVICE)
    modes = ("raw", "lambda") if args.mode == "all" else (args.mode,)
    cycles_per_ns = calibrate_tsc_cycles_per_ns() if args.timer == "tsc" else 1.0
    results = [
        run_mode(
            mode_name=mode,
            kernel_name=args.kernel,
            tasks=args.tasks,
            warmup_tasks=args.warmup_tasks,
            work_items=args.work_items,
            pre_layer_sleep_ms=args.pre_layer_sleep_ms,
            timer_name=args.timer,
            cycles_per_ns=cycles_per_ns,
            stream=stream,
        )
        for mode in modes
    ]

    summary = {
        "env": {
            "TORCH_MCPU_STREAM_WORKER_CORE": os.environ.get(
                "TORCH_MCPU_STREAM_WORKER_CORE", ""
            ),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", ""),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "timer": args.timer,
        "cycles_per_ns": cycles_per_ns if args.timer == "tsc" else 0.0,
        "results": results,
    }
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Pointer gap benchmark complete")
    for result in results:
        print_summary(result)
    print(f"summary={summary_file}")


if __name__ == "__main__":
    main()
