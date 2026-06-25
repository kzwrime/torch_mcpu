#!/usr/bin/env python3
"""Measure MCPU stream-side kernel body gaps for a small MLP+sigmoid workload."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch.nn as nn

import torch_mcpu  # noqa: F401 - registers and patches the mcpu backend


DEVICE = torch.device("mcpu")


class SmallSigmoidMLP(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int, execution_mode: str) -> None:
        super().__init__()
        self.execution_mode = execution_mode
        self.w1 = nn.Parameter(torch.empty(hidden_dim, model_dim))
        self.b1 = nn.Parameter(torch.empty(hidden_dim))
        self.w2 = nn.Parameter(torch.empty(model_dim, hidden_dim))
        self.b2 = nn.Parameter(torch.empty(model_dim))
        self.register_buffer("_hidden_buf", None, persistent=False)
        self.register_buffer("_out_buf", None, persistent=False)
        self._w1_t: torch.Tensor | None = None
        self._w2_t: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.w1, mean=0.0, std=0.02)
        nn.init.zeros_(self.b1)
        nn.init.normal_(self.w2, mean=0.0, std=0.02)
        nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.execution_mode == "eager":
            hidden = torch.mm(x, self.w1.t()) + self.b1
            hidden = torch.sigmoid(hidden)
            return torch.mm(hidden, self.w2.t()) + self.b2

        self._ensure_cached_views_and_buffers(x)
        if self.execution_mode == "preallocated":
            torch.mm(x, self._w1_t, out=self._hidden_buf)
            torch.add(self._hidden_buf, self.b1, out=self._hidden_buf)
            torch.sigmoid(self._hidden_buf, out=self._hidden_buf)
            torch.mm(self._hidden_buf, self._w2_t, out=self._out_buf)
            torch.add(self._out_buf, self.b2, out=self._out_buf)
            return self._out_buf

        if self.execution_mode == "addmm":
            torch.addmm(self.b1, x, self._w1_t, out=self._hidden_buf)
            torch.sigmoid(self._hidden_buf, out=self._hidden_buf)
            torch.addmm(self.b2, self._hidden_buf, self._w2_t, out=self._out_buf)
            return self._out_buf

        raise RuntimeError(f"unknown execution_mode={self.execution_mode!r}")

    def _ensure_cached_views_and_buffers(self, x: torch.Tensor) -> None:
        if self._w1_t is None or self._w1_t.device != self.w1.device:
            self._w1_t = self.w1.t()
        if self._w2_t is None or self._w2_t.device != self.w2.device:
            self._w2_t = self.w2.t()

        hidden_shape = (x.size(0), self.w1.size(0))
        out_shape = (x.size(0), self.w2.size(0))
        if (
            self._hidden_buf is None
            or tuple(self._hidden_buf.shape) != hidden_shape
            or self._hidden_buf.device != x.device
            or self._hidden_buf.dtype != x.dtype
        ):
            self._hidden_buf = torch.empty(hidden_shape, device=x.device, dtype=x.dtype)
        if (
            self._out_buf is None
            or tuple(self._out_buf.shape) != out_shape
            or self._out_buf.device != x.device
            or self._out_buf.dtype != x.dtype
        ):
            self._out_buf = torch.empty(out_shape, device=x.device, dtype=x.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--profile-iters", type=int, default=200)
    parser.add_argument("--pre-layer-sleep-ms", type=int, default=100)
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "preallocated", "addmm"),
        default="preallocated",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "traces",
    )
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument(
        "--use-profiler",
        action="store_true",
        help=(
            "also run torch.profiler with CPU+PrivateUse1 activities and print "
            "the key_averages table sorted by self_privateuse1_time_total"
        ),
    )
    parser.add_argument(
        "--profiler-row-limit",
        type=int,
        default=20,
        help="row limit for the printed profiler key_averages table",
    )
    parser.add_argument(
        "--export-profiler-trace",
        action="store_true",
        help="export a Chrome trace when --use-profiler is set",
    )
    parser.add_argument(
        "--torch-profiler-with-stack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="collect Python stack metadata in torch.profiler traces",
    )
    parser.add_argument(
        "--torch-profiler-record-shapes",
        action="store_true",
        help="record tensor shapes in torch.profiler traces",
    )
    parser.add_argument(
        "--torch-profiler-with-memory",
        action="store_true",
        help="collect memory events in torch.profiler traces",
    )
    parser.add_argument(
        "--torch-profiler-with-flops",
        action="store_true",
        help="collect FLOP estimates in torch.profiler traces where supported",
    )
    parser.add_argument(
        "--torch-profiler-verbose-stack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write collected Python stacks into the Chrome trace metadata",
    )
    parser.add_argument(
        "--torch-profiler-all-threads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="profile host work from all Python threads when supported",
    )
    return parser.parse_args()


def synchronize_mcpu() -> None:
    if hasattr(torch, "mcpu") and hasattr(torch.mcpu, "synchronize"):
        torch.mcpu.synchronize()


def summarize_ns(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean_ns": 0.0,
            "median_ns": 0.0,
            "p90_ns": 0.0,
            "p99_ns": 0.0,
            "min_ns": 0.0,
            "max_ns": 0.0,
            "over_1us": 0.0,
            "over_10us": 0.0,
        }
    ordered = sorted(values)
    return {
        "count": float(len(values)),
        "mean_ns": statistics.fmean(values),
        "median_ns": statistics.median(values),
        "p90_ns": ordered[min(len(ordered) - 1, int(len(ordered) * 0.90))],
        "p99_ns": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
        "min_ns": ordered[0],
        "max_ns": ordered[-1],
        "over_1us": float(sum(1 for value in values if value > 1_000.0)),
        "over_10us": float(sum(1 for value in values if value > 10_000.0)),
    }


def collect_kernel_events() -> list[dict[str, Any]]:
    threads = torch.mcpu.get_kernel_timing()
    best_thread = max(threads, key=lambda item: len(item.get("events", [])), default=None)
    if best_thread is None:
        return []

    events = []
    for index, event in enumerate(best_thread.get("events", [])):
        begin_time = int(event["begin_time"])
        end_time = int(event["end_time"])
        if begin_time == 0 or end_time <= begin_time:
            continue
        events.append(
            {
                "index": index,
                "name": str(event["name"]),
                "begin_time": begin_time,
                "end_time": end_time,
                "body_ns": max(0.0, end_time - begin_time),
            }
        )
    return events


def summarize_kernel_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = [
        max(0.0, int(curr["begin_time"]) - int(prev["end_time"]))
        for prev, curr in zip(events, events[1:])
        if int(curr["begin_time"]) >= int(prev["end_time"])
    ]
    body_by_op: dict[str, list[float]] = {}
    gap_by_edge: dict[str, list[float]] = {}
    for event in events:
        body_by_op.setdefault(str(event["name"]), []).append(float(event["body_ns"]))
    for prev, curr in zip(events, events[1:]):
        if int(curr["begin_time"]) < int(prev["end_time"]):
            continue
        edge = f"{prev['name']} -> {curr['name']}"
        gap_by_edge.setdefault(edge, []).append(
            int(curr["begin_time"]) - int(prev["end_time"])
        )
    return {
        "event_count": len(events),
        "gap_ns": summarize_ns(gaps),
        "body_ns_by_op": {
            name: summarize_ns(values) for name, values in sorted(body_by_op.items())
        },
        "gap_ns_by_edge": {
            name: summarize_ns(values) for name, values in sorted(gap_by_edge.items())
        },
    }


def build_profiler_table(
    profiler: torch.profiler.profile,
    row_limit: int,
) -> str:
    return profiler.key_averages().table(
        sort_by="self_privateuse1_time_total",
        row_limit=row_limit,
    )


def count_trace_x_events(path: Path, category: str) -> int:
    pattern = f'"ph":"X","cat":"{category}"'.encode("utf-8")
    pretty_pattern = f'"ph": "X", "cat": "{category}"'.encode("utf-8")
    count = 0
    tail = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            data = tail + chunk
            count += data.count(pattern)
            count += data.count(pretty_pattern)
            tail = data[-len(pretty_pattern) :]
    return count


def main() -> None:
    args = parse_args()
    if not torch.mcpu.is_available():
        raise RuntimeError("mcpu backend is not available")

    trace_dir = args.trace_dir.resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    summary_file = trace_dir / "mlp_sigmoid_kernel_timing_summary.json"
    events_file = trace_dir / "mlp_sigmoid_kernel_timing_events.json"
    profiler_table_file = trace_dir / "mlp_sigmoid_profiler_table.txt"
    profiler_trace_file = trace_dir / "mlp_sigmoid.pt.trace.json"

    torch.manual_seed(2026)
    x_cpu = torch.randn(args.tokens, args.model_dim)
    model_cpu = SmallSigmoidMLP(args.model_dim, args.hidden_dim, "eager").eval()
    model = SmallSigmoidMLP(args.model_dim, args.hidden_dim, args.execution_mode).eval()
    model.load_state_dict(model_cpu.state_dict())
    model = model.to(DEVICE)
    x = x_cpu.to(DEVICE)
    stream = torch.Stream(device=DEVICE)

    with torch.inference_mode():
        expected = model_cpu(x_cpu)

    torch.mcpu.set_kernel_timing_enabled(False)
    with torch.inference_mode(), stream:
        for _ in range(args.warmup_iters):
            y = model(x)
    stream.synchronize()
    synchronize_mcpu()

    sleep_marker = torch.empty(1, dtype=torch.int64, device=DEVICE)
    torch.mcpu.reset_kernel_timing()
    torch.mcpu.set_kernel_timing_enabled(True)

    profiler = None
    if args.use_profiler:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.PrivateUse1,
            ],
            record_shapes=args.torch_profiler_record_shapes,
            profile_memory=args.torch_profiler_with_memory,
            with_stack=args.torch_profiler_with_stack,
            with_flops=args.torch_profiler_with_flops,
            with_modules=True,
            experimental_config=torch.profiler._ExperimentalConfig(
                verbose=(
                    args.torch_profiler_with_stack
                    and args.torch_profiler_verbose_stack
                ),
                profile_all_threads=args.torch_profiler_all_threads,
            ),
            acc_events=True,
        )

    submit_start = time.perf_counter()
    with profiler if profiler is not None else nullcontext():
        with torch.inference_mode(), stream:
            if args.pre_layer_sleep_ms > 0:
                torch.ops.mcpu.stream_sleep_fill_(
                    sleep_marker, 1, args.pre_layer_sleep_ms
                )
            for step in range(args.profile_iters):
                if torch.accelerator.current_stream() != stream:
                    raise RuntimeError(f"not on explicit stream before layer {step}")
                y = model(x)
                if torch.accelerator.current_stream() != stream:
                    raise RuntimeError(f"left explicit stream after layer {step}")
        submit_elapsed = time.perf_counter() - submit_start

        sync_start = time.perf_counter()
        stream.synchronize()
        sync_elapsed = time.perf_counter() - sync_start
        total_elapsed = time.perf_counter() - submit_start
    torch.mcpu.set_kernel_timing_enabled(False)

    actual = y.cpu()
    max_diff = float((actual - expected).abs().max().item())
    if not torch.allclose(actual, expected, rtol=args.rtol, atol=args.atol):
        raise RuntimeError(
            f"mcpu result differs from CPU reference: max_diff={max_diff:.8e}, "
            f"rtol={args.rtol}, atol={args.atol}"
        )

    events = collect_kernel_events()
    profiler_table = ""
    profiler_mcpu_stream_events = 0
    if profiler is not None:
        profiler_table = build_profiler_table(profiler, args.profiler_row_limit)
        profiler_table_file.write_text(profiler_table, encoding="utf-8")
        if args.export_profiler_trace:
            profiler.export_chrome_trace(str(profiler_trace_file))
            profiler_mcpu_stream_events = count_trace_x_events(
                profiler_trace_file, "mcpu_kernel"
            )

    summary = {
        "shape": {
            "tokens": args.tokens,
            "model_dim": args.model_dim,
            "hidden_dim": args.hidden_dim,
        },
        "execution_mode": args.execution_mode,
        "profile_iters": args.profile_iters,
        "warmup_iters": args.warmup_iters,
        "pre_layer_sleep_ms": args.pre_layer_sleep_ms,
        "stream_id": stream.stream_id,
        "submit_elapsed_s": submit_elapsed,
        "sync_elapsed_s": sync_elapsed,
        "total_elapsed_s": total_elapsed,
        "max_diff": max_diff,
        "kernel_timing": summarize_kernel_events(events),
        "profiler": {
            "enabled": args.use_profiler,
            "table_file": str(profiler_table_file) if profiler is not None else "",
            "trace_file": (
                str(profiler_trace_file)
                if profiler is not None and args.export_profiler_trace
                else ""
            ),
            "mcpu_stream_events": profiler_mcpu_stream_events,
        },
    }
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    events_file.write_text(json.dumps(events, indent=2), encoding="utf-8")

    gap = summary["kernel_timing"]["gap_ns"]
    print(
        "MLP sigmoid kernel timing complete: "
        f"shape=({args.tokens}x{args.model_dim}) hidden={args.hidden_dim} "
        f"mode={args.execution_mode} iters={args.profile_iters}"
    )
    print(
        f"submit={submit_elapsed:.6f}s sync={sync_elapsed:.6f}s "
        f"total={total_elapsed:.6f}s stream_id={stream.stream_id}"
    )
    print(f"reference_check: PASS max_diff={max_diff:.8e}")
    if profiler_table:
        print("torch_profiler_key_averages_self_mcpu:")
        print(profiler_table)
        print(f"profiler_table={profiler_table_file}")
        if args.export_profiler_trace:
            print(f"profiler_trace={profiler_trace_file}")
            print(f"profiler_mcpu_stream_events={profiler_mcpu_stream_events}")
    print(
        "kernel_gap_ns "
        f"mean={gap['mean_ns']:.1f} median={gap['median_ns']:.1f} "
        f"p90={gap['p90_ns']:.1f} p99={gap['p99_ns']:.1f} "
        f"min={gap['min_ns']:.1f} max={gap['max_ns']:.1f} "
        f"over_1us={int(gap['over_1us'])} over_10us={int(gap['over_10us'])} "
        f"count={int(gap['count'])}"
    )
    print("kernel_body_ns_by_op:")
    for name, values in summary["kernel_timing"]["body_ns_by_op"].items():
        print(
            f"  {name}: mean={values['mean_ns']:.1f} "
            f"median={values['median_ns']:.1f} p90={values['p90_ns']:.1f} "
            f"p99={values['p99_ns']:.1f} "
            f"over_1us={int(values['over_1us'])} count={int(values['count'])}"
        )
    print("kernel_gap_ns_by_edge:")
    for name, values in summary["kernel_timing"]["gap_ns_by_edge"].items():
        print(
            f"  {name}: mean={values['mean_ns']:.1f} "
            f"median={values['median_ns']:.1f} p90={values['p90_ns']:.1f} "
            f"p99={values['p99_ns']:.1f} "
            f"over_1us={int(values['over_1us'])} count={int(values['count'])}"
        )
    print(f"summary={summary_file}")
    print(f"events={events_file}")


if __name__ == "__main__":
    main()
