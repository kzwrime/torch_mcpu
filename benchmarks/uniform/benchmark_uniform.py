#!/usr/bin/env python3

import argparse
import json
import statistics
import time

import torch
import torch_mcpu  # noqa: F401


DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark mcpu uniform_ enqueue latency and completion throughput."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="PyTorch intra-op threads; must be between 1 and 16.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1_000, 1_000_000, 16_000_000],
    )
    parser.add_argument(
        "--dtypes",
        choices=tuple(DTYPES),
        nargs="+",
        default=["float32", "bfloat16"],
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be between 1 and 16")
    if min(args.sizes) < 1:
        parser.error("--sizes values must be positive")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    return args


def median(values: list[float]) -> float:
    return statistics.median(values)


def benchmark_mcpu(
    numel: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    tensor = torch.empty(numel, device="mcpu", dtype=dtype)
    generator = torch.Generator(device="mcpu").manual_seed(1234)

    for _ in range(warmup):
        tensor.uniform_(-1.0, 1.0, generator=generator)
        torch.mcpu.synchronize()

    enqueue_times = []
    completion_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        tensor.uniform_(-1.0, 1.0, generator=generator)
        submitted = time.perf_counter()
        torch.mcpu.synchronize()
        completed = time.perf_counter()
        enqueue_times.append(submitted - start)
        completion_times.append(completed - start)

    enqueue = median(enqueue_times)
    completion = median(completion_times)
    return {
        "enqueue_us": enqueue * 1e6,
        "completion_ms": completion * 1e3,
        "throughput_melem_s": numel / completion / 1e6,
    }


def benchmark_cpu(
    numel: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    tensor = torch.empty(numel, device="cpu", dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(1234)

    for _ in range(warmup):
        tensor.uniform_(-1.0, 1.0, generator=generator)

    completion_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        tensor.uniform_(-1.0, 1.0, generator=generator)
        completion_times.append(time.perf_counter() - start)

    completion = median(completion_times)
    return {
        "completion_ms": completion * 1e3,
        "throughput_melem_s": numel / completion / 1e6,
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    results = []

    for dtype_name in args.dtypes:
        dtype = DTYPES[dtype_name]
        for numel in args.sizes:
            cpu = benchmark_cpu(numel, dtype, args.warmup, args.repeats)
            mcpu = benchmark_mcpu(numel, dtype, args.warmup, args.repeats)
            results.append(
                {
                    "threads": args.threads,
                    "dtype": dtype_name,
                    "numel": numel,
                    "cpu": cpu,
                    "mcpu": mcpu,
                    "mcpu_vs_cpu": (
                        mcpu["throughput_melem_s"] / cpu["throughput_melem_s"]
                    ),
                }
            )

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(
        "threads dtype     numel      enqueue_us completion_ms "
        "throughput_Melem/s vs_cpu"
    )
    for row in results:
        print(
            f"{row['threads']:>7} "
            f"{row['dtype']:<9} "
            f"{row['numel']:>10} "
            f"{row['mcpu']['enqueue_us']:>11.2f} "
            f"{row['mcpu']['completion_ms']:>13.3f} "
            f"{row['mcpu']['throughput_melem_s']:>18.1f} "
            f"{row['mcpu_vs_cpu']:>7.2f}"
        )


if __name__ == "__main__":
    main()
