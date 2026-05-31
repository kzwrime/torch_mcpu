#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${BENCH_DIR}/../.." && pwd)"
BUILD_DIR="${BENCH_DIR}/build"
CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O3 -DNDEBUG -std=c++17 -pthread}"
TASKS=500
ITERATIONS=2000
WARMUP=200
PIN_CORE=""
RUN_ONLY=""
STREAM_SPIN_ONLY=0

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --tasks N            Tasks per batch. Default: ${TASKS}
  --iterations N       Measured iterations. Default: ${ITERATIONS}
  --warmup N           Warmup iterations. Default: ${WARMUP}
  --pin-core N         Pin worker threads to CPU N where supported.
  --pin-core auto      Pick a high-numbered CPU automatically.
  --stream-spin-only   Keep the openreg stream worker spinning instead of sleeping.
  --run NAME           Run one benchmark: overhead, breakdown, patterns, progression.
  --build-only         Compile only.
  --help              Show this help.

Environment:
  CXX                  C++ compiler. Default: g++
  CXXFLAGS             Compile flags. Default: -O3 -DNDEBUG -std=c++17 -pthread
EOF
}

BUILD_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)
      TASKS="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --pin-core)
      PIN_CORE="$2"
      shift 2
      ;;
    --run)
      RUN_ONLY="$2"
      shift 2
      ;;
    --stream-spin-only)
      STREAM_SPIN_ONLY=1
      shift
      ;;
    --build-only)
      BUILD_ONLY=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

auto_pin_core() {
  if command -v nproc >/dev/null 2>&1; then
    local count
    count="$(nproc)"
    if [[ "${count}" =~ ^[0-9]+$ && "${count}" -gt 1 ]]; then
      echo $((count - 1))
      return
    fi
  fi
  echo ""
}

if [[ "${PIN_CORE}" == "auto" ]]; then
  PIN_CORE="$(auto_pin_core)"
fi

mkdir -p "${BUILD_DIR}"

compile() {
  local name="$1"
  shift
  echo "[build] ${name}"
  # shellcheck disable=SC2086
  "${CXX}" ${CXXFLAGS} "$@" -o "${BUILD_DIR}/${name}"
}

compile \
  openreg_queue_overhead \
  -DTORCH_MCPU_ENABLE_MEMORY_PROTECTION=0 \
  -DTORCH_MCPU_ENABLE_ASYNC_LAUNCH=1 \
  -I"${BENCH_DIR}" \
  -I"${ROOT_DIR}/third_party/openreg" \
  "${BENCH_DIR}/openreg_queue_overhead.cpp" \
  "${ROOT_DIR}/third_party/openreg/csrc/device.cpp" \
  "${ROOT_DIR}/third_party/openreg/csrc/stream.cpp" \
  "${ROOT_DIR}/third_party/openreg/csrc/memory.cpp"

compile \
  queue_cost_breakdown \
  -I"${BENCH_DIR}" \
  "${BENCH_DIR}/queue_cost_breakdown.cpp"

compile \
  kernel_dispatch_patterns \
  -I"${BENCH_DIR}" \
  "${BENCH_DIR}/kernel_dispatch_patterns.cpp"

compile \
  stream_progression_benchmark \
  -I"${BENCH_DIR}" \
  "${BENCH_DIR}/stream_progression_benchmark.cpp"

if [[ "${BUILD_ONLY}" -eq 1 ]]; then
  exit 0
fi

common_args=(--tasks "${TASKS}" --iterations "${ITERATIONS}" --warmup "${WARMUP}")
pin_args=()
if [[ -n "${PIN_CORE}" ]]; then
  pin_args=(--pin-core "${PIN_CORE}")
  export TORCH_MCPU_STREAM_WORKER_CORE="${PIN_CORE}"
fi
if [[ "${STREAM_SPIN_ONLY}" -eq 1 ]]; then
  export TORCH_MCPU_STREAM_SPIN_ONLY=1
fi

run_bench() {
  local name="$1"
  shift
  if [[ -n "${RUN_ONLY}" && "${RUN_ONLY}" != "${name}" ]]; then
    return
  fi
  echo
  echo "[run] ${name}"
  "$@"
}

run_bench \
  overhead \
  "${BUILD_DIR}/openreg_queue_overhead" \
  "${common_args[@]}"

run_bench \
  breakdown \
  "${BUILD_DIR}/queue_cost_breakdown" \
  "${common_args[@]}" \
  "${pin_args[@]}"

run_bench \
  patterns \
  "${BUILD_DIR}/kernel_dispatch_patterns" \
  "${common_args[@]}" \
  "${pin_args[@]}"

progression_args=("${common_args[@]}")
if [[ -n "${PIN_CORE}" ]]; then
  progression_args+=(--worker-core "${PIN_CORE}")
fi

run_bench \
  progression \
  "${BUILD_DIR}/stream_progression_benchmark" \
  "${progression_args[@]}"
