#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

pick_pair() {
  for d in /sys/devices/system/cpu/cpu[0-9]*; do
    cpu="${d##*cpu}"
    topo="$d/topology"
    pkg="$(cat "$topo/physical_package_id" 2>/dev/null || echo 0)"
    core="$(cat "$topo/core_id" 2>/dev/null || echo "$cpu")"
    sib="$(cat "$topo/thread_siblings_list" 2>/dev/null || echo "$cpu")"
    printf "%s %s %s %s\n" "$cpu" "$pkg" "$core" "$sib"
  done | sort -n | awk '
    !seen[$2 ":" $3]++ { cpu[++n]=$1; pkg[n]=$2; core[n]=$3; sib[n]=$4 }
    END {
      for (gap=1; gap<1000000; gap++) {
        for (i=1; i<=n; i++) for (j=i+1; j<=n; j++) {
          if (pkg[i] != pkg[j] || sib[i] == sib[j]) continue;
          d = core[i] > core[j] ? core[i] - core[j] : core[j] - core[i];
          if (d == gap) { print cpu[i], cpu[j]; exit 0; }
        }
      }
      exit 1;
    }'
}

if [[ "${1:-}" == "--main-cpu" ]]; then
  MAIN_CPU="$2"
  shift 2
else
  read -r MAIN_CPU WORKER_CPU < <(pick_pair)
fi

if [[ "${1:-}" == "--worker-cpu" ]]; then
  WORKER_CPU="$2"
  shift 2
else
  WORKER_CPU="${WORKER_CPU:-}"
fi

if [[ -z "${WORKER_CPU}" ]]; then
  echo "failed to choose worker CPU" >&2
  exit 1
fi

export TORCH_MCPU_STREAM_WORKER_CORE="$WORKER_CPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "[pin] main_cpu=$MAIN_CPU worker_cpu=$WORKER_CPU"
exec taskset -c "$MAIN_CPU" python example/03_mlp_sigmoid/run.py "$@"
