#!/bin/bash -l
# Submit a short smoke run, the default two-stage run, or resume a saved run.
set -euo pipefail
mode="${1:-smoke}"
case "$mode" in smoke|train|resume) ;; *) echo "Usage: bash scripts/scc_submit.sh smoke|train|resume" >&2; exit 2 ;; esac
config="${SCC_CONFIG:-$(cd "$(dirname "$0")" && pwd)/scc_config.local.sh}"
if [[ ! -f "$config" ]]; then
    echo "Missing SCC configuration: $config" >&2
    exit 2
fi
config="$(realpath "$config")"
source "$config"
: "${SCC_PROJECT:?Set SCC_PROJECT in your local SCC config}"
if [[ ! -x "$SCC_ENV/bin/python" ]]; then
    echo "Run bash scripts/scc_setup.sh first." >&2
    exit 2
fi
if [[ "$mode" == resume ]]; then
    : "${SCC_RUN_DIR:?Set SCC_RUN_DIR to the existing run directory for resume}"
    [[ -f "$SCC_RUN_DIR/config.json" ]] || { echo "Missing run configuration" >&2; exit 2; }
else
    revision="$(git -C "$SCC_REPO" rev-parse --short HEAD)"
    SCC_RUN_DIR="$OUTPUT_ROOT/$(date -u +%Y%m%dT%H%M%SZ)_${revision}_${mode}"
fi
mkdir -p "$SCC_RUN_DIR"
SCC_RUN_DIR="$(realpath "$SCC_RUN_DIR")"
if [[ "$config$SCC_RUN_DIR" == *,* ]]; then
    echo "SGE environment paths must not contain commas." >&2
    exit 2
fi
walltime="$SCC_WALLTIME"
[[ "$mode" != smoke ]] || walltime=00:30:00
options=(-P "$SCC_PROJECT" -N "ec523_$mode" -j y -o "$SCC_RUN_DIR/training.log"
    -wd "$SCC_REPO" -pe omp "$SCC_CPUS" -l "gpus=1,gpu_c=$SCC_GPU_CAPABILITY,gpu_memory=$SCC_GPU_MEMORY,h_rt=$walltime"
    -v "SCC_CONFIG=$config,SCC_RUN_DIR=$SCC_RUN_DIR,SCC_MODE=$mode")
if [[ -n "${SCC_GPU_TYPE:-}" ]]; then
    options+=(-l "gpu_type=$SCC_GPU_TYPE")
fi
echo "Run directory: $SCC_RUN_DIR"
qsub "${options[@]}" "$SCC_REPO/scripts/scc_job.sge"

