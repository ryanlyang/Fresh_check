#!/usr/bin/env bash
# Slurm executes a spool copy. Resolve code from --tool-root, never BASH_SOURCE.
set -euo pipefail
python_bin= tool_root= output= phase= model=
while (($#)); do
  (($# >= 2)) || { echo 'Missing worker argument value.' >&2; exit 2; }
  case "$1" in
    --python) python_bin=$2 ;;
    --tool-root) tool_root=$2 ;;
    --output) output=$2 ;;
    --command) phase=$2 ;;
    --model) model=$2 ;;
    *) echo "Unexpected worker option: $1" >&2; exit 2 ;;
  esac
  shift 2
done
[[ "$python_bin" == /* && -x "$python_bin" && "$tool_root" == /* && "$output" == /* ]] || exit 2
[[ -f "$tool_root/scripts/part_ev_oscar.py" && -d "$output" ]] || exit 2
case "$phase" in export-model|prepare|parity|evaluate|metrics|report) ;; *) exit 2 ;; esac
unset PYTHONPATH RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PATH="$(dirname -- "$python_bin"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$python_bin")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR="$output/tmp" XDG_CACHE_HOME="$output/cache" PIP_CACHE_DIR="$output/cache/pip"
export TORCH_HOME="$output/cache/torch" CUDA_CACHE_PATH="$output/cache/cuda"
export TORCH_EXTENSIONS_DIR="$output/cache/extensions" MPLCONFIGDIR="$output/cache/matplotlib"
for worker_dir in "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR" "$MPLCONFIGDIR"; do
  [[ -d "$worker_dir" ]] || mkdir -p -- "$worker_dir"
done
cd -- "$output"
args=("$python_bin" -B "$tool_root/scripts/part_ev_oscar.py" "$phase" --output "$output")
[[ -z "$model" ]] || args+=(--model "$model")
# exec makes B:USR1 reach Python directly; a timed-out file is recomputed on resume.
exec "${args[@]}"
