#!/usr/bin/env bash
# Executed from Slurm's spool: use only explicitly pinned campaign paths.
set -euo pipefail
: "${PART_FULLSCALE_OUTPUT:?Missing absolute campaign output}"
: "${PART_FULLSCALE_PHASE:?Missing phase}"
[[ "$PART_FULLSCALE_OUTPUT" == /* ]] || { echo 'Campaign output must be absolute' >&2; exit 2; }
source_root="$PART_FULLSCALE_OUTPUT/source"
[[ -f "$PART_FULLSCALE_OUTPUT/launch.env" ]] || { echo 'Pinned launch environment missing' >&2; exit 2; }
# shellcheck disable=SC1090
unset PART_FULLSCALE_COMPARISON
source "$PART_FULLSCALE_OUTPUT/launch.env"
PART_FULLSCALE_COMPARISON="${PART_FULLSCALE_COMPARISON:-standard}"
case "$PART_FULLSCALE_COMPARISON" in standard|plain) ;; *) echo 'Invalid frozen comparison' >&2; exit 2 ;; esac
[[ "$(uname -m)" == aarch64 ]] || { echo 'This worker requires Tigris aarch64; cross-platform execution is not authorized' >&2; exit 2; }
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$source_root"
env_bin="$(dirname -- "$PART_FULLSCALE_PYTHON")"
export PATH="$env_bin:$PATH" LD_LIBRARY_PATH="$env_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS=1
cd -- "$source_root"
sha256sum --quiet -c "$PART_FULLSCALE_OUTPUT/source.sha256"
driver="$source_root/scripts/run_part_fullscale.py"
args=("$PART_FULLSCALE_PHASE" --output "$PART_FULLSCALE_OUTPUT")
case "$PART_FULLSCALE_PHASE" in
  prepare) args+=(--data-root "$PART_FULLSCALE_DATA_ROOT" --comparison "$PART_FULLSCALE_COMPARISON") ;;
  benchmark) args+=(--steps "${PART_FULLSCALE_STEPS:-200}" --warmup "${PART_FULLSCALE_WARMUP:-20}") ;;
  train|test)
    : "${SLURM_ARRAY_TASK_ID:?An array task index is required}"
    [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-5]$ ]] || { echo 'Task index must be 0..5' >&2; exit 2; }
    args+=(--task-index "$SLURM_ARRAY_TASK_ID")
    [[ "$PART_FULLSCALE_PHASE" != test ]] || args+=(--test-data "${PART_FULLSCALE_TEST_DATA:?Explicit test data path required}") ;;
  report) ;;
  *) echo "Unknown phase: $PART_FULLSCALE_PHASE" >&2; exit 2 ;;
esac
case "$PART_FULLSCALE_PHASE" in
  prepare|report) exec "$PART_FULLSCALE_PYTHON" -u "$driver" "${args[@]}" ;;
  *)
    : "${SLURM_JOB_NODELIST:?Missing Slurm allocation}"
    : "${SLURM_JOB_ID:?Missing Slurm job ID}"
    export MASTER_ADDR
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
    [[ -n "$MASTER_ADDR" ]] || { echo 'Could not resolve the master node' >&2; exit 2; }
    export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
    export WORLD_SIZE="${SLURM_NTASKS:-${SLURM_NNODES:-1}}"
    [[ "$WORLD_SIZE" == "${SLURM_NNODES:-1}" ]] || { echo 'Expected exactly one rank/GPU per node' >&2; exit 2; }
    exec srun --kill-on-bad-exit=1 --ntasks="$WORLD_SIZE" --ntasks-per-node=1 \
      bash -c 'export RANK="${SLURM_PROCID:?}" LOCAL_RANK=0; exec "$@"' \
      part-fullscale-rank "$PART_FULLSCALE_PYTHON" -u "$driver" "${args[@]}" ;;
esac
