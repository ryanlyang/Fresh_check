#!/usr/bin/env bash
set -euo pipefail
: "${EVSEG_CHAIN:?Missing chain}"
: "${EVSEG_PHASE:?Missing phase}"
[[ "$EVSEG_CHAIN" == /* && "$(uname -m)" == x86_64 ]] || exit 2
(cd -- "$EVSEG_CHAIN"; sha256sum --quiet -c config.sha256)
source "$EVSEG_CHAIN/config.sh"
cd -- "$EVSEG_CHAIN/source"
sha256sum --quiet -c "$EVSEG_CHAIN/source.sha256"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$EVSEG_CHAIN/source"
export PATH="$(dirname -- "$EVSEG_PYTHON"):$PATH" LD_LIBRARY_PATH="$(dirname -- "$EVSEG_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
driver=scripts/run_part_ev_ddp_segments.py
args=(--phase "$EVSEG_PHASE" --chain "$EVSEG_CHAIN")
case "$EVSEG_PHASE" in
  gate|report)
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
    exec "$EVSEG_PYTHON" "$driver" "${args[@]}" ;;
  segment)
    : "${EVSEG_INDEX:?Missing segment index}"
    : "${SLURM_JOB_START_TIME:?Missing allocation start}"
    : "${SLURM_JOB_END_TIME:?Missing allocation end}"
    gpus=2; args+=(--index "$EVSEG_INDEX") ;;
  evaluate) gpus=1 ;;
  *) echo 'Invalid worker phase' >&2; exit 2 ;;
esac
export OMP_NUM_THREADS="$((4/gpus))" MKL_NUM_THREADS="$((4/gpus))" OPENBLAS_NUM_THREADS=1
exec "$EVSEG_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --max-restarts=0 \
  --nproc-per-node="$gpus" "$driver" "${args[@]}"
