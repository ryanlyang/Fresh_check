#!/usr/bin/env bash
set -euo pipefail
: "${EVTG_CHAIN:?Missing chain}"
: "${EVTG_PHASE:?Missing phase}"
[[ "$(uname -m)" == aarch64 && "$EVTG_CHAIN" == /* ]] || exit 2
(cd -- "$EVTG_CHAIN"; sha256sum --quiet -c config.sha256)
source "$EVTG_CHAIN/config.sh"
(cd -- "$EVTG_CAMPAIGN"; sha256sum --quiet -c launch.sha256)
source "$EVTG_CAMPAIGN/launch.env"
cd -- "$EVTG_CHAIN/source"
sha256sum --quiet -c "$EVTG_CHAIN/source.sha256"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$EVTG_CHAIN/source"
export PATH="$(dirname -- "$EVTG_PYTHON"):$PATH" LD_LIBRARY_PATH="$(dirname -- "$EVTG_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
args=(--phase "$EVTG_PHASE" --chain "$EVTG_CHAIN")
case "$EVTG_PHASE" in
  segment)
    : "${SLURM_JOB_START_TIME:?Missing allocation start}" "${SLURM_JOB_END_TIME:?Missing allocation end}"
    [[ "${EVTG_INDEX:-}" =~ ^[0-7]$ ]] || exit 2
    args+=(--index "$EVTG_INDEX") ;;
  admit|gate|evaluate|report) ;;
  *) echo 'Invalid production phase' >&2; exit 2 ;;
esac
exec "$EVTG_PYTHON" scripts/run_part_ev_tigris_segments.py "${args[@]}"
