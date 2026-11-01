#!/usr/bin/env bash
set -euo pipefail
: "${EVGH_OUTPUT:?Missing diagnostic directory}"
: "${EVGH_PHASE:?Missing phase}"
[[ "$EVGH_OUTPUT" == /* && "$(uname -m)" == aarch64 ]] || exit 2
(cd -- "$EVGH_OUTPUT"; sha256sum --quiet -c launch.sha256)
source "$EVGH_OUTPUT/launch.env"
cd -- "$EVGH_OUTPUT/source"
sha256sum --quiet -c "$EVGH_OUTPUT/source.sha256"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$EVGH_OUTPUT/source"
export PATH="$(dirname -- "$EVGH_PYTHON"):$PATH" LD_LIBRARY_PATH="$(dirname -- "$EVGH_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
case "$EVGH_PHASE" in prepare|benchmark) ;; *) echo 'Invalid benchmark phase' >&2; exit 2 ;; esac
exec "$EVGH_PYTHON" scripts/benchmark_part_ev_state_tigris.py \
  --phase "$EVGH_PHASE" --output "$EVGH_OUTPUT" --donor "$EVGH_DONOR"
