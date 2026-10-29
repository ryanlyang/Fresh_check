#!/usr/bin/env bash
set -euo pipefail
: "${EVDDP_OUTPUT:?Missing campaign}"
: "${EVDDP_PHASE:?Missing phase}"
[[ "$EVDDP_OUTPUT" == /* && "$(uname -m)" == x86_64 ]] || exit 2
(cd -- "$EVDDP_OUTPUT"; sha256sum --quiet -c config.sha256)
source "$EVDDP_OUTPUT/config.sh"
cd -- "$EVDDP_OUTPUT/source"
sha256sum --quiet -c "$EVDDP_OUTPUT/source.sha256"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$EVDDP_OUTPUT/source"
export PATH="$(dirname -- "$EVDDP_PYTHON"):$PATH" LD_LIBRARY_PATH="$(dirname -- "$EVDDP_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
driver=scripts/run_part_ev_layerwise_ddp.py
case "$EVDDP_PHASE" in
  report) export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1; exec "$EVDDP_PYTHON" "$driver" --phase report --output "$EVDDP_OUTPUT" ;;
  preflight|train) gpus=2 ;;
  evaluate) gpus=1 ;;
  *) echo 'Invalid worker phase' >&2; exit 2 ;;
esac
export OMP_NUM_THREADS="$((4/gpus))" MKL_NUM_THREADS="$((4/gpus))" OPENBLAS_NUM_THREADS=1
exec "$EVDDP_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --max-restarts=0 \
  --nproc-per-node="$gpus" "$driver" --phase "$EVDDP_PHASE" --output "$EVDDP_OUTPUT"
