#!/usr/bin/env bash
set -euo pipefail
: "${EVSCALE_OUTPUT:?Missing diagnostic directory}"
: "${EVSCALE_GPUS:?Missing GPU count}"
[[ "$EVSCALE_OUTPUT" == /* && "$EVSCALE_GPUS" =~ ^[12]$ ]] || exit 2
[[ "$(uname -m)" == x86_64 ]] || exit 2
(cd -- "$EVSCALE_OUTPUT"; sha256sum --quiet -c config.sha256)
source "$EVSCALE_OUTPUT/config.sh"
cd -- "$EVSCALE_OUTPUT/source"
sha256sum --quiet -c "$EVSCALE_OUTPUT/source.sha256"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$EVSCALE_OUTPUT/source"
export PATH="$(dirname -- "$EVSCALE_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$EVSCALE_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="$((4 / EVSCALE_GPUS))" MKL_NUM_THREADS="$((4 / EVSCALE_GPUS))" OPENBLAS_NUM_THREADS=1
# Do not inherit rank/world settings from an earlier distributed shell.
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
if command -v nvidia-smi >/dev/null; then
  nvidia-smi -L
  nvidia-smi topo -m || true
fi
# torchrun owns both processes on this single Slurm node; random rendezvous port.
exec "$EVSCALE_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --max-restarts=0 \
  --nproc-per-node="$EVSCALE_GPUS" scripts/benchmark_part_ev_gpu_scaling.py \
  --phase benchmark --output "$EVSCALE_OUTPUT" --gpus "$EVSCALE_GPUS"
