#!/usr/bin/env bash
set -euo pipefail
: "${PAIRMEM_OUTPUT:?Missing campaign}"
: "${PAIRMEM_PHASE:?Missing phase}"
[[ "$PAIRMEM_OUTPUT" == /* && "$(uname -m)" == x86_64 ]] || exit 2
(cd -- "$PAIRMEM_OUTPUT"; sha256sum --quiet -c config.sha256)
source "$PAIRMEM_OUTPUT/config.sh"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PAIRMEM_OUTPUT/source"
export PATH="$(dirname -- "$PAIRMEM_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$PAIRMEM_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd -- "$PAIRMEM_OUTPUT/source"
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
# Restore exactly the submission-time NCCL options, not an unrelated shell's.
for name in ${!NCCL_@}; do unset "$name"; done
nccl_exports="$("$PAIRMEM_PYTHON" -B -c 'import json,re,shlex,sys; row=json.load(open(sys.argv[1])); env=row["nccl_environment"]; assert all(re.fullmatch(r"NCCL_[A-Za-z0-9_]+",k) and isinstance(v,str) for k,v in env.items()); print("\n".join("export "+k+"="+shlex.quote(v) for k,v in env.items()))' "$PAIRMEM_OUTPUT/launch.json")"
eval "$nccl_exports"
"$PAIRMEM_PYTHON" -B -c 'import sys; from part_ev_pair_memory.protocol import verify_snapshot; verify_snapshot(sys.argv[1])' "$PAIRMEM_OUTPUT"
case "$PAIRMEM_PHASE" in
  prepare|summary)
    exec "$PAIRMEM_PYTHON" -B scripts/run_part_ev_pair_memory.py --phase "$PAIRMEM_PHASE" --output "$PAIRMEM_OUTPUT" ;;
  benchmark)
    exec "$PAIRMEM_PYTHON" -B -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
      scripts/run_part_ev_pair_memory.py --phase benchmark --output "$PAIRMEM_OUTPUT" ;;
  *) echo 'Unsupported phase; no production mode exists' >&2; exit 2 ;;
esac
