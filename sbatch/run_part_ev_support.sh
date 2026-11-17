#!/usr/bin/env bash
set -euo pipefail
support_output=${1:?Missing output}
support_phase=${2:?Missing phase}
[[ "$support_output" == /* && "$(uname -m)" == x86_64 ]] || exit 2
(cd -- "$support_output"; sha256sum --quiet -c launch.sha256)
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONHOME
support_python="$(python3 -S -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$support_output/launch.json")"
export PATH="$(dirname -- "$support_python"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$support_python")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$support_output/source" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
for name in ${!NCCL_@}; do unset "$name"; done
support_nccl="$("$support_python" -B -c 'import json,re,shlex,sys; row=json.load(open(sys.argv[1])); env=row["nccl_environment"]; assert all(re.fullmatch(r"NCCL_[A-Za-z0-9_]+",k) and isinstance(v,str) for k,v in env.items()); print("\n".join("export "+k+"="+shlex.quote(v) for k,v in env.items()))' "$support_output/launch.json")"
eval "$support_nccl"
cd -- "$support_output/source"
"$support_python" -B -c 'import sys; from part_ev_support.protocol import launch_config; launch_config(sys.argv[1])' "$support_output"
case "$support_phase" in
  prepare|summary)
    exec "$support_python" -B scripts/run_part_ev_support.py --phase "$support_phase" --output "$support_output" ;;
  benchmark)
    exec "$support_python" -B -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
      scripts/run_part_ev_support.py --phase benchmark --output "$support_output" ;;
  *) echo 'Unsupported phase; no production mode exists' >&2; exit 2 ;;
esac
