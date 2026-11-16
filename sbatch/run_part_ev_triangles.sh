#!/usr/bin/env bash
set -euo pipefail
triangle_output=${1:?Missing output}
triangle_phase=${2:?Missing phase}
[[ "$triangle_output" == /* && "$(uname -m)" == x86_64 ]] || exit 2
(cd -- "$triangle_output"; sha256sum --quiet -c launch.sha256)
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONHOME
triangle_python="$(python3 -S -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$triangle_output/launch.json")"
export PATH="$(dirname -- "$triangle_python"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$triangle_python")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$triangle_output/source" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
for name in ${!NCCL_@}; do unset "$name"; done
triangle_nccl="$("$triangle_python" -B -c 'import json,re,shlex,sys; row=json.load(open(sys.argv[1])); env=row["nccl_environment"]; assert all(re.fullmatch(r"NCCL_[A-Za-z0-9_]+",k) and isinstance(v,str) for k,v in env.items()); print("\n".join("export "+k+"="+shlex.quote(v) for k,v in env.items()))' "$triangle_output/launch.json")"
eval "$triangle_nccl"
cd -- "$triangle_output/source"
"$triangle_python" -B -c 'import sys; from part_ev_triangles.protocol import launch_config; launch_config(sys.argv[1])' "$triangle_output"
case "$triangle_phase" in
  prepare|summary)
    exec "$triangle_python" -B scripts/run_part_ev_triangles.py --phase "$triangle_phase" --output "$triangle_output" ;;
  benchmark)
    exec "$triangle_python" -B -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
      scripts/run_part_ev_triangles.py --phase benchmark --output "$triangle_output" ;;
  *) echo 'Unsupported phase; no production mode exists' >&2; exit 2 ;;
esac
