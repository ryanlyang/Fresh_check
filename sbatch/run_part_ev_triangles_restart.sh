#!/usr/bin/env bash
set -euo pipefail
evtri_output=${1:?Missing diagnostic output}
[[ "$evtri_output" == /* && "$(uname -m)" == x86_64 ]] || exit 2
(cd -- "$evtri_output"; sha256sum --quiet -c diagnostic.sha256)
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONHOME
evtri_python="$(python3 -S -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$evtri_output/plan.json")"
evtri_campaign="$(python3 -S -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["campaign"])' "$evtri_output/plan.json")"
export PATH="$(dirname -- "$evtri_python"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$evtri_python")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$evtri_output/source" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
for evtri_name in ${!NCCL_@}; do unset "$evtri_name"; done
evtri_nccl="$("$evtri_python" -B -c 'import json,re,shlex,sys; row=json.load(open(sys.argv[1])); env=row["nccl_environment"]; assert all(re.fullmatch(r"NCCL_[A-Za-z0-9_]+",k) and isinstance(v,str) for k,v in env.items()); print("\n".join("export "+k+"="+shlex.quote(v) for k,v in env.items()))' "$evtri_campaign/launch.json")"
eval "$evtri_nccl"
cd -- "$evtri_output"
exec "$evtri_python" -B -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  "$evtri_output/source/scripts/diagnose_part_ev_triangles_restart.py" --output "$evtri_output"
