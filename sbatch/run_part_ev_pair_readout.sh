#!/usr/bin/env bash
set -euo pipefail
readout_output=${1:?Missing output}
readout_phase=${2:?Missing phase}
[[ "$readout_output" == /* && "$(uname -m)" == x86_64 ]] || exit 2
(cd -- "$readout_output"; sha256sum --quiet -c launch.sha256)
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONHOME
readout_python="$(python3 -S -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$readout_output/launch.json")"
export PATH="$(dirname -- "$readout_python"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$readout_python")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$readout_output/source" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
for name in ${!NCCL_@}; do unset "$name"; done
readout_nccl="$("$readout_python" -B -c 'import json,re,shlex,sys; row=json.load(open(sys.argv[1])); env=row["nccl_environment"]; assert all(re.fullmatch(r"NCCL_[A-Za-z0-9_]+",k) and isinstance(v,str) for k,v in env.items()); print("\n".join("export "+k+"="+shlex.quote(v) for k,v in env.items()))' "$readout_output/launch.json")"
eval "$readout_nccl"
cd -- "$readout_output/source"
"$readout_python" -B -c 'import sys; from part_ev_pair_readout.protocol import launch_config; launch_config(sys.argv[1])' "$readout_output"
case "$readout_phase" in
  prepare|summary)
    exec "$readout_python" -B scripts/run_part_ev_pair_readout.py --phase "$readout_phase" --output "$readout_output" ;;
  benchmark)
    exec "$readout_python" -B -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
      scripts/run_part_ev_pair_readout.py --phase benchmark --output "$readout_output" ;;
  *) echo 'Unsupported phase; no production mode exists' >&2; exit 2 ;;
esac
