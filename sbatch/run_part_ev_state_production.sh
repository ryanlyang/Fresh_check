#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && "$1" == /* && "$(uname -m)" == x86_64 ]] || exit 2
evprod_output="$(realpath -e -- "$1")"
evprod_phase="$2"
case "$evprod_phase" in train|evaluate|report) ;; *) exit 2 ;; esac
cd -- "$evprod_output"
sha256sum --quiet -c snapshot.sha256
evprod_python="$(python3 -B -c 'import json; print(json.load(open("campaign.json"))["python"])')"
evprod_science="$(python3 -B -c 'import json; print(json.load(open("campaign.json"))["benchmark"]+"/source")')"
evprod_bin="$(dirname -- "$evprod_python")"
export PATH="$evprod_bin:$PATH" LD_LIBRARY_PATH="$evprod_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$evprod_science" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
for evprod_name in ${!NCCL_@}; do unset "$evprod_name"; done
evprod_exports="$("$evprod_python" -B -c 'import json,re,shlex; env=json.load(open("campaign.json"))["expected_gpu"]["nccl_environment"]; assert all(re.fullmatch(r"NCCL_[A-Za-z0-9_]+",k) and isinstance(v,str) for k,v in env.items()); print("\n".join("export "+k+"="+shlex.quote(v) for k,v in env.items()))')"
eval "$evprod_exports"
export EVPROD_STOP_FILE="$evprod_output/logs/stop_${SLURM_JOB_ID:?}_${SLURM_RESTART_COUNT:-0}"
# Keep the batch shell alive. USR1 must NOT kill the torchrun agent before
# both ranks can commit their full restart state. Workers also check EndTime.
trap 'touch -- "$EVPROD_STOP_FILE"' USR1 TERM
evprod_args=("$evprod_output/source/scripts/run_part_ev_state_production.py" --phase "$evprod_phase" --output "$evprod_output")
if [[ "$evprod_phase" == train ]]; then
  "$evprod_python" -B -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 "${evprod_args[@]}" &
else
  "$evprod_python" -B -u "${evprod_args[@]}" &
fi
evprod_pid=$!
while true; do
  if wait "$evprod_pid"; then exit 0; else evprod_status=$?; fi
  # wait can be interrupted by the trapped Slurm warning; keep supervising.
  if ! kill -0 "$evprod_pid" 2>/dev/null; then exit "$evprod_status"; fi
done
