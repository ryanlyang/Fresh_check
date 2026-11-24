#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 2 && $# -le 3 && "$1" == /* ]] || exit 2
evws_output="$(realpath -e -- "$1")"
evws_phase="$2"
evws_arm="${3:-}"
[[ -z "${evws_arm}" || "${evws_arm}" == control || "${evws_arm}" == workspace ]] || exit 2
case "${evws_phase}" in prepare|benchmark|summary|dispatch|train|evaluate|report) ;; *) exit 2 ;; esac
cd -- "${evws_output}"
sha256sum --quiet -c snapshot.sha256
mapfile -t evws_fields < <(python3 -B -c 'import json; r=json.load(open("launch.json")); print(r["python"]); print(r["resources"]["machine"]); print(r["resources"]["gpus"]); print(r["resources"]["threads"])')
[[ "$(uname -m)" == "${evws_fields[1]}" ]] || exit 2
evws_python="${evws_fields[0]}"
evws_bin="$(dirname -- "${evws_python}")"
export PATH="${evws_bin}:$PATH" LD_LIBRARY_PATH="${evws_bin}/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${evws_output}/source" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${evws_fields[3]}" MKL_NUM_THREADS="${evws_fields[3]}" OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
for evws_name in ${!NCCL_@} ${!TORCH_NCCL_@}; do unset "${evws_name}"; done
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Preserve Slurm's CUDA_VISIBLE_DEVICES; a marker safely reaches both workers.
export EVWS_STOP_FILE="${evws_output}/logs/.stop_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}_$$"
evws_stop() { : > "${EVWS_STOP_FILE}"; }
trap evws_stop USR1 TERM
evws_args=(-B -u "${evws_output}/source/scripts/run_part_ev_workspace.py" --output "${evws_output}" --phase "${evws_phase}")
evws_worker_args=(--output "${evws_output}" --phase "${evws_phase}")
[[ -z "${evws_arm}" ]] || evws_args+=(--arm "${evws_arm}")
[[ -z "${evws_arm}" ]] || evws_worker_args+=(--arm "${evws_arm}")
if [[ "${evws_fields[2]}" == 2 && ( "${evws_phase}" == benchmark || "${evws_phase}" == train ) ]]; then
  "${evws_python}" -B -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
    "${evws_output}/source/scripts/run_part_ev_workspace.py" "${evws_worker_args[@]}" &
else
  "${evws_python}" "${evws_args[@]}" &
fi
evws_child=$!
set +e
while true; do
  wait "${evws_child}"
  evws_code=$?
  kill -0 "${evws_child}" 2>/dev/null || break
done
exit "${evws_code}"
