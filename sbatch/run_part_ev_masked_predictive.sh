#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 2 && $# -le 3 && "$1" == /* ]] || exit 2
evmp_output="$(realpath -e -- "$1")"
evmp_phase="$2"
evmp_arm="${3:-}"
[[ -z "${evmp_arm}" || "${evmp_arm}" == control || "${evmp_arm}" == masked ]] || exit 2
case "${evmp_phase}" in prepare|benchmark|summary|dispatch|train|evaluate|report) ;; *) exit 2 ;; esac
cd -- "${evmp_output}"
sha256sum --quiet -c snapshot.sha256
mapfile -t evmp_fields < <(python3 -B -c 'import json; r=json.load(open("launch.json")); print(r["python"]); print(r["resources"]["machine"]); print(r["resources"]["gpus"]); print(r["resources"]["threads"])')
[[ "$(uname -m)" == "${evmp_fields[1]}" ]] || exit 2
evmp_python="${evmp_fields[0]}"
evmp_bin="$(dirname -- "${evmp_python}")"
export PATH="${evmp_bin}:$PATH" LD_LIBRARY_PATH="${evmp_bin}/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${evmp_output}/source" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${evmp_fields[3]}" MKL_NUM_THREADS="${evmp_fields[3]}" OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
for evmp_name in ${!NCCL_@} ${!TORCH_NCCL_@}; do unset "${evmp_name}"; done
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Preserve Slurm's CUDA_VISIBLE_DEVICES; a marker safely reaches both workers.
export EVMP_STOP_FILE="${evmp_output}/logs/.stop_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}_$$"
evmp_stop() { : > "${EVMP_STOP_FILE}"; }
trap evmp_stop USR1 TERM
evmp_args=(-B -u "${evmp_output}/source/scripts/run_part_ev_masked_predictive.py" --output "${evmp_output}" --phase "${evmp_phase}")
evmp_worker_args=(--output "${evmp_output}" --phase "${evmp_phase}")
[[ -z "${evmp_arm}" ]] || evmp_args+=(--arm "${evmp_arm}")
[[ -z "${evmp_arm}" ]] || evmp_worker_args+=(--arm "${evmp_arm}")
if [[ "${evmp_fields[2]}" == 2 && ( "${evmp_phase}" == benchmark || "${evmp_phase}" == train ) ]]; then
  "${evmp_python}" -B -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
    "${evmp_output}/source/scripts/run_part_ev_masked_predictive.py" "${evmp_worker_args[@]}" &
else
  "${evmp_python}" "${evmp_args[@]}" &
fi
evmp_child=$!
set +e
while true; do
  wait "${evmp_child}"
  evmp_code=$?
  kill -0 "${evmp_child}" 2>/dev/null || break
done
exit "${evmp_code}"
