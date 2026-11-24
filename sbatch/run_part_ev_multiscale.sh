#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 2 && $# -le 3 && "$1" == /* ]] || exit 2
evms_output="$(realpath -e -- "$1")"
evms_phase="$2"
evms_arm="${3:-}"
[[ -z "${evms_arm}" || "${evms_arm}" == control || "${evms_arm}" == multiscale ]] || exit 2
case "${evms_phase}" in prepare|benchmark|summary|dispatch|train|evaluate|report) ;; *) exit 2 ;; esac
cd -- "${evms_output}"
sha256sum --quiet -c snapshot.sha256
mapfile -t evms_fields < <(python3 -B -c 'import json; r=json.load(open("launch.json")); print(r["python"]); print(r["resources"]["machine"]); print(r["resources"]["gpus"]); print(r["resources"]["threads"])')
[[ "$(uname -m)" == "${evms_fields[1]}" ]] || exit 2
evms_python="${evms_fields[0]}"
evms_bin="$(dirname -- "${evms_python}")"
export PATH="${evms_bin}:$PATH" LD_LIBRARY_PATH="${evms_bin}/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${evms_output}/source" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${evms_fields[3]}" MKL_NUM_THREADS="${evms_fields[3]}" OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
for evms_name in ${!NCCL_@} ${!TORCH_NCCL_@}; do unset "${evms_name}"; done
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Preserve Slurm's CUDA_VISIBLE_DEVICES; a marker safely reaches both workers.
export EVMS_STOP_FILE="${evms_output}/logs/.stop_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}_$$"
evms_stop() { : > "${EVMS_STOP_FILE}"; }
trap evms_stop USR1 TERM
evms_args=(-B -u "${evms_output}/source/scripts/run_part_ev_multiscale.py" --output "${evms_output}" --phase "${evms_phase}")
evms_worker_args=(--output "${evms_output}" --phase "${evms_phase}")
[[ -z "${evms_arm}" ]] || evms_args+=(--arm "${evms_arm}")
[[ -z "${evms_arm}" ]] || evms_worker_args+=(--arm "${evms_arm}")
if [[ "${evms_fields[2]}" == 2 && ( "${evms_phase}" == benchmark || "${evms_phase}" == train ) ]]; then
  "${evms_python}" -B -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
    "${evms_output}/source/scripts/run_part_ev_multiscale.py" "${evms_worker_args[@]}" &
else
  "${evms_python}" "${evms_args[@]}" &
fi
evms_child=$!
set +e
while true; do
  wait "${evms_child}"
  evms_code=$?
  kill -0 "${evms_child}" 2>/dev/null || break
done
exit "${evms_code}"

