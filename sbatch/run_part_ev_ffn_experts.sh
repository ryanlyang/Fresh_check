#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 2 && $# -le 3 && "$1" == /* ]] || exit 2
evex_output="$(realpath -e -- "$1")"
evex_phase="$2"
evex_arm="${3:-}"
[[ -z "${evex_arm}" || "${evex_arm}" == control || "${evex_arm}" == experts ]] || exit 2
case "${evex_phase}" in prepare|benchmark|summary|dispatch|train|evaluate|report) ;; *) exit 2 ;; esac
cd -- "${evex_output}"
sha256sum --quiet -c snapshot.sha256
mapfile -t evex_fields < <(python3 -B -c 'import json; r=json.load(open("launch.json")); print(r["python"]); print(r["resources"]["machine"]); print(r["resources"]["gpus"]); print(r["resources"]["threads"])')
[[ "$(uname -m)" == "${evex_fields[1]}" ]] || exit 2
evex_python="${evex_fields[0]}"
evex_bin="$(dirname -- "${evex_python}")"
export PATH="${evex_bin}:$PATH" LD_LIBRARY_PATH="${evex_bin}/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${evex_output}/source" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${evex_fields[3]}" MKL_NUM_THREADS="${evex_fields[3]}" OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
for evex_name in ${!NCCL_@} ${!TORCH_NCCL_@}; do unset "${evex_name}"; done
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Preserve Slurm's CUDA_VISIBLE_DEVICES; a marker safely reaches both workers.
export EVEX_STOP_FILE="${evex_output}/logs/.stop_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}_$$"
evex_stop() { : > "${EVEX_STOP_FILE}"; }
trap evex_stop USR1 TERM
evex_args=(-B -u "${evex_output}/source/scripts/run_part_ev_ffn_experts.py" --output "${evex_output}" --phase "${evex_phase}")
evex_worker_args=(--output "${evex_output}" --phase "${evex_phase}")
[[ -z "${evex_arm}" ]] || evex_args+=(--arm "${evex_arm}")
[[ -z "${evex_arm}" ]] || evex_worker_args+=(--arm "${evex_arm}")
if [[ "${evex_fields[2]}" == 2 && ( "${evex_phase}" == benchmark || "${evex_phase}" == train ) ]]; then
  "${evex_python}" -B -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
    "${evex_output}/source/scripts/run_part_ev_ffn_experts.py" "${evex_worker_args[@]}" &
else
  "${evex_python}" "${evex_args[@]}" &
fi
evex_child=$!
set +e
while true; do
  wait "${evex_child}"
  evex_code=$?
  kill -0 "${evex_child}" 2>/dev/null || break
done
exit "${evex_code}"
