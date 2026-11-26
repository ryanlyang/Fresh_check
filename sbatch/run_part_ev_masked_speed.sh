#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && "$1" == /* ]] || exit 2
speed_root="$(realpath -e -- "$1")"
speed_phase="$2"
case "${speed_phase}" in benchmark|summary) ;; *) exit 2 ;; esac
cd -- "${speed_root}"
sha256sum --quiet -c snapshot.sha256
speed_python="$(python3 -B -c 'import json; print(json.load(open("launch.json"))["python"])')"
speed_bin="$(dirname -- "${speed_python}")"
export PATH="${speed_bin}:$PATH" LD_LIBRARY_PATH="${speed_bin}/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${speed_root}/source" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
for speed_name in ${!NCCL_@} ${!TORCH_NCCL_@}; do unset "${speed_name}"; done
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export EVMP_STOP_FILE="${speed_root}/logs/.stop_${SLURM_JOB_ID:-manual}_$$"
speed_stop() { : > "${EVMP_STOP_FILE}"; }
trap speed_stop USR1 TERM
"${speed_python}" -B -u "${speed_root}/source/scripts/run_part_ev_masked_speed.py" \
  --phase "${speed_phase}" --output "${speed_root}" &
speed_child=$!
set +e
while true; do
  wait "${speed_child}"
  speed_code=$?
  kill -0 "${speed_child}" 2>/dev/null || break
done
exit "${speed_code}"
