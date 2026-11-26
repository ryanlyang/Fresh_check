#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && "$1" == /* ]] || exit 2
fast_root="$(realpath -e -- "$1")"
fast_phase="$2"
case "${fast_phase}" in preflight|summary|dispatch|train|evaluate|report) ;; *) exit 2 ;; esac
cd -- "${fast_root}"
sha256sum --quiet -c snapshot.sha256
fast_python="$(python3 -B -c 'import json; print(json.load(open("launch.json"))["python"])')"
fast_bin="$(dirname -- "${fast_python}")"
export PATH="${fast_bin}:$PATH" LD_LIBRARY_PATH="${fast_bin}/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${fast_root}/source" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
for fast_name in ${!NCCL_@} ${!TORCH_NCCL_@}; do unset "${fast_name}"; done
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export EVMP_STOP_FILE="${fast_root}/logs/.stop_${SLURM_JOB_ID:-manual}_$$"
fast_stop() { : > "${EVMP_STOP_FILE}"; }
trap fast_stop USR1 TERM
"${fast_python}" -B -u "${fast_root}/source/scripts/run_part_ev_masked_fast_production.py" \
  --phase "${fast_phase}" --output "${fast_root}" &
fast_child=$!
set +e
while true; do
  wait "${fast_child}"
  fast_code=$?
  kill -0 "${fast_child}" 2>/dev/null || break
done
exit "${fast_code}"
