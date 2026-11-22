#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && "$1" == /* && "$(uname -m)" == aarch64 ]] || exit 2
evrts_output="$(realpath -e -- "$1")"
evrts_phase="$2"
case "${evrts_phase}" in prepare|benchmark|summary|dispatch|train|evaluate|report|recover_profile) ;; *) exit 2 ;; esac
cd -- "${evrts_output}"
sha256sum --quiet -c snapshot.sha256
evrts_python="$(python3 -B -c 'import json; print(json.load(open("launch.json"))["python"])')"
evrts_bin="$(dirname -- "${evrts_python}")"
export PATH="${evrts_bin}:$PATH" LD_LIBRARY_PATH="${evrts_bin}/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${evrts_output}/source" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
for evrts_name in ${!NCCL_@}; do unset "${evrts_name}"; done
exec "${evrts_python}" -B -u "${evrts_output}/source/scripts/run_part_ev_rts_tigris.py" \
  --phase "${evrts_phase}" --output "${evrts_output}"


