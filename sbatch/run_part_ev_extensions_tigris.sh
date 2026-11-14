#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && "$1" == /* && "$(uname -m)" == aarch64 ]] || exit 2
evext_output="$(realpath -e -- "$1")"
evext_phase="$2"
case "${evext_phase}" in prepare|benchmark|summary) ;; *) exit 2 ;; esac
cd -- "${evext_output}"
sha256sum --quiet -c snapshot.sha256
evext_python="$(python3 -B -c 'import json; print(json.load(open("launch.json"))["python"])')"
evext_bin="$(dirname -- "${evext_python}")"
export PATH="${evext_bin}:$PATH" LD_LIBRARY_PATH="${evext_bin}/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${evext_output}/source" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
for evext_name in ${!NCCL_@}; do unset "${evext_name}"; done
exec "${evext_python}" -B -u "${evext_output}/source/scripts/run_part_ev_extensions_tigris.py" \
  --phase "${evext_phase}" --output "${evext_output}"
