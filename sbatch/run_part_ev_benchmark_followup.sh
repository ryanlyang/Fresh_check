#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 && "$1" == /* && "$(uname -m)" == x86_64 ]] || exit 2
evfollow_output="$(realpath -e -- "$1")"
cd -- "$evfollow_output"
sha256sum --quiet -c snapshot.sha256
evfollow_python="$(python3 -B -c 'import json; print(json.load(open("followup.json"))["benchmark_identity"]["python"])')"
evfollow_bin="$(dirname -- "$evfollow_python")"
export PATH="$evfollow_bin:$PATH" LD_LIBRARY_PATH="$evfollow_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$evfollow_output/source" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
exec "$evfollow_python" -B -u "$evfollow_output/source/scripts/run_part_ev_benchmark_followup.py" --output "$evfollow_output"
