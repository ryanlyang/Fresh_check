#!/usr/bin/env bash
# This file is executed from the Slurm spool; never derive paths from $0.
set -euo pipefail

: "${RPT_UNIFIED_ROOT:?Set the absolute new SPORC campaign root}"
: "${RPT_UNIFIED_PHASE:?Set build, validate, infer, aggregate, or report}"
: "${CONDA_BASE:=/home/ryreu/miniconda3}"
: "${CONDA_ENV:=atlas_kd_sporc}"
[[ "${RPT_UNIFIED_ROOT}" == /* && -f "${RPT_UNIFIED_ROOT}/unified_campaign.json" ]] || { echo 'Missing absolute SPORC campaign root/manifest' >&2; exit 2; }
[[ "$(uname -m)" == x86_64 ]] || { echo 'SPORC worker requires x86_64; ARM execution is forbidden' >&2; exit 2; }
case "${CONDA_BASE}" in *aarch64*|*arm64*) echo 'ARM Conda prefix is forbidden' >&2; exit 2 ;; esac
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MAX_JOBS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export TORCH_EXTENSIONS_DIR="${RPT_UNIFIED_ROOT}/torch_extensions"
python -c 'import platform,sys; assert platform.machine() == "x86_64"; assert sys.version_info >= (3,10); print("SPORC Python:", sys.executable, sys.version.split()[0])'

source_root="${RPT_UNIFIED_ROOT}/source"
driver="${source_root}/scripts/run_relational_part_sporc_unified.py"
[[ -f "${driver}" ]] || { echo 'Frozen SPORC driver is missing' >&2; exit 2; }
cd -- "${source_root}"
# Ensure imports come from the frozen source, never an inherited Tigris checkout.
export PYTHONPATH="${source_root}"
args=("${RPT_UNIFIED_PHASE}" --output "${RPT_UNIFIED_ROOT}")
case "${RPT_UNIFIED_PHASE}" in
  build|validate|report) ;;
  infer|aggregate)
    : "${SLURM_ARRAY_TASK_ID:?Array phase requires SLURM_ARRAY_TASK_ID}"
    [[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || { echo 'Invalid array index' >&2; exit 2; }
    args+=(--task-index "${SLURM_ARRAY_TASK_ID}") ;;
  *) echo "Unknown SPORC phase: ${RPT_UNIFIED_PHASE}" >&2; exit 2 ;;
esac
exec python -u "${driver}" "${args[@]}"
