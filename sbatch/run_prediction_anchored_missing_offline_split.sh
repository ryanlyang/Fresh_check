#!/usr/bin/env bash
#SBATCH --job-name=pab_stack_train_offline
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --time=1-00:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${PAB_SOURCE_BASE:?missing PAB_SOURCE_BASE}"
export PYTHONNOUSERSITE=1
: "${CONDA_ENV:=weaver}"
: "${CONDA_BASE:=/home/ryreu/miniconda3}"
direct_python="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
[[ -x "${direct_python}" ]] || {
  echo "Missing direct conda-environment Python: ${direct_python}" >&2
  exit 2
}
export SKIP_CONDA=1 PYTHON_BIN="${direct_python}"
export CONDA_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
source "${PROJECT_DIR}/sbatch/common.sh"
fresh_setup
cd "${PROJECT_DIR}"

manifest="${PAB_SOURCE_BASE}/inputs/split_manifest/split_manifest.json.gz"
output="${PAB_SOURCE_BASE}/inputs/offline_cache"
fresh_require_file "${manifest}"
fresh_require_dir "${output}"
[[ ! -e "${output}/stack_train_offline.npz" ]] || {
  echo "stack_train offline cache already exists; refusing replacement" >&2
  exit 2
}
fresh_run "${PYTHON_BIN}" -u scripts/cache_architecture_view_offline_inputs.py \
  --manifest-path "${manifest}" --output-dir "${output}" \
  --splits stack_train --read-chunk-size 50000
fresh_require_file "${output}/stack_train_offline.npz"
fresh_require_file "${output}/stack_train_offline_metadata.json"
