#!/usr/bin/env bash
#SBATCH --job-name=pab_hd_manifest
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=24

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${PAB_HIGH_DATA_PARENT_MANIFEST:?missing high-data parent manifest}"
: "${PAB_HIGH_DATA_PREFLIGHT_ROOT:?missing high-data preflight root}"
: "${PAB_HIGH_DATA_DATA_DIR:?missing high-data source data directory}"
: "${PAB_HIGH_DATA_SHARD_EVENTS:=100000}"
: "${PAB_CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${PAB_CONDA_ENV:=atlas_kd_tigris}"

export CONDA_BASE="${PAB_CONDA_BASE}"
export CONDA_ENV="${PAB_CONDA_ENV}"
export PYTHONNOUSERSITE=1

source "${PROJECT_DIR}/sbatch/common.sh"
fresh_setup
cd "${PROJECT_DIR}"

fresh_require_file "${PAB_HIGH_DATA_PARENT_MANIFEST}"
fresh_refuse_existing_path "${PAB_HIGH_DATA_PREFLIGHT_ROOT}"

fresh_run "${PYTHON_BIN}" -u \
  scripts/prepare_prediction_anchored_high_data_manifest.py \
  --parent-manifest "${PAB_HIGH_DATA_PARENT_MANIFEST}" \
  --output-dir "${PAB_HIGH_DATA_PREFLIGHT_ROOT}" \
  --source-data-dir "${PAB_HIGH_DATA_DATA_DIR}" \
  --shard-events "${PAB_HIGH_DATA_SHARD_EVENTS}"

