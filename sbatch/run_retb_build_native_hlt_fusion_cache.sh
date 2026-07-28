#!/usr/bin/env bash
#SBATCH --job-name=retb_native_hlt_cache
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_SPLIT:?RETB_SPLIT is required}"
: "${RETB_PIPELINE_SEED:?RETB_PIPELINE_SEED is required}"
: "${RETB_SHAPE_ID:?RETB_SHAPE_ID is required}"
: "${RETB_REALIZATION_POLICY:?RETB_REALIZATION_POLICY is required}"
: "${RETB_IDENTITY_MANIFEST:?RETB_IDENTITY_MANIFEST is required}"
: "${RETB_LABEL_MANIFEST:?RETB_LABEL_MANIFEST is required}"
: "${RETB_OUTPUT_DIR:?RETB_OUTPUT_DIR is required}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

arguments=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --split "${RETB_SPLIT}"
  --pipeline-seed "${RETB_PIPELINE_SEED}"
  --shape-id "${RETB_SHAPE_ID}"
  --realization-policy "${RETB_REALIZATION_POLICY}"
  --identity-manifest "${RETB_IDENTITY_MANIFEST}"
  --label-manifest "${RETB_LABEL_MANIFEST}"
  --output-dir "${RETB_OUTPUT_DIR}"
)
for expert in BASE4 PT TRACK PID CHARGE DENSITY REGION; do
  registration_variable="RETB_${expert}_REGISTRATION"
  output_manifest_variable="RETB_${expert}_OUTPUT_MANIFEST"
  registration_path="${!registration_variable:-}"
  output_manifest_path="${!output_manifest_variable:-}"
  if [[ -z "${registration_path}" || -z "${output_manifest_path}" ]]; then
    echo "${registration_variable} and ${output_manifest_variable} are required" >&2
    exit 2
  fi
  arguments+=(
    --expert-registration "${expert}=${registration_path}"
    --expert-output-manifest "${expert}=${output_manifest_path}"
  )
  for replica in 0 1 2 3; do
    output_variable="RETB_${expert}_${RETB_SPLIT^^}_R${replica}_OUTPUT"
    if [[ -n "${!output_variable:-}" ]]; then
      arguments+=(--expert-output "${expert}:${replica}=${!output_variable}")
    fi
  done
done
for replica in 0 1 2 3; do
  manifest_variable="RETB_HLT_CACHE_R${replica}_MANIFEST"
  if [[ -n "${!manifest_variable:-}" ]]; then
    arguments+=(--hlt-cache-manifest "${replica}=${!manifest_variable}")
  fi
done
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then
  arguments+=(--dry-run)
fi
python scripts/build_retb_native_hlt_fusion_cache.py "${arguments[@]}"
