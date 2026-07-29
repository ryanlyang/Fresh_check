#!/usr/bin/env bash
#SBATCH --job-name=retb_stage_i
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_PREDICTOR_BUNDLE_LOCK:?RETB_PREDICTOR_BUNDLE_LOCK is required}"
: "${RETB_STAGE_I_POLICY:?RETB_STAGE_I_POLICY is required}"
: "${RETB_STAGE_I_INPUT:?RETB_STAGE_I_INPUT is required}"
: "${RETB_STAGE_I_CONFIGURATION:?RETB_STAGE_I_CONFIGURATION is required}"
: "${RETB_STAGE_I_FUSION_CHECKPOINT:?RETB_STAGE_I_FUSION_CHECKPOINT is required}"
: "${RETB_STAGE_I_ORACLE_TARGET_CACHE:?RETB_STAGE_I_ORACLE_TARGET_CACHE is required}"
: "${RETB_STAGE_I_OUTPUT:?RETB_STAGE_I_OUTPUT is required}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

python scripts/evaluate_retb_stage_i_substitutions.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --bundle-lock "${RETB_PREDICTOR_BUNDLE_LOCK}" \
  --stage-i-policy "${RETB_STAGE_I_POLICY}" \
  --input-npz "${RETB_STAGE_I_INPUT}" \
  --configuration "${RETB_STAGE_I_CONFIGURATION}" \
  --fusion-checkpoint "${RETB_STAGE_I_FUSION_CHECKPOINT}" \
  --oracle-target-cache "${RETB_STAGE_I_ORACLE_TARGET_CACHE}" \
  --output "${RETB_STAGE_I_OUTPUT}"
