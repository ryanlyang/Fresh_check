#!/usr/bin/env bash
#SBATCH --job-name=retb_predictor
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_PREDICTOR_RUN:?RETB_PREDICTOR_RUN is required}"
: "${RETB_PREDICTOR_MODEL_TRAIN:?RETB_PREDICTOR_MODEL_TRAIN is required}"
: "${RETB_PREDICTOR_VAL_STOP:?RETB_PREDICTOR_VAL_STOP is required}"
: "${RETB_TARGET_NORMALIZER:?RETB_TARGET_NORMALIZER is required}"
: "${RETB_TARGET_CHECKPOINT:?RETB_TARGET_CHECKPOINT is required}"
: "${RETB_FUSION_CHECKPOINT:?RETB_FUSION_CHECKPOINT is required}"
: "${RETB_PREDICTOR_OUTPUT:?RETB_PREDICTOR_OUTPUT is required}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

arguments=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --run "${RETB_PREDICTOR_RUN}"
  --model-train "${RETB_PREDICTOR_MODEL_TRAIN}"
  --val-stop "${RETB_PREDICTOR_VAL_STOP}"
  --target-normalizer "${RETB_TARGET_NORMALIZER}"
  --target-checkpoint "${RETB_TARGET_CHECKPOINT}"
  --fusion-checkpoint "${RETB_FUSION_CHECKPOINT}"
  --fusion-variant "${RETB_FUSION_VARIANT:-F_TOKEN_TRANSFORMER}"
  --output-dir "${RETB_PREDICTOR_OUTPUT}"
  --microbatch-size "${RETB_PREDICTOR_MICROBATCH_SIZE:-256}"
  --gradient-accumulation-steps "${RETB_PREDICTOR_ACCUMULATION_STEPS:-1}"
)
if [[ -n "${RETB_PREDICTOR_VAL_DESIGN:-}" ]]; then
  : "${RETB_PREDICTOR_VAL_DESIGN_OUTPUT:?RETB_PREDICTOR_VAL_DESIGN_OUTPUT is required}"
  : "${RETB_PREDICTOR_CALIBRATION_OUTPUT:?RETB_PREDICTOR_CALIBRATION_OUTPUT is required}"
  arguments+=(
    --val-design "${RETB_PREDICTOR_VAL_DESIGN}"
    --val-design-output "${RETB_PREDICTOR_VAL_DESIGN_OUTPUT}"
    --calibration-output "${RETB_PREDICTOR_CALIBRATION_OUTPUT}"
  )
fi
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then
  arguments+=(--dry-run)
fi
python scripts/train_retb_predictor.py "${arguments[@]}"
