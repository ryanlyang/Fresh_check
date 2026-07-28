#!/usr/bin/env bash
#SBATCH --job-name=retb_native_hlt_expert
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_RUN_ID:?RETB_RUN_ID is required}"
: "${RETB_TRAIN_LABELS:?RETB_TRAIN_LABELS is required}"
: "${RETB_VAL_STOP_LABELS:?RETB_VAL_STOP_LABELS is required}"
: "${RETB_OFFLINE_REGISTRATION:?RETB_OFFLINE_REGISTRATION is required}"
: "${RETB_OFFLINE_CHECKPOINT:?RETB_OFFLINE_CHECKPOINT is required}"
: "${RETB_MICROBATCH_SIZE:=64}"
: "${RETB_GRADIENT_ACCUMULATION_STEPS:=2}"
: "${RETB_DEVICE:=auto}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

arguments=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --run-id "${RETB_RUN_ID}"
  --train-labels "${RETB_TRAIN_LABELS}"
  --val-stop-labels "${RETB_VAL_STOP_LABELS}"
  --offline-registration "${RETB_OFFLINE_REGISTRATION}"
  --offline-checkpoint "${RETB_OFFLINE_CHECKPOINT}"
  --microbatch-size "${RETB_MICROBATCH_SIZE}"
  --gradient-accumulation-steps "${RETB_GRADIENT_ACCUMULATION_STEPS}"
  --device "${RETB_DEVICE}"
)
if [[ -n "${RETB_CONFIRMATION_REGISTRY:-}" ]]; then
  arguments+=(--confirmation-registry "${RETB_CONFIRMATION_REGISTRY}")
fi
for replica in 0 1 2 3; do
  variable="RETB_TRAIN_CACHE_R${replica}"
  if [[ -n "${!variable:-}" ]]; then
    arguments+=(--train-cache "${replica}=${!variable}")
  fi
  variable="RETB_VAL_STOP_CACHE_R${replica}"
  if [[ -n "${!variable:-}" ]]; then
    arguments+=(--val-stop-cache "${replica}=${!variable}")
  fi
done
optional=(
  RETB_UNIFORM_SHAPES --uniform-shapes
  RETB_HETEROGENEOUS_SHAPES --heterogeneous-shapes
  RETB_OFFLINE_TRAIN_TARGETS --offline-train-targets
  RETB_RELATION_NORMALIZATION --relation-normalization
  RETB_REGION_NORMALIZATION --region-normalization
  RETB_REGION_TREE_ROOT --region-tree-root
  RETB_OUTPUT_DIR --output-dir
)
for ((index=0; index<${#optional[@]}; index+=2)); do
  variable="${optional[index]}"
  if [[ -n "${!variable:-}" ]]; then
    arguments+=("${optional[index+1]}" "${!variable}")
  fi
done
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then
  arguments+=(--dry-run)
fi
python scripts/train_retb_native_hlt_expert.py "${arguments[@]}"
