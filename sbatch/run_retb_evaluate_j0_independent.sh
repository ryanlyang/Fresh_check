#!/usr/bin/env bash
#SBATCH --job-name=retb_j0_eval
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
: "${RETB_STAGE_J_RUN:?RETB_STAGE_J_RUN is required}"
: "${RETB_PREDICTOR_BUNDLE_LOCK:?RETB_PREDICTOR_BUNDLE_LOCK is required}"
: "${RETB_JOINT_GRAPH_TEMPLATE:?RETB_JOINT_GRAPH_TEMPLATE is required}"
: "${RETB_JOINT_VAL_DESIGN_CACHE:?RETB_JOINT_VAL_DESIGN_CACHE is required}"
: "${RETB_JOINT_OUTPUT:?RETB_JOINT_OUTPUT is required}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

python scripts/evaluate_retb_j0_independent.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --run "${RETB_STAGE_J_RUN}" \
  --predictor-bundle-lock "${RETB_PREDICTOR_BUNDLE_LOCK}" \
  --graph-template "${RETB_JOINT_GRAPH_TEMPLATE}" \
  --val-design-cache "${RETB_JOINT_VAL_DESIGN_CACHE}" \
  --output-dir "${RETB_JOINT_OUTPUT}" \
  --batch-size "${RETB_JOINT_EVAL_BATCH_SIZE:-128}"
