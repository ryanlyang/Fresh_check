#!/usr/bin/env bash
#SBATCH --job-name=retb_joint_graph
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_STAGE_J_RUN:?RETB_STAGE_J_RUN is required}"
: "${RETB_PREDICTOR_BUNDLE_LOCK:?RETB_PREDICTOR_BUNDLE_LOCK is required}"
: "${RETB_PREPARED_JOINT_GRAPH:?RETB_PREPARED_JOINT_GRAPH is required}"
: "${RETB_PREPARED_JOINT_GRAPH_SHA256:?RETB_PREPARED_JOINT_GRAPH_SHA256 is required}"
: "${RETB_JOINT_COMPONENT_PARENTS:?RETB_JOINT_COMPONENT_PARENTS is required}"
: "${RETB_JOINT_GRAPH_OUTPUT:?RETB_JOINT_GRAPH_OUTPUT is required}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

python scripts/register_retb_joint_graph_template.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --run "${RETB_STAGE_J_RUN}" \
  --predictor-bundle-lock "${RETB_PREDICTOR_BUNDLE_LOCK}" \
  --prepared-graph "${RETB_PREPARED_JOINT_GRAPH}" \
  --prepared-graph-sha256 "${RETB_PREPARED_JOINT_GRAPH_SHA256}" \
  --component-parents "${RETB_JOINT_COMPONENT_PARENTS}" \
  --output-dir "${RETB_JOINT_GRAPH_OUTPUT}"
