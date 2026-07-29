#!/usr/bin/env bash
#SBATCH --job-name=retb_capacity
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_DEPLOYMENT_MANIFEST:?RETB_DEPLOYMENT_MANIFEST is required}"
: "${RETB_COMPLETE_GRAPH_ID:?RETB_COMPLETE_GRAPH_ID is required}"
: "${RETB_CAPACITY_COMPONENTS:?RETB_CAPACITY_COMPONENTS is required}"
: "${RETB_CAPACITY_COMPONENTS_SHA256:?RETB_CAPACITY_COMPONENTS_SHA256 is required}"
: "${RETB_ANALYTICAL_FLOPS:?RETB_ANALYTICAL_FLOPS is required}"
: "${RETB_MEASURED_DIAGNOSTICS:?RETB_MEASURED_DIAGNOSTICS is required}"
: "${RETB_CAPACITY_OUTPUT:?RETB_CAPACITY_OUTPUT is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
python scripts/attest_retb_complete_graph_capacity.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --deployment-manifest "${RETB_DEPLOYMENT_MANIFEST}" \
  --graph-id "${RETB_COMPLETE_GRAPH_ID}" \
  --prepared-components "${RETB_CAPACITY_COMPONENTS}" \
  --prepared-components-sha256 "${RETB_CAPACITY_COMPONENTS_SHA256}" \
  --analytical-flops "${RETB_ANALYTICAL_FLOPS}" \
  --measured-diagnostics "${RETB_MEASURED_DIAGNOSTICS}" \
  --output "${RETB_CAPACITY_OUTPUT}"
