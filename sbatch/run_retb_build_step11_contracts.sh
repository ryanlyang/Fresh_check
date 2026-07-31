#!/usr/bin/env bash
#SBATCH --job-name=retb_step11
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_PREDICTOR_BUNDLE_LOCK:=${CAMPAIGN_ROOT}/selection/predictor_bundle/predictor_bundle_lock.json}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

arguments=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --predictor-bundle-lock "${RETB_PREDICTOR_BUNDLE_LOCK}"
)
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then
  arguments+=(--dry-run)
fi
python scripts/build_retb_step11_contracts.py "${arguments[@]}"
if [[ "${RETB_DRY_RUN:-0}" != "1" ]]; then
  python scripts/attest_retb_direct_node_completion.py \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --node-id step11_joint_bridge_contracts \
    --output-artifact "${CAMPAIGN_ROOT}/registry/retb_step11_joint_bridge_bundle.json" >/dev/null
  python scripts/produce_retb_downstream_manifest_plans.py \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --producer-node-id step11_joint_bridge_contracts >/dev/null
  python scripts/materialize_retb_downstream_manifests.py \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --producer-node-id step11_joint_bridge_contracts >/dev/null
fi
