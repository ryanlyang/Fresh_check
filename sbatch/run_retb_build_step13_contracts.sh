#!/usr/bin/env bash
#SBATCH --job-name=retb_step13
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
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
arguments=(--campaign-root "${CAMPAIGN_ROOT}")
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then arguments+=(--dry-run); fi
python scripts/build_retb_step13_contracts.py "${arguments[@]}"
if [[ "${RETB_DRY_RUN:-0}" != "1" ]]; then
  python scripts/emit_retb_stage_l_registration_factory_input.py \
    --campaign-root "${CAMPAIGN_ROOT}"
  python scripts/attest_retb_direct_node_completion.py \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --node-id step13_confirmation_contracts \
    --output-artifact "${CAMPAIGN_ROOT}/registry/retb_step13_confirmation_shortlist_bundle.json" \
    --output-artifact "${CAMPAIGN_ROOT}/job_ledgers/factory_inputs/stage_l_graph_registration.json" >/dev/null
  python scripts/produce_retb_downstream_manifest_plans.py \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --producer-node-id step13_confirmation_contracts >/dev/null
  python scripts/materialize_retb_downstream_manifests.py \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --producer-node-id step13_confirmation_contracts >/dev/null
fi
