#!/usr/bin/env bash
#SBATCH --job-name=pab_finalize_submit
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${PAB_SOURCE_BASE:?missing PAB_SOURCE_BASE}"
: "${PAB_PREFLIGHT_ROOT:?missing PAB_PREFLIGHT_ROOT}"
: "${PREDICTION_ANCHORED_ARTIFACT_ROOT:?missing campaign artifact root}"
: "${PAB_SPLIT_PROFILE:=pilot_250k}"
: "${PAB_BUDGET_GIB:=5}"
: "${PAB_RECON_PHASE2_EPOCHS:=40}"
export PYTHONNOUSERSITE=1
: "${PAB_CONDA_ENV:=atlas_kd_tigris}"
: "${PAB_CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
export CONDA_ENV="${PAB_CONDA_ENV}"
export CONDA_BASE="${PAB_CONDA_BASE}"

source "${PROJECT_DIR}/sbatch/common.sh"
fresh_setup
cd "${PROJECT_DIR}"

parent="${PAB_SOURCE_BASE}/inputs/split_manifest/split_manifest.json.gz"
hlt_cache="${PAB_SOURCE_BASE}/inputs/hlt_cache"
offline_cache="${PAB_SOURCE_BASE}/inputs/offline_cache"
baseline="${PAB_SOURCE_BASE}/taggers/A0/best_model_val.pt"

for path in "${parent}" "${baseline}" \
  "${hlt_cache}/stack_train_fixed_hlt.npz" \
  "${offline_cache}/stack_train_offline.npz"; do
  fresh_require_file "${path}"
done

fresh_run "${PYTHON_BIN}" -u scripts/bootstrap_prediction_anchored_bridge_preflight.py \
  --parent-manifest "${parent}" \
  --hlt-cache-dir "${hlt_cache}" \
  --offline-cache-dir "${offline_cache}" \
  --baseline-checkpoint "${baseline}" \
  --output-dir "${PAB_PREFLIGHT_ROOT}" \
  --artifact-root "${PREDICTION_ANCHORED_ARTIFACT_ROOT}" \
  --budget-gib "${PAB_BUDGET_GIB}" \
  --split-profile "${PAB_SPLIT_PROFILE}"

export PREDICTION_ANCHORED_GRAPH="${PAB_PREFLIGHT_ROOT}/prediction_anchored_tigris_graph.json"
export PAB_REGISTRY="${PAB_PREFLIGHT_ROOT}/campaign_registry_step8.json"
export PAB_RESERVATIONS="${PAB_PREFLIGHT_ROOT}/campaign_reservations.json"
export PAB_EXECUTION_SPEC="${PAB_PREFLIGHT_ROOT}/prediction_anchored_execution_spec.json"
export PAB_REPRESENTATIVE_RESOURCE_REFERENCE="${PAB_PREFLIGHT_ROOT}/representative_architecture_resource_reference.json"
export PREDICTION_ANCHORED_EXECUTE=1
export PAB_SPLIT_PROFILE PAB_RECON_PHASE2_EPOCHS

# The finalizer submits B0--B6 only after every immutable binding validates.
fresh_run bash sbatch/submit_prediction_anchored_bridge_pilot.sh
