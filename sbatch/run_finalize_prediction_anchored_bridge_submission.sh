#!/usr/bin/env bash
#SBATCH --job-name=pab_finalize_submit
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${PAB_SOURCE_BASE:?missing PAB_SOURCE_BASE}"
: "${PAB_PREFLIGHT_ROOT:?missing PAB_PREFLIGHT_ROOT}"
: "${PREDICTION_ANCHORED_ARTIFACT_ROOT:?missing campaign artifact root}"
export PYTHONNOUSERSITE=1
: "${CONDA_ENV:=weaver}"
: "${CONDA_BASE:=/home/ryreu/miniconda3}"
direct_python="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
[[ -x "${direct_python}" ]] || {
  echo "Missing direct conda-environment Python: ${direct_python}" >&2
  exit 2
}
export SKIP_CONDA=1 PYTHON_BIN="${direct_python}"
export CONDA_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

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
  --budget-gib 5

export PREDICTION_ANCHORED_GRAPH="${PAB_PREFLIGHT_ROOT}/prediction_anchored_tigris_graph.json"
export PAB_REGISTRY="${PAB_PREFLIGHT_ROOT}/campaign_registry_step8.json"
export PAB_RESERVATIONS="${PAB_PREFLIGHT_ROOT}/campaign_reservations.json"
export PAB_EXECUTION_SPEC="${PAB_PREFLIGHT_ROOT}/prediction_anchored_execution_spec.json"
export PREDICTION_ANCHORED_EXECUTE=1

# The finalizer submits B0--B6 only after every immutable binding validates.
fresh_run bash sbatch/submit_prediction_anchored_bridge_pilot.sh
