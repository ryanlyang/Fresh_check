#!/usr/bin/env bash
# Fit one locked group/candidate pair; final-test is inaccessible to this job.
#SBATCH --job-name=lprf_fuse_fit
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=1-00:00:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"
GROUP_ID="${1:?Usage: run_fit_local_residual_field_fusion_candidate.sh <F_method|F_seed> <candidate> [screening|stability]}"
CANDIDATE_ID="${2:?candidate ID required}"
PHASE="${3:-screening}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ID:?LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ID is required}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_fusion/p7b_seed_control_${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ID}}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT:=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/source_artifact_audit.json}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_SOURCES:=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/predictions/development_prediction_sources.json}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_DIR:=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/representations}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_STABILITY_PLAN:=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/selection/stability_plan.json}"
OUTPUT_DIR="${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/candidates/${GROUP_ID}/${CANDIDATE_ID}"
fresh_setup "$@"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_SOURCES}"
cmd=("${PYTHON_BIN}" -u scripts/run_local_residual_field_fusion_campaign.py
  --campaign-id "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ID}" --group-id "${GROUP_ID}"
  --candidate-id "${CANDIDATE_ID}" --output-dir "${OUTPUT_DIR}"
  --prediction-sources "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_SOURCES}"
  --source-artifact-audit "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}"
  --feature-root "${LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_DIR}" --phase "${PHASE}" --device "${DEVICE}")
if [[ "${PHASE}" == "stability" ]]; then
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_FUSION_STABILITY_PLAN}"
  cmd+=(--stability-plan "${LOCAL_RESIDUAL_FIELD_FUSION_STABILITY_PLAN}")
fi
fresh_run "${cmd[@]}"
