#!/usr/bin/env bash
# Freeze both champions for both matched groups.
#SBATCH --job-name=lprf_fuse_select
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ID:?campaign ID required}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_fusion/p7b_seed_control_${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ID}}"
fresh_setup "$@"
cmd=("${PYTHON_BIN}" -u scripts/select_local_residual_field_fusion.py --campaign-id "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ID}"
  --candidates-root "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/candidates"
  --prediction-sources "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/predictions/development_prediction_sources.json"
  --source-artifact-audit "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/source_artifact_audit.json"
  --output-path "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/selection/selected_fusion.json")
fresh_run "${cmd[@]}"
