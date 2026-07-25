#!/usr/bin/env bash
# Freeze and validate the reused inputs for the matched A0/P7b seed study.

#SBATCH --job-name=lprf_seed_pre
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"

: "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_CAMPAIGN_ID:?seed-study campaign ID is required}"
: "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT:?seed-study campaign root is required}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT:?source curriculum root is required}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT:?source fusion root is required}"
: "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_MANIFEST:=${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}/study_manifest.json}"

fresh_setup "$@"
cmd=(
  "${PYTHON_BIN}" -u scripts/prepare_local_residual_field_seed_study.py
  --campaign-id "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_CAMPAIGN_ID}"
  --campaign-root "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}"
  --curriculum-root "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT}"
  --fusion-root "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}"
  --output "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_MANIFEST}"
)
fresh_write_run_config "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}/preflight" \
  "local_residual_field_seed_study_preflight" "${cmd[@]}"
fresh_run "${cmd[@]}"
if ! fresh_is_dry_run; then
  fresh_assert_json_ok "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_MANIFEST}"
fi
