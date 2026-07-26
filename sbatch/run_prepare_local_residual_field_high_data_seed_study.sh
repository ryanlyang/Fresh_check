#!/usr/bin/env bash
# Audit shared 3M sources and freeze the high-data matched-seed manifest.

#SBATCH --job-name=lprf_hd_pre
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

: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_CAMPAIGN_ID:?high-data campaign ID is required}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT:?high-data campaign root is required}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_REFERENCE_ROOT:?low-data P7b reference root is required}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST:=${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/study_manifest.json}"

fresh_setup "$@"
cmd=(
  "${PYTHON_BIN}" -u scripts/prepare_local_residual_field_high_data_seed_study.py
  --campaign-id "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_CAMPAIGN_ID}"
  --campaign-root "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}"
  --reference-curriculum-root "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_REFERENCE_ROOT}"
  --output "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST}"
)
fresh_write_run_config "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/preflight" \
  "local_residual_field_high_data_seed_study_preflight" "${cmd[@]}"
fresh_run "${cmd[@]}"
if ! fresh_is_dry_run; then
  fresh_assert_json_ok "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST}"
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/selected_consumer.json"
fi
