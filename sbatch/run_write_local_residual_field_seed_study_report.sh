#!/usr/bin/env bash
# Aggregate model_val and stack_val metrics for the five matched pairs.

#SBATCH --job-name=lprf_seed_report
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

: "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT:?seed-study campaign root is required}"
: "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_MANIFEST:=${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}/study_manifest.json}"
REPORT_DIR="${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}/final_report"

fresh_setup "$@"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_MANIFEST}"
cmd=(
  "${PYTHON_BIN}" -u scripts/write_local_residual_field_seed_study_report.py
  --study-manifest "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_MANIFEST}"
  --output-dir "${REPORT_DIR}"
)
fresh_write_run_config "${REPORT_DIR}" "local_residual_field_seed_study_report" "${cmd[@]}"
fresh_run "${cmd[@]}"
if ! fresh_is_dry_run; then
  fresh_assert_json_ok "${REPORT_DIR}/run_report.json"
  fresh_require_file "${REPORT_DIR}/seed_metrics.csv"
fi
