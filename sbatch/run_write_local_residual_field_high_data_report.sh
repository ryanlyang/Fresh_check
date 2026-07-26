#!/usr/bin/env bash
# Aggregate the locked high-data validation or explicitly confirmed final test.

#SBATCH --job-name=lprf_hd_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"

MODE="${1:-validation}"
case "${MODE}" in validation|final_test) ;; *) echo "mode must be validation or final_test" >&2; exit 2 ;; esac
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT:?high-data campaign root is required}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST:=${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/study_manifest.json}"

fresh_setup
fresh_require_file "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST}"
if [[ "${MODE}" == "validation" ]]; then
  OUTPUT_DIR="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/validation_report"
  cmd=(
    "${PYTHON_BIN}" -u scripts/write_local_residual_field_high_data_report.py
    --study-manifest "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST}"
    --output-dir "${OUTPUT_DIR}"
  )
else
  fresh_bool_enabled "${CONFIRM_FINAL_TEST}" || {
    echo "final-test reporting requires CONFIRM_FINAL_TEST=1" >&2
    exit 2
  }
  OUTPUT_DIR="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/final_report"
  cmd=(
    "${PYTHON_BIN}" -u scripts/write_local_residual_field_high_data_report.py
    --study-manifest "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST}"
    --output-dir "${OUTPUT_DIR}"
    --validation-report "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/validation_report/run_report.json"
    --final-test-predictions-root "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/final_test_predictions"
  )
fi
fresh_write_run_config "${OUTPUT_DIR}" "local_residual_field_high_data_${MODE}_report" "${cmd[@]}"
fresh_run "${cmd[@]}"
if ! fresh_is_dry_run; then
  fresh_assert_json_ok "${OUTPUT_DIR}/run_report.json"
fi
