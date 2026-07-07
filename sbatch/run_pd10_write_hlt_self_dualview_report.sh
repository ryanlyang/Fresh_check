#!/usr/bin/env bash
# Write the final PD10 HLT self-dualview report.

#SBATCH --job-name=pd10_hlt_sdv_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=04:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_setup "$@"
if ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing HLT-SDV final report without CONFIRM_FINAL_TEST=1." >&2
  exit 2
fi
fresh_claim_new_dir "${PD10_HLT_SDV_FINAL_REPORT_DIR}"

fresh_split_words variant_args "${PD10_HLT_SDV_VARIANTS} ${PD10_HLT_SDV_CONTROL_VARIANTS}"
fresh_split_words strength_args "${PD10_HLT_SDV_STRENGTHS}"
cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_pd10_hlt_self_dualview_report.py"
  --pd10-root "${PD10_ROOT}"
  --output-dir "${PD10_HLT_SDV_FINAL_REPORT_DIR}"
  --sdv-models-dir "${PD10_HLT_SDV_MODELS_DIR}"
  --pd10-teacher-logits-dir "${PD10_TEACHER_LOGITS_DIR}"
  --pd10-students-dir "${PD10_STUDENTS_DIR}"
  --pd10-teachers-dir "${PD10_TEACHERS_DIR}"
  --pd10-final-report-json "${PD10_FINAL_REPORT_DIR}/pd10_report.json"
  --variants "${variant_args[@]}"
  --strengths "${strength_args[@]}"
  --primary-strength "${PD10_HLT_SDV_PRIMARY_STRENGTH}"
  --confirm-final-test
)
fresh_append_flag_if_enabled cmd --skip-prediction-metrics "${PD10_HLT_SDV_REPORT_SKIP_PREDICTION_METRICS}"
fresh_append_flag_if_enabled cmd --allow-missing-sdv-variants "${PD10_HLT_SDV_REPORT_ALLOW_MISSING_SDV_VARIANTS}"
fresh_append_flag_if_enabled cmd --require-anchors "${PD10_HLT_SDV_REPORT_REQUIRE_ANCHORS}"

fresh_write_run_config "${PD10_HLT_SDV_FINAL_REPORT_DIR}" "pd10_hlt_self_dualview_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${PD10_HLT_SDV_FINAL_REPORT_DIR}/summary.json"
  fresh_require_file "${PD10_HLT_SDV_FINAL_REPORT_DIR}/hlt_self_dualview_report.json"
  fresh_require_file "${PD10_HLT_SDV_FINAL_REPORT_DIR}/hlt_self_dualview_report.md"
  fresh_require_file "${PD10_HLT_SDV_FINAL_REPORT_DIR}/run_report.json"
  fresh_assert_json_ok "${PD10_HLT_SDV_FINAL_REPORT_DIR}/run_report.json"
fi
