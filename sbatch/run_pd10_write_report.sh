#!/usr/bin/env bash
# Write the PD10 final report after the student matrix finishes.

#SBATCH --job-name=pd10_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_setup "$@"
fresh_require_dir "${PD10_TEACHERS_DIR}"
fresh_require_dir "${PD10_STUDENTS_DIR}"
fresh_require_dir "${PD10_TEACHER_LOGITS_DIR}"
fresh_require_file "${PD10_STEP2_AUDIT_DIR}/pd10_step2_audit_report.json"
fresh_claim_new_dir "${PD10_FINAL_REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_pd10_report.py"
  --output-dir "${PD10_FINAL_REPORT_DIR}"
  --teachers-dir "${PD10_TEACHERS_DIR}"
  --students-dir "${PD10_STUDENTS_DIR}"
  --teacher-logit-dir "${PD10_TEACHER_LOGITS_DIR}"
  --audit-dir "${PD10_STEP2_AUDIT_DIR}"
  --confirm-final-test
)
if [[ -n "${PD10_REPORT_STUDENT_VARIANTS}" ]]; then
  fresh_split_words report_variant_args "${PD10_REPORT_STUDENT_VARIANTS}"
  cmd+=(--student-variants "${report_variant_args[@]}")
fi
fresh_append_flag_if_enabled cmd --core-only "${PD10_REPORT_CORE_ONLY}"
fresh_append_flag_if_enabled cmd --allow-missing-core-students "${PD10_REPORT_ALLOW_MISSING_CORE_STUDENTS}"
fresh_append_flag_if_enabled cmd --allow-missing-teacher-reports "${PD10_REPORT_ALLOW_MISSING_TEACHER_REPORTS}"
fresh_append_flag_if_enabled cmd --allow-missing-audit "${PD10_REPORT_ALLOW_MISSING_AUDIT}"
fresh_append_flag_if_enabled cmd --skip-prediction-metrics "${PD10_REPORT_SKIP_PREDICTION_METRICS}"

fresh_write_run_config "${PD10_FINAL_REPORT_DIR}" "pd10_final_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/pd10_report.json"
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/pd10_report.md"
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/run_report.json"
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/teacher_metrics.csv"
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/student_metrics.csv"
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/student_core_matrix.csv"
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/warm_start_comparisons.csv"
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/scratch_comparisons.csv"
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/teacher_target_comparison.csv"
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/topk_confidence_ablations.csv"
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/binary_projection_table.csv"
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/gap_closure_table.csv"
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/calibration_table.csv"
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/class_pair_improvements.csv"
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/leakage_audit_summary.csv"
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/v2_comparisons.csv"
  fresh_require_file "${PD10_FINAL_REPORT_DIR}/v2_diagnostics.csv"
  fresh_assert_json_ok "${PD10_FINAL_REPORT_DIR}/run_report.json"
fi
