#!/usr/bin/env bash
#SBATCH --job-name=fresh_final_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${SUBSTANTIAL_ACCURACY_DELTA:=0.01}"
: "${ALLOW_MISSING_FINAL_INPUTS:=0}"
: "${ALLOW_CROSS_ENTROPY_WORSE:=0}"

fresh_setup "$@"
fresh_require_file "${RECO7_AUDIT_DIR}/audit_report.json"
fresh_assert_json_ok "${RECO7_AUDIT_DIR}/audit_report.json"
if [[ -f "${HLT5_AUDIT_DIR}/audit_report.json" ]]; then
  fresh_assert_json_ok "${HLT5_AUDIT_DIR}/audit_report.json"
fi
fresh_refuse_existing_dir "${FINAL_REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" "scripts/write_final_report.py"
  --output-dir "${FINAL_REPORT_DIR}"
  --hlt-baseline-report "${HLT_BASELINE_DIR}/model_val_report.json"
  --offline-teacher-report "${OFFLINE_TEACHER_DIR}/model_val_report.json"
  --reco7-fusion-report "${RECO7_FUSION_DIR}/fusion_report.json"
  --hlt5-fusion-report "${HLT5_FUSION_DIR}/fusion_report.json"
  --reco7-audit-report "${RECO7_AUDIT_DIR}/audit_report.json"
  --hlt5-audit-report "${HLT5_AUDIT_DIR}/audit_report.json"
  --substantial-accuracy-delta "${SUBSTANTIAL_ACCURACY_DELTA}"
)
fresh_append_flag_if_enabled cmd --allow-missing "${ALLOW_MISSING_FINAL_INPUTS}"
fresh_append_flag_if_enabled cmd --allow-cross-entropy-worse "${ALLOW_CROSS_ENTROPY_WORSE}"

fresh_write_run_config "${FINAL_REPORT_DIR}" "write_final_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${FINAL_REPORT_DIR}/FINAL_FRESH_START_REPORT.md"
  fresh_require_file "${FINAL_REPORT_DIR}/final_report_summary.json"
fi

