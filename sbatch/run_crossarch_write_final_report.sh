#!/usr/bin/env bash
# Write a compact cross-architecture final report from fusion JSON outputs.

#SBATCH --job-name=crossarch_final
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_setup "$@"
fresh_require_file "scripts/write_crossarch_final_report.py"
fresh_require_file "${CROSSARCH_FUSION_DIR}/fusion_report.json"
fresh_claim_new_dir "${CROSSARCH_FINAL_REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_crossarch_final_report.py"
  --fusion-report "${CROSSARCH_FUSION_DIR}/fusion_report.json"
  --output-dir "${CROSSARCH_FINAL_REPORT_DIR}"
  --root-dir "${CROSSARCH_ROOT}"
  --prediction-dir "${CROSSARCH_PREDICTION_DIR}"
)

fresh_write_run_config "${CROSSARCH_FINAL_REPORT_DIR}" "crossarch_final_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${CROSSARCH_FINAL_REPORT_DIR}/crossarch_final_report.json"
  fresh_require_file "${CROSSARCH_FINAL_REPORT_DIR}/crossarch_final_report.md"
  fresh_assert_json_ok "${CROSSARCH_FINAL_REPORT_DIR}/crossarch_final_report.json"
fi
