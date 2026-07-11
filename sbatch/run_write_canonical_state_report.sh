#!/usr/bin/env bash
# Write the canonical-state final metrics report.

#SBATCH --job-name=cstate_report
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

: "${CANONICAL_STATE_ROOT:=${OUTPUT_ROOT}/canonical_multi_scale_jet_state}"
: "${CANONICAL_STATE_RUN_ROOT:=${CANONICAL_STATE_ROOT}/runs}"
: "${CANONICAL_STATE_REPORT_DIR:=${CANONICAL_STATE_ROOT}/final_report}"
: "${CANONICAL_STATE_REPORT_RUN_IDS:=A0 A1 A2 A3 B0 B1 B2 B3 C0 C1 C2 C3 C4 C5 C6 D0 D1 D2 D3 D4 D5 E0 E1 E2 E3 E4 E5 E6 F0 F1 F2 F3 F4 Fseed Fshuffle G0 G1 G2 G3}"
: "${CANONICAL_STATE_BASELINE_RUN_ID:=A0}"
: "${CANONICAL_STATE_REPORT_ALLOW_MISSING_RUNS:=0}"
: "${CANONICAL_STATE_REPORT_REQUIRE_ALL_RUNS:=1}"

fresh_setup "$@"
fresh_require_file "scripts/write_canonical_state_report.py"
fresh_split_words run_id_args "${CANONICAL_STATE_REPORT_RUN_IDS}"
if ! fresh_bool_enabled "${CANONICAL_STATE_REPORT_ALLOW_MISSING_RUNS}"; then
  for run_id in "${run_id_args[@]}"; do
    fresh_require_file "${CANONICAL_STATE_RUN_ROOT}/${run_id}/run_report.json"
  done
fi
fresh_claim_new_dir "${CANONICAL_STATE_REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_canonical_state_report.py"
  --output-dir "${CANONICAL_STATE_REPORT_DIR}"
  --run-root "${CANONICAL_STATE_RUN_ROOT}"
  --run-ids "${run_id_args[@]}"
  --baseline-run-id "${CANONICAL_STATE_BASELINE_RUN_ID}"
)
fresh_append_flag_if_enabled cmd --allow-missing-runs "${CANONICAL_STATE_REPORT_ALLOW_MISSING_RUNS}"
if ! fresh_bool_enabled "${CANONICAL_STATE_REPORT_REQUIRE_ALL_RUNS}"; then
  cmd+=(--no-require-all-runs)
fi
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"

fresh_write_run_config "${CANONICAL_STATE_REPORT_DIR}" "canonical_state_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${CANONICAL_STATE_REPORT_DIR}/canonical_state_report.json"
  fresh_require_file "${CANONICAL_STATE_REPORT_DIR}/canonical_state_report.md"
  fresh_require_file "${CANONICAL_STATE_REPORT_DIR}/tagging_metrics.csv"
  fresh_require_file "${CANONICAL_STATE_REPORT_DIR}/fusion_comparison.csv"
  fresh_assert_json_ok "${CANONICAL_STATE_REPORT_DIR}/canonical_state_report.json"
fi
