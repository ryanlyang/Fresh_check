#!/usr/bin/env bash
#SBATCH --job-name=hltv2_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${HLT_V2_BASELINE_SWEEP_ROOT:=${OUTPUT_ROOT}/hlt_v2_baseline_sweep}"
: "${HLT_V2_BASELINE_SWEEP_REPORT_DIR:=${HLT_V2_BASELINE_SWEEP_ROOT}/baseline_sweep_report}"
: "${HLT_V2_BASELINE_SWEEP_STRENGTHS:=0.0 0.75 1.0 1.25}"

fresh_setup "$@"

fresh_split_words strength_args "${HLT_V2_BASELINE_SWEEP_STRENGTHS}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_hlt_v2_baseline_sweep_report.py"
  --sweep-root "${HLT_V2_BASELINE_SWEEP_ROOT}"
  --output-dir "${HLT_V2_BASELINE_SWEEP_REPORT_DIR}"
  --strengths "${strength_args[@]}"
)

fresh_write_run_config "${HLT_V2_BASELINE_SWEEP_REPORT_DIR}" "hlt_v2_baseline_sweep_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${HLT_V2_BASELINE_SWEEP_REPORT_DIR}/hlt_v2_baseline_sweep_model_val.json"
  fresh_require_file "${HLT_V2_BASELINE_SWEEP_REPORT_DIR}/hlt_v2_baseline_sweep_model_val.csv"
  fresh_require_file "${HLT_V2_BASELINE_SWEEP_REPORT_DIR}/hlt_v2_baseline_sweep_model_val.md"
fi
