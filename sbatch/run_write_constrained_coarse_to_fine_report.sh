#!/usr/bin/env bash
# Write the strict final campaign report from completed artifacts.

#SBATCH --job-name=c2f_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${CONSTRAINED_C2F_ROOT:=${OUTPUT_ROOT}/constrained_coarse_to_fine}"
: "${CONSTRAINED_C2F_PREDICTION_DIR:=${CONSTRAINED_C2F_ROOT}/predictions}"
: "${CONSTRAINED_C2F_FUSION_DIR:=${CONSTRAINED_C2F_ROOT}/fusion}"
: "${CONSTRAINED_C2F_REPORT_DIR:=${CONSTRAINED_C2F_ROOT}/final_report}"
: "${CONSTRAINED_C2F_REPORT_RECON_RUN_IDS:=B0 B1 B2 B3 B4 B5 B6 B7 C0 C1 C2 C3 C4 C5 C6 C5-B1 C5-B2 C5-B3}"
: "${CONSTRAINED_C2F_REPORT_TAGGER_RUN_IDS:=A0 D5 D5-B1 D5-B2 D5-B3 D6 D8 D8-seed1 D8-seed2}"
: "${CONSTRAINED_C2F_REQUIRED_FUSION_GROUPS:=F0 F1 F2 F3 F4 F5}"
: "${CONSTRAINED_C2F_REPORT_ALLOW_MISSING_RUNS:=0}"

fresh_setup "$@"
fresh_require_dir "${CONSTRAINED_C2F_PREDICTION_DIR}"
fresh_require_file "${CONSTRAINED_C2F_FUSION_DIR}/fusion_report.json"
fresh_claim_new_dir "${CONSTRAINED_C2F_REPORT_DIR}"
fresh_split_words recon_args "${CONSTRAINED_C2F_REPORT_RECON_RUN_IDS}"
fresh_split_words tagger_args "${CONSTRAINED_C2F_REPORT_TAGGER_RUN_IDS}"
fresh_split_words fusion_args "${CONSTRAINED_C2F_REQUIRED_FUSION_GROUPS}"

cmd=(
  "${PYTHON_BIN}" -u scripts/write_constrained_coarse_to_fine_report.py
  --campaign-root "${CONSTRAINED_C2F_ROOT}"
  --prediction-dir "${CONSTRAINED_C2F_PREDICTION_DIR}"
  --output-dir "${CONSTRAINED_C2F_REPORT_DIR}"
  --fusion-report "${CONSTRAINED_C2F_FUSION_DIR}/fusion_report.json"
  --reconstructor-runs "${recon_args[@]}"
  --tagger-runs "${tagger_args[@]}"
  --required-fusion-groups "${fusion_args[@]}"
)
fresh_append_flag_if_enabled cmd --allow-missing-runs "${CONSTRAINED_C2F_REPORT_ALLOW_MISSING_RUNS}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"

fresh_write_run_config "${CONSTRAINED_C2F_REPORT_DIR}" constrained_c2f_report "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then fresh_require_file "${CONSTRAINED_C2F_REPORT_DIR}/final_report.json"; fi
