#!/usr/bin/env bash
# Write strict model-selection or final-claim campaign report.

#SBATCH --job-name=abph_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
MODE="${1:?Usage: run_adaptive_binary_report.sh <selection|final_claim>}"
: "${ABPH_ROOT:=${OUTPUT_ROOT}/adaptive_binary_pseudooffline}"
: "${ABPH_STORAGE_PROFILE:=cache_heavy_v1}"
export PYTHONNOUSERSITE=1
fresh_setup
cmd=("${PYTHON_BIN}" -u scripts/write_adaptive_binary_pseudooffline_report.py --campaign-root "${ABPH_ROOT}")
destination_dir="${ABPH_ROOT}/report"
if [[ "${MODE}" == "final_claim" ]]; then
  [[ -n "${ABPH_FINAL_CLAIM_CONTRACT:-}" ]] || { echo "final report lacks frozen claim contract" >&2; exit 2; }
  [[ -n "${ABPH_SELECTION_REPORT_PATH:-}" ]] || { echo "final report lacks frozen selection report" >&2; exit 2; }
  destination_dir="${ABPH_ROOT}/final_claim_report"
  cmd+=(--confirm-final-test --output-dir "${destination_dir}"
    --selection-report "${ABPH_SELECTION_REPORT_PATH}"
    --final-claim-contract "${ABPH_FINAL_CLAIM_CONTRACT}")
else
  cmd+=(--output-dir "${destination_dir}")
fi
if [[ "${ABPH_STORAGE_PROFILE}" == "streaming_30gb_v1" ]]; then
  source "${PROJECT_DIR}/sbatch/adaptive_binary_ram_workspace.sh"
  abph_setup_ram_workspace
  staged_report="${ABPH_RAM_WORKSPACE}/$(basename "${destination_dir}")"
  for ((index=0; index<${#cmd[@]}; index++)); do
    if [[ "${cmd[index]}" == "--output-dir" ]]; then
      cmd[index+1]="${staged_report}"
      break
    fi
  done
  cmd+=(--logical-output-dir "${destination_dir}")
fi
fresh_run "${cmd[@]}"
if [[ "${ABPH_STORAGE_PROFILE}" == "streaming_30gb_v1" ]]; then
  publish_cmd=("${PYTHON_BIN}" -u scripts/publish_adaptive_binary_quota_tree.py
    --campaign-root "${ABPH_ROOT}"
    --source-dir "${staged_report}"
    --destination-dir "${destination_dir}"
    --artifact-role "campaign_report"
    --run-id "report-${MODE}-${SLURM_JOB_ID:-local}")
  fresh_append_flag_if_enabled publish_cmd --overwrite "${OVERWRITE:-0}"
  fresh_run "${publish_cmd[@]}"
fi
