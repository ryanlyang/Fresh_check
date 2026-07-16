#!/usr/bin/env bash
# Fit/apply one immutable G-tier fusion artifact.

#SBATCH --job-name=abph_fusion
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=16

set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
VARIANT="${1:?Usage: run_adaptive_binary_fusion.sh <G2-G5> <members...> [--apply-split SPLIT]}"; shift
: "${ABPH_ROOT:=${OUTPUT_ROOT}/adaptive_binary_pseudooffline}"
: "${ABPH_FUSION_EXECUTOR:=${PROJECT_DIR}/scripts/fuse_adaptive_binary_pseudooffline.py}"
export PYTHONNOUSERSITE=1
fresh_setup
fresh_require_file "${ABPH_FUSION_EXECUTOR}"
cmd=("${PYTHON_BIN}" -u "${ABPH_FUSION_EXECUTOR}" --variant "${VARIANT}" --campaign-root "${ABPH_ROOT}")
apply_split=""
while (( $# )); do
  if [[ "$1" == "--apply-split" ]]; then
    shift
    (( $# )) || { echo "--apply-split requires a split" >&2; exit 2; }
    apply_split="$1"
    cmd+=(--apply-split "$1" --frozen-artifact "${ABPH_ROOT}/fusion/${VARIANT}/frozen_fusion.json")
  else
    cmd+=(--member "$1")
  fi
  shift
done
if [[ "${apply_split}" == "final_test" ]]; then
  [[ "${ABPH_CONFIRM_FINAL_TEST:-0}" == "1" ]] || { echo "final fusion lacks approval" >&2; exit 2; }
  [[ -n "${ABPH_FINAL_CLAIM_CONTRACT:-}" && -n "${ABPH_SELECTION_REPORT_PATH:-}" ]] \
    || { echo "final fusion lacks frozen claim inputs" >&2; exit 2; }
  "${PYTHON_BIN}" scripts/validate_adaptive_binary_orchestration.py final-claim \
    --path "${ABPH_FINAL_CLAIM_CONTRACT}" --selection-report "${ABPH_SELECTION_REPORT_PATH}" \
    --fusion-artifact "${VARIANT}=${ABPH_ROOT}/fusion/${VARIANT}/frozen_fusion.json"
fi
fresh_run "${cmd[@]}"
