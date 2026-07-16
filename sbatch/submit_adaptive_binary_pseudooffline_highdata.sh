#!/usr/bin/env bash
# Queue approved high-data training; the canonical submitter validates the pilot report.
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
[[ "${ABPH_APPROVE_HIGHDATA:-0}" == "1" ]] || { echo "Set ABPH_APPROVE_HIGHDATA=1" >&2; exit 2; }
[[ -n "${ABPH_PILOT_REPORT_PATH:-}" ]] || { echo "Set ABPH_PILOT_REPORT_PATH" >&2; exit 2; }
export ABPH_CAMPAIGN_MODE=highdata ABPH_STAGE_MODE=full CONFIRM_FINAL_TEST=0
exec bash "${PROJECT_DIR}/sbatch/submit_adaptive_binary_pseudooffline.sh" "$@"
