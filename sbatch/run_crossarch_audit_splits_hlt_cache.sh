#!/usr/bin/env bash
#SBATCH --job-name=crossarch_audit_splits
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=06:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_setup "$@"
fresh_require_file "${CROSSARCH_MANIFEST_PATH}"
fresh_require_dir "${CROSSARCH_HLT_CACHE_DIR}"
fresh_claim_new_dir "${CROSSARCH_STEP2_AUDIT_DIR}"

cmd=(
  "${PYTHON_BIN}" "scripts/audit_crossarch_step2_splits_hlt_cache.py"
  --manifest "${CROSSARCH_MANIFEST_PATH}"
  --hlt-cache-dir "${CROSSARCH_HLT_CACHE_DIR}"
  --output-dir "${CROSSARCH_STEP2_AUDIT_DIR}"
)

fresh_write_run_config "${CROSSARCH_STEP2_AUDIT_DIR}" "crossarch_audit_splits_hlt_cache" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${CROSSARCH_STEP2_AUDIT_DIR}/crossarch_step2_audit_report.json"
  fresh_assert_json_ok "${CROSSARCH_STEP2_AUDIT_DIR}/crossarch_step2_audit_report.json"
fi
