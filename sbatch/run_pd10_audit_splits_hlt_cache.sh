#!/usr/bin/env bash
#SBATCH --job-name=pd10_audit_splits
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=08:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_setup "$@"
fresh_require_file "${PD10_MANIFEST_PATH}"
fresh_require_dir "${PD10_HLT_CACHE_DIR}"
fresh_claim_new_dir "${PD10_STEP2_AUDIT_DIR}"

cmd=(
  "${PYTHON_BIN}" "scripts/audit_pd10_step2_splits_hlt_cache.py"
  --manifest "${PD10_MANIFEST_PATH}"
  --hlt-cache-dir "${PD10_HLT_CACHE_DIR}"
  --output-dir "${PD10_STEP2_AUDIT_DIR}"
  --expected-model-train "${PD10_MODEL_TRAIN_SIZE}"
  --expected-model-val "${PD10_MODEL_VAL_SIZE}"
  --expected-final-test "${PD10_FINAL_TEST_SIZE}"
  --expected-stack-train "${PD10_STACK_TRAIN_SIZE}"
  --expected-stack-val "${PD10_STACK_VAL_SIZE}"
)

fresh_write_run_config "${PD10_STEP2_AUDIT_DIR}" "pd10_audit_splits_hlt_cache" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${PD10_STEP2_AUDIT_DIR}/pd10_step2_audit_report.json"
  fresh_assert_json_ok "${PD10_STEP2_AUDIT_DIR}/pd10_step2_audit_report.json"
fi
