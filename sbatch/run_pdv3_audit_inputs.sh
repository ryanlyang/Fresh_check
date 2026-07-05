#!/usr/bin/env bash
#SBATCH --job-name=pdv3_audit_inputs
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
fresh_require_file "${PDV3_MANIFEST_PATH}"
fresh_require_dir "${PDV3_HLT_CACHE_DIR}"
fresh_require_dir "${PDV3_OFFLINE_CACHE_DIR}"
fresh_claim_new_dir "${PDV3_STEP1_AUDIT_DIR}"

cmd=(
  "${PYTHON_BIN}" "scripts/audit_pdv3_step1_inputs.py"
  --manifest "${PDV3_MANIFEST_PATH}"
  --hlt-cache-dir "${PDV3_HLT_CACHE_DIR}"
  --offline-cache-dir "${PDV3_OFFLINE_CACHE_DIR}"
  --output-dir "${PDV3_STEP1_AUDIT_DIR}"
  --expected-model-train "${PDV3_MODEL_TRAIN_SIZE}"
  --expected-model-val "${PDV3_MODEL_VAL_SIZE}"
  --expected-final-test "${PDV3_FINAL_TEST_SIZE}"
  --expected-stack-train "${PDV3_STACK_TRAIN_SIZE}"
  --expected-stack-val "${PDV3_STACK_VAL_SIZE}"
)

fresh_write_run_config "${PDV3_STEP1_AUDIT_DIR}" "pdv3_audit_inputs" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${PDV3_STEP1_AUDIT_DIR}/pdv3_step1_input_audit_report.json"
  fresh_assert_json_ok "${PDV3_STEP1_AUDIT_DIR}/pdv3_step1_input_audit_report.json"
fi
