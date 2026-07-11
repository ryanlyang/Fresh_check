#!/usr/bin/env bash
# Audit canonical-state split and HLT v2 strength-2.5 cache inputs.

#SBATCH --job-name=cstate_audit
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
: "${CANONICAL_STATE_MANIFEST_PATH:=${CANONICAL_STATE_ROOT}/inputs/split_manifest.json.gz}"
: "${CANONICAL_STATE_HLT_CACHE_DIR:=${CANONICAL_STATE_ROOT}/inputs/hlt_cache}"
: "${CANONICAL_STATE_AUDIT_DIR:=${CANONICAL_STATE_ROOT}/inputs/audits}"
: "${CANONICAL_STATE_MODEL_TRAIN_SIZE:=5000000}"
: "${CANONICAL_STATE_MODEL_VAL_SIZE:=1000000}"
: "${CANONICAL_STATE_STACK_TRAIN_SIZE:=3000000}"
: "${CANONICAL_STATE_STACK_VAL_SIZE:=1000000}"
: "${CANONICAL_STATE_FINAL_TEST_SIZE:=1000000}"

fresh_setup "$@"
fresh_require_file "${CANONICAL_STATE_MANIFEST_PATH}"
fresh_require_dir "${CANONICAL_STATE_HLT_CACHE_DIR}"
fresh_claim_new_dir "${CANONICAL_STATE_AUDIT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/audit_canonical_state_step1_inputs.py"
  --manifest "${CANONICAL_STATE_MANIFEST_PATH}"
  --hlt-cache-dir "${CANONICAL_STATE_HLT_CACHE_DIR}"
  --output-dir "${CANONICAL_STATE_AUDIT_DIR}"
  --expected-model-train "${CANONICAL_STATE_MODEL_TRAIN_SIZE}"
  --expected-model-val "${CANONICAL_STATE_MODEL_VAL_SIZE}"
  --expected-stack-train "${CANONICAL_STATE_STACK_TRAIN_SIZE}"
  --expected-stack-val "${CANONICAL_STATE_STACK_VAL_SIZE}"
  --expected-final-test "${CANONICAL_STATE_FINAL_TEST_SIZE}"
)

fresh_write_run_config "${CANONICAL_STATE_AUDIT_DIR}" "canonical_state_audit_inputs" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${CANONICAL_STATE_AUDIT_DIR}/canonical_state_step1_input_audit_report.json"
  fresh_require_file "${CANONICAL_STATE_AUDIT_DIR}/canonical_state_step1_input_audit_summary.md"
  fresh_assert_json_ok "${CANONICAL_STATE_AUDIT_DIR}/canonical_state_step1_input_audit_report.json"
fi
