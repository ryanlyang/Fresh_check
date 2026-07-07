#!/usr/bin/env bash
# Audit one deterministic second-layer HLT cache for the PD10 HLT self-dualview study.

#SBATCH --job-name=pd10_hlt2_audit
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=08:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

STRENGTH="${1:-${PD10_HLT_SDV_PRIMARY_STRENGTH}}"
STRENGTH_TAG="$(fresh_pd10_hlt_sdv_strength_tag "${STRENGTH}")"
HLT2_CACHE_DIR="$(fresh_pd10_hlt_sdv_hlt2_cache_dir "${STRENGTH}")"
AUDIT_OUTPUT_DIR="$(fresh_pd10_hlt_sdv_hlt2_audit_dir "${STRENGTH}")"

fresh_setup "$@"
fresh_require_file "${PD10_MANIFEST_PATH}"
fresh_require_dir "${PD10_HLT_CACHE_DIR}"
fresh_require_dir "${HLT2_CACHE_DIR}"
fresh_require_file "${HLT2_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${HLT2_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${HLT2_CACHE_DIR}/final_test_fixed_hlt_metadata.json"

fresh_split_words split_args "${PD10_HLT_SPLITS}"
cmd=(
  "${PYTHON_BIN}" "-u" "scripts/audit_pd10_hlt_self_dualview_inputs.py"
  --pd10-root "${PD10_ROOT}"
  --manifest "${PD10_MANIFEST_PATH}"
  --source-hlt-cache-dir "${PD10_HLT_CACHE_DIR}"
  --hlt2-cache-dir "${HLT2_CACHE_DIR}"
  --output-dir "${AUDIT_OUTPUT_DIR}"
  --strength "${STRENGTH}"
  --splits "${split_args[@]}"
  --expected-model-train "${PD10_MODEL_TRAIN_SIZE}"
  --expected-model-val "${PD10_MODEL_VAL_SIZE}"
  --expected-final-test "${PD10_FINAL_TEST_SIZE}"
)

fresh_write_run_config "${AUDIT_OUTPUT_DIR}" "pd10_hlt_self_dualview_hlt2_audit_${STRENGTH_TAG}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${AUDIT_OUTPUT_DIR}/hlt2_cache_audit_report.json"
  fresh_assert_json_ok "${AUDIT_OUTPUT_DIR}/hlt2_cache_audit_report.json"
fi
