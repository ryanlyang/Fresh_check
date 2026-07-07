#!/usr/bin/env bash
# Build one deterministic second-layer HLT cache for the PD10 HLT self-dualview study.

#SBATCH --job-name=pd10_hlt2_cache
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=2-00:00:00
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
fresh_require_file "${PD10_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${PD10_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${PD10_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_claim_new_dir "${HLT2_CACHE_DIR}"

fresh_split_words split_args "${PD10_HLT_SPLITS}"
cmd=(
  "${PYTHON_BIN}" "-u" "scripts/build_pd10_hlt2_cache.py"
  --pd10-root "${PD10_ROOT}"
  --manifest "${PD10_MANIFEST_PATH}"
  --source-hlt-cache-dir "${PD10_HLT_CACHE_DIR}"
  --hlt2-cache-dir "${HLT2_CACHE_DIR}"
  --audit-output-dir "${AUDIT_OUTPUT_DIR}"
  --strength "${STRENGTH}"
  --splits "${split_args[@]}"
  --hlt2-seed "${PD10_HLT_SDV_HLT2_SEED}"
  --max-model-train "${PD10_MODEL_TRAIN_SIZE}"
  --max-model-val "${PD10_MODEL_VAL_SIZE}"
  --max-final-test "${PD10_FINAL_TEST_SIZE}"
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --show-progress "${PD10_HLT_SDV_HLT2_SHOW_PROGRESS}"

fresh_write_run_config "${HLT2_CACHE_DIR}" "pd10_hlt_self_dualview_hlt2_cache_${STRENGTH_TAG}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  for split in "${split_args[@]}"; do
    fresh_require_file "${HLT2_CACHE_DIR}/${split}_fixed_hlt.npz"
    fresh_require_file "${HLT2_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
  done
  fresh_require_file "${AUDIT_OUTPUT_DIR}/hlt2_cache_audit_report.json"
  fresh_assert_json_ok "${AUDIT_OUTPUT_DIR}/hlt2_cache_audit_report.json"
fi
