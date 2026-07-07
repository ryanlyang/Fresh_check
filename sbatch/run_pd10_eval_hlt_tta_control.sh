#!/usr/bin/env bash
# Evaluate the deployable PD10 HLT/HLT2 logit-average TTA control.

#SBATCH --job-name=pd10_hlt_tta
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=12:00:00
#SBATCH --mem=220G
#SBATCH --gres=gpu:1
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
VARIANT_NAME="${2:-tta_hlt_part_hlt_plus_hlt2_${STRENGTH_TAG}}"
OUTPUT_DIR="${PD10_HLT_SDV_MODELS_DIR}/${VARIANT_NAME}"
HLT2_CACHE_DIR="$(fresh_pd10_hlt_sdv_hlt2_cache_dir "${STRENGTH}")"

fresh_setup "$@"
fresh_require_dir "${PD10_HLT_CACHE_DIR}"
fresh_require_file "${PD10_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${PD10_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_dir "${HLT2_CACHE_DIR}"
fresh_require_file "${HLT2_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${HLT2_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_file "${PD10_HLT_SDV_HLT_TEACHER_CHECKPOINT}"
if ! fresh_bool_enabled "${PD10_HLT_TTA_SKIP_FINAL_TEST}" && ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing HLT TTA final-test evaluation without CONFIRM_FINAL_TEST=1." >&2
  exit 2
fi
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/evaluate_pd10_hlt_tta_control.py"
  --pd10-root "${PD10_ROOT}"
  --strength "${STRENGTH}"
  --variant "${VARIANT_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --hlt-cache-dir "${PD10_HLT_CACHE_DIR}"
  --hlt2-cache-dir "${HLT2_CACHE_DIR}"
  --hlt-teacher-checkpoint "${PD10_HLT_SDV_HLT_TEACHER_CHECKPOINT}"
  --batch-size "${PD10_HLT_TTA_BATCH_SIZE}"
  --num-workers "${PD10_HLT_TTA_NUM_WORKERS}"
  --device "${PD10_HLT_TTA_DEVICE}"
  --seed "${PD10_HLT_TTA_SEED}"
  --max-val-jets "${PD10_MODEL_VAL_SIZE}"
  --max-final-test-jets "${PD10_FINAL_TEST_SIZE}"
)
fresh_append_optional_arg cmd --max-val-batches "${PD10_HLT_TTA_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${PD10_HLT_TTA_MAX_FINAL_TEST_BATCHES}"
fresh_append_flag_if_enabled cmd --skip-final-test "${PD10_HLT_TTA_SKIP_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"

fresh_write_run_config "${OUTPUT_DIR}" "pd10_hlt_self_dualview_${VARIANT_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_assert_json_ok "${OUTPUT_DIR}/run_report.json"
  if ! fresh_bool_enabled "${PD10_HLT_TTA_SKIP_FINAL_TEST}"; then
    fresh_require_file "${OUTPUT_DIR}/final_test_report.json"
  fi
fi
