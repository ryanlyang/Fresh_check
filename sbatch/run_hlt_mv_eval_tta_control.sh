#!/usr/bin/env bash
# Evaluate one HLT-MV HLT+HLT2 logit-average TTA control.

#SBATCH --job-name=hlt_mv_tta
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

: "${HLT_MV_PDV3_EXPERIMENT_NAME:=privileged_distill_v3_av10_adapter_fixed_hlt_v2_realistic_s1p0_highdata_20260705_190747}"
: "${HLT_MV_PDV3_ROOT:=${OUTPUT_ROOT}/${HLT_MV_PDV3_EXPERIMENT_NAME}}"
: "${HLT_MV_ROOT:=${HLT_MV_PDV3_ROOT}/hlt_multiview_source_fusion}"
: "${HLT_MV_HLT_CACHE_DIR:=${HLT_MV_PDV3_ROOT}/inputs/hlt_cache}"
: "${HLT_MV_HLT2_CACHE_ROOT:=${HLT_MV_PDV3_ROOT}/hlt_self_dualview/hlt2_cache}"
: "${HLT_MV_SOURCE_MODELS_DIR:=${HLT_MV_ROOT}/source_models}"
: "${HLT_MV_CONTROLS_DIR:=${HLT_MV_ROOT}/controls}"
: "${HLT_MV_CANONICAL_HLT_SOURCE_NAME:=hlt_part_seed8801}"
: "${HLT_MV_TTA_HLT_CHECKPOINT:=${HLT_MV_SOURCE_MODELS_DIR}/${HLT_MV_CANONICAL_HLT_SOURCE_NAME}/best_model_val.pt}"
: "${HLT_MV_TTA_OUTPUT_DIR:=}"
: "${HLT_MV_TTA_SEED:=8821}"
: "${HLT_MV_TTA_BATCH_SIZE:=128}"
: "${HLT_MV_TTA_NUM_WORKERS:=${NUM_WORKERS}}"
: "${HLT_MV_TTA_DEVICE:=${DEVICE}}"
: "${HLT_MV_TTA_SKIP_FINAL_TEST:=0}"
: "${HLT_MV_TTA_MAX_VAL_BATCHES:=}"
: "${HLT_MV_TTA_MAX_FINAL_TEST_BATCHES:=}"
: "${HLT_MV_TTA_VAL_SIZE:=1000000}"
: "${HLT_MV_TTA_FINAL_TEST_SIZE:=1000000}"

STRENGTH="${1:?strength is required, e.g. 0.20}"
STRENGTH_TAG="$(fresh_pd10_hlt_sdv_strength_tag "${STRENGTH}")"
VARIANT_NAME="${2:-tta_hlt_part_hlt_plus_hlt2_${STRENGTH_TAG}}"
HLT2_CACHE_DIR="${HLT_MV_HLT2_CACHE_ROOT}/hlt_second_degrade_mild_v1_${STRENGTH_TAG}"
OUTPUT_DIR="${HLT_MV_TTA_OUTPUT_DIR:-${HLT_MV_CONTROLS_DIR}/${VARIANT_NAME}}"

fresh_setup "$@"
fresh_require_dir "${HLT_MV_HLT_CACHE_DIR}"
fresh_require_file "${HLT_MV_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${HLT_MV_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_dir "${HLT2_CACHE_DIR}"
fresh_require_file "${HLT2_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${HLT2_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_file "${HLT_MV_TTA_HLT_CHECKPOINT}"
if ! fresh_bool_enabled "${HLT_MV_TTA_SKIP_FINAL_TEST}" && ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing HLT-MV TTA final-test evaluation without CONFIRM_FINAL_TEST=1." >&2
  exit 2
fi
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/evaluate_pd10_hlt_tta_control.py"
  --pd10-root "${HLT_MV_PDV3_ROOT}"
  --strength "${STRENGTH}"
  --variant "${VARIANT_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --hlt-cache-dir "${HLT_MV_HLT_CACHE_DIR}"
  --hlt2-cache-dir "${HLT2_CACHE_DIR}"
  --hlt-teacher-checkpoint "${HLT_MV_TTA_HLT_CHECKPOINT}"
  --batch-size "${HLT_MV_TTA_BATCH_SIZE}"
  --num-workers "${HLT_MV_TTA_NUM_WORKERS}"
  --device "${HLT_MV_TTA_DEVICE}"
  --seed "${HLT_MV_TTA_SEED}"
  --max-val-jets "${HLT_MV_TTA_VAL_SIZE}"
  --max-final-test-jets "${HLT_MV_TTA_FINAL_TEST_SIZE}"
)
fresh_append_optional_arg cmd --max-val-batches "${HLT_MV_TTA_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${HLT_MV_TTA_MAX_FINAL_TEST_BATCHES}"
fresh_append_flag_if_enabled cmd --skip-final-test "${HLT_MV_TTA_SKIP_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"

fresh_write_run_config "${OUTPUT_DIR}" "hlt_mv_tta_control_${VARIANT_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_assert_json_ok "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/predictions/${VARIANT_NAME}/model_val_predictions.npz"
  fresh_require_file "${OUTPUT_DIR}/predictions/${VARIANT_NAME}/model_val_predictions_metadata.json"
  if ! fresh_bool_enabled "${HLT_MV_TTA_SKIP_FINAL_TEST}"; then
    fresh_require_file "${OUTPUT_DIR}/final_test_report.json"
    fresh_require_file "${OUTPUT_DIR}/predictions/${VARIANT_NAME}/final_test_predictions.npz"
    fresh_require_file "${OUTPUT_DIR}/predictions/${VARIANT_NAME}/final_test_predictions_metadata.json"
  fi
fi
