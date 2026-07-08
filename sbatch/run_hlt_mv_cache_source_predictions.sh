#!/usr/bin/env bash
# Cache model-val/final-test predictions for one trained HLT-MV source model.

#SBATCH --job-name=hlt_mv_pred
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=12:00:00
#SBATCH --mem=120G
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
: "${HLT_MV_RANDOM_HLT_CONTROLS_DIR:=${HLT_MV_ROOT}/hlt_random_seed_controls}"
: "${HLT_MV_SOURCE_OUTPUT_DIR:=}"
: "${HLT_MV_SOURCE_CACHE_DIR:=}"
: "${HLT_MV_SOURCE_VIEW:=}"
: "${HLT_MV_SOURCE_PREDICTION_SPLITS:=model_val final_test}"
: "${HLT_MV_SOURCE_EVAL_BATCH_SIZE:=128}"
: "${HLT_MV_SOURCE_NUM_WORKERS:=${NUM_WORKERS}}"
: "${HLT_MV_SOURCE_DEVICE:=${DEVICE}}"
: "${HLT_MV_SOURCE_MAX_VAL_BATCHES:=}"
: "${HLT_MV_SOURCE_MAX_FINAL_TEST_BATCHES:=}"
: "${HLT_MV_SOURCE_VAL_SIZE:=1000000}"
: "${HLT_MV_SOURCE_FINAL_TEST_SIZE:=1000000}"

SOURCE_NAME="${1:?source name is required, e.g. hlt_part_seed8801 or hlt2_part_s0p35_seed8831}"
SOURCE_VIEW_INFERRED=""
CACHE_DIR_INFERRED=""
OUTPUT_DIR_INFERRED=""
SOURCE_SEED=""

case "${SOURCE_NAME}" in
  hlt_part_seed8801)
    SOURCE_VIEW_INFERRED="fixed_hlt"
    CACHE_DIR_INFERRED="${HLT_MV_HLT_CACHE_DIR}"
    OUTPUT_DIR_INFERRED="${HLT_MV_SOURCE_MODELS_DIR}/${SOURCE_NAME}"
    SOURCE_SEED="8801"
    ;;
  hlt_part_seed9101|hlt_part_seed9102|hlt_part_seed9103|hlt_part_seed9104)
    SOURCE_VIEW_INFERRED="fixed_hlt"
    CACHE_DIR_INFERRED="${HLT_MV_HLT_CACHE_DIR}"
    OUTPUT_DIR_INFERRED="${HLT_MV_RANDOM_HLT_CONTROLS_DIR}/${SOURCE_NAME}"
    SOURCE_SEED="${SOURCE_NAME#hlt_part_seed}"
    ;;
  hlt2_part_s0p10_seed8811)
    SOURCE_VIEW_INFERRED="hlt2"
    CACHE_DIR_INFERRED="${HLT_MV_HLT2_CACHE_ROOT}/hlt_second_degrade_mild_v1_s0p10"
    OUTPUT_DIR_INFERRED="${HLT_MV_SOURCE_MODELS_DIR}/${SOURCE_NAME}"
    SOURCE_SEED="8811"
    ;;
  hlt2_part_s0p20_seed8821)
    SOURCE_VIEW_INFERRED="hlt2"
    CACHE_DIR_INFERRED="${HLT_MV_HLT2_CACHE_ROOT}/hlt_second_degrade_mild_v1_s0p20"
    OUTPUT_DIR_INFERRED="${HLT_MV_SOURCE_MODELS_DIR}/${SOURCE_NAME}"
    SOURCE_SEED="8821"
    ;;
  hlt2_part_s0p35_seed8831)
    SOURCE_VIEW_INFERRED="hlt2"
    CACHE_DIR_INFERRED="${HLT_MV_HLT2_CACHE_ROOT}/hlt_second_degrade_mild_v1_s0p35"
    OUTPUT_DIR_INFERRED="${HLT_MV_SOURCE_MODELS_DIR}/${SOURCE_NAME}"
    SOURCE_SEED="8831"
    ;;
  hlt2_part_s1p00_seed8841)
    SOURCE_VIEW_INFERRED="hlt2"
    CACHE_DIR_INFERRED="${HLT_MV_HLT2_CACHE_ROOT}/hlt_second_degrade_mild_v1_s1p00"
    OUTPUT_DIR_INFERRED="${HLT_MV_SOURCE_MODELS_DIR}/${SOURCE_NAME}"
    SOURCE_SEED="8841"
    ;;
  *)
    echo "Unknown HLT-MV source model: ${SOURCE_NAME}" >&2
    exit 2
    ;;
esac

SOURCE_VIEW="${HLT_MV_SOURCE_VIEW:-${SOURCE_VIEW_INFERRED}}"
CACHE_DIR="${HLT_MV_SOURCE_CACHE_DIR:-${CACHE_DIR_INFERRED}}"
OUTPUT_DIR="${HLT_MV_SOURCE_OUTPUT_DIR:-${OUTPUT_DIR_INFERRED}}"
CACHE_RUN_DIR="${OUTPUT_DIR}/prediction_cache_job"

fresh_setup "$@"
fresh_require_dir "${OUTPUT_DIR}"
fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
fresh_require_file "${OUTPUT_DIR}/run_report.json"
fresh_assert_json_ok "${OUTPUT_DIR}/run_report.json"
fresh_require_dir "${CACHE_DIR}"
fresh_require_file "${CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${CACHE_DIR}/final_test_fixed_hlt_metadata.json"
if [[ "${HLT_MV_SOURCE_PREDICTION_SPLITS}" == *"final_test"* ]] && ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing HLT-MV source final-test prediction caching without CONFIRM_FINAL_TEST=1." >&2
  exit 2
fi

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/cache_hlt_mv_source_predictions.py"
  --output-root "${OUTPUT_ROOT}"
  --pdv3-experiment-name "${HLT_MV_PDV3_EXPERIMENT_NAME}"
  --source-name "${SOURCE_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --cache-dir "${CACHE_DIR}"
  --source-view "${SOURCE_VIEW}"
  --splits
)
prediction_splits=()
fresh_split_words prediction_splits "${HLT_MV_SOURCE_PREDICTION_SPLITS}"
cmd+=("${prediction_splits[@]}")
cmd+=(
  --eval-batch-size "${HLT_MV_SOURCE_EVAL_BATCH_SIZE}"
  --num-workers "${HLT_MV_SOURCE_NUM_WORKERS}"
  --device "${HLT_MV_SOURCE_DEVICE}"
  --seed "${SOURCE_SEED}"
  --max-val-jets "${HLT_MV_SOURCE_VAL_SIZE}"
  --max-final-test-jets "${HLT_MV_SOURCE_FINAL_TEST_SIZE}"
)
fresh_append_optional_arg cmd --max-val-batches "${HLT_MV_SOURCE_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${HLT_MV_SOURCE_MAX_FINAL_TEST_BATCHES}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"

fresh_write_run_config "${CACHE_RUN_DIR}" "hlt_mv_source_predictions_${SOURCE_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/prediction_cache_report.json"
  fresh_assert_json_ok "${OUTPUT_DIR}/prediction_cache_report.json"
  for split in "${prediction_splits[@]}"; do
    fresh_require_file "${OUTPUT_DIR}/predictions/${SOURCE_NAME}/${split}_predictions.npz"
    fresh_require_file "${OUTPUT_DIR}/predictions/${SOURCE_NAME}/${split}_predictions_metadata.json"
  done
fi
