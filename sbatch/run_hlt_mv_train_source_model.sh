#!/usr/bin/env bash
# Train one scratch source ParT model for deployable HLT multiview source/fusion.

#SBATCH --job-name=hlt_mv_src
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=3-00:00:00
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
: "${HLT_MV_RANDOM_HLT_CONTROLS_DIR:=${HLT_MV_ROOT}/hlt_random_seed_controls}"
: "${HLT_MV_CANONICAL_HLT_SOURCE_NAME:=hlt_part_seed8801}"
: "${HLT_MV_SOURCE_OUTPUT_DIR:=}"
: "${HLT_MV_SOURCE_CACHE_DIR:=}"
: "${HLT_MV_SOURCE_VIEW:=}"
: "${HLT_MV_SOURCE_EPOCHS:=10}"
: "${HLT_MV_SOURCE_BATCH_SIZE:=128}"
: "${HLT_MV_SOURCE_EVAL_BATCH_SIZE:=128}"
: "${HLT_MV_SOURCE_LR:=0.0003}"
: "${HLT_MV_SOURCE_WEIGHT_DECAY:=0.0001}"
: "${HLT_MV_SOURCE_EARLY_STOP_PATIENCE:=3}"
: "${HLT_MV_SOURCE_GRAD_CLIP_NORM:=1.0}"
: "${HLT_MV_SOURCE_NUM_WORKERS:=${NUM_WORKERS}}"
: "${HLT_MV_SOURCE_DEVICE:=${DEVICE}}"
: "${HLT_MV_SOURCE_MODEL_SIZE:=base}"
: "${HLT_MV_SOURCE_AMP:=0}"
: "${HLT_MV_SOURCE_COMPILE_MODEL:=0}"
: "${HLT_MV_SOURCE_SKIP_MODEL_VAL_PREDICTIONS:=0}"
: "${HLT_MV_SOURCE_SKIP_FINAL_TEST:=0}"
: "${HLT_MV_SOURCE_MAX_TRAIN_BATCHES:=}"
: "${HLT_MV_SOURCE_MAX_VAL_BATCHES:=}"
: "${HLT_MV_SOURCE_MAX_FINAL_TEST_BATCHES:=}"
: "${HLT_MV_SOURCE_TRAIN_SIZE:=5000000}"
: "${HLT_MV_SOURCE_VAL_SIZE:=1000000}"
: "${HLT_MV_SOURCE_FINAL_TEST_SIZE:=1000000}"

SOURCE_NAME="${1:?source name is required, e.g. hlt_part_seed8801 or hlt2_part_s0p35_seed8831}"
SOURCE_VIEW_INFERRED=""
CACHE_DIR_INFERRED=""
OUTPUT_DIR_INFERRED=""
SOURCE_SEED=""

if [[ "${SOURCE_NAME}" =~ ^hlt_part_seed([0-9]+)$ ]]; then
  SOURCE_VIEW_INFERRED="fixed_hlt"
  CACHE_DIR_INFERRED="${HLT_MV_HLT_CACHE_DIR}"
  SOURCE_SEED="${BASH_REMATCH[1]}"
  if [[ "${SOURCE_NAME}" == "${HLT_MV_CANONICAL_HLT_SOURCE_NAME}" ]]; then
    OUTPUT_DIR_INFERRED="${HLT_MV_SOURCE_MODELS_DIR}/${SOURCE_NAME}"
  else
    OUTPUT_DIR_INFERRED="${HLT_MV_RANDOM_HLT_CONTROLS_DIR}/${SOURCE_NAME}"
  fi
elif [[ "${SOURCE_NAME}" =~ ^hlt2_part_(s[0-9]+p[0-9]+)_seed([0-9]+)$ ]]; then
  SOURCE_VIEW_INFERRED="hlt2"
  CACHE_DIR_INFERRED="${HLT_MV_HLT2_CACHE_ROOT}/hlt_second_degrade_mild_v1_${BASH_REMATCH[1]}"
  OUTPUT_DIR_INFERRED="${HLT_MV_SOURCE_MODELS_DIR}/${SOURCE_NAME}"
  SOURCE_SEED="${BASH_REMATCH[2]}"
else
  echo "Unknown HLT-MV source model: ${SOURCE_NAME}" >&2
  exit 2
fi

SOURCE_VIEW="${HLT_MV_SOURCE_VIEW:-${SOURCE_VIEW_INFERRED}}"
CACHE_DIR="${HLT_MV_SOURCE_CACHE_DIR:-${CACHE_DIR_INFERRED}}"
OUTPUT_DIR="${HLT_MV_SOURCE_OUTPUT_DIR:-${OUTPUT_DIR_INFERRED}}"

fresh_setup "$@"
fresh_require_dir "${CACHE_DIR}"
fresh_require_file "${CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${CACHE_DIR}/final_test_fixed_hlt_metadata.json"
if ! fresh_bool_enabled "${HLT_MV_SOURCE_SKIP_FINAL_TEST}" && ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing HLT-MV source final-test evaluation without CONFIRM_FINAL_TEST=1." >&2
  exit 2
fi
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_hlt_mv_source_model.py"
  --output-root "${OUTPUT_ROOT}"
  --pdv3-experiment-name "${HLT_MV_PDV3_EXPERIMENT_NAME}"
  --source-name "${SOURCE_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --cache-dir "${CACHE_DIR}"
  --source-view "${SOURCE_VIEW}"
  --epochs "${HLT_MV_SOURCE_EPOCHS}"
  --batch-size "${HLT_MV_SOURCE_BATCH_SIZE}"
  --eval-batch-size "${HLT_MV_SOURCE_EVAL_BATCH_SIZE}"
  --lr "${HLT_MV_SOURCE_LR}"
  --weight-decay "${HLT_MV_SOURCE_WEIGHT_DECAY}"
  --early-stop-patience "${HLT_MV_SOURCE_EARLY_STOP_PATIENCE}"
  --grad-clip-norm "${HLT_MV_SOURCE_GRAD_CLIP_NORM}"
  --num-workers "${HLT_MV_SOURCE_NUM_WORKERS}"
  --device "${HLT_MV_SOURCE_DEVICE}"
  --seed "${SOURCE_SEED}"
  --model-size "${HLT_MV_SOURCE_MODEL_SIZE}"
  --max-train-jets "${HLT_MV_SOURCE_TRAIN_SIZE}"
  --max-val-jets "${HLT_MV_SOURCE_VAL_SIZE}"
  --max-final-test-jets "${HLT_MV_SOURCE_FINAL_TEST_SIZE}"
)
fresh_append_optional_arg cmd --max-train-batches "${HLT_MV_SOURCE_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${HLT_MV_SOURCE_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${HLT_MV_SOURCE_MAX_FINAL_TEST_BATCHES}"
fresh_append_flag_if_enabled cmd --amp "${HLT_MV_SOURCE_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${HLT_MV_SOURCE_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-model-val-predictions "${HLT_MV_SOURCE_SKIP_MODEL_VAL_PREDICTIONS}"
fresh_append_flag_if_enabled cmd --skip-final-test "${HLT_MV_SOURCE_SKIP_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"

fresh_write_run_config "${OUTPUT_DIR}" "hlt_mv_source_${SOURCE_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_assert_json_ok "${OUTPUT_DIR}/run_report.json"
  if ! fresh_bool_enabled "${HLT_MV_SOURCE_SKIP_MODEL_VAL_PREDICTIONS}"; then
    fresh_require_file "${OUTPUT_DIR}/predictions/${SOURCE_NAME}/model_val_predictions.npz"
    fresh_require_file "${OUTPUT_DIR}/predictions/${SOURCE_NAME}/model_val_predictions_metadata.json"
  fi
  if ! fresh_bool_enabled "${HLT_MV_SOURCE_SKIP_FINAL_TEST}"; then
    fresh_require_file "${OUTPUT_DIR}/final_test_report.json"
    fresh_require_file "${OUTPUT_DIR}/predictions/${SOURCE_NAME}/final_test_predictions.npz"
    fresh_require_file "${OUTPUT_DIR}/predictions/${SOURCE_NAME}/final_test_predictions_metadata.json"
  fi
fi
