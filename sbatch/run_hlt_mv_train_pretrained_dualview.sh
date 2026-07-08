#!/usr/bin/env bash
# Train one pretrained-branch particle dual-view model for HLT multiview fusion.

#SBATCH --job-name=hlt_mv_pdv
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
: "${HLT_MV_PRETRAINED_DUALVIEW_DIR:=${HLT_MV_ROOT}/particle_dualview_pretrained}"
: "${HLT_MV_CANONICAL_HLT_SOURCE_NAME:=hlt_part_seed8801}"
: "${HLT_MV_SOURCE_NAMES:=hlt_part_seed8801 hlt2_part_s0p10_seed8811 hlt2_part_s0p20_seed8821 hlt2_part_s0p35_seed8831 hlt2_part_s1p00_seed8841}"
: "${HLT_MV_PRETRAINED_DUALVIEW_OUTPUT_DIR:=}"
: "${HLT_MV_PRETRAINED_DUALVIEW_HLT_CHECKPOINT:=}"
: "${HLT_MV_PRETRAINED_DUALVIEW_HLT2_CHECKPOINT:=}"
: "${HLT_MV_PRETRAINED_DUALVIEW_EPOCHS:=10}"
: "${HLT_MV_PRETRAINED_DUALVIEW_HEAD_WARMUP_EPOCHS:=1}"
: "${HLT_MV_PRETRAINED_DUALVIEW_BATCH_SIZE:=128}"
: "${HLT_MV_PRETRAINED_DUALVIEW_EVAL_BATCH_SIZE:=128}"
: "${HLT_MV_PRETRAINED_DUALVIEW_HEAD_WARMUP_LR:=0.0003}"
: "${HLT_MV_PRETRAINED_DUALVIEW_BRANCH_LR:=0.00003}"
: "${HLT_MV_PRETRAINED_DUALVIEW_HEAD_LR:=0.0003}"
: "${HLT_MV_PRETRAINED_DUALVIEW_WEIGHT_DECAY:=0.0001}"
: "${HLT_MV_PRETRAINED_DUALVIEW_DROPOUT:=0.05}"
: "${HLT_MV_PRETRAINED_DUALVIEW_FUSION_HIDDEN_DIM:=512}"
: "${HLT_MV_PRETRAINED_DUALVIEW_REPRESENTATION_DIM:=256}"
: "${HLT_MV_PRETRAINED_DUALVIEW_EARLY_STOP_PATIENCE:=3}"
: "${HLT_MV_PRETRAINED_DUALVIEW_GRAD_CLIP_NORM:=1.0}"
: "${HLT_MV_PRETRAINED_DUALVIEW_NUM_WORKERS:=${NUM_WORKERS}}"
: "${HLT_MV_PRETRAINED_DUALVIEW_DEVICE:=${DEVICE}}"
: "${HLT_MV_PRETRAINED_DUALVIEW_SEED:=8801}"
: "${HLT_MV_PRETRAINED_DUALVIEW_MODEL_SIZE:=base}"
: "${HLT_MV_PRETRAINED_DUALVIEW_AMP:=0}"
: "${HLT_MV_PRETRAINED_DUALVIEW_COMPILE_MODEL:=0}"
: "${HLT_MV_PRETRAINED_DUALVIEW_SKIP_MODEL_VAL_PREDICTIONS:=0}"
: "${HLT_MV_PRETRAINED_DUALVIEW_SKIP_FINAL_TEST:=0}"
: "${HLT_MV_PRETRAINED_DUALVIEW_MAX_TRAIN_BATCHES:=}"
: "${HLT_MV_PRETRAINED_DUALVIEW_MAX_VAL_BATCHES:=}"
: "${HLT_MV_PRETRAINED_DUALVIEW_MAX_FINAL_TEST_BATCHES:=}"
: "${HLT_MV_PRETRAINED_DUALVIEW_TRAIN_SIZE:=5000000}"
: "${HLT_MV_PRETRAINED_DUALVIEW_VAL_SIZE:=1000000}"
: "${HLT_MV_PRETRAINED_DUALVIEW_FINAL_TEST_SIZE:=1000000}"

VARIANT_NAME="${1:?variant is required, e.g. sdv_hlt_hlt2_s0p20}"
HLT_CHECKPOINT_INFERRED="${HLT_MV_SOURCE_MODELS_DIR}/${HLT_MV_CANONICAL_HLT_SOURCE_NAME}/best_model_val.pt"
HLT2_CACHE_DIR=""
HLT2_CHECKPOINT_INFERRED=""

hlt_mv_source_name_for_tag() {
  local tag="$1"
  local names=()
  local source_name
  fresh_split_words names "${HLT_MV_SOURCE_NAMES}"
  for source_name in "${names[@]}"; do
    if [[ "${source_name}" =~ ^hlt2_part_${tag}_seed[0-9]+$ ]]; then
      printf '%s\n' "${source_name}"
      return 0
    fi
  done
  echo "No HLT-MV source name configured for HLT2 tag ${tag}." >&2
  return 2
}

if [[ "${VARIANT_NAME}" =~ ^sdv_hlt_hlt2_(s[0-9]+p[0-9]+)$ ]]; then
  hlt2_tag="${BASH_REMATCH[1]}"
  hlt2_source_name="$(hlt_mv_source_name_for_tag "${hlt2_tag}")"
  HLT2_CACHE_DIR="${HLT_MV_HLT2_CACHE_ROOT}/hlt_second_degrade_mild_v1_${hlt2_tag}"
  HLT2_CHECKPOINT_INFERRED="${HLT_MV_SOURCE_MODELS_DIR}/${hlt2_source_name}/best_model_val.pt"
else
  echo "Unknown HLT-MV pretrained dual-view variant: ${VARIANT_NAME}" >&2
  exit 2
fi

OUTPUT_DIR="${HLT_MV_PRETRAINED_DUALVIEW_OUTPUT_DIR:-${HLT_MV_PRETRAINED_DUALVIEW_DIR}/${VARIANT_NAME}}"
HLT_CHECKPOINT="${HLT_MV_PRETRAINED_DUALVIEW_HLT_CHECKPOINT:-${HLT_CHECKPOINT_INFERRED}}"
HLT2_CHECKPOINT="${HLT_MV_PRETRAINED_DUALVIEW_HLT2_CHECKPOINT:-${HLT2_CHECKPOINT_INFERRED}}"

fresh_setup "$@"
fresh_require_dir "${HLT_MV_HLT_CACHE_DIR}"
fresh_require_file "${HLT_MV_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${HLT_MV_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${HLT_MV_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_dir "${HLT2_CACHE_DIR}"
fresh_require_file "${HLT2_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${HLT2_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${HLT2_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CHECKPOINT}"
fresh_require_file "${HLT2_CHECKPOINT}"
if [[ "${HLT_MV_PRETRAINED_DUALVIEW_HEAD_WARMUP_EPOCHS}" -lt 1 ]]; then
  echo "Pretrained dual-view runs require HLT_MV_PRETRAINED_DUALVIEW_HEAD_WARMUP_EPOCHS >= 1." >&2
  exit 2
fi
if ! fresh_bool_enabled "${HLT_MV_PRETRAINED_DUALVIEW_SKIP_FINAL_TEST}" && ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing HLT-MV pretrained dual-view final-test evaluation without CONFIRM_FINAL_TEST=1." >&2
  exit 2
fi
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_hlt_mv_pretrained_dualview.py"
  --output-root "${OUTPUT_ROOT}"
  --pdv3-experiment-name "${HLT_MV_PDV3_EXPERIMENT_NAME}"
  --variant "${VARIANT_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --hlt-cache-dir "${HLT_MV_HLT_CACHE_DIR}"
  --hlt2-cache-dir "${HLT2_CACHE_DIR}"
  --hlt-checkpoint "${HLT_CHECKPOINT}"
  --hlt2-checkpoint "${HLT2_CHECKPOINT}"
  --epochs "${HLT_MV_PRETRAINED_DUALVIEW_EPOCHS}"
  --head-warmup-epochs "${HLT_MV_PRETRAINED_DUALVIEW_HEAD_WARMUP_EPOCHS}"
  --batch-size "${HLT_MV_PRETRAINED_DUALVIEW_BATCH_SIZE}"
  --eval-batch-size "${HLT_MV_PRETRAINED_DUALVIEW_EVAL_BATCH_SIZE}"
  --head-warmup-lr "${HLT_MV_PRETRAINED_DUALVIEW_HEAD_WARMUP_LR}"
  --branch-lr "${HLT_MV_PRETRAINED_DUALVIEW_BRANCH_LR}"
  --head-lr "${HLT_MV_PRETRAINED_DUALVIEW_HEAD_LR}"
  --weight-decay "${HLT_MV_PRETRAINED_DUALVIEW_WEIGHT_DECAY}"
  --dropout "${HLT_MV_PRETRAINED_DUALVIEW_DROPOUT}"
  --fusion-hidden-dim "${HLT_MV_PRETRAINED_DUALVIEW_FUSION_HIDDEN_DIM}"
  --representation-dim "${HLT_MV_PRETRAINED_DUALVIEW_REPRESENTATION_DIM}"
  --early-stop-patience "${HLT_MV_PRETRAINED_DUALVIEW_EARLY_STOP_PATIENCE}"
  --grad-clip-norm "${HLT_MV_PRETRAINED_DUALVIEW_GRAD_CLIP_NORM}"
  --num-workers "${HLT_MV_PRETRAINED_DUALVIEW_NUM_WORKERS}"
  --device "${HLT_MV_PRETRAINED_DUALVIEW_DEVICE}"
  --seed "${HLT_MV_PRETRAINED_DUALVIEW_SEED}"
  --model-size "${HLT_MV_PRETRAINED_DUALVIEW_MODEL_SIZE}"
  --max-train-jets "${HLT_MV_PRETRAINED_DUALVIEW_TRAIN_SIZE}"
  --max-val-jets "${HLT_MV_PRETRAINED_DUALVIEW_VAL_SIZE}"
  --max-final-test-jets "${HLT_MV_PRETRAINED_DUALVIEW_FINAL_TEST_SIZE}"
)
fresh_append_optional_arg cmd --max-train-batches "${HLT_MV_PRETRAINED_DUALVIEW_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${HLT_MV_PRETRAINED_DUALVIEW_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${HLT_MV_PRETRAINED_DUALVIEW_MAX_FINAL_TEST_BATCHES}"
fresh_append_flag_if_enabled cmd --amp "${HLT_MV_PRETRAINED_DUALVIEW_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${HLT_MV_PRETRAINED_DUALVIEW_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-model-val-predictions "${HLT_MV_PRETRAINED_DUALVIEW_SKIP_MODEL_VAL_PREDICTIONS}"
fresh_append_flag_if_enabled cmd --skip-final-test "${HLT_MV_PRETRAINED_DUALVIEW_SKIP_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"

fresh_write_run_config "${OUTPUT_DIR}" "hlt_mv_pretrained_dualview_${VARIANT_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/hlt_mv_pretrained_dualview_report.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_assert_json_ok "${OUTPUT_DIR}/run_report.json"
  if ! fresh_bool_enabled "${HLT_MV_PRETRAINED_DUALVIEW_SKIP_MODEL_VAL_PREDICTIONS}"; then
    fresh_require_file "${OUTPUT_DIR}/predictions/${VARIANT_NAME}/model_val_predictions.npz"
    fresh_require_file "${OUTPUT_DIR}/predictions/${VARIANT_NAME}/model_val_predictions_metadata.json"
  fi
  if ! fresh_bool_enabled "${HLT_MV_PRETRAINED_DUALVIEW_SKIP_FINAL_TEST}"; then
    fresh_require_file "${OUTPUT_DIR}/final_test_report.json"
    fresh_require_file "${OUTPUT_DIR}/predictions/${VARIANT_NAME}/final_test_predictions.npz"
    fresh_require_file "${OUTPUT_DIR}/predictions/${VARIANT_NAME}/final_test_predictions_metadata.json"
  fi
fi
