#!/usr/bin/env bash
# Train an HLT-MV N-view particle fusion model.

#SBATCH --job-name=hlt_mv_nview
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=3-00:00:00
#SBATCH --mem=260G
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
: "${HLT_MV_MULTIVIEW_DIR:=${HLT_MV_ROOT}/particle_multiview}"
: "${HLT_MV_CANONICAL_HLT_SOURCE_NAME:=hlt_part_seed8801}"
: "${HLT_MV_SOURCE_NAMES:=hlt_part_seed8801 hlt2_part_s0p10_seed8811 hlt2_part_s0p20_seed8821 hlt2_part_s0p35_seed8831 hlt2_part_s1p00_seed8841}"
: "${HLT_MV_MULTIVIEW_MODEL_NAME:=${1:-five_hlt_hlt2_s0p10_s0p20_s0p35_s1p00}}"
: "${HLT_MV_MULTIVIEW_OUTPUT_DIR:=}"
: "${HLT_MV_MULTIVIEW_SEED:=9301}"
: "${HLT_MV_MULTIVIEW_EPOCHS:=10}"
: "${HLT_MV_MULTIVIEW_HEAD_WARMUP_EPOCHS:=1}"
: "${HLT_MV_MULTIVIEW_BATCH_SIZE:=48}"
: "${HLT_MV_MULTIVIEW_EVAL_BATCH_SIZE:=64}"
: "${HLT_MV_MULTIVIEW_HEAD_WARMUP_LR:=0.0003}"
: "${HLT_MV_MULTIVIEW_BRANCH_LR:=0.00002}"
: "${HLT_MV_MULTIVIEW_HEAD_LR:=0.0003}"
: "${HLT_MV_MULTIVIEW_WEIGHT_DECAY:=0.0001}"
: "${HLT_MV_MULTIVIEW_DROPOUT:=0.05}"
: "${HLT_MV_MULTIVIEW_FUSION_HIDDEN_DIM:=512}"
: "${HLT_MV_MULTIVIEW_REPRESENTATION_DIM:=256}"
: "${HLT_MV_MULTIVIEW_EARLY_STOP_PATIENCE:=3}"
: "${HLT_MV_MULTIVIEW_GRAD_CLIP_NORM:=1.0}"
: "${HLT_MV_MULTIVIEW_NUM_WORKERS:=${NUM_WORKERS}}"
: "${HLT_MV_MULTIVIEW_DEVICE:=${DEVICE}}"
: "${HLT_MV_MULTIVIEW_MODEL_SIZE:=base}"
: "${HLT_MV_MULTIVIEW_AMP:=0}"
: "${HLT_MV_MULTIVIEW_COMPILE_MODEL:=0}"
: "${HLT_MV_MULTIVIEW_SKIP_MODEL_VAL_PREDICTIONS:=0}"
: "${HLT_MV_MULTIVIEW_SKIP_FINAL_TEST:=0}"
: "${HLT_MV_MULTIVIEW_MAX_TRAIN_BATCHES:=}"
: "${HLT_MV_MULTIVIEW_MAX_VAL_BATCHES:=}"
: "${HLT_MV_MULTIVIEW_MAX_FINAL_TEST_BATCHES:=}"
: "${HLT_MV_MULTIVIEW_TRAIN_SIZE:=5000000}"
: "${HLT_MV_MULTIVIEW_VAL_SIZE:=1000000}"
: "${HLT_MV_MULTIVIEW_FINAL_TEST_SIZE:=1000000}"

MODEL_NAME="${HLT_MV_MULTIVIEW_MODEL_NAME}"
OUTPUT_DIR="${HLT_MV_MULTIVIEW_OUTPUT_DIR:-${HLT_MV_MULTIVIEW_DIR}/${MODEL_NAME}}"

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

expected_hlt2_views=0
hlt2_tags_text=""
if [[ "${MODEL_NAME}" == quad_hlt_hlt2_* ]]; then
  expected_hlt2_views=3
  hlt2_tags_text="${MODEL_NAME#quad_hlt_hlt2_}"
elif [[ "${MODEL_NAME}" == five_hlt_hlt2_* ]]; then
  expected_hlt2_views=4
  hlt2_tags_text="${MODEL_NAME#five_hlt_hlt2_}"
else
  echo "HLT-MV multiview model must be quad_hlt_hlt2_<3 tags> or five_hlt_hlt2_<4 tags>; got ${MODEL_NAME}" >&2
  exit 2
fi

IFS='_' read -r -a hlt2_tags <<< "${hlt2_tags_text}"
IFS=$'\n\t'
if [[ "${#hlt2_tags[@]}" -ne "${expected_hlt2_views}" ]]; then
  echo "HLT-MV ${MODEL_NAME} expected ${expected_hlt2_views} HLT2 tags, got ${#hlt2_tags[@]}: ${hlt2_tags_text}" >&2
  exit 2
fi

fresh_setup "$@"
fresh_require_dir "${HLT_MV_HLT_CACHE_DIR}"
fresh_require_file "${HLT_MV_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${HLT_MV_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${HLT_MV_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"

hlt_checkpoint="${HLT_MV_SOURCE_MODELS_DIR}/${HLT_MV_CANONICAL_HLT_SOURCE_NAME}/best_model_val.pt"
fresh_require_file "${hlt_checkpoint}"

view_args=(
  --view "hlt,fixed_hlt,${HLT_MV_HLT_CACHE_DIR},${hlt_checkpoint}"
)
for tag in "${hlt2_tags[@]}"; do
  if [[ ! "${tag}" =~ ^s[0-9]+p[0-9]+$ ]]; then
    echo "Bad HLT2 tag in ${MODEL_NAME}: ${tag}" >&2
    exit 2
  fi
  source_name="$(hlt_mv_source_name_for_tag "${tag}")"
  cache_dir="${HLT_MV_HLT2_CACHE_ROOT}/hlt_second_degrade_mild_v1_${tag}"
  checkpoint="${HLT_MV_SOURCE_MODELS_DIR}/${source_name}/best_model_val.pt"
  fresh_require_dir "${cache_dir}"
  fresh_require_file "${cache_dir}/model_train_fixed_hlt_metadata.json"
  fresh_require_file "${cache_dir}/model_val_fixed_hlt_metadata.json"
  fresh_require_file "${cache_dir}/final_test_fixed_hlt_metadata.json"
  fresh_require_file "${checkpoint}"
  view_args+=(--view "hlt2_${tag},hlt2,${cache_dir},${checkpoint}")
done

if [[ "${HLT_MV_MULTIVIEW_HEAD_WARMUP_EPOCHS}" -lt 1 ]]; then
  echo "HLT-MV multiview runs require HLT_MV_MULTIVIEW_HEAD_WARMUP_EPOCHS >= 1." >&2
  exit 2
fi
if ! fresh_bool_enabled "${HLT_MV_MULTIVIEW_SKIP_FINAL_TEST}" && ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing HLT-MV multiview final-test evaluation without CONFIRM_FINAL_TEST=1." >&2
  exit 2
fi
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_hlt_mv_multiview.py"
  --model-name "${MODEL_NAME}"
  --output-dir "${OUTPUT_DIR}"
  "${view_args[@]}"
  --epochs "${HLT_MV_MULTIVIEW_EPOCHS}"
  --head-warmup-epochs "${HLT_MV_MULTIVIEW_HEAD_WARMUP_EPOCHS}"
  --batch-size "${HLT_MV_MULTIVIEW_BATCH_SIZE}"
  --eval-batch-size "${HLT_MV_MULTIVIEW_EVAL_BATCH_SIZE}"
  --head-warmup-lr "${HLT_MV_MULTIVIEW_HEAD_WARMUP_LR}"
  --branch-lr "${HLT_MV_MULTIVIEW_BRANCH_LR}"
  --head-lr "${HLT_MV_MULTIVIEW_HEAD_LR}"
  --weight-decay "${HLT_MV_MULTIVIEW_WEIGHT_DECAY}"
  --dropout "${HLT_MV_MULTIVIEW_DROPOUT}"
  --fusion-hidden-dim "${HLT_MV_MULTIVIEW_FUSION_HIDDEN_DIM}"
  --representation-dim "${HLT_MV_MULTIVIEW_REPRESENTATION_DIM}"
  --early-stop-patience "${HLT_MV_MULTIVIEW_EARLY_STOP_PATIENCE}"
  --grad-clip-norm "${HLT_MV_MULTIVIEW_GRAD_CLIP_NORM}"
  --num-workers "${HLT_MV_MULTIVIEW_NUM_WORKERS}"
  --device "${HLT_MV_MULTIVIEW_DEVICE}"
  --seed "${HLT_MV_MULTIVIEW_SEED}"
  --model-size "${HLT_MV_MULTIVIEW_MODEL_SIZE}"
  --max-train-jets "${HLT_MV_MULTIVIEW_TRAIN_SIZE}"
  --max-val-jets "${HLT_MV_MULTIVIEW_VAL_SIZE}"
  --max-final-test-jets "${HLT_MV_MULTIVIEW_FINAL_TEST_SIZE}"
)
fresh_append_optional_arg cmd --max-train-batches "${HLT_MV_MULTIVIEW_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${HLT_MV_MULTIVIEW_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${HLT_MV_MULTIVIEW_MAX_FINAL_TEST_BATCHES}"
fresh_append_flag_if_enabled cmd --amp "${HLT_MV_MULTIVIEW_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${HLT_MV_MULTIVIEW_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-model-val-predictions "${HLT_MV_MULTIVIEW_SKIP_MODEL_VAL_PREDICTIONS}"
fresh_append_flag_if_enabled cmd --skip-final-test "${HLT_MV_MULTIVIEW_SKIP_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"

fresh_write_run_config "${OUTPUT_DIR}" "hlt_mv_multiview_${MODEL_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/hlt_multiview_report.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_assert_json_ok "${OUTPUT_DIR}/run_report.json"
  if ! fresh_bool_enabled "${HLT_MV_MULTIVIEW_SKIP_MODEL_VAL_PREDICTIONS}"; then
    fresh_require_file "${OUTPUT_DIR}/predictions/${MODEL_NAME}/model_val_predictions.npz"
    fresh_require_file "${OUTPUT_DIR}/predictions/${MODEL_NAME}/model_val_predictions_metadata.json"
  fi
  if ! fresh_bool_enabled "${HLT_MV_MULTIVIEW_SKIP_FINAL_TEST}"; then
    fresh_require_file "${OUTPUT_DIR}/final_test_report.json"
    fresh_require_file "${OUTPUT_DIR}/predictions/${MODEL_NAME}/final_test_predictions.npz"
    fresh_require_file "${OUTPUT_DIR}/predictions/${MODEL_NAME}/final_test_predictions_metadata.json"
  fi
fi
