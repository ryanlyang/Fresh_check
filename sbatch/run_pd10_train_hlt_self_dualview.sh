#!/usr/bin/env bash
# Train one deployable PD10 HLT self-dualview fusion model.

#SBATCH --job-name=pd10_hlt_sdv
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

VARIANT_NAME="${1:-sdv_hlt_hlt2_s0p20}"
OUTPUT_DIR="${PD10_HLT_SDV_MODELS_DIR}/${VARIANT_NAME}"
HLT2_CACHE_DIR=""

fresh_setup "$@"
fresh_require_dir "${PD10_HLT_CACHE_DIR}"
fresh_require_file "${PD10_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${PD10_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${PD10_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_file "${PD10_HLT_SDV_HLT_TEACHER_CHECKPOINT}"
if [[ "${VARIANT_NAME}" != "sdv_hlt_hlt_same_view" ]]; then
  STRENGTH="$(fresh_pd10_hlt_sdv_strength_from_variant "${VARIANT_NAME}")"
  HLT2_CACHE_DIR="$(fresh_pd10_hlt_sdv_hlt2_cache_dir "${STRENGTH}")"
  fresh_require_dir "${HLT2_CACHE_DIR}"
  fresh_require_file "${HLT2_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
  fresh_require_file "${HLT2_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
  fresh_require_file "${HLT2_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fi
if ! fresh_bool_enabled "${PD10_HLT_SDV_SKIP_FINAL_TEST}" && ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing HLT-SDV final-test training/evaluation without CONFIRM_FINAL_TEST=1." >&2
  exit 2
fi
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_pd10_hlt_self_dualview.py"
  --pd10-root "${PD10_ROOT}"
  --variant "${VARIANT_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --hlt-cache-dir "${PD10_HLT_CACHE_DIR}"
  --hlt-teacher-checkpoint "${PD10_HLT_SDV_HLT_TEACHER_CHECKPOINT}"
  --epochs "${PD10_HLT_SDV_EPOCHS}"
  --head-warmup-epochs "${PD10_HLT_SDV_HEAD_WARMUP_EPOCHS}"
  --batch-size "${PD10_HLT_SDV_BATCH_SIZE}"
  --eval-batch-size "${PD10_HLT_SDV_EVAL_BATCH_SIZE}"
  --head-warmup-lr "${PD10_HLT_SDV_HEAD_WARMUP_LR}"
  --branch-lr "${PD10_HLT_SDV_BRANCH_LR}"
  --head-lr "${PD10_HLT_SDV_HEAD_LR}"
  --weight-decay "${PD10_HLT_SDV_WEIGHT_DECAY}"
  --dropout "${PD10_HLT_SDV_DROPOUT}"
  --fusion-hidden-dim "${PD10_HLT_SDV_FUSION_HIDDEN_DIM}"
  --representation-dim "${PD10_HLT_SDV_REPRESENTATION_DIM}"
  --early-stop-patience "${PD10_HLT_SDV_EARLY_STOP_PATIENCE}"
  --num-workers "${PD10_HLT_SDV_NUM_WORKERS}"
  --device "${PD10_HLT_SDV_DEVICE}"
  --seed "${PD10_HLT_SDV_SEED}"
  --model-size "${PD10_HLT_SDV_MODEL_SIZE}"
  --max-train-jets "${PD10_MODEL_TRAIN_SIZE}"
  --max-val-jets "${PD10_MODEL_VAL_SIZE}"
  --max-final-test-jets "${PD10_FINAL_TEST_SIZE}"
)
fresh_append_optional_arg cmd --hlt2-cache-dir "${HLT2_CACHE_DIR}"
fresh_append_optional_arg cmd --max-train-batches "${PD10_HLT_SDV_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${PD10_HLT_SDV_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${PD10_HLT_SDV_MAX_FINAL_TEST_BATCHES}"
fresh_append_flag_if_enabled cmd --no-amp "${PD10_HLT_SDV_NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${PD10_HLT_SDV_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --no-branch-init "${PD10_HLT_SDV_NO_BRANCH_INIT}"
fresh_append_flag_if_enabled cmd --skip-model-val-predictions "${PD10_HLT_SDV_SKIP_MODEL_VAL_PREDICTIONS}"
fresh_append_flag_if_enabled cmd --skip-final-test "${PD10_HLT_SDV_SKIP_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"

fresh_write_run_config "${OUTPUT_DIR}" "pd10_hlt_self_dualview_${VARIANT_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_assert_json_ok "${OUTPUT_DIR}/run_report.json"
  if ! fresh_bool_enabled "${PD10_HLT_SDV_SKIP_FINAL_TEST}"; then
    fresh_require_file "${OUTPUT_DIR}/final_test_report.json"
  fi
fi
