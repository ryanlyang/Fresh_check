#!/usr/bin/env bash
# Cache AV10 logits for trained Architecture-View variants.

#SBATCH --job-name=av10_cache
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${ARCHITECTURE_VIEW_10CLASS_ROOT:=${OUTPUT_ROOT}/architecture_view_10class_hlt0p6}"
: "${ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/taggers}"
: "${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/inputs/hlt_cache}"
: "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/prediction_cache}"
: "${ARCHITECTURE_VIEW_10CLASS_VARIANTS:=av10_baseline_recheck av10_pn_context_to_part av10_pfn_context_to_part av10_pcnn_context_to_part av10_all_views_to_part av10_random_view_control av10_context_mlp_control}"
: "${ARCHITECTURE_VIEW_10CLASS_CACHE_BATCH_SIZE:=128}"
: "${ARCHITECTURE_VIEW_10CLASS_CACHE_NUM_WORKERS:=4}"
: "${ARCHITECTURE_VIEW_10CLASS_CACHE_DEVICE:=${DEVICE}}"
: "${ARCHITECTURE_VIEW_10CLASS_MODEL_VAL_SIZE:=150000}"
: "${ARCHITECTURE_VIEW_10CLASS_STACK_TRAIN_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_10CLASS_STACK_VAL_SIZE:=150000}"
: "${ARCHITECTURE_VIEW_10CLASS_FINAL_TEST_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_10CLASS_CONFIRM_FINAL_TEST:=1}"
: "${ARCHITECTURE_VIEW_10CLASS_SKIP_HLT_HASH_CHECK:=0}"
: "${ARCHITECTURE_VIEW_10CLASS_OVERWRITE_PREDICTIONS:=0}"
: "${ARCHITECTURE_VIEW_10CLASS_NO_SKIP_EXISTING:=0}"
: "${ARCHITECTURE_VIEW_10CLASS_CACHE_SEED:=7207}"

fresh_setup "$@"
fresh_require_file "scripts/cache_architecture_view_10class_predictions.py"
for split in model_val stack_train stack_val final_test; do
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
fresh_split_words variant_args "${ARCHITECTURE_VIEW_10CLASS_VARIANTS}"
for variant in "${variant_args[@]}"; do
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT}/${variant}/best_model_val.pt"
done
fresh_claim_new_dir "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/cache_architecture_view_10class_predictions.py"
  --output-dir "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT}"
  --hlt-cache-dir "${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR}"
  --checkpoint-root "${ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT}"
  --variants "${variant_args[@]}"
  --splits model_val stack_train stack_val final_test
  --batch-size "${ARCHITECTURE_VIEW_10CLASS_CACHE_BATCH_SIZE}"
  --num-workers "${ARCHITECTURE_VIEW_10CLASS_CACHE_NUM_WORKERS}"
  --device "${ARCHITECTURE_VIEW_10CLASS_CACHE_DEVICE}"
  --max-model-val-jets "${ARCHITECTURE_VIEW_10CLASS_MODEL_VAL_SIZE}"
  --max-stack-train-jets "${ARCHITECTURE_VIEW_10CLASS_STACK_TRAIN_SIZE}"
  --max-stack-val-jets "${ARCHITECTURE_VIEW_10CLASS_STACK_VAL_SIZE}"
  --max-final-test-jets "${ARCHITECTURE_VIEW_10CLASS_FINAL_TEST_SIZE}"
  --seed "${ARCHITECTURE_VIEW_10CLASS_CACHE_SEED}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${ARCHITECTURE_VIEW_10CLASS_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${ARCHITECTURE_VIEW_10CLASS_SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --overwrite "${ARCHITECTURE_VIEW_10CLASS_OVERWRITE_PREDICTIONS}"
fresh_append_flag_if_enabled cmd --no-skip-existing "${ARCHITECTURE_VIEW_10CLASS_NO_SKIP_EXISTING}"

fresh_write_run_config "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT}" "architecture_view_10class_prediction_cache" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT}/prediction_manifest.json"
  for variant in "${variant_args[@]}"; do
    for split in model_val stack_train stack_val final_test; do
      fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT}/predictions/${variant}/${split}_logits.npz"
      fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT}/predictions/${variant}/${split}_metadata.json"
    done
  done
fi
