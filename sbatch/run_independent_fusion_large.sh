#!/usr/bin/env bash
# Handoff independent fusion: large split sizes, frozen HLT + reco7 predictions.

#SBATCH --job-name=ifuse_large
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=05:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${FUSION_BATCH_SIZE:=128}"
: "${FUSION_NUM_WORKERS:=4}"
: "${FUSION_DEVICE:=${DEVICE}}"
: "${FUSION_STACK_TRAIN_SIZE:=250000}"
: "${FUSION_STACK_VAL_SIZE:=50000}"
: "${FUSION_FINAL_TEST_SIZE:=500000}"

RUN_OUTPUT_DIR="${FUSION_MODEL_LOADING_LARGE_DIR}"

fresh_setup "$@"
fresh_require_file "${HLT_CACHE_DIR}/stack_train_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/stack_val_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CHECKPOINT}"

fresh_split_words variant_args "${FUSION_MODEL_LOADING_VARIANTS}"
for variant in "${variant_args[@]}"; do
  fresh_require_file "${RECO7_ROOT}/${variant}/stage2_dual_view/best_model_val.pt"
done

fresh_claim_new_dir "${RUN_OUTPUT_DIR}"
fresh_split_words feature_mode_args "${FUSION_MODEL_LOADING_FEATURE_MODES}"
model_args=(hlt_baseline "${variant_args[@]}")

demo_common=(
  "${PYTHON_BIN}" "-u" "scripts/demo_load_and_score_models_no_fusion.py"
  --hlt-cache-dir "${HLT_CACHE_DIR}"
  --hlt-checkpoint "${HLT_CHECKPOINT}"
  --reco-root "${RECO7_ROOT}"
  --output-dir "${RUN_OUTPUT_DIR}"
  --variants "${variant_args[@]}"
  --batch-size "${FUSION_BATCH_SIZE}"
  --num-workers "${FUSION_NUM_WORKERS}"
  --device "${FUSION_DEVICE}"
  --feature-mode logits_probs
)
if fresh_bool_enabled "${OVERWRITE}"; then
  demo_common+=(--overwrite-predictions)
fi

fusion_cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_independent_fusion_from_predictions.py"
  --prediction-dir "${RUN_OUTPUT_DIR}/predictions"
  --output-dir "${RUN_OUTPUT_DIR}/fusion"
  --model-names "${model_args[@]}"
  --feature-modes "${feature_mode_args[@]}"
  --max-iter "${FUSION_MODEL_LOADING_MAX_ITER}"
  --control-seed "${FUSION_MODEL_LOADING_CONTROL_SEED}"
  --confirm-final-test
)
fresh_append_flag_if_enabled fusion_cmd --skip-controls "${FUSION_MODEL_LOADING_SKIP_CONTROLS}"
if [[ -n "${FUSION_MODEL_LOADING_C_GRID}" ]]; then
  fresh_split_words c_grid_args "${FUSION_MODEL_LOADING_C_GRID}"
  fusion_cmd+=(--c-grid "${c_grid_args[@]}")
fi

fresh_write_run_config \
  "${RUN_OUTPUT_DIR}" \
  "independent_fusion_large_250k_50k_500k" \
  "demo_load_and_score_models_no_fusion.py per split then run_independent_fusion_from_predictions.py"

fresh_run "${demo_common[@]}" --splits stack_train --max-jets-per-split "${FUSION_STACK_TRAIN_SIZE}"
fresh_run "${demo_common[@]}" --splits stack_val --max-jets-per-split "${FUSION_STACK_VAL_SIZE}"
fresh_run "${demo_common[@]}" --splits final_test --confirm-final-test --max-jets-per-split "${FUSION_FINAL_TEST_SIZE}"
fresh_run "${fusion_cmd[@]}"

if ! fresh_is_dry_run; then
  for model_name in "${model_args[@]}"; do
    fresh_require_file "${RUN_OUTPUT_DIR}/predictions/${model_name}/stack_train_predictions.npz"
    fresh_require_file "${RUN_OUTPUT_DIR}/predictions/${model_name}/stack_val_predictions.npz"
    fresh_require_file "${RUN_OUTPUT_DIR}/predictions/${model_name}/final_test_predictions.npz"
  done
  fresh_require_file "${RUN_OUTPUT_DIR}/model_loading_demo_report.json"
  fresh_require_file "${RUN_OUTPUT_DIR}/fusion/fusion_report.json"
  fresh_require_file "${RUN_OUTPUT_DIR}/fusion/raw_source_metrics.csv"
  fresh_require_file "${RUN_OUTPUT_DIR}/fusion/group_fusion_metrics.csv"
  fresh_require_file "${RUN_OUTPUT_DIR}/fusion/singleton_stacker_metrics.csv"
  fresh_require_file "${RUN_OUTPUT_DIR}/fusion/controls.json"
  fresh_require_file "${RUN_OUTPUT_DIR}/fusion/stack_split_hash_audit.json"
fi
