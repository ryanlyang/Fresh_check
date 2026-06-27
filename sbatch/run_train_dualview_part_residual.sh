#!/usr/bin/env bash
# Train one reliability-gated dual-view ParT residual tagger.

#SBATCH --job-name=dualview_part
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

VARIANT="${1:?Usage: sbatch run_train_dualview_part_residual.sh <frozen_anchor_pn_residual|frozen_anchor_shuffled_pn_control|warm_anchor_pn_residual>}"
OUTPUT_DIR="${DUALVIEW_PART_TAGGER_ROOT}/${VARIANT}"
DIAGNOSTICS_DIR="$(fresh_diagnostics_dir_for "${OUTPUT_DIR}")"

: "${DUALVIEW_PART_NO_AMP:=${NO_AMP:-0}}"
: "${DUALVIEW_PART_COMPILE_MODEL:=${COMPILE_MODEL:-0}}"
: "${DUALVIEW_PART_SKIP_HLT_HASH_CHECK:=${SKIP_HLT_HASH_CHECK:-0}}"
: "${DUALVIEW_PART_ALLOW_NONCANONICAL_ANCHOR:=0}"
: "${DUALVIEW_PART_ALLOW_NONCANONICAL_DATASET:=0}"
: "${DUALVIEW_PART_ENFORCE_SPLIT_SIZE:=0}"
: "${DUALVIEW_PART_DISABLE_PN_CONFIDENCE:=0}"
: "${DUALVIEW_PART_DISABLE_ANCHOR_CONTEXT:=0}"
: "${DUALVIEW_PART_DISABLE_RELIABILITY_FEATURES:=0}"
: "${DUALVIEW_PART_NON_STRICT_ANCHOR:=0}"
: "${DUALVIEW_PART_ANCHOR_MODEL_SIZE:=base}"
: "${DUALVIEW_PART_ANCHOR_CONTEXT_DIM:=128}"
: "${DUALVIEW_PART_ANCHOR_SUMMARY_HIDDEN_DIM:=128}"
: "${DUALVIEW_PART_ANCHOR_SUMMARY_DROPOUT:=0.0}"
: "${DUALVIEW_PART_PN_VIEW_SHUFFLE_SEED:=${DUALVIEW_PART_SEED}}"

shuffle_pn_view=0
warm_anchor=0
case "${VARIANT}" in
  frozen_anchor_pn_residual)
    ;;
  frozen_anchor_shuffled_pn_control)
    shuffle_pn_view=1
    ;;
  warm_anchor_pn_residual)
    warm_anchor=1
    ;;
  *)
    echo "Unknown dual-view ParT residual variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

fresh_setup "$@"
fresh_require_file "scripts/train_dualview_part_residual.py"
fresh_require_file "${DUALVIEW_PART_HLT_ANCHOR_CHECKPOINT}"
fresh_split_words label_name_args "${DUALVIEW_PART_LABEL_NAMES}"
for split in stack_train stack_val final_test; do
  fresh_require_file "${DUALVIEW_PART_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${DUALVIEW_PART_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
  fresh_require_file "${DUALVIEW_PART_PN_RECONSTRUCTED_VIEW_DIR}/${split}_reconstructed_view.npz"
  fresh_require_file "${DUALVIEW_PART_PN_RECONSTRUCTED_VIEW_DIR}/${split}_reconstructed_view_metadata.json"
done
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_dualview_part_residual.py"
  --output-dir "${OUTPUT_DIR}"
  --hlt-anchor-checkpoint "${DUALVIEW_PART_HLT_ANCHOR_CHECKPOINT}"
  --hlt-cache-dir "${DUALVIEW_PART_HLT_CACHE_DIR}"
  --pn-reconstructed-view-dir "${DUALVIEW_PART_PN_RECONSTRUCTED_VIEW_DIR}"
  --experiment-dir "${DUALVIEW_PART_EXPERIMENT_DIR}"
  --diagnostics-mirror-dir "${DIAGNOSTICS_DIR}"
  --train-split stack_train
  --val-split stack_val
  --final-test-split final_test
  --confirm-split-settings
  --seed "${DUALVIEW_PART_SEED}"
  --batch-size "${DUALVIEW_PART_BATCH_SIZE}"
  --eval-batch-size "${DUALVIEW_PART_EVAL_BATCH_SIZE}"
  --epochs "${DUALVIEW_PART_EPOCHS}"
  --lr "${DUALVIEW_PART_LR}"
  --weight-decay "${DUALVIEW_PART_WEIGHT_DECAY}"
  --num-workers "${DUALVIEW_PART_NUM_WORKERS}"
  --device "${DUALVIEW_PART_DEVICE}"
  --grad-clip-norm "${DUALVIEW_PART_GRAD_CLIP_NORM}"
  --early-stop-patience "${DUALVIEW_PART_EARLY_STOP_PATIENCE}"
  --selection-metric "${DUALVIEW_PART_SELECTION_METRIC}"
  --max-train-jets "${DUALVIEW_PART_STACK_TRAIN_SIZE}"
  --max-val-jets "${DUALVIEW_PART_STACK_VAL_SIZE}"
  --max-final-test-jets "${DUALVIEW_PART_FINAL_TEST_SIZE}"
  --anchor-model-size "${DUALVIEW_PART_ANCHOR_MODEL_SIZE}"
  --anchor-context-dim "${DUALVIEW_PART_ANCHOR_CONTEXT_DIM}"
  --anchor-summary-hidden-dim "${DUALVIEW_PART_ANCHOR_SUMMARY_HIDDEN_DIM}"
  --anchor-summary-dropout "${DUALVIEW_PART_ANCHOR_SUMMARY_DROPOUT}"
  --max-hlt-constits "${DUALVIEW_PART_MAX_HLT_CONSTITS}"
  --hlt-weight-threshold "${DUALVIEW_PART_HLT_WEIGHT_THRESHOLD}"
  --max-pn-tokens "${DUALVIEW_PART_MAX_PN_TOKENS}"
  --min-pn-tokens "${DUALVIEW_PART_MIN_PN_TOKENS}"
  --confidence-threshold "${DUALVIEW_PART_CONFIDENCE_THRESHOLD}"
  --selection-mode "${DUALVIEW_PART_SELECTION_MODE}"
  --pn-embed-dim "${DUALVIEW_PART_PN_EMBED_DIM}"
  --pn-layers "${DUALVIEW_PART_PN_LAYERS}"
  --pn-heads "${DUALVIEW_PART_PN_HEADS}"
  --pn-mlp-ratio "${DUALVIEW_PART_PN_MLP_RATIO}"
  --pn-dropout "${DUALVIEW_PART_PN_DROPOUT}"
  --pn-attention-dropout "${DUALVIEW_PART_PN_ATTENTION_DROPOUT}"
  --residual-hidden-dim "${DUALVIEW_PART_RESIDUAL_HIDDEN_DIM}"
  --residual-layers "${DUALVIEW_PART_RESIDUAL_LAYERS}"
  --residual-dropout "${DUALVIEW_PART_RESIDUAL_DROPOUT}"
  --gate-bias-init "${DUALVIEW_PART_GATE_BIAS_INIT}"
  --initialization-check-batches "${DUALVIEW_PART_INITIALIZATION_CHECK_BATCHES}"
  --max-case-rows-per-type "${DUALVIEW_PART_MAX_CASE_ROWS_PER_TYPE}"
  --pn-view-shuffle-seed "${DUALVIEW_PART_PN_VIEW_SHUFFLE_SEED}"
  --label-names "${label_name_args[@]}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${DUALVIEW_PART_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --warm-anchor "${warm_anchor}"
fresh_append_flag_if_enabled cmd --allow-noncanonical-anchor "${DUALVIEW_PART_ALLOW_NONCANONICAL_ANCHOR}"
fresh_append_flag_if_enabled cmd --allow-noncanonical-dataset "${DUALVIEW_PART_ALLOW_NONCANONICAL_DATASET}"
fresh_append_flag_if_enabled cmd --enforce-split-size "${DUALVIEW_PART_ENFORCE_SPLIT_SIZE}"
fresh_append_flag_if_enabled cmd --non-strict-anchor "${DUALVIEW_PART_NON_STRICT_ANCHOR}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${DUALVIEW_PART_SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --no-amp "${DUALVIEW_PART_NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${DUALVIEW_PART_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --disable-pn-confidence "${DUALVIEW_PART_DISABLE_PN_CONFIDENCE}"
fresh_append_flag_if_enabled cmd --disable-anchor-context "${DUALVIEW_PART_DISABLE_ANCHOR_CONTEXT}"
fresh_append_flag_if_enabled cmd --disable-reliability-features "${DUALVIEW_PART_DISABLE_RELIABILITY_FEATURES}"
fresh_append_flag_if_enabled cmd --skip-initialization-check "${DUALVIEW_PART_SKIP_INITIALIZATION_CHECK}"
fresh_append_flag_if_enabled cmd --shuffle-pn-view "${shuffle_pn_view}"
fresh_append_optional_arg cmd --anchor-lr "${DUALVIEW_PART_ANCHOR_LR}"
fresh_append_optional_arg cmd --max-train-batches "${DUALVIEW_PART_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${DUALVIEW_PART_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${DUALVIEW_PART_MAX_FINAL_TEST_BATCHES}"

fresh_write_run_config "${OUTPUT_DIR}" "dualview_part_residual_${VARIANT}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/epoch_metrics.csv"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/per_class_metrics.csv"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/residual_diagnostics.json"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/gate_by_class.csv"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/gate_by_hlt_confidence.csv"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/gate_by_hlt_correctness.csv"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/prediction_change_summary.csv"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/fix_break_cases.csv"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/fix_cases.csv"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/break_cases.csv"
fi
