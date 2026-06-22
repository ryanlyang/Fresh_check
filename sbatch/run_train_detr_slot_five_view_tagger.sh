#!/usr/bin/env bash
# Train one DETR-backed five-view tagger or ablation variant.

#SBATCH --job-name=detrslot_tagger
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=2-00:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

VARIANT="${1:?Usage: sbatch run_train_detr_slot_five_view_tagger.sh <hlt_only|hlt_plus_gt|hlt_plus_pn|hlt_plus_pfn|hlt_plus_pcnn|five_view_plain|five_view_geometry|five_view_no_confidence|view_label_shuffle_control>}"
OUTPUT_DIR="${DETR_SLOT_TAGGER_ROOT}/${VARIANT}"

: "${NO_AMP:=0}"
: "${COMPILE_MODEL:=0}"
: "${SKIP_HLT_HASH_CHECK:=0}"
: "${DETR_SLOT_TAGGER_DISABLE_CONFIDENCE:=0}"
: "${DETR_SLOT_TAGGER_DISABLE_VIEW_EMBEDDING:=0}"
: "${DETR_SLOT_TAGGER_DISABLE_SOURCE_EMBEDDING:=0}"
: "${DETR_SLOT_TAGGER_DISABLE_VIEW_SUMMARIES:=0}"

fresh_setup "$@"
fresh_require_file "scripts/train_detr_slot_five_view_tagger.py"
fresh_split_words label_filter_args "${DETR_SLOT_LABEL_FILTER_NAMES}"
fresh_split_words label_name_args "${DETR_SLOT_LABEL_NAMES}"
fresh_split_words reco_args "${DETR_SLOT_ARCHITECTURES}"
for split in stack_train stack_val final_test; do
  fresh_require_file "${DETR_SLOT_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
for architecture in "${reco_args[@]}"; do
  for split in stack_train stack_val final_test; do
    fresh_require_file "${DETR_SLOT_RECONSTRUCTED_VIEW_DIR}/${architecture}/${split}_reconstructed_view.npz"
    fresh_require_file "${DETR_SLOT_RECONSTRUCTED_VIEW_DIR}/${architecture}/${split}_reconstructed_view_metadata.json"
  done
done
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_detr_slot_five_view_tagger.py"
  --variants "${VARIANT}"
  --output-root "${DETR_SLOT_TAGGER_ROOT}"
  --experiment-dir "${DETR_SLOT_ROOT}"
  --hlt-cache-dir "${DETR_SLOT_HLT_CACHE_DIR}"
  --reconstructed-view-dir "${DETR_SLOT_RECONSTRUCTED_VIEW_DIR}"
  --train-split stack_train
  --val-split stack_val
  --final-test-split final_test
  --confirm-split-settings
  --seed "${DETR_SLOT_TAGGER_SEED}"
  --batch-size "${DETR_SLOT_TAGGER_BATCH_SIZE}"
  --epochs "${DETR_SLOT_TAGGER_EPOCHS}"
  --lr "${DETR_SLOT_TAGGER_LR}"
  --weight-decay "${DETR_SLOT_TAGGER_WEIGHT_DECAY}"
  --num-workers "${DETR_SLOT_TAGGER_NUM_WORKERS}"
  --device "${DETR_SLOT_TAGGER_DEVICE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --early-stop-patience "${DETR_SLOT_TAGGER_EARLY_STOP_PATIENCE}"
  --max-train-jets "${DETR_SLOT_STACK_TRAIN_SIZE}"
  --max-val-jets "${DETR_SLOT_STACK_VAL_SIZE}"
  --max-final-test-jets "${DETR_SLOT_FINAL_TEST_SIZE}"
  --selection-metric "${DETR_SLOT_TAGGER_SELECTION_METRIC}"
  --max-tokens-per-view "${DETR_SLOT_MAX_TOKENS_PER_VIEW}"
  --min-tokens-per-view "${DETR_SLOT_MIN_TOKENS_PER_VIEW}"
  --confidence-threshold "${DETR_SLOT_CONFIDENCE_THRESHOLD}"
  --selection-mode topk_or_threshold
  --view-label-shuffle-seed "${DETR_SLOT_TAGGER_SEED}"
  --embed-dim "${DETR_SLOT_TAGGER_EMBED_DIM}"
  --stage1-layers "${DETR_SLOT_TAGGER_STAGE1_LAYERS}"
  --stage1-heads "${DETR_SLOT_TAGGER_STAGE1_HEADS}"
  --stage2-layers "${DETR_SLOT_TAGGER_STAGE2_LAYERS}"
  --stage2-heads "${DETR_SLOT_TAGGER_STAGE2_HEADS}"
  --mlp-ratio "${DETR_SLOT_TAGGER_MLP_RATIO}"
  --dropout "${DETR_SLOT_TAGGER_DROPOUT}"
  --attention-dropout "${DETR_SLOT_TAGGER_ATTENTION_DROPOUT}"
  --geometry-hidden-dim "${DETR_SLOT_TAGGER_GEOMETRY_HIDDEN_DIM}"
  --geometry-dropout "${DETR_SLOT_TAGGER_GEOMETRY_DROPOUT}"
)
if [[ -n "${DETR_SLOT_NUM_CLASSES}" ]]; then
  cmd+=(--num-classes "${DETR_SLOT_NUM_CLASSES}")
fi
if ((${#label_name_args[@]})); then
  cmd+=(--label-names "${label_name_args[@]}")
fi
if ((${#label_filter_args[@]})); then
  cmd+=(--label-filter-names "${label_filter_args[@]}")
fi
fresh_append_flag_if_enabled cmd --confirm-final-test "${DETR_SLOT_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --disable-confidence "${DETR_SLOT_TAGGER_DISABLE_CONFIDENCE}"
fresh_append_flag_if_enabled cmd --disable-view-embedding "${DETR_SLOT_TAGGER_DISABLE_VIEW_EMBEDDING}"
fresh_append_flag_if_enabled cmd --disable-source-embedding "${DETR_SLOT_TAGGER_DISABLE_SOURCE_EMBEDDING}"
fresh_append_flag_if_enabled cmd --disable-view-summaries "${DETR_SLOT_TAGGER_DISABLE_VIEW_SUMMARIES}"
fresh_append_optional_arg cmd --classifier-hidden-dim "${DETR_SLOT_TAGGER_CLASSIFIER_HIDDEN_DIM}"
fresh_append_optional_arg cmd --max-train-batches "${DETR_SLOT_TAGGER_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${DETR_SLOT_TAGGER_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${DETR_SLOT_TAGGER_MAX_FINAL_TEST_BATCHES}"

fresh_write_run_config "${OUTPUT_DIR}" "detr_slot_five_view_tagger_${VARIANT}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/detr_five_view_tagger_report.json"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/per_class_metrics.csv"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/view_ablation_metrics.json"
fi
