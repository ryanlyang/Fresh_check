#!/usr/bin/env bash
# Train one five-view set-matching tagger or ablation variant.

#SBATCH --job-name=setmatch_tagger
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

VARIANT="${1:?Usage: sbatch run_train_five_view_tagger.sh <hlt_only|hlt_plus_gt|hlt_plus_pn|hlt_plus_pfn|hlt_plus_pcnn|five_view_plain|five_view_geometry|five_view_no_confidence|view_label_shuffle_control>}"
OUTPUT_DIR="${SET_MATCHING_TAGGER_ROOT}/${VARIANT}"

: "${NO_AMP:=0}"
: "${COMPILE_MODEL:=0}"
: "${SKIP_HLT_HASH_CHECK:=0}"

drop_views=()
use_geometry_attention=0
disable_confidence=0
shuffle_view_labels=0
case "${VARIANT}" in
  hlt_only)
    drop_views=(gt_reco pn_reco pfn_reco pcnn_reco)
    ;;
  hlt_plus_gt)
    drop_views=(pn_reco pfn_reco pcnn_reco)
    ;;
  hlt_plus_pn)
    drop_views=(gt_reco pfn_reco pcnn_reco)
    ;;
  hlt_plus_pfn)
    drop_views=(gt_reco pn_reco pcnn_reco)
    ;;
  hlt_plus_pcnn)
    drop_views=(gt_reco pn_reco pfn_reco)
    ;;
  five_view_plain)
    ;;
  five_view_geometry)
    use_geometry_attention=1
    ;;
  five_view_no_confidence)
    disable_confidence=1
    ;;
  view_label_shuffle_control)
    shuffle_view_labels=1
    ;;
  *)
    echo "Unknown five-view tagger variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

fresh_setup "$@"
fresh_require_file "scripts/train_five_view_tagger.py"
fresh_split_words label_filter_args "${SET_MATCHING_LABEL_FILTER_NAMES}"
fresh_split_words label_name_args "${SET_MATCHING_LABEL_NAMES}"
for split in stack_train stack_val final_test; do
  fresh_require_file "${SET_MATCHING_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
fresh_split_words reco_args "${SET_MATCHING_RECO_ARCHITECTURES}"
for architecture in "${reco_args[@]}"; do
  for split in stack_train stack_val final_test; do
    fresh_require_file "${SET_MATCHING_RECONSTRUCTED_VIEW_DIR}/${architecture}/${split}_reconstructed_view.npz"
    fresh_require_file "${SET_MATCHING_RECONSTRUCTED_VIEW_DIR}/${architecture}/${split}_reconstructed_view_metadata.json"
  done
done
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_five_view_tagger.py"
  --output-dir "${OUTPUT_DIR}"
  --experiment-dir "${SET_MATCHING_ROOT}"
  --hlt-cache-dir "${SET_MATCHING_HLT_CACHE_DIR}"
  --reconstructed-view-dir "${SET_MATCHING_RECONSTRUCTED_VIEW_DIR}"
  --train-split stack_train
  --val-split stack_val
  --final-test-split final_test
  --confirm-split-settings
  --seed "${SET_MATCHING_TAGGER_SEED}"
  --batch-size "${SET_MATCHING_TAGGER_BATCH_SIZE}"
  --epochs "${SET_MATCHING_TAGGER_EPOCHS}"
  --lr "${SET_MATCHING_TAGGER_LR}"
  --weight-decay "${SET_MATCHING_TAGGER_WEIGHT_DECAY}"
  --num-workers "${SET_MATCHING_TAGGER_NUM_WORKERS}"
  --device "${SET_MATCHING_TAGGER_DEVICE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --early-stop-patience "${SET_MATCHING_TAGGER_EARLY_STOP_PATIENCE}"
  --max-train-jets "${SET_MATCHING_STACK_TRAIN_SIZE}"
  --max-val-jets "${SET_MATCHING_STACK_VAL_SIZE}"
  --max-final-test-jets "${SET_MATCHING_FINAL_TEST_SIZE}"
  --max-tokens-per-view "${SET_MATCHING_MAX_TOKENS_PER_VIEW}"
  --min-tokens-per-view "${SET_MATCHING_MIN_TOKENS_PER_VIEW}"
  --confidence-threshold "${SET_MATCHING_CONFIDENCE_THRESHOLD}"
  --selection-mode topk_or_threshold
  --view-label-shuffle-seed "${SET_MATCHING_TAGGER_SEED}"
  --embed-dim "${SET_MATCHING_TAGGER_EMBED_DIM}"
  --stage1-layers "${SET_MATCHING_TAGGER_STAGE1_LAYERS}"
  --stage1-heads "${SET_MATCHING_TAGGER_STAGE1_HEADS}"
  --stage2-layers "${SET_MATCHING_TAGGER_STAGE2_LAYERS}"
  --stage2-heads "${SET_MATCHING_TAGGER_STAGE2_HEADS}"
  --mlp-ratio "${SET_MATCHING_TAGGER_MLP_RATIO}"
  --dropout "${SET_MATCHING_TAGGER_DROPOUT}"
  --attention-dropout "${SET_MATCHING_TAGGER_ATTENTION_DROPOUT}"
  --geometry-hidden-dim "${SET_MATCHING_TAGGER_GEOMETRY_HIDDEN_DIM}"
  --geometry-dropout "${SET_MATCHING_TAGGER_GEOMETRY_DROPOUT}"
)
if [[ -n "${SET_MATCHING_NUM_CLASSES}" ]]; then
  cmd+=(--num-classes "${SET_MATCHING_NUM_CLASSES}")
fi
if ((${#label_name_args[@]})); then
  cmd+=(--label-names "${label_name_args[@]}")
fi
if ((${#label_filter_args[@]})); then
  cmd+=(--label-filter-names "${label_filter_args[@]}")
fi
if ((${#drop_views[@]})); then
  cmd+=(--drop-views "${drop_views[@]}")
fi
fresh_append_flag_if_enabled cmd --confirm-final-test "${SET_MATCHING_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --use-geometry-attention "${use_geometry_attention}"
fresh_append_flag_if_enabled cmd --disable-confidence "${disable_confidence}"
fresh_append_flag_if_enabled cmd --shuffle-view-labels "${shuffle_view_labels}"
fresh_append_optional_arg cmd --classifier-hidden-dim "${SET_MATCHING_TAGGER_CLASSIFIER_HIDDEN_DIM}"
fresh_append_optional_arg cmd --max-train-batches "${SET_MATCHING_TAGGER_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${SET_MATCHING_TAGGER_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${SET_MATCHING_TAGGER_MAX_FINAL_TEST_BATCHES}"

fresh_write_run_config "${OUTPUT_DIR}" "set_matching_five_view_tagger_${VARIANT}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/per_class_metrics.csv"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/view_ablation_metrics.json"
fi
