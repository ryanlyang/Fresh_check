#!/usr/bin/env bash
# Run the Step 13/14 HLT ParT vs subtoken Version A compatibility comparison.

#SBATCH --job-name=subtoken_part
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=2-12:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${SUBTOKEN_PART_ROOT:=${OUTPUT_ROOT}/subtoken_part_qcd_hgg_binary}"
: "${SUBTOKEN_PART_COMPAT_DIR:=${SUBTOKEN_PART_ROOT}/version_a_comparison}"
: "${SUBTOKEN_PART_HLT_CACHE_DIR:=${HLT_CACHE_DIR}}"
: "${SUBTOKEN_PART_VARIANTS:=hlt_part_baseline subtoken_no_gate subtoken_gate_local_only subtoken_gate_context}"
: "${SUBTOKEN_PART_LABEL_FILTER_NAMES:=0 1}"
: "${SUBTOKEN_PART_LABEL_NAMES:=QCD Hgg}"
: "${SUBTOKEN_PART_NUM_CLASSES:=2}"
: "${SUBTOKEN_PART_SEED:=2607}"
: "${SUBTOKEN_PART_BATCH_SIZE:=64}"
: "${SUBTOKEN_PART_EVAL_BATCH_SIZE:=128}"
: "${SUBTOKEN_PART_EPOCHS:=45}"
: "${SUBTOKEN_PART_LR:=0.0003}"
: "${SUBTOKEN_PART_WEIGHT_DECAY:=0.0001}"
: "${SUBTOKEN_PART_NUM_WORKERS:=${NUM_WORKERS}}"
: "${SUBTOKEN_PART_DEVICE:=${DEVICE}}"
: "${SUBTOKEN_PART_GRAD_CLIP_NORM:=${GRAD_CLIP_NORM}}"
: "${SUBTOKEN_PART_EARLY_STOP_PATIENCE:=6}"
: "${SUBTOKEN_PART_MAX_TRAIN_BATCHES:=}"
: "${SUBTOKEN_PART_MAX_VAL_BATCHES:=}"
: "${SUBTOKEN_PART_MAX_STACK_VAL_BATCHES:=}"
: "${SUBTOKEN_PART_MAX_FINAL_TEST_BATCHES:=}"
: "${SUBTOKEN_PART_MODEL_TRAIN_SIZE:=500000}"
: "${SUBTOKEN_PART_MODEL_VAL_SIZE:=150000}"
: "${SUBTOKEN_PART_STACK_VAL_SIZE:=150000}"
: "${SUBTOKEN_PART_FINAL_TEST_SIZE:=500000}"
: "${SUBTOKEN_PART_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"
: "${SUBTOKEN_PART_FINAL_REPORT_DIR:=${SUBTOKEN_PART_COMPAT_DIR}/final_report}"
: "${SUBTOKEN_PART_REPORT_PRIMARY_METRIC:=}"
: "${SUBTOKEN_PART_REPORT_COMPARISON_SPLIT:=}"
: "${SUBTOKEN_PART_REPORT_SKIP_PARAMETER_COUNTS:=0}"
: "${SUBTOKEN_PART_BASELINE_MODEL_SIZE:=base}"
: "${SUBTOKEN_PART_EMBED_DIM:=128}"
: "${SUBTOKEN_PART_LOCAL_LAYERS:=1}"
: "${SUBTOKEN_PART_LOCAL_HEADS:=4}"
: "${SUBTOKEN_PART_CONTEXT_LAYERS:=2}"
: "${SUBTOKEN_PART_CONTEXT_HEADS:=4}"
: "${SUBTOKEN_PART_GLOBAL_LAYERS:=6}"
: "${SUBTOKEN_PART_GLOBAL_HEADS:=8}"
: "${SUBTOKEN_PART_LOCAL_POOL_MODE:=learned_query}"
: "${SUBTOKEN_PART_MODALITY_DROPOUT:=0.0}"
: "${SUBTOKEN_PART_DROPOUT:=0.05}"
: "${SUBTOKEN_PART_ATTENTION_DROPOUT:=0.05}"
: "${SUBTOKEN_PART_ANCHOR_SOURCE:=raw}"
: "${SUBTOKEN_PART_CONFIRM_FINAL_TEST:=1}"
: "${SUBTOKEN_PART_REPORT_CONFIRM_FINAL_TEST:=${SUBTOKEN_PART_CONFIRM_FINAL_TEST}}"
: "${SUBTOKEN_PART_NO_AMP:=0}"
: "${SUBTOKEN_PART_COMPILE_MODEL:=0}"
: "${SUBTOKEN_PART_SKIP_HLT_HASH_CHECK:=0}"
: "${SUBTOKEN_PART_DISABLE_PARTICLE_ANCHOR:=0}"
: "${SUBTOKEN_PART_DISABLE_MODALITY_TYPE_EMBEDDINGS:=0}"
: "${SUBTOKEN_PART_USE_PT_RANK_EMBEDDING:=0}"
: "${SUBTOKEN_PART_DISABLE_PART_STYLE_DERIVED_FEATURES:=0}"

fresh_setup "$@"
fresh_require_file "scripts/run_subtoken_part_compat.py"
fresh_require_file "scripts/write_subtoken_part_report.py"
for split in model_train model_val stack_val final_test; do
  fresh_require_file "${SUBTOKEN_PART_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${SUBTOKEN_PART_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
fresh_claim_new_dir "${SUBTOKEN_PART_COMPAT_DIR}"

fresh_split_words variant_args "${SUBTOKEN_PART_VARIANTS}"
fresh_split_words label_filter_args "${SUBTOKEN_PART_LABEL_FILTER_NAMES}"
fresh_split_words label_name_args "${SUBTOKEN_PART_LABEL_NAMES}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_subtoken_part_compat.py"
  --output-dir "${SUBTOKEN_PART_COMPAT_DIR}"
  --hlt-cache-dir "${SUBTOKEN_PART_HLT_CACHE_DIR}"
  --variants "${variant_args[@]}"
  --train-split model_train
  --val-split model_val
  --stack-val-split stack_val
  --final-test-split final_test
  --confirm-split-settings
  --seed "${SUBTOKEN_PART_SEED}"
  --batch-size "${SUBTOKEN_PART_BATCH_SIZE}"
  --eval-batch-size "${SUBTOKEN_PART_EVAL_BATCH_SIZE}"
  --epochs "${SUBTOKEN_PART_EPOCHS}"
  --lr "${SUBTOKEN_PART_LR}"
  --weight-decay "${SUBTOKEN_PART_WEIGHT_DECAY}"
  --num-workers "${SUBTOKEN_PART_NUM_WORKERS}"
  --device "${SUBTOKEN_PART_DEVICE}"
  --grad-clip-norm "${SUBTOKEN_PART_GRAD_CLIP_NORM}"
  --early-stop-patience "${SUBTOKEN_PART_EARLY_STOP_PATIENCE}"
  --max-train-jets "${SUBTOKEN_PART_MODEL_TRAIN_SIZE}"
  --max-val-jets "${SUBTOKEN_PART_MODEL_VAL_SIZE}"
  --max-stack-val-jets "${SUBTOKEN_PART_STACK_VAL_SIZE}"
  --max-final-test-jets "${SUBTOKEN_PART_FINAL_TEST_SIZE}"
  --selection-metric "${SUBTOKEN_PART_SELECTION_METRIC}"
  --num-classes "${SUBTOKEN_PART_NUM_CLASSES}"
  --label-names "${label_name_args[@]}"
  --label-filter-names "${label_filter_args[@]}"
  --baseline-model-size "${SUBTOKEN_PART_BASELINE_MODEL_SIZE}"
  --embed-dim "${SUBTOKEN_PART_EMBED_DIM}"
  --local-layers "${SUBTOKEN_PART_LOCAL_LAYERS}"
  --local-heads "${SUBTOKEN_PART_LOCAL_HEADS}"
  --context-layers "${SUBTOKEN_PART_CONTEXT_LAYERS}"
  --context-heads "${SUBTOKEN_PART_CONTEXT_HEADS}"
  --global-layers "${SUBTOKEN_PART_GLOBAL_LAYERS}"
  --global-heads "${SUBTOKEN_PART_GLOBAL_HEADS}"
  --local-pool-mode "${SUBTOKEN_PART_LOCAL_POOL_MODE}"
  --modality-dropout "${SUBTOKEN_PART_MODALITY_DROPOUT}"
  --dropout "${SUBTOKEN_PART_DROPOUT}"
  --attention-dropout "${SUBTOKEN_PART_ATTENTION_DROPOUT}"
  --anchor-source "${SUBTOKEN_PART_ANCHOR_SOURCE}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${SUBTOKEN_PART_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --no-amp "${SUBTOKEN_PART_NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${SUBTOKEN_PART_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${SUBTOKEN_PART_SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --disable-particle-anchor "${SUBTOKEN_PART_DISABLE_PARTICLE_ANCHOR}"
fresh_append_flag_if_enabled cmd --disable-modality-type-embeddings "${SUBTOKEN_PART_DISABLE_MODALITY_TYPE_EMBEDDINGS}"
fresh_append_flag_if_enabled cmd --use-pt-rank-embedding "${SUBTOKEN_PART_USE_PT_RANK_EMBEDDING}"
fresh_append_flag_if_enabled cmd --disable-part-style-derived-features "${SUBTOKEN_PART_DISABLE_PART_STYLE_DERIVED_FEATURES}"
fresh_append_optional_arg cmd --max-train-batches "${SUBTOKEN_PART_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${SUBTOKEN_PART_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-stack-val-batches "${SUBTOKEN_PART_MAX_STACK_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${SUBTOKEN_PART_MAX_FINAL_TEST_BATCHES}"

fresh_write_run_config "${SUBTOKEN_PART_COMPAT_DIR}" "subtoken_part_compat_version_a" "${cmd[@]}"
fresh_run "${cmd[@]}"

report_cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_subtoken_part_report.py"
  --experiment-dir "${SUBTOKEN_PART_COMPAT_DIR}"
  --output-dir "${SUBTOKEN_PART_FINAL_REPORT_DIR}"
  --baseline-variant hlt_part_baseline
  --variants "${variant_args[@]}"
)
fresh_append_optional_arg report_cmd --primary-metric "${SUBTOKEN_PART_REPORT_PRIMARY_METRIC}"
fresh_append_optional_arg report_cmd --comparison-split "${SUBTOKEN_PART_REPORT_COMPARISON_SPLIT}"
fresh_append_flag_if_enabled report_cmd --skip-parameter-counts "${SUBTOKEN_PART_REPORT_SKIP_PARAMETER_COUNTS}"
fresh_append_flag_if_enabled report_cmd --confirm-final-test "${SUBTOKEN_PART_REPORT_CONFIRM_FINAL_TEST}"

fresh_write_run_config "${SUBTOKEN_PART_FINAL_REPORT_DIR}" "subtoken_part_final_report" "${report_cmd[@]}"
fresh_run "${report_cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${SUBTOKEN_PART_COMPAT_DIR}/run_report.json"
  fresh_require_file "${SUBTOKEN_PART_COMPAT_DIR}/model_val_report.json"
  fresh_require_file "${SUBTOKEN_PART_COMPAT_DIR}/diagnostics/comparison_metrics.csv"
  fresh_require_file "${SUBTOKEN_PART_FINAL_REPORT_DIR}/subtoken_part_final_report.json"
  fresh_require_file "${SUBTOKEN_PART_FINAL_REPORT_DIR}/subtoken_part_final_report.md"
  fresh_require_file "${SUBTOKEN_PART_FINAL_REPORT_DIR}/metric_table.csv"
  fresh_require_file "${SUBTOKEN_PART_FINAL_REPORT_DIR}/gate_diagnostics.csv"
  fresh_require_file "${SUBTOKEN_PART_FINAL_REPORT_DIR}/parameter_counts.csv"
  fresh_require_file "${SUBTOKEN_PART_FINAL_REPORT_DIR}/runtime_summary.csv"
  for variant in "${variant_args[@]}"; do
    fresh_require_file "${SUBTOKEN_PART_COMPAT_DIR}/${variant}/best_model_val.pt"
    fresh_require_file "${SUBTOKEN_PART_COMPAT_DIR}/${variant}/run_report.json"
    fresh_require_file "${SUBTOKEN_PART_COMPAT_DIR}/${variant}/diagnostics/summary_metrics.csv"
  done
fi
