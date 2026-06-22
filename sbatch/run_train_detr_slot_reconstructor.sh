#!/usr/bin/env bash
# Train one DETR/free-slot reconstructor.

#SBATCH --job-name=detrslot_reco_train
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
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ARCHITECTURE="${1:?Usage: sbatch run_train_detr_slot_reconstructor.sh <gt|pn|pfn|pcnn>}"
OUTPUT_DIR="${DETR_SLOT_RECONSTRUCTOR_DIR}/${ARCHITECTURE}"

: "${NO_AMP:=0}"
: "${COMPILE_MODEL:=0}"
: "${SKIP_HLT_HASH_CHECK:=0}"
: "${VERIFY_LABEL_BRANCHES:=0}"
: "${READ_CHUNK_SIZE:=50000}"
: "${DETR_SLOT_NO_TRIM_TO_VALID:=0}"

fresh_setup "$@"
fresh_require_data_dir
fresh_require_file "scripts/train_detr_slot_reconstructor.py"
fresh_require_file "${DETR_SLOT_MANIFEST_PATH}"
fresh_require_file "${DETR_SLOT_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${DETR_SLOT_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_claim_new_dir "${OUTPUT_DIR}"

fresh_split_words edgeconv_dim_args "${DETR_SLOT_EDGECONV_DIMS}"
fresh_split_words phi_dim_args "${DETR_SLOT_PHI_DIMS}"
fresh_split_words context_mlp_dim_args "${DETR_SLOT_CONTEXT_MLP_DIMS}"
fresh_split_words kernel_size_args "${DETR_SLOT_KERNEL_SIZES}"
fresh_split_words dilation_args "${DETR_SLOT_DILATIONS}"
fresh_split_words label_filter_args "${DETR_SLOT_LABEL_FILTER_NAMES}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_detr_slot_reconstructor.py"
  --architecture "${ARCHITECTURE}"
  --hlt-cache-dir "${DETR_SLOT_HLT_CACHE_DIR}"
  --data-dir "${DATA_DIR}"
  --manifest-path "${DETR_SLOT_MANIFEST_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --train-split model_train
  --val-split model_val
  --confirm-split-settings
  --seed "${DETR_SLOT_RECO_SEED}"
  --batch-size "${DETR_SLOT_RECO_BATCH_SIZE}"
  --epochs "${DETR_SLOT_RECO_EPOCHS}"
  --lr "${DETR_SLOT_RECO_LR}"
  --weight-decay "${DETR_SLOT_RECO_WEIGHT_DECAY}"
  --num-workers "${DETR_SLOT_RECO_NUM_WORKERS}"
  --device "${DETR_SLOT_RECO_DEVICE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --early-stop-patience "${DETR_SLOT_RECO_EARLY_STOP_PATIENCE}"
  --max-train-jets "${DETR_SLOT_MODEL_TRAIN_SIZE}"
  --max-val-jets "${DETR_SLOT_MODEL_VAL_SIZE}"
  --read-chunk-size "${READ_CHUNK_SIZE}"
  --num-slots "${DETR_SLOT_NUM_SLOTS}"
  --export-max-tokens "${DETR_SLOT_EXPORT_MAX_TOKENS}"
  --memory-dim "${DETR_SLOT_MEMORY_DIM}"
  --embed-dim "${DETR_SLOT_EMBED_DIM}"
  --dropout "${DETR_SLOT_DROPOUT}"
  --max-abs-eta "${DETR_SLOT_MAX_ABS_ETA}"
  --decoder-layers "${DETR_SLOT_DECODER_LAYERS}"
  --decoder-heads "${DETR_SLOT_DECODER_HEADS}"
  --decoder-mlp-ratio "${DETR_SLOT_DECODER_MLP_RATIO}"
  --head-hidden-dim "${DETR_SLOT_HEAD_HIDDEN_DIM}"
  --existence-bias "${DETR_SLOT_EXISTENCE_BIAS}"
  --core-output-scale "${DETR_SLOT_CORE_OUTPUT_SCALE}"
  --gt-layers "${DETR_SLOT_GT_LAYERS}"
  --gt-heads "${DETR_SLOT_GT_HEADS}"
  --gt-mlp-ratio "${DETR_SLOT_GT_MLP_RATIO}"
  --edgeconv-dims "${edgeconv_dim_args[@]}"
  --k "${DETR_SLOT_K}"
  --phi-dims "${phi_dim_args[@]}"
  --context-mlp-dims "${context_mlp_dim_args[@]}"
  --hidden-channels "${DETR_SLOT_HIDDEN_CHANNELS}"
  --kernel-sizes "${kernel_size_args[@]}"
  --dilations "${dilation_args[@]}"
  --assignment-aux-weight "${DETR_SLOT_ASSIGNMENT_AUX_WEIGHT}"
  --matched-core-weight "${DETR_SLOT_MATCHED_CORE_WEIGHT}"
  --matched-aux-weight "${DETR_SLOT_MATCHED_AUX_WEIGHT}"
  --existence-weight "${DETR_SLOT_EXISTENCE_WEIGHT}"
  --existence-positive-weight "${DETR_SLOT_EXISTENCE_POSITIVE_WEIGHT}"
  --existence-negative-weight "${DETR_SLOT_EXISTENCE_NEGATIVE_WEIGHT}"
  --count-weight "${DETR_SLOT_COUNT_WEIGHT}"
  --jet-summary-weight "${DETR_SLOT_JET_SUMMARY_WEIGHT}"
  --duplicate-weight "${DETR_SLOT_DUPLICATE_WEIGHT}"
  --hlt-support-weight "${DETR_SLOT_HLT_SUPPORT_WEIGHT}"
  --max-nearest-hlt-delta-r "${DETR_SLOT_MAX_NEAREST_HLT_DELTA_R}"
  --duplicate-delta-r-scale "${DETR_SLOT_DUPLICATE_DELTA_R_SCALE}"
  --duplicate-probability-threshold "${DETR_SLOT_DUPLICATE_PROBABILITY_THRESHOLD}"
  --max-count-for-summary "${DETR_SLOT_MAX_COUNT_FOR_SUMMARY}"
  --huber-beta "${DETR_SLOT_HUBER_BETA}"
  --brute-force-fallback-limit "${DETR_SLOT_BRUTE_FORCE_FALLBACK_LIMIT}"
)
if [[ -n "${DETR_SLOT_CONTEXT_DIM}" ]]; then
  cmd+=(--context-dim "${DETR_SLOT_CONTEXT_DIM}")
fi
if ((${#label_filter_args[@]})); then
  cmd+=(--label-filter-names "${label_filter_args[@]}")
fi
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${VERIFY_LABEL_BRANCHES}"
fresh_append_flag_if_enabled cmd --no-trim-to-valid "${DETR_SLOT_NO_TRIM_TO_VALID}"
fresh_append_flag_if_enabled cmd --allow-bruteforce-fallback "${DETR_SLOT_ALLOW_BRUTEFORCE_FALLBACK}"
fresh_append_optional_arg cmd --max-train-batches "${DETR_SLOT_RECO_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${DETR_SLOT_RECO_MAX_VAL_BATCHES}"

fresh_write_run_config "${OUTPUT_DIR}" "detr_slot_reco_train_${ARCHITECTURE}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
fi
