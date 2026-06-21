#!/usr/bin/env bash
# Train one set-matching reconstructor for the multi-view branch.

#SBATCH --job-name=setmatch_reco_train
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

ARCHITECTURE="${1:?Usage: sbatch run_train_set_matching_reconstructor.sh <gt|pn|pfn|pcnn>}"
OUTPUT_DIR="${SET_MATCHING_RECONSTRUCTOR_DIR}/${ARCHITECTURE}"

: "${NO_AMP:=0}"
: "${COMPILE_MODEL:=0}"
: "${SKIP_HLT_HASH_CHECK:=0}"
: "${VERIFY_LABEL_BRANCHES:=0}"
: "${READ_CHUNK_SIZE:=50000}"
: "${SKIP_CORE_NORMALIZATION:=0}"

fresh_setup "$@"
fresh_require_data_dir
fresh_require_file "scripts/train_set_matching_reconstructor.py"
fresh_require_file "${SET_MATCHING_MANIFEST_PATH}"
fresh_require_file "${SET_MATCHING_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${SET_MATCHING_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_set_matching_reconstructor.py"
  --architecture "${ARCHITECTURE}"
  --hlt-cache-dir "${SET_MATCHING_HLT_CACHE_DIR}"
  --data-dir "${DATA_DIR}"
  --manifest-path "${SET_MATCHING_MANIFEST_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --train-split model_train
  --val-split model_val
  --confirm-split-settings
  --seed "${SET_MATCHING_RECO_SEED}"
  --batch-size "${SET_MATCHING_RECO_BATCH_SIZE}"
  --epochs "${SET_MATCHING_RECO_EPOCHS}"
  --lr "${SET_MATCHING_RECO_LR}"
  --weight-decay "${SET_MATCHING_RECO_WEIGHT_DECAY}"
  --num-workers "${SET_MATCHING_RECO_NUM_WORKERS}"
  --device "${SET_MATCHING_RECO_DEVICE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --early-stop-patience "${SET_MATCHING_RECO_EARLY_STOP_PATIENCE}"
  --max-train-jets "${SET_MATCHING_MODEL_TRAIN_SIZE}"
  --max-val-jets "${SET_MATCHING_MODEL_VAL_SIZE}"
  --read-chunk-size "${READ_CHUNK_SIZE}"
  --max-slots "${SET_MATCHING_MAX_SLOTS}"
  --hidden-dim "${SET_MATCHING_RECO_HIDDEN_DIM}"
  --num-layers "${SET_MATCHING_RECO_NUM_LAYERS}"
  --num-heads "${SET_MATCHING_RECO_NUM_HEADS}"
  --k "${SET_MATCHING_RECO_K}"
  --context-dim "${SET_MATCHING_RECO_CONTEXT_DIM}"
  --hidden-channels "${SET_MATCHING_RECO_HIDDEN_CHANNELS}"
  --num-blocks "${SET_MATCHING_RECO_NUM_BLOCKS}"
  --embedding-dim "${SET_MATCHING_RECO_EMBEDDING_DIM}"
  --dropout "${SET_MATCHING_RECO_DROPOUT}"
  --num-extra-candidates "${SET_MATCHING_RECO_NUM_EXTRA_CANDIDATES}"
  --matched-core-weight "${SET_MATCHING_MATCHED_CORE_WEIGHT}"
  --matched-aux-weight "${SET_MATCHING_MATCHED_AUX_WEIGHT}"
  --existence-weight "${SET_MATCHING_EXISTENCE_WEIGHT}"
  --existence-positive-weight "${SET_MATCHING_EXISTENCE_POSITIVE_WEIGHT}"
  --count-weight "${SET_MATCHING_COUNT_WEIGHT}"
  --jet-summary-weight "${SET_MATCHING_JET_SUMMARY_WEIGHT}"
  --correction-budget-weight "${SET_MATCHING_CORRECTION_BUDGET_WEIGHT}"
  --chamfer-weight "${SET_MATCHING_CHAMFER_WEIGHT}"
  --missing-target-weight "${SET_MATCHING_MISSING_TARGET_WEIGHT}"
  --huber-beta "${SET_MATCHING_HUBER_BETA}"
  --max-abs-eta "${SET_MATCHING_MAX_ABS_ETA}"
  --hlt-support-budget-weight "${SET_MATCHING_HLT_SUPPORT_BUDGET_WEIGHT}"
  --max-nearest-hlt-delta-r "${SET_MATCHING_MAX_NEAREST_HLT_DELTA_R}"
)
fresh_split_words edgeconv_dim_args "${SET_MATCHING_RECO_EDGECONV_DIMS}"
fresh_split_words phi_dim_args "${SET_MATCHING_RECO_PHI_DIMS}"
fresh_split_words context_mlp_dim_args "${SET_MATCHING_RECO_CONTEXT_MLP_DIMS}"
fresh_split_words kernel_size_args "${SET_MATCHING_RECO_KERNEL_SIZES}"
fresh_split_words dilation_args "${SET_MATCHING_RECO_DILATIONS}"
fresh_split_words label_filter_args "${SET_MATCHING_LABEL_FILTER_NAMES}"
cmd+=(
  --edgeconv-dims "${edgeconv_dim_args[@]}"
  --phi-dims "${phi_dim_args[@]}"
  --context-mlp-dims "${context_mlp_dim_args[@]}"
  --kernel-sizes "${kernel_size_args[@]}"
  --dilations "${dilation_args[@]}"
)
if ((${#label_filter_args[@]})); then
  cmd+=(--label-filter-names "${label_filter_args[@]}")
fi
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${VERIFY_LABEL_BRANCHES}"
fresh_append_flag_if_enabled cmd --skip-core-normalization "${SKIP_CORE_NORMALIZATION}"
fresh_append_optional_arg cmd --max-train-batches "${SET_MATCHING_RECO_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${SET_MATCHING_RECO_MAX_VAL_BATCHES}"

fresh_write_run_config "${OUTPUT_DIR}" "set_matching_reco_train_${ARCHITECTURE}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
fi
