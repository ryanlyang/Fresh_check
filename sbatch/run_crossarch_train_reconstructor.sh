#!/usr/bin/env bash
# Train one cross-architecture teacher-logit reconstructor.

#SBATCH --job-name=crossarch_reco_train
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

RECO_ARCHITECTURE="${1:?Usage: sbatch run_crossarch_train_reconstructor.sh <gt|pn|pfn|pcnn> <part|pn|pfn|pcnn>}"
TEACHER_ARCHITECTURE="${2:?Usage: sbatch run_crossarch_train_reconstructor.sh <gt|pn|pfn|pcnn> <part|pn|pfn|pcnn>}"

MODEL_NAME="$(fresh_crossarch_reco_model_name "${RECO_ARCHITECTURE}" "${TEACHER_ARCHITECTURE}")"
TRAIN_SCRIPT="$(fresh_crossarch_reco_train_script "${RECO_ARCHITECTURE}")"
OUTPUT_DIR="${CROSSARCH_RECO_MODEL_DIR}/${RECO_ARCHITECTURE}/${TEACHER_ARCHITECTURE}"
TEACHER_CHECKPOINT="${CROSSARCH_OFFLINE_TEACHER_DIR}/${TEACHER_ARCHITECTURE}/best_model_val.pt"

: "${NO_AMP:=0}"
: "${COMPILE_MODEL:=0}"
: "${SKIP_HLT_HASH_CHECK:=0}"
: "${VERIFY_LABEL_BRANCHES:=0}"
: "${READ_CHUNK_SIZE:=50000}"
: "${MAX_CONSTITS:=128}"
: "${TEACHER_WEIGHT_THRESHOLD:=0.0}"

fresh_setup "$@"
fresh_require_data_dir
fresh_require_file "${TRAIN_SCRIPT}"
fresh_require_file "${CROSSARCH_MANIFEST_PATH}"
fresh_require_file "${CROSSARCH_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${CROSSARCH_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${TEACHER_CHECKPOINT}"
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "${TRAIN_SCRIPT}"
  --output-dir "${OUTPUT_DIR}"
  --manifest-path "${CROSSARCH_MANIFEST_PATH}"
  --hlt-cache-dir "${CROSSARCH_HLT_CACHE_DIR}"
  --data-dir "${DATA_DIR}"
  --teacher-checkpoint "${TEACHER_CHECKPOINT}"
  --teacher-architecture "${TEACHER_ARCHITECTURE}"
  --train-split model_train
  --val-split model_val
  --seed "${CROSSARCH_RECO_SEED}"
  --batch-size "${CROSSARCH_RECO_BATCH_SIZE}"
  --epochs "${CROSSARCH_RECO_EPOCHS}"
  --lr "${CROSSARCH_RECO_LR}"
  --weight-decay "${CROSSARCH_RECO_WEIGHT_DECAY}"
  --num-workers "${CROSSARCH_RECO_NUM_WORKERS}"
  --device "${CROSSARCH_RECO_DEVICE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --early-stop-patience "${CROSSARCH_RECO_EARLY_STOP_PATIENCE}"
  --max-train-jets "${CROSSARCH_RECO_MAX_TRAIN_JETS}"
  --max-val-jets "${CROSSARCH_RECO_MAX_VAL_JETS}"
  --read-chunk-size "${READ_CHUNK_SIZE}"
  --max-constits "${MAX_CONSTITS}"
  --teacher-weight-threshold "${TEACHER_WEIGHT_THRESHOLD}"
)
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${VERIFY_LABEL_BRANCHES}"
fresh_append_optional_arg cmd --max-train-batches "${CROSSARCH_RECO_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${CROSSARCH_RECO_MAX_VAL_BATCHES}"

case "${RECO_ARCHITECTURE}" in
  gt)
    cmd+=(
      --hidden-dim "${TEACHER_LOGIT_GT_HIDDEN_DIM}"
      --num-layers "${TEACHER_LOGIT_GT_NUM_LAYERS}"
      --num-heads "${TEACHER_LOGIT_GT_NUM_HEADS}"
      --num-extra-candidates "${TEACHER_LOGIT_GT_NUM_EXTRA_CANDIDATES}"
      --dropout "${TEACHER_LOGIT_GT_DROPOUT}"
    )
    ;;
  pn)
    fresh_split_words edgeconv_dim_args "${TEACHER_LOGIT_PN_EDGECONV_DIMS}"
    cmd+=(
      --edgeconv-dims "${edgeconv_dim_args[@]}"
      --k "${TEACHER_LOGIT_PN_K}"
      --num-extra-candidates "${TEACHER_LOGIT_PN_NUM_EXTRA_CANDIDATES}"
      --dropout "${TEACHER_LOGIT_PN_DROPOUT}"
    )
    ;;
  pfn)
    fresh_split_words phi_dim_args "${TEACHER_LOGIT_PFN_PHI_DIMS}"
    fresh_split_words pfn_context_dim_args "${TEACHER_LOGIT_PFN_CONTEXT_DIMS}"
    fresh_split_words pfn_decoder_dim_args "${TEACHER_LOGIT_PFN_DECODER_DIMS}"
    cmd+=(
      --phi-dims "${phi_dim_args[@]}"
      --context-dim "${TEACHER_LOGIT_PFN_CONTEXT_DIM}"
      --context-dims "${pfn_context_dim_args[@]}"
      --decoder-dims "${pfn_decoder_dim_args[@]}"
      --num-extra-candidates "${TEACHER_LOGIT_PFN_NUM_EXTRA_CANDIDATES}"
      --dropout "${TEACHER_LOGIT_PFN_DROPOUT}"
    )
    fresh_append_optional_arg cmd --slot-dim "${TEACHER_LOGIT_PFN_SLOT_DIM}"
    ;;
  pcnn)
    fresh_split_words kernel_size_args "${TEACHER_LOGIT_PCNN_KERNEL_SIZES}"
    fresh_split_words dilation_args "${TEACHER_LOGIT_PCNN_DILATIONS}"
    fresh_split_words pcnn_context_dim_args "${TEACHER_LOGIT_PCNN_CONTEXT_DIMS}"
    fresh_split_words pcnn_decoder_dim_args "${TEACHER_LOGIT_PCNN_DECODER_DIMS}"
    cmd+=(
      --hidden-channels "${TEACHER_LOGIT_PCNN_HIDDEN_CHANNELS}"
      --num-blocks "${TEACHER_LOGIT_PCNN_NUM_BLOCKS}"
      --kernel-sizes "${kernel_size_args[@]}"
      --dilations "${dilation_args[@]}"
      --context-dim "${TEACHER_LOGIT_PCNN_CONTEXT_DIM}"
      --context-dims "${pcnn_context_dim_args[@]}"
      --decoder-dims "${pcnn_decoder_dim_args[@]}"
      --num-extra-candidates "${TEACHER_LOGIT_PCNN_NUM_EXTRA_CANDIDATES}"
      --dropout "${TEACHER_LOGIT_PCNN_DROPOUT}"
    )
    fresh_append_optional_arg cmd --slot-dim "${TEACHER_LOGIT_PCNN_SLOT_DIM}"
    ;;
esac

fresh_write_run_config "${OUTPUT_DIR}" "crossarch_reco_train_${MODEL_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
fi
