#!/usr/bin/env bash
# Train one aggressive cross-architecture teacher-logit reconstructor.

#SBATCH --job-name=crossarch_aggr_reco_train
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

RECO_ARCHITECTURE="${1:?Usage: sbatch run_crossarch_aggressive_train_reconstructor.sh <aggt|agpn|agpfn|agpcnn> <part|pn|pfn|pcnn>}"
TEACHER_ARCHITECTURE="${2:?Usage: sbatch run_crossarch_aggressive_train_reconstructor.sh <aggt|agpn|agpfn|agpcnn> <part|pn|pfn|pcnn>}"

MODEL_NAME="$(fresh_crossarch_aggressive_reco_model_name "${RECO_ARCHITECTURE}" "${TEACHER_ARCHITECTURE}")"
RECO_CLI_ARCHITECTURE="$(fresh_crossarch_aggressive_reco_cli_architecture "${RECO_ARCHITECTURE}")"
TRAIN_SCRIPT="$(fresh_crossarch_aggressive_reco_train_script "${RECO_ARCHITECTURE}")"
OUTPUT_DIR="${CROSSARCH_AGGRESSIVE_RECO_MODEL_DIR}/${RECO_ARCHITECTURE}/${TEACHER_ARCHITECTURE}"
TEACHER_CHECKPOINT="${CROSSARCH_OFFLINE_TEACHER_DIR}/${TEACHER_ARCHITECTURE}/best_model_val.pt"

: "${NO_AMP:=0}"
: "${COMPILE_MODEL:=0}"
: "${SKIP_HLT_HASH_CHECK:=0}"
: "${VERIFY_LABEL_BRANCHES:=0}"
: "${MAX_CONSTITS:=128}"
: "${TEACHER_WEIGHT_THRESHOLD:=0.0}"

fresh_setup "$@"
fresh_require_file "${TRAIN_SCRIPT}"
fresh_require_file "${CROSSARCH_MANIFEST_PATH}"
fresh_require_file "${CROSSARCH_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${CROSSARCH_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${TEACHER_CHECKPOINT}"
fresh_claim_new_dir "${OUTPUT_DIR}"

fresh_split_words edgeconv_dim_args "${CROSSARCH_AGGRESSIVE_RECO_EDGECONV_DIMS}"
fresh_split_words phi_dim_args "${CROSSARCH_AGGRESSIVE_RECO_PHI_DIMS}"
fresh_split_words context_mlp_dim_args "${CROSSARCH_AGGRESSIVE_RECO_CONTEXT_MLP_DIMS}"
fresh_split_words kernel_size_args "${CROSSARCH_AGGRESSIVE_RECO_KERNEL_SIZES}"
fresh_split_words dilation_args "${CROSSARCH_AGGRESSIVE_RECO_DILATIONS}"

cmd=(
  "${PYTHON_BIN}" "-u" "${TRAIN_SCRIPT}"
  --output-dir "${OUTPUT_DIR}"
  --manifest-path "${CROSSARCH_MANIFEST_PATH}"
  --hlt-cache-dir "${CROSSARCH_HLT_CACHE_DIR}"
  --data-dir "${DATA_DIR}"
  --teacher-checkpoint "${TEACHER_CHECKPOINT}"
  --teacher-architecture "${TEACHER_ARCHITECTURE}"
  --reco-architecture "${RECO_CLI_ARCHITECTURE}"
  --seed "${CROSSARCH_AGGRESSIVE_RECO_SEED}"
  --batch-size "${CROSSARCH_AGGRESSIVE_RECO_BATCH_SIZE}"
  --epochs "${CROSSARCH_AGGRESSIVE_RECO_EPOCHS}"
  --lr "${CROSSARCH_AGGRESSIVE_RECO_LR}"
  --weight-decay "${CROSSARCH_AGGRESSIVE_RECO_WEIGHT_DECAY}"
  --num-workers "${CROSSARCH_AGGRESSIVE_RECO_NUM_WORKERS}"
  --device "${CROSSARCH_AGGRESSIVE_RECO_DEVICE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --early-stop-patience "${CROSSARCH_AGGRESSIVE_RECO_EARLY_STOP_PATIENCE}"
  --max-train-jets "${CROSSARCH_AGGRESSIVE_RECO_MAX_TRAIN_JETS}"
  --max-val-jets "${CROSSARCH_AGGRESSIVE_RECO_MAX_VAL_JETS}"
  --max-constits "${MAX_CONSTITS}"
  --teacher-weight-threshold "${TEACHER_WEIGHT_THRESHOLD}"
  --hidden-dim "${CROSSARCH_AGGRESSIVE_RECO_HIDDEN_DIM}"
  --num-layers "${CROSSARCH_AGGRESSIVE_RECO_NUM_LAYERS}"
  --num-heads "${CROSSARCH_AGGRESSIVE_RECO_NUM_HEADS}"
  --edgeconv-dims "${edgeconv_dim_args[@]}"
  --k "${CROSSARCH_AGGRESSIVE_RECO_K}"
  --phi-dims "${phi_dim_args[@]}"
  --context-dim "${CROSSARCH_AGGRESSIVE_RECO_CONTEXT_DIM}"
  --context-mlp-dims "${context_mlp_dim_args[@]}"
  --hidden-channels "${CROSSARCH_AGGRESSIVE_RECO_HIDDEN_CHANNELS}"
  --num-blocks "${CROSSARCH_AGGRESSIVE_RECO_NUM_BLOCKS}"
  --kernel-sizes "${kernel_size_args[@]}"
  --dilations "${dilation_args[@]}"
  --embedding-dim "${CROSSARCH_AGGRESSIVE_RECO_EMBEDDING_DIM}"
  --dropout "${CROSSARCH_AGGRESSIVE_RECO_DROPOUT}"
  --num-extra-candidates "${CROSSARCH_AGGRESSIVE_RECO_NUM_EXTRA_CANDIDATES}"
  --max-delta-logpt "${CROSSARCH_AGGRESSIVE_RECO_MAX_DELTA_LOGPT}"
  --max-delta-eta "${CROSSARCH_AGGRESSIVE_RECO_MAX_DELTA_ETA}"
  --max-delta-phi "${CROSSARCH_AGGRESSIVE_RECO_MAX_DELTA_PHI}"
  --max-delta-loge "${CROSSARCH_AGGRESSIVE_RECO_MAX_DELTA_LOGE}"
  --parent-weight-bias "${CROSSARCH_AGGRESSIVE_RECO_PARENT_WEIGHT_BIAS}"
  --extra-weight-bias "${CROSSARCH_AGGRESSIVE_RECO_EXTRA_WEIGHT_BIAS}"
  --max-total-extra-pt-fraction "${CROSSARCH_AGGRESSIVE_RECO_MAX_TOTAL_EXTRA_PT_FRACTION}"
  --max-extra-delta-eta "${CROSSARCH_AGGRESSIVE_RECO_MAX_EXTRA_DELTA_ETA}"
  --max-extra-delta-phi "${CROSSARCH_AGGRESSIVE_RECO_MAX_EXTRA_DELTA_PHI}"
  --max-global-logpt-scale "${CROSSARCH_AGGRESSIVE_RECO_MAX_GLOBAL_LOGPT_SCALE}"
  --max-global-loge-scale "${CROSSARCH_AGGRESSIVE_RECO_MAX_GLOBAL_LOGE_SCALE}"
  --max-global-eta-shift "${CROSSARCH_AGGRESSIVE_RECO_MAX_GLOBAL_ETA_SHIFT}"
  --max-global-phi-shift "${CROSSARCH_AGGRESSIVE_RECO_MAX_GLOBAL_PHI_SHIFT}"
  --extra-usage-weight-threshold "${CROSSARCH_AGGRESSIVE_RECO_EXTRA_USAGE_WEIGHT_THRESHOLD}"
  --eta-limit "${CROSSARCH_AGGRESSIVE_RECO_ETA_LIMIT}"
  --min-pt "${CROSSARCH_AGGRESSIVE_RECO_MIN_PT}"
  --teacher-kl-weight "${CROSSARCH_AGGRESSIVE_RECO_TEACHER_KL_WEIGHT}"
  --ce-weight "${CROSSARCH_AGGRESSIVE_RECO_CE_WEIGHT}"
  --correction-budget-weight "${CROSSARCH_AGGRESSIVE_RECO_CORRECTION_BUDGET_WEIGHT}"
  --jet-summary-weight "${CROSSARCH_AGGRESSIVE_RECO_JET_SUMMARY_WEIGHT}"
  --temperature "${CROSSARCH_AGGRESSIVE_RECO_TEMPERATURE}"
  --aggressive-extra-budget-weight "${CROSSARCH_AGGRESSIVE_RECO_EXTRA_BUDGET_WEIGHT}"
  --aggressive-parent-weight-budget-weight "${CROSSARCH_AGGRESSIVE_RECO_PARENT_WEIGHT_BUDGET_WEIGHT}"
  --aggressive-global-calibration-budget-weight "${CROSSARCH_AGGRESSIVE_RECO_GLOBAL_CALIBRATION_BUDGET_WEIGHT}"
  --extra-count-budget-weight "${CROSSARCH_AGGRESSIVE_RECO_EXTRA_COUNT_BUDGET_WEIGHT}"
  --min-parent-weight-fraction "${CROSSARCH_AGGRESSIVE_RECO_MIN_PARENT_WEIGHT_FRACTION}"
  --parent-prune-budget-weight "${CROSSARCH_AGGRESSIVE_RECO_PARENT_PRUNE_BUDGET_WEIGHT}"
)
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${VERIFY_LABEL_BRANCHES}"
fresh_append_optional_arg cmd --max-train-batches "${CROSSARCH_AGGRESSIVE_RECO_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${CROSSARCH_AGGRESSIVE_RECO_MAX_VAL_BATCHES}"

fresh_write_run_config "${OUTPUT_DIR}" "crossarch_aggressive_reco_train_${MODEL_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_assert_json_ok "${OUTPUT_DIR}/run_report.json"
fi
