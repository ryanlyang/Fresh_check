#!/usr/bin/env bash
# Train one local particle residual-field reconstructor variant.

#SBATCH --job-name=lprf_reco
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

RUN_ID="${1:?Usage: sbatch run_train_local_residual_reconstructor.sh <C0-C6>}"

: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field}"
: "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/split_manifest.json.gz}"
: "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/hlt_cache}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/targets}"
: "${LOCAL_RESIDUAL_FIELD_RECON_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/reconstructors}"
: "${LOCAL_RESIDUAL_FIELD_RECO_SEED:=10421}"
: "${LOCAL_RESIDUAL_FIELD_RECO_BATCH_SIZE:=128}"
: "${LOCAL_RESIDUAL_FIELD_RECO_EVAL_BATCH_SIZE:=256}"
: "${LOCAL_RESIDUAL_FIELD_RECO_EPOCHS:=60}"
: "${LOCAL_RESIDUAL_FIELD_RECO_LR:=0.0003}"
: "${LOCAL_RESIDUAL_FIELD_RECO_WEIGHT_DECAY:=0.0001}"
: "${LOCAL_RESIDUAL_FIELD_RECO_NUM_WORKERS:=4}"
: "${LOCAL_RESIDUAL_FIELD_RECO_GRAD_CLIP_NORM:=1.0}"
: "${LOCAL_RESIDUAL_FIELD_RECO_EARLY_STOP_PATIENCE:=8}"
: "${LOCAL_RESIDUAL_FIELD_RECO_D_MODEL:=160}"
: "${LOCAL_RESIDUAL_FIELD_RECO_NUM_HEADS:=5}"
: "${LOCAL_RESIDUAL_FIELD_RECO_NUM_LAYERS:=4}"
: "${LOCAL_RESIDUAL_FIELD_RECO_CONTEXT_LAYERS:=1}"
: "${LOCAL_RESIDUAL_FIELD_RECO_MLP_RATIO:=2.0}"
: "${LOCAL_RESIDUAL_FIELD_RECO_DROPOUT:=0.05}"
: "${LOCAL_RESIDUAL_FIELD_RECO_ATTENTION_DROPOUT:=0.05}"
: "${LOCAL_RESIDUAL_FIELD_RECO_LOCAL_RADIUS:=0.12}"
: "${LOCAL_RESIDUAL_FIELD_RECO_HARD_LOCAL_RADIUS:=0.08}"
: "${LOCAL_RESIDUAL_FIELD_RECO_MAX_TRAIN_JETS:=}"
: "${LOCAL_RESIDUAL_FIELD_RECO_MAX_VAL_JETS:=}"
: "${LOCAL_RESIDUAL_FIELD_RECO_MAX_STACK_VAL_JETS:=}"
: "${LOCAL_RESIDUAL_FIELD_RECO_SAVE_LAST_CHECKPOINT:=1}"
: "${LOCAL_RESIDUAL_FIELD_RECO_DISABLE_AMP:=0}"
: "${LOCAL_RESIDUAL_FIELD_NO_VERIFY_HASH:=0}"

variant="${RUN_ID}"
consistency_weight="0.0"
uncertainty_weight="1.0"
case "${RUN_ID}" in
  C0) variant="C0" ;;
  C1) variant="C1" ;;
  C2) variant="C2" ;;
  C3) variant="C3" ;;
  C4) variant="C4" ;;
  C5) variant="C5"; uncertainty_weight="${LOCAL_RESIDUAL_FIELD_C5_UNCERTAINTY_LOSS_WEIGHT:=1.0}" ;;
  C6) variant="C6"; consistency_weight="${LOCAL_RESIDUAL_FIELD_C6_CONSISTENCY_LOSS_WEIGHT:=0.25}" ;;
  *)
    echo "Unknown local residual-field reconstructor RUN_ID: ${RUN_ID}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${LOCAL_RESIDUAL_FIELD_RECON_ROOT}/${RUN_ID}"

fresh_setup "$@"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_local_residual_reconstructor.py"
  --output-dir "${OUTPUT_DIR}"
  --hlt-cache-dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
  --target-cache-dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
  --manifest-path "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
  --seed "${LOCAL_RESIDUAL_FIELD_RECO_SEED}"
  --batch-size "${LOCAL_RESIDUAL_FIELD_RECO_BATCH_SIZE}"
  --eval-batch-size "${LOCAL_RESIDUAL_FIELD_RECO_EVAL_BATCH_SIZE}"
  --epochs "${LOCAL_RESIDUAL_FIELD_RECO_EPOCHS}"
  --lr "${LOCAL_RESIDUAL_FIELD_RECO_LR}"
  --weight-decay "${LOCAL_RESIDUAL_FIELD_RECO_WEIGHT_DECAY}"
  --num-workers "${LOCAL_RESIDUAL_FIELD_RECO_NUM_WORKERS}"
  --device "${DEVICE}"
  --grad-clip-norm "${LOCAL_RESIDUAL_FIELD_RECO_GRAD_CLIP_NORM}"
  --early-stop-patience "${LOCAL_RESIDUAL_FIELD_RECO_EARLY_STOP_PATIENCE}"
  --variant "${variant}"
  --d-model "${LOCAL_RESIDUAL_FIELD_RECO_D_MODEL}"
  --num-heads "${LOCAL_RESIDUAL_FIELD_RECO_NUM_HEADS}"
  --num-layers "${LOCAL_RESIDUAL_FIELD_RECO_NUM_LAYERS}"
  --context-layers "${LOCAL_RESIDUAL_FIELD_RECO_CONTEXT_LAYERS}"
  --mlp-ratio "${LOCAL_RESIDUAL_FIELD_RECO_MLP_RATIO}"
  --dropout "${LOCAL_RESIDUAL_FIELD_RECO_DROPOUT}"
  --attention-dropout "${LOCAL_RESIDUAL_FIELD_RECO_ATTENTION_DROPOUT}"
  --local-radius "${LOCAL_RESIDUAL_FIELD_RECO_LOCAL_RADIUS}"
  --hard-local-radius "${LOCAL_RESIDUAL_FIELD_RECO_HARD_LOCAL_RADIUS}"
  --uncertainty-loss-weight "${uncertainty_weight}"
  --consistency-loss-weight "${consistency_weight}"
)
fresh_append_flag_if_enabled cmd --disable-amp "${LOCAL_RESIDUAL_FIELD_RECO_DISABLE_AMP}"
fresh_append_flag_if_enabled cmd --no-verify-hash "${LOCAL_RESIDUAL_FIELD_NO_VERIFY_HASH}"
if ! fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_RECO_SAVE_LAST_CHECKPOINT}"; then cmd+=(--no-save-last-checkpoint); fi
if [[ -n "${LOCAL_RESIDUAL_FIELD_RECO_MAX_TRAIN_JETS}" ]]; then cmd+=(--max-train-jets "${LOCAL_RESIDUAL_FIELD_RECO_MAX_TRAIN_JETS}"); fi
if [[ -n "${LOCAL_RESIDUAL_FIELD_RECO_MAX_VAL_JETS}" ]]; then cmd+=(--max-val-jets "${LOCAL_RESIDUAL_FIELD_RECO_MAX_VAL_JETS}"); fi
if [[ -n "${LOCAL_RESIDUAL_FIELD_RECO_MAX_STACK_VAL_JETS}" ]]; then cmd+=(--max-stack-val-jets "${LOCAL_RESIDUAL_FIELD_RECO_MAX_STACK_VAL_JETS}"); fi

fresh_write_run_config "${OUTPUT_DIR}" "local_residual_reconstructor_${RUN_ID}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
fi
