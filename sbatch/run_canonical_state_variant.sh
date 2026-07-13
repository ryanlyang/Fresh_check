#!/usr/bin/env bash
# Run one canonical-state A0-G3 variant job.

#SBATCH --job-name=cstate_variant
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=3-00:00:00
#SBATCH --mem=220G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

RUN_ID="${1:?Usage: sbatch run_canonical_state_variant.sh <A0-G3 run_id>}"

: "${CANONICAL_STATE_ROOT:=${OUTPUT_ROOT}/canonical_multi_scale_jet_state}"
: "${CANONICAL_STATE_MANIFEST_PATH:=${CANONICAL_STATE_ROOT}/inputs/split_manifest.json.gz}"
: "${CANONICAL_STATE_HLT_CACHE_DIR:=${CANONICAL_STATE_ROOT}/inputs/hlt_cache}"
: "${CANONICAL_STATE_PHI_HLT_CACHE_DIR:=${CANONICAL_STATE_ROOT}/phi_cache/hlt}"
: "${CANONICAL_STATE_PHI_OFFLINE_CACHE_DIR:=${CANONICAL_STATE_ROOT}/phi_cache/offline}"
: "${CANONICAL_STATE_RUN_ROOT:=${CANONICAL_STATE_ROOT}/runs}"
: "${CANONICAL_STATE_BASELINE_CHECKPOINT:=${CANONICAL_STATE_RUN_ROOT}/A0/best_model_val.pt}"
: "${CANONICAL_STATE_EMIT_PLANNING_STUB:=0}"
: "${CANONICAL_STATE_SEED:=10101}"
: "${CANONICAL_STATE_BATCH_SIZE:=64}"
: "${CANONICAL_STATE_EVAL_BATCH_SIZE:=128}"
: "${CANONICAL_STATE_EPOCHS:=45}"
: "${CANONICAL_STATE_WARMUP_EPOCHS:=2}"
: "${CANONICAL_STATE_ADAPTER_WARMUP_EPOCHS:=2}"
: "${CANONICAL_STATE_PART_LR:=0.00003}"
: "${CANONICAL_STATE_ADAPTER_LR:=0.0003}"
: "${CANONICAL_STATE_PREDICTOR_LR:=0.0003}"
: "${CANONICAL_STATE_HEAD_LR:=0.0001}"
: "${CANONICAL_STATE_WEIGHT_DECAY:=0.0001}"
: "${CANONICAL_STATE_NUM_WORKERS:=4}"
: "${DEVICE:=auto}"
: "${CANONICAL_STATE_DISABLE_AMP:=0}"
: "${CANONICAL_STATE_GRAD_CLIP_NORM:=1.0}"
: "${CANONICAL_STATE_EARLY_STOP_PATIENCE:=6}"
: "${CANONICAL_STATE_MODEL_SIZE:=base}"
: "${CANONICAL_STATE_MODEL_TRAIN_SIZE:=}"
: "${CANONICAL_STATE_MODEL_VAL_SIZE:=}"
: "${CANONICAL_STATE_STACK_TRAIN_SIZE:=}"
: "${CANONICAL_STATE_STACK_VAL_SIZE:=}"
: "${CANONICAL_STATE_FINAL_TEST_SIZE:=}"
: "${CANONICAL_STATE_CHECKPOINT_POLICY:=all}"
: "${CANONICAL_STATE_SAVE_LAST_CHECKPOINT:=1}"

OUTPUT_DIR="${CANONICAL_STATE_RUN_ROOT}/${RUN_ID}"

fresh_setup "$@"
fresh_require_file "${CANONICAL_STATE_MANIFEST_PATH}"
fresh_require_dir "${CANONICAL_STATE_HLT_CACHE_DIR}"
fresh_require_dir "${CANONICAL_STATE_PHI_HLT_CACHE_DIR}"
case "${RUN_ID}" in
  C*|D*|E*|F*|G*)
    fresh_require_dir "${CANONICAL_STATE_PHI_OFFLINE_CACHE_DIR}"
    ;;
esac
if [[ "${RUN_ID}" != "A0" && -n "${CANONICAL_STATE_BASELINE_CHECKPOINT}" ]] && ! fresh_bool_enabled "${CANONICAL_STATE_EMIT_PLANNING_STUB}"; then
  fresh_require_file "${CANONICAL_STATE_BASELINE_CHECKPOINT}"
fi
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_canonical_state_variant.py"
  --run-id "${RUN_ID}"
  --output-dir "${OUTPUT_DIR}"
  --manifest "${CANONICAL_STATE_MANIFEST_PATH}"
  --hlt-cache-dir "${CANONICAL_STATE_HLT_CACHE_DIR}"
  --phi-hlt-cache-dir "${CANONICAL_STATE_PHI_HLT_CACHE_DIR}"
  --phi-offline-cache-dir "${CANONICAL_STATE_PHI_OFFLINE_CACHE_DIR}"
  --baseline-checkpoint "${CANONICAL_STATE_BASELINE_CHECKPOINT}"
  --variant-root "${CANONICAL_STATE_RUN_ROOT}"
  --seed "${CANONICAL_STATE_SEED}"
  --batch-size "${CANONICAL_STATE_BATCH_SIZE}"
  --eval-batch-size "${CANONICAL_STATE_EVAL_BATCH_SIZE}"
  --epochs "${CANONICAL_STATE_EPOCHS}"
  --warmup-epochs "${CANONICAL_STATE_WARMUP_EPOCHS}"
  --adapter-warmup-epochs "${CANONICAL_STATE_ADAPTER_WARMUP_EPOCHS}"
  --part-lr "${CANONICAL_STATE_PART_LR}"
  --adapter-lr "${CANONICAL_STATE_ADAPTER_LR}"
  --predictor-lr "${CANONICAL_STATE_PREDICTOR_LR}"
  --head-lr "${CANONICAL_STATE_HEAD_LR}"
  --weight-decay "${CANONICAL_STATE_WEIGHT_DECAY}"
  --num-workers "${CANONICAL_STATE_NUM_WORKERS}"
  --device "${DEVICE}"
  --grad-clip-norm "${CANONICAL_STATE_GRAD_CLIP_NORM}"
  --early-stop-patience "${CANONICAL_STATE_EARLY_STOP_PATIENCE}"
  --model-size "${CANONICAL_STATE_MODEL_SIZE}"
  --checkpoint-policy "${CANONICAL_STATE_CHECKPOINT_POLICY}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --emit-planning-stub "${CANONICAL_STATE_EMIT_PLANNING_STUB}"
fresh_append_flag_if_enabled cmd --disable-amp "${CANONICAL_STATE_DISABLE_AMP}"
if ! fresh_bool_enabled "${CANONICAL_STATE_SAVE_LAST_CHECKPOINT}"; then cmd+=(--no-save-last-checkpoint); fi
if [[ -n "${CANONICAL_STATE_MODEL_TRAIN_SIZE}" ]]; then cmd+=(--max-train-jets "${CANONICAL_STATE_MODEL_TRAIN_SIZE}"); fi
if [[ -n "${CANONICAL_STATE_MODEL_VAL_SIZE}" ]]; then cmd+=(--max-val-jets "${CANONICAL_STATE_MODEL_VAL_SIZE}"); fi
if [[ -n "${CANONICAL_STATE_STACK_TRAIN_SIZE}" ]]; then cmd+=(--max-stack-train-jets "${CANONICAL_STATE_STACK_TRAIN_SIZE}"); fi
if [[ -n "${CANONICAL_STATE_STACK_VAL_SIZE}" ]]; then cmd+=(--max-stack-val-jets "${CANONICAL_STATE_STACK_VAL_SIZE}"); fi
if [[ -n "${CANONICAL_STATE_FINAL_TEST_SIZE}" ]]; then cmd+=(--max-final-test-jets "${CANONICAL_STATE_FINAL_TEST_SIZE}"); fi

fresh_write_run_config "${OUTPUT_DIR}" "canonical_state_variant_${RUN_ID}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
fi
