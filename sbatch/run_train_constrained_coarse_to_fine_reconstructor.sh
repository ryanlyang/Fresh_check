#!/usr/bin/env bash
# Train one B/C constrained coarse-to-fine reconstructor.

#SBATCH --job-name=c2f_reco
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=3-00:00:00
#SBATCH --mem=220G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

RUN_ID="${1:?Usage: sbatch run_train_constrained_coarse_to_fine_reconstructor.sh <B0-B7|C0-C6|C5-B1|C5-B2|C5-B3|C5-no-slot|Cdirect-unconstrained>}"
variant="${RUN_ID}"
slot_loss_weight="1.0"
unconstrained_slot_accounting=0
direct_particle_decoding=0
hierarchy_loss_weight="1.0"
case "${RUN_ID}" in
  B[0-7]|C[0-6]|C5-B1|C5-B2|C5-B3) ;;
  C5-no-slot) variant=C5; slot_loss_weight="0.0" ;;
  C5-unconstrained|Cdirect-unconstrained)
    variant=C5
    unconstrained_slot_accounting=1
    direct_particle_decoding=1
    hierarchy_loss_weight="0.0"
    ;;
  *) echo "Unsupported constrained coarse-to-fine reconstructor RUN_ID: ${RUN_ID}" >&2; exit 2 ;;
esac

: "${CONSTRAINED_C2F_ROOT:=${OUTPUT_ROOT}/constrained_coarse_to_fine}"
: "${CONSTRAINED_C2F_MANIFEST_PATH:=${CONSTRAINED_C2F_ROOT}/inputs/split_manifest/split_manifest.json.gz}"
: "${CONSTRAINED_C2F_HLT_CACHE_DIR:=${CONSTRAINED_C2F_ROOT}/inputs/hlt_cache}"
: "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR:=${CONSTRAINED_C2F_ROOT}/inputs/offline_cache}"
: "${CONSTRAINED_C2F_TARGET_CACHE_DIR:=${CONSTRAINED_C2F_ROOT}/targets}"
: "${CONSTRAINED_C2F_RECON_ROOT:=${CONSTRAINED_C2F_ROOT}/reconstructors}"
: "${CONSTRAINED_C2F_RECO_OUTPUT_ID:=${RUN_ID}}"
: "${CONSTRAINED_C2F_RECO_SEED:=22031}"
: "${CONSTRAINED_C2F_RUNTIME_PROFILE:=fp32_reference}"
: "${CONSTRAINED_C2F_RECO_PRECISION_MODE:=}"
: "${CONSTRAINED_C2F_RECO_PREFETCH_FACTOR:=}"
: "${CONSTRAINED_C2F_RECO_LR_SCHEDULE:=constant}"
: "${CONSTRAINED_C2F_RECO_WARMUP_FRACTION:=0.10}"
: "${CONSTRAINED_C2F_RECO_MIN_LR_RATIO:=0.05}"
: "${CONSTRAINED_C2F_RECO_MIN_EPOCHS:=0}"
: "${CONSTRAINED_C2F_RECO_LEARNING_RATE:=2.0e-4}"
: "${CONSTRAINED_C2F_RECO_HLT_ENCODER_LR_SCALE:=0.05}"
: "${CONSTRAINED_C2F_RECO_WEIGHT_DECAY:=1.0e-4}"
: "${CONSTRAINED_C2F_RECO_GRAD_CLIP_NORM:=1.0}"
: "${CONSTRAINED_C2F_RECO_EARLY_STOP_PATIENCE:=6}"
: "${CONSTRAINED_C2F_RECO_MAX_NONFINITE_BATCHES:=8}"
: "${CONSTRAINED_C2F_RECO_FIXED_HORIZON:=0}"
: "${CONSTRAINED_C2F_HUNGARIAN_WORKERS:=1}"
: "${CONSTRAINED_C2F_HUNGARIAN_EXECUTOR:=serial}"
: "${CONSTRAINED_C2F_RECO_EPOCHS:=30}"
: "${CONSTRAINED_C2F_RECO_BATCH_SIZE:=}"
: "${CONSTRAINED_C2F_RECO_EVAL_BATCH_SIZE:=}"
: "${CONSTRAINED_C2F_RECO_NUM_WORKERS:=4}"
: "${CONSTRAINED_C2F_RECO_PROGRESS_INTERVAL_BATCHES:=100}"
: "${CONSTRAINED_C2F_RECO_RESUME_FROM:=}"
: "${CONSTRAINED_C2F_RECO_MAX_TRAIN_JETS:=}"
: "${CONSTRAINED_C2F_RECO_MAX_VAL_JETS:=}"
: "${CONSTRAINED_C2F_RECO_MAX_STACK_VAL_JETS:=}"
# An explicitly empty value is meaningful for runtime benchmarks, which build
# only model_train/model_val calibration inputs. Use '=' so an empty export is
# preserved instead of being replaced by the normal campaign default.
: "${CONSTRAINED_C2F_RECO_STACK_VAL_SPLIT=stack_val}"
: "${CONSTRAINED_C2F_RECO_SAVE_BEST_CHECKPOINT:=1}"
: "${CONSTRAINED_C2F_RECO_AMP:=0}"
: "${CONSTRAINED_C2F_TORCH_NATIVE_TRITON:=auto}"
: "${CONSTRAINED_C2F_TORCH_NATIVE_TRITON_PROBE:=1}"
: "${CONSTRAINED_C2F_RUNTIME_PROFILE_ARTIFACT:=}"
: "${CONSTRAINED_C2F_RUNTIME_PROFILE_EXPECTED_STATUS:=${CONSTRAINED_C2F_RUNTIME_PROFILE}}"
export CONSTRAINED_C2F_TORCH_NATIVE_TRITON CONSTRAINED_C2F_TORCH_NATIVE_TRITON_PROBE

if [[ -n "${CONSTRAINED_C2F_RUNTIME_PROFILE_ARTIFACT}" ]]; then
  # This output is generated only from the validated, typed profile schema.
  # It intentionally supersedes all caller-provided execution tunables.
  profile_exports="$("${PYTHON_BIN}" scripts/validate_constrained_coarse_to_fine_runtime_profile.py \
    --profile "${CONSTRAINED_C2F_RUNTIME_PROFILE_ARTIFACT}" \
    --expected-status "${CONSTRAINED_C2F_RUNTIME_PROFILE_EXPECTED_STATUS}" \
    --run-id "${RUN_ID}" --emit-shell)"
  eval "${profile_exports}"
fi

if [[ -z "${CONSTRAINED_C2F_RECO_PRECISION_MODE}" ]]; then
  case "${CONSTRAINED_C2F_RUNTIME_PROFILE}" in
    fp32_reference) CONSTRAINED_C2F_RECO_PRECISION_MODE=fp32 ;;
    fp16_diagnostic) CONSTRAINED_C2F_RECO_PRECISION_MODE=fp16_forward_fp32_loss ;;
    bf16_calibration|bf16_exploratory_pilot_v1|accelerated_candidate_v1|accelerated_approved_v1)
      CONSTRAINED_C2F_RECO_PRECISION_MODE=bf16_forward_fp32_loss
      ;;
    *) echo "Unsupported CONSTRAINED_C2F_RUNTIME_PROFILE: ${CONSTRAINED_C2F_RUNTIME_PROFILE}" >&2; exit 2 ;;
  esac
fi

if [[ "${CONSTRAINED_C2F_RUNTIME_PROFILE}" == "accelerated_candidate_v1" || "${CONSTRAINED_C2F_RUNTIME_PROFILE}" == "accelerated_approved_v1" ]]; then
  : "${CONSTRAINED_C2F_RECO_SAVE_LAST_CHECKPOINT:=1}"
  if ! fresh_bool_enabled "${CONSTRAINED_C2F_RECO_SAVE_LAST_CHECKPOINT}"; then
    echo "${CONSTRAINED_C2F_RUNTIME_PROFILE} requires CONSTRAINED_C2F_RECO_SAVE_LAST_CHECKPOINT=1" >&2
    exit 2
  fi
else
  : "${CONSTRAINED_C2F_RECO_SAVE_LAST_CHECKPOINT:=0}"
fi
if [[ "${CONSTRAINED_C2F_RUNTIME_PROFILE}" == "accelerated_candidate_v1" || "${CONSTRAINED_C2F_RUNTIME_PROFILE}" == "accelerated_approved_v1" ]]; then
  CONSTRAINED_C2F_RECO_MAX_NONFINITE_BATCHES=0
fi

case "${CONSTRAINED_C2F_HUNGARIAN_EXECUTOR}" in
  serial|thread) ;;
  *) echo "CONSTRAINED_C2F_HUNGARIAN_EXECUTOR must be serial or thread" >&2; exit 2 ;;
esac
if ! [[ "${CONSTRAINED_C2F_HUNGARIAN_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONSTRAINED_C2F_HUNGARIAN_WORKERS must be a positive integer" >&2
  exit 2
fi
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]] && (( CONSTRAINED_C2F_HUNGARIAN_WORKERS > SLURM_CPUS_PER_TASK )); then
  echo "CONSTRAINED_C2F_HUNGARIAN_WORKERS exceeds SLURM_CPUS_PER_TASK" >&2
  exit 2
fi

# The slot decoder retains one decoded particle view per requested view.  The
# C6 four-view variant therefore has a substantially larger activation peak
# than B-tier or single-view C-tier models.  Preserve explicit operator
# settings, but select resource-aware defaults for unattended campaign runs.
if [[ -z "${CONSTRAINED_C2F_RECO_BATCH_SIZE}" ]]; then
  case "${RUN_ID}" in
    C6) CONSTRAINED_C2F_RECO_BATCH_SIZE=8 ;;
    C[0-6]|C5-B1|C5-B2|C5-B3|C5-no-slot|Cdirect-unconstrained)
      CONSTRAINED_C2F_RECO_BATCH_SIZE=16
      ;;
    *) CONSTRAINED_C2F_RECO_BATCH_SIZE=64 ;;
  esac
fi
if [[ -z "${CONSTRAINED_C2F_RECO_EVAL_BATCH_SIZE}" ]]; then
  case "${RUN_ID}" in
    C6) CONSTRAINED_C2F_RECO_EVAL_BATCH_SIZE=16 ;;
    C[0-6]|C5-B1|C5-B2|C5-B3|C5-no-slot|Cdirect-unconstrained)
      CONSTRAINED_C2F_RECO_EVAL_BATCH_SIZE=32
      ;;
    *) CONSTRAINED_C2F_RECO_EVAL_BATCH_SIZE=128 ;;
  esac
fi

OUTPUT_DIR="${CONSTRAINED_C2F_RECON_ROOT}/${CONSTRAINED_C2F_RECO_OUTPUT_ID}"
fresh_setup "$@"
fresh_require_file "${CONSTRAINED_C2F_MANIFEST_PATH}"
fresh_require_dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"
fresh_require_dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}"
fresh_require_file "${CONSTRAINED_C2F_TARGET_CACHE_DIR}/hierarchy_target_cache_manifest.json"
if [[ -n "${CONSTRAINED_C2F_RECO_RESUME_FROM}" ]]; then
  fresh_require_file "${CONSTRAINED_C2F_RECO_RESUME_FROM}"
else
  fresh_claim_new_dir "${OUTPUT_DIR}"
fi

memory_cmd=(
  "${PYTHON_BIN}" scripts/audit_constrained_coarse_to_fine_memory.py
  --hlt-cache-dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"
  --offline-cache-dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}"
  --target-cache-dir "${CONSTRAINED_C2F_TARGET_CACHE_DIR}"
  --splits model_train model_val
  --allocated-memory-mb "${SLURM_MEM_PER_NODE:-0}"
  --output "${OUTPUT_DIR}/memory_preflight.json"
)
fresh_run "${memory_cmd[@]}"

cmd=(
  "${PYTHON_BIN}" -u scripts/train_constrained_coarse_to_fine.py
  --output-dir "${OUTPUT_DIR}"
  --manifest "${CONSTRAINED_C2F_MANIFEST_PATH}"
  --hlt-cache-dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"
  --offline-cache-dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}"
  --target-cache-dir "${CONSTRAINED_C2F_TARGET_CACHE_DIR}"
  --variant "${variant}"
  --hierarchy-loss-weight "${hierarchy_loss_weight}"
  --slot-loss-weight "${slot_loss_weight}"
  --seed "${CONSTRAINED_C2F_RECO_SEED}"
  --runtime-profile "${CONSTRAINED_C2F_RUNTIME_PROFILE}"
  --precision-mode "${CONSTRAINED_C2F_RECO_PRECISION_MODE}"
  --lr-schedule "${CONSTRAINED_C2F_RECO_LR_SCHEDULE}"
  --warmup-fraction "${CONSTRAINED_C2F_RECO_WARMUP_FRACTION}"
  --min-lr-ratio "${CONSTRAINED_C2F_RECO_MIN_LR_RATIO}"
  --min-epochs "${CONSTRAINED_C2F_RECO_MIN_EPOCHS}"
  --learning-rate "${CONSTRAINED_C2F_RECO_LEARNING_RATE}"
  --hlt-encoder-lr-scale "${CONSTRAINED_C2F_RECO_HLT_ENCODER_LR_SCALE}"
  --weight-decay "${CONSTRAINED_C2F_RECO_WEIGHT_DECAY}"
  --grad-clip-norm "${CONSTRAINED_C2F_RECO_GRAD_CLIP_NORM}"
  --early-stop-patience "${CONSTRAINED_C2F_RECO_EARLY_STOP_PATIENCE}"
  --max-nonfinite-batches "${CONSTRAINED_C2F_RECO_MAX_NONFINITE_BATCHES}"
  --hungarian-workers "${CONSTRAINED_C2F_HUNGARIAN_WORKERS}"
  --hungarian-executor "${CONSTRAINED_C2F_HUNGARIAN_EXECUTOR}"
  --epochs "${CONSTRAINED_C2F_RECO_EPOCHS}"
  --batch-size "${CONSTRAINED_C2F_RECO_BATCH_SIZE}"
  --eval-batch-size "${CONSTRAINED_C2F_RECO_EVAL_BATCH_SIZE}"
  --num-workers "${CONSTRAINED_C2F_RECO_NUM_WORKERS}"
  --progress-interval-batches "${CONSTRAINED_C2F_RECO_PROGRESS_INTERVAL_BATCHES}"
  --device "${DEVICE}"
)
if ! fresh_bool_enabled "${CONSTRAINED_C2F_RECO_SAVE_BEST_CHECKPOINT}"; then
  cmd+=(--no-save-best-checkpoint)
fi
if [[ -n "${CONSTRAINED_C2F_RECO_STACK_VAL_SPLIT}" ]]; then
  cmd+=(--stack-val-split "${CONSTRAINED_C2F_RECO_STACK_VAL_SPLIT}")
fi
if fresh_bool_enabled "${unconstrained_slot_accounting}"; then cmd+=(--unconstrained-slot-accounting); fi
if fresh_bool_enabled "${direct_particle_decoding}"; then cmd+=(--direct-particle-decoding); fi
if fresh_bool_enabled "${CONSTRAINED_C2F_RECO_AMP}"; then cmd+=(--amp); else cmd+=(--no-amp); fi
if ! fresh_bool_enabled "${CONSTRAINED_C2F_RECO_SAVE_LAST_CHECKPOINT}"; then cmd+=(--no-save-last-checkpoint); fi
if [[ -n "${CONSTRAINED_C2F_RECO_RESUME_FROM}" ]]; then cmd+=(--resume-from "${CONSTRAINED_C2F_RECO_RESUME_FROM}"); fi
if [[ -n "${CONSTRAINED_C2F_RECO_PREFETCH_FACTOR}" ]]; then cmd+=(--prefetch-factor "${CONSTRAINED_C2F_RECO_PREFETCH_FACTOR}"); fi
if fresh_bool_enabled "${CONSTRAINED_C2F_RECO_FIXED_HORIZON}"; then cmd+=(--fixed-horizon); else cmd+=(--no-fixed-horizon); fi
if [[ -n "${CONSTRAINED_C2F_RECO_MAX_TRAIN_JETS}" ]]; then cmd+=(--max-train-jets "${CONSTRAINED_C2F_RECO_MAX_TRAIN_JETS}"); fi
if [[ -n "${CONSTRAINED_C2F_RECO_MAX_VAL_JETS}" ]]; then cmd+=(--max-val-jets "${CONSTRAINED_C2F_RECO_MAX_VAL_JETS}"); fi
if [[ -n "${CONSTRAINED_C2F_RECO_MAX_STACK_VAL_JETS}" ]]; then cmd+=(--max-stack-val-jets "${CONSTRAINED_C2F_RECO_MAX_STACK_VAL_JETS}"); fi

fresh_write_run_config "${OUTPUT_DIR}" "constrained_c2f_reconstructor_${CONSTRAINED_C2F_RECO_OUTPUT_ID}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  if fresh_bool_enabled "${CONSTRAINED_C2F_RECO_SAVE_BEST_CHECKPOINT}"; then
    fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fi
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
fi
