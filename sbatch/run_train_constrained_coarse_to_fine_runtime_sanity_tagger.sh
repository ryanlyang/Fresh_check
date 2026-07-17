#!/usr/bin/env bash
# Train one member of the fixed-row Step 9 C5-B3/C6 tagger sanity pair.

#SBATCH --job-name=c2f_rt_tagger
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

PATH_NAME="${1:?Usage: sbatch run_train_constrained_coarse_to_fine_runtime_sanity_tagger.sh <C5-B3|C6> <accelerated|fp32_reference>}"
MEMBER="${2:?Usage: sbatch run_train_constrained_coarse_to_fine_runtime_sanity_tagger.sh <C5-B3|C6> <accelerated|fp32_reference>}"
case "${PATH_NAME}" in C5-B3|C6) ;; *) echo "Unsupported runtime sanity path: ${PATH_NAME}" >&2; exit 2;; esac
case "${MEMBER}" in accelerated|fp32_reference) ;; *) echo "Unsupported sanity member: ${MEMBER}" >&2; exit 2;; esac

: "${CONSTRAINED_C2F_CALIBRATION_ROOT:?Set CONSTRAINED_C2F_CALIBRATION_ROOT}"
: "${CONSTRAINED_C2F_MANIFEST_PATH:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/calibration_manifest.json.gz}"
: "${CONSTRAINED_C2F_HLT_CACHE_DIR:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/hlt_cache}"
: "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/offline_cache}"
: "${CONSTRAINED_C2F_TARGET_CACHE_DIR:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/targets}"
: "${CONSTRAINED_C2F_RUNTIME_SANITY_ROOT:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/runtime_tagger_sanity}"
: "${CONSTRAINED_C2F_TAGGER_ROOT:=${CONSTRAINED_C2F_RUNTIME_SANITY_ROOT}/taggers}"
: "${CONSTRAINED_C2F_SANITY_CANDIDATE_PROFILE:?Set CONSTRAINED_C2F_SANITY_CANDIDATE_PROFILE}"
: "${CONSTRAINED_C2F_SANITY_HLT_WARM_START_CHECKPOINT:?Set CONSTRAINED_C2F_SANITY_HLT_WARM_START_CHECKPOINT}"
: "${CONSTRAINED_C2F_SANITY_TAGGER_SEED:=28031}"
: "${CONSTRAINED_C2F_SANITY_TAGGER_EPOCHS:=12}"
: "${CONSTRAINED_C2F_SANITY_TAGGER_BATCH_SIZE:=32}"
: "${CONSTRAINED_C2F_SANITY_TAGGER_EVAL_BATCH_SIZE:=64}"
: "${CONSTRAINED_C2F_SANITY_TAGGER_NUM_WORKERS:=4}"
: "${CONSTRAINED_C2F_SANITY_TAGGER_LEARNING_RATE:=2.0e-4}"
: "${CONSTRAINED_C2F_SANITY_TAGGER_WEIGHT_DECAY:=1.0e-4}"
: "${CONSTRAINED_C2F_SANITY_TAGGER_AMP:=0}"
: "${CONSTRAINED_C2F_SANITY_TAGGER_SAVE_LAST_CHECKPOINT:=0}"

if [[ "${PATH_NAME}" == "C5-B3" ]]; then
  : "${CONSTRAINED_C2F_SANITY_C5_ACCELERATED_CHECKPOINT:?Set CONSTRAINED_C2F_SANITY_C5_ACCELERATED_CHECKPOINT}"
  : "${CONSTRAINED_C2F_SANITY_C5_FP32_REFERENCE_CHECKPOINT:?Set CONSTRAINED_C2F_SANITY_C5_FP32_REFERENCE_CHECKPOINT}"
  if [[ "${MEMBER}" == "accelerated" ]]; then CHECKPOINT="${CONSTRAINED_C2F_SANITY_C5_ACCELERATED_CHECKPOINT}"; else CHECKPOINT="${CONSTRAINED_C2F_SANITY_C5_FP32_REFERENCE_CHECKPOINT}"; fi
  VARIANT="D5"
  source_args=(--reconstructor-source "canonical=${CHECKPOINT}" --reconstructor-variant canonical=C5-B3)
else
  : "${CONSTRAINED_C2F_SANITY_C6_ACCELERATED_CHECKPOINT:?Set CONSTRAINED_C2F_SANITY_C6_ACCELERATED_CHECKPOINT}"
  : "${CONSTRAINED_C2F_SANITY_C6_FP32_REFERENCE_CHECKPOINT:?Set CONSTRAINED_C2F_SANITY_C6_FP32_REFERENCE_CHECKPOINT}"
  if [[ "${MEMBER}" == "accelerated" ]]; then CHECKPOINT="${CONSTRAINED_C2F_SANITY_C6_ACCELERATED_CHECKPOINT}"; else CHECKPOINT="${CONSTRAINED_C2F_SANITY_C6_FP32_REFERENCE_CHECKPOINT}"; fi
  VARIANT="D6"
  source_args=()
  for index in 0 1 2 3; do
    source_args+=(--reconstructor-source "stochastic_${index}@${index}=${CHECKPOINT}" --reconstructor-variant "stochastic_${index}=C6")
  done
fi

RUN_ID="${PATH_NAME}_${MEMBER}"
OUTPUT_DIR="${CONSTRAINED_C2F_TAGGER_ROOT}/${RUN_ID}"
fresh_setup "$@"
for required in \
  "${CONSTRAINED_C2F_MANIFEST_PATH}" \
  "${CONSTRAINED_C2F_TARGET_CACHE_DIR}/hierarchy_target_cache_manifest.json" \
  "${CONSTRAINED_C2F_SANITY_CANDIDATE_PROFILE}" \
  "${CONSTRAINED_C2F_SANITY_HLT_WARM_START_CHECKPOINT}" \
  "${CHECKPOINT}"; do fresh_require_file "${required}"; done
fresh_require_dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"
fresh_require_dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}"
fresh_claim_new_dir "${OUTPUT_DIR}"

memory_cmd=(
  "${PYTHON_BIN}" scripts/audit_constrained_coarse_to_fine_memory.py
  --hlt-cache-dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"
  --offline-cache-dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}"
  --target-cache-dir "${CONSTRAINED_C2F_TARGET_CACHE_DIR}"
  --splits model_train model_val
  --loading-mode sequential
  --allocated-memory-mb "${SLURM_MEM_PER_NODE:-0}"
  --output "${OUTPUT_DIR}/memory_preflight.json"
)
fresh_run "${memory_cmd[@]}"

cmd=(
  "${PYTHON_BIN}" -u scripts/train_constrained_coarse_to_fine_end_to_end.py
  --output-dir "${OUTPUT_DIR}"
  --manifest "${CONSTRAINED_C2F_MANIFEST_PATH}"
  --hlt-cache-dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"
  --offline-cache-dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}"
  --target-cache-dir "${CONSTRAINED_C2F_TARGET_CACHE_DIR}"
  --variant "${VARIANT}"
  "${source_args[@]}"
  --hlt-warm-start-checkpoint "${CONSTRAINED_C2F_SANITY_HLT_WARM_START_CHECKPOINT}"
  --seed "${CONSTRAINED_C2F_SANITY_TAGGER_SEED}"
  --epochs "${CONSTRAINED_C2F_SANITY_TAGGER_EPOCHS}"
  --batch-size "${CONSTRAINED_C2F_SANITY_TAGGER_BATCH_SIZE}"
  --eval-batch-size "${CONSTRAINED_C2F_SANITY_TAGGER_EVAL_BATCH_SIZE}"
  --num-workers "${CONSTRAINED_C2F_SANITY_TAGGER_NUM_WORKERS}"
  --learning-rate "${CONSTRAINED_C2F_SANITY_TAGGER_LEARNING_RATE}"
  --weight-decay "${CONSTRAINED_C2F_SANITY_TAGGER_WEIGHT_DECAY}"
  --device "${DEVICE}"
)
if fresh_bool_enabled "${CONSTRAINED_C2F_SANITY_TAGGER_AMP}"; then cmd+=(--amp); else cmd+=(--no-amp); fi
if ! fresh_bool_enabled "${CONSTRAINED_C2F_SANITY_TAGGER_SAVE_LAST_CHECKPOINT}"; then cmd+=(--no-save-last-checkpoint); fi

fresh_write_run_config "${OUTPUT_DIR}" "constrained_c2f_runtime_sanity_${RUN_ID}" "${cmd[@]}"
fresh_run "${cmd[@]}"
if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
fi
