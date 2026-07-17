#!/usr/bin/env bash
# Write and gate one fixed-row Step 9 accelerated-vs-FP32 tagger comparison.

#SBATCH --job-name=c2f_rt_sanity
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

PATH_NAME="${1:?Usage: sbatch run_write_constrained_coarse_to_fine_runtime_tagger_sanity.sh <C5-B3|C6>}"
case "${PATH_NAME}" in C5-B3|C6) ;; *) echo "Unsupported runtime sanity path: ${PATH_NAME}" >&2; exit 2;; esac
: "${CONSTRAINED_C2F_CALIBRATION_ROOT:?Set CONSTRAINED_C2F_CALIBRATION_ROOT}"
: "${CONSTRAINED_C2F_RUNTIME_SANITY_ROOT:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/runtime_tagger_sanity}"
: "${CONSTRAINED_C2F_TAGGER_ROOT:=${CONSTRAINED_C2F_RUNTIME_SANITY_ROOT}/taggers}"
: "${CONSTRAINED_C2F_PREDICTION_DIR:=${CONSTRAINED_C2F_RUNTIME_SANITY_ROOT}/predictions}"
: "${CONSTRAINED_C2F_SANITY_CANDIDATE_PROFILE:?Set CONSTRAINED_C2F_SANITY_CANDIDATE_PROFILE}"

if [[ "${PATH_NAME}" == "C5-B3" ]]; then
  : "${CONSTRAINED_C2F_SANITY_C5_ACCELERATED_CHECKPOINT:?Set CONSTRAINED_C2F_SANITY_C5_ACCELERATED_CHECKPOINT}"
  : "${CONSTRAINED_C2F_SANITY_C5_FP32_REFERENCE_CHECKPOINT:?Set CONSTRAINED_C2F_SANITY_C5_FP32_REFERENCE_CHECKPOINT}"
  ACC_CHECKPOINT="${CONSTRAINED_C2F_SANITY_C5_ACCELERATED_CHECKPOINT}"
  REF_CHECKPOINT="${CONSTRAINED_C2F_SANITY_C5_FP32_REFERENCE_CHECKPOINT}"
else
  : "${CONSTRAINED_C2F_SANITY_C6_ACCELERATED_CHECKPOINT:?Set CONSTRAINED_C2F_SANITY_C6_ACCELERATED_CHECKPOINT}"
  : "${CONSTRAINED_C2F_SANITY_C6_FP32_REFERENCE_CHECKPOINT:?Set CONSTRAINED_C2F_SANITY_C6_FP32_REFERENCE_CHECKPOINT}"
  ACC_CHECKPOINT="${CONSTRAINED_C2F_SANITY_C6_ACCELERATED_CHECKPOINT}"
  REF_CHECKPOINT="${CONSTRAINED_C2F_SANITY_C6_FP32_REFERENCE_CHECKPOINT}"
fi

OUTPUT_DIR="${CONSTRAINED_C2F_RUNTIME_SANITY_ROOT}/reports"
OUTPUT_PATH="${OUTPUT_DIR}/${PATH_NAME}_tagger_sanity.json"
fresh_setup "$@"
for required in \
  "${CONSTRAINED_C2F_SANITY_CANDIDATE_PROFILE}" \
  "${ACC_CHECKPOINT}" \
  "${REF_CHECKPOINT}" \
  "${CONSTRAINED_C2F_TAGGER_ROOT}/${PATH_NAME}_accelerated/run_report.json" \
  "${CONSTRAINED_C2F_TAGGER_ROOT}/${PATH_NAME}_fp32_reference/run_report.json" \
  "${CONSTRAINED_C2F_PREDICTION_DIR}/${PATH_NAME}_accelerated/model_val_predictions.npz" \
  "${CONSTRAINED_C2F_PREDICTION_DIR}/${PATH_NAME}_accelerated/model_val_predictions_metadata.json" \
  "${CONSTRAINED_C2F_PREDICTION_DIR}/${PATH_NAME}_fp32_reference/model_val_predictions.npz" \
  "${CONSTRAINED_C2F_PREDICTION_DIR}/${PATH_NAME}_fp32_reference/model_val_predictions_metadata.json"; do fresh_require_file "${required}"; done
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" scripts/write_constrained_coarse_to_fine_runtime_tagger_sanity.py
  --path "${PATH_NAME}"
  --accelerated-tagger-dir "${CONSTRAINED_C2F_TAGGER_ROOT}/${PATH_NAME}_accelerated"
  --fp32-tagger-dir "${CONSTRAINED_C2F_TAGGER_ROOT}/${PATH_NAME}_fp32_reference"
  --accelerated-prediction-dir "${CONSTRAINED_C2F_PREDICTION_DIR}"
  --fp32-prediction-dir "${CONSTRAINED_C2F_PREDICTION_DIR}"
  --accelerated-model-name "${PATH_NAME}_accelerated"
  --fp32-model-name "${PATH_NAME}_fp32_reference"
  --accelerated-reconstructor-checkpoint "${ACC_CHECKPOINT}"
  --fp32-reconstructor-checkpoint "${REF_CHECKPOINT}"
  --candidate-profile "${CONSTRAINED_C2F_SANITY_CANDIDATE_PROFILE}"
  --output "${OUTPUT_PATH}"
)
fresh_write_run_config "${OUTPUT_DIR}" "constrained_c2f_runtime_tagger_sanity_${PATH_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"
if ! fresh_is_dry_run; then fresh_require_file "${OUTPUT_PATH}"; fi
