#!/usr/bin/env bash
# Submit the fixed-row Step 9 C5-B3/C6 downstream tagger sanity gates.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_prepare_submitter
fresh_activate_env

: "${CONSTRAINED_C2F_CALIBRATION_ROOT:?Set CONSTRAINED_C2F_CALIBRATION_ROOT}"
: "${CONSTRAINED_C2F_RUNTIME_SANITY_ROOT:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/runtime_tagger_sanity}"
: "${CONSTRAINED_C2F_SANITY_CANDIDATE_PROFILE:?Set CONSTRAINED_C2F_SANITY_CANDIDATE_PROFILE}"
: "${CONSTRAINED_C2F_SANITY_HLT_WARM_START_CHECKPOINT:?Set CONSTRAINED_C2F_SANITY_HLT_WARM_START_CHECKPOINT}"
: "${CONSTRAINED_C2F_SANITY_C5_ACCELERATED_CHECKPOINT:?Set CONSTRAINED_C2F_SANITY_C5_ACCELERATED_CHECKPOINT}"
: "${CONSTRAINED_C2F_SANITY_C5_FP32_REFERENCE_CHECKPOINT:?Set CONSTRAINED_C2F_SANITY_C5_FP32_REFERENCE_CHECKPOINT}"
: "${CONSTRAINED_C2F_SANITY_C6_ACCELERATED_CHECKPOINT:?Set CONSTRAINED_C2F_SANITY_C6_ACCELERATED_CHECKPOINT}"
: "${CONSTRAINED_C2F_SANITY_C6_FP32_REFERENCE_CHECKPOINT:?Set CONSTRAINED_C2F_SANITY_C6_FP32_REFERENCE_CHECKPOINT}"
: "${CONSTRAINED_C2F_SANITY_CPUS:=12}"
: "${CONSTRAINED_C2F_SANITY_MEM:=220G}"
: "${CONSTRAINED_C2F_SANITY_GPU_GRES:=gpu:1}"
: "${CONSTRAINED_C2F_SBATCH_ACCOUNT:=}"
: "${CONSTRAINED_C2F_SBATCH_PARTITION:=}"

for required in \
  "${CONSTRAINED_C2F_CALIBRATION_ROOT}/calibration_manifest.json.gz" \
  "${CONSTRAINED_C2F_CALIBRATION_ROOT}/targets/hierarchy_target_cache_manifest.json" \
  "${CONSTRAINED_C2F_SANITY_CANDIDATE_PROFILE}" \
  "${CONSTRAINED_C2F_SANITY_HLT_WARM_START_CHECKPOINT}" \
  "${CONSTRAINED_C2F_SANITY_C5_ACCELERATED_CHECKPOINT}" \
  "${CONSTRAINED_C2F_SANITY_C5_FP32_REFERENCE_CHECKPOINT}" \
  "${CONSTRAINED_C2F_SANITY_C6_ACCELERATED_CHECKPOINT}" \
  "${CONSTRAINED_C2F_SANITY_C6_FP32_REFERENCE_CHECKPOINT}"; do fresh_require_file "${required}"; done
fresh_require_dir "${CONSTRAINED_C2F_CALIBRATION_ROOT}/hlt_cache"
fresh_require_dir "${CONSTRAINED_C2F_CALIBRATION_ROOT}/offline_cache"
if ! fresh_is_dry_run && [[ -e "${CONSTRAINED_C2F_RUNTIME_SANITY_ROOT}" ]]; then
  echo "Refusing to reuse runtime sanity root: ${CONSTRAINED_C2F_RUNTIME_SANITY_ROOT}" >&2
  exit 2
fi

export CONSTRAINED_C2F_RUNTIME_SANITY_ROOT
export CONSTRAINED_C2F_ROOT="${CONSTRAINED_C2F_CALIBRATION_ROOT}"
export CONSTRAINED_C2F_MANIFEST_PATH="${CONSTRAINED_C2F_CALIBRATION_ROOT}/calibration_manifest.json.gz"
export CONSTRAINED_C2F_HLT_CACHE_DIR="${CONSTRAINED_C2F_CALIBRATION_ROOT}/hlt_cache"
export CONSTRAINED_C2F_OFFLINE_CACHE_DIR="${CONSTRAINED_C2F_CALIBRATION_ROOT}/offline_cache"
export CONSTRAINED_C2F_TARGET_CACHE_DIR="${CONSTRAINED_C2F_CALIBRATION_ROOT}/targets"
export CONSTRAINED_C2F_TAGGER_ROOT="${CONSTRAINED_C2F_RUNTIME_SANITY_ROOT}/taggers"
export CONSTRAINED_C2F_PREDICTION_DIR="${CONSTRAINED_C2F_RUNTIME_SANITY_ROOT}/predictions"
export CONSTRAINED_C2F_PREDICT_SPLITS="model_val"
export CONFIRM_FINAL_TEST=0
export OVERWRITE=0

gpu_args=(--cpus-per-task="${CONSTRAINED_C2F_SANITY_CPUS}" --mem="${CONSTRAINED_C2F_SANITY_MEM}" --gres="${CONSTRAINED_C2F_SANITY_GPU_GRES}")
[[ -n "${CONSTRAINED_C2F_SBATCH_ACCOUNT}" ]] && gpu_args+=(--account="${CONSTRAINED_C2F_SBATCH_ACCOUNT}")
[[ -n "${CONSTRAINED_C2F_SBATCH_PARTITION}" ]] && gpu_args+=(--partition="${CONSTRAINED_C2F_SBATCH_PARTITION}")
report_args=(--cpus-per-task=4 --mem=32G)
[[ -n "${CONSTRAINED_C2F_SBATCH_ACCOUNT}" ]] && report_args+=(--account="${CONSTRAINED_C2F_SBATCH_ACCOUNT}")
[[ -n "${CONSTRAINED_C2F_SBATCH_PARTITION}" ]] && report_args+=(--partition="${CONSTRAINED_C2F_SBATCH_PARTITION}")

if fresh_is_dry_run; then
  for path in C5-B3 C6; do
    for member in accelerated fp32_reference; do
      fresh_print_shell_command sbatch "${gpu_args[@]}" "${SCRIPT_DIR}/run_train_constrained_coarse_to_fine_runtime_sanity_tagger.sh" "${path}" "${member}"
    done
  done
  exit 0
fi

mkdir -p "${CONSTRAINED_C2F_RUNTIME_SANITY_ROOT}"
submission="${CONSTRAINED_C2F_RUNTIME_SANITY_ROOT}/submission.tsv"
printf 'kind\tpath\tmember\tjob_id\n' > "${submission}"
for path in C5-B3 C6; do
  declare -A train_jids=()
  declare -A prediction_jids=()
  for member in accelerated fp32_reference; do
    submitted="$(sbatch "${gpu_args[@]}" "${SCRIPT_DIR}/run_train_constrained_coarse_to_fine_runtime_sanity_tagger.sh" "${path}" "${member}")"
    echo "${submitted}"
    jid="${submitted##* }"; [[ "${jid}" =~ ^[0-9]+$ ]] || { echo "Could not parse job id: ${submitted}" >&2; exit 2; }
    train_jids["${member}"]="${jid}"
    printf 'tagger\t%s\t%s\t%s\n' "${path}" "${member}" "${jid}" >> "${submission}"
    submitted="$(sbatch "${gpu_args[@]}" --dependency="afterok:${jid}" "${SCRIPT_DIR}/run_cache_constrained_coarse_to_fine_predictions.sh" "${path}_${member}")"
    echo "${submitted}"
    pred_jid="${submitted##* }"; [[ "${pred_jid}" =~ ^[0-9]+$ ]] || { echo "Could not parse job id: ${submitted}" >&2; exit 2; }
    prediction_jids["${member}"]="${pred_jid}"
    printf 'prediction\t%s\t%s\t%s\n' "${path}" "${member}" "${pred_jid}" >> "${submission}"
  done
  dependency="afterok:${prediction_jids[accelerated]}:${prediction_jids[fp32_reference]}"
  submitted="$(sbatch "${report_args[@]}" --dependency="${dependency}" "${SCRIPT_DIR}/run_write_constrained_coarse_to_fine_runtime_tagger_sanity.sh" "${path}")"
  echo "${submitted}"
  report_jid="${submitted##* }"; [[ "${report_jid}" =~ ^[0-9]+$ ]] || { echo "Could not parse job id: ${submitted}" >&2; exit 2; }
  printf 'report\t%s\t\t%s\n' "${path}" "${report_jid}" >> "${submission}"
done
echo "constrained_c2f_runtime_tagger_sanity_submission_complete:"
echo "  root: ${CONSTRAINED_C2F_RUNTIME_SANITY_ROOT}"
echo "  submission: ${submission}"
