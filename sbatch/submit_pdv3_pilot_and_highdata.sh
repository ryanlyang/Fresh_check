#!/usr/bin/env bash
# Submit the PDV3 Step 7 pilot and high-data campaigns.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_prepare_submitter
echo "wrapper_dry_run=$(fresh_is_dry_run && echo 1 || echo 0)"

SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_pdv3_full_experiment.sh"
: "${PDV3_RUN_STAMP:=$(date +%Y%m%d_%H%M%S)}"
: "${PDV3_SUBMIT_PILOT:=1}"
: "${PDV3_SUBMIT_HIGHDATA:=1}"
: "${PDV3_HIGHDATA_AFTER_PILOT:=1}"

: "${PDV3_PILOT_TAG:=pilot_${PDV3_RUN_STAMP}}"
: "${PDV3_PILOT_ROOT:=${OUTPUT_ROOT}/privileged_distill_v3_av10_adapter_hlt0p2_${PDV3_PILOT_TAG}}"
: "${PDV3_PILOT_MODEL_TRAIN_SIZE:=500000}"
: "${PDV3_PILOT_MODEL_VAL_SIZE:=150000}"
: "${PDV3_PILOT_STACK_TRAIN_SIZE:=10}"
: "${PDV3_PILOT_STACK_VAL_SIZE:=10}"
: "${PDV3_PILOT_FINAL_TEST_SIZE:=150000}"
: "${PDV3_PILOT_STUDENT_EPOCHS:=20}"
: "${PDV3_PILOT_EARLY_STOP_PATIENCE:=4}"

: "${PDV3_HIGHDATA_TAG:=highdata_${PDV3_RUN_STAMP}}"
: "${PDV3_HIGHDATA_ROOT:=${OUTPUT_ROOT}/privileged_distill_v3_av10_adapter_hlt0p2_${PDV3_HIGHDATA_TAG}}"
: "${PDV3_HIGHDATA_MODEL_TRAIN_SIZE:=5000000}"
: "${PDV3_HIGHDATA_MODEL_VAL_SIZE:=1000000}"
: "${PDV3_HIGHDATA_STACK_TRAIN_SIZE:=10}"
: "${PDV3_HIGHDATA_STACK_VAL_SIZE:=10}"
: "${PDV3_HIGHDATA_FINAL_TEST_SIZE:=1000000}"
: "${PDV3_HIGHDATA_STUDENT_EPOCHS:=45}"
: "${PDV3_HIGHDATA_EARLY_STOP_PATIENCE:=6}"

extract_final_report_job() {
  local output="$1"
  echo "${output}" | awk -F': ' '/^[[:space:]]+final_report:/ {print $2; exit}'
}

submit_campaign() {
  local label="$1"
  local root="$2"
  local train_size="$3"
  local val_size="$4"
  local stack_train_size="$5"
  local stack_val_size="$6"
  local final_test_size="$7"
  local epochs="$8"
  local patience="$9"
  local dependency="${10:-}"

  echo "submitting ${label}:"
  echo "  root=${root}"
  echo "  sizes=${train_size}/${val_size}/${final_test_size}"
  echo "  upstream_dependency=${dependency:-none}"

  PDV3_ROOT="${root}" \
  PDV3_MODEL_TRAIN_SIZE="${train_size}" \
  PDV3_MODEL_VAL_SIZE="${val_size}" \
  PDV3_STACK_TRAIN_SIZE="${stack_train_size}" \
  PDV3_STACK_VAL_SIZE="${stack_val_size}" \
  PDV3_FINAL_TEST_SIZE="${final_test_size}" \
  PDV3_STUDENT_EPOCHS="${epochs}" \
  PDV3_STUDENT_EARLY_STOP_PATIENCE="${patience}" \
  PDV3_UPSTREAM_DEPENDENCY="${dependency}" \
    bash "${SUBMIT_SCRIPT}"
}

pilot_report_job=""
if fresh_bool_enabled "${PDV3_SUBMIT_PILOT}"; then
  pilot_output="$(submit_campaign \
    "PDV3 HLT0.2 pilot" \
    "${PDV3_PILOT_ROOT}" \
    "${PDV3_PILOT_MODEL_TRAIN_SIZE}" \
    "${PDV3_PILOT_MODEL_VAL_SIZE}" \
    "${PDV3_PILOT_STACK_TRAIN_SIZE}" \
    "${PDV3_PILOT_STACK_VAL_SIZE}" \
    "${PDV3_PILOT_FINAL_TEST_SIZE}" \
    "${PDV3_PILOT_STUDENT_EPOCHS}" \
    "${PDV3_PILOT_EARLY_STOP_PATIENCE}")"
  echo "${pilot_output}"
  pilot_report_job="$(extract_final_report_job "${pilot_output}")"
fi

highdata_dependency=""
if fresh_bool_enabled "${PDV3_HIGHDATA_AFTER_PILOT}" && [[ -n "${pilot_report_job}" && "${pilot_report_job}" != "skipped_or_existing" ]]; then
  highdata_dependency="${pilot_report_job}"
fi

if fresh_bool_enabled "${PDV3_SUBMIT_HIGHDATA}"; then
  submit_campaign \
    "PDV3 HLT0.2 high-data" \
    "${PDV3_HIGHDATA_ROOT}" \
    "${PDV3_HIGHDATA_MODEL_TRAIN_SIZE}" \
    "${PDV3_HIGHDATA_MODEL_VAL_SIZE}" \
    "${PDV3_HIGHDATA_STACK_TRAIN_SIZE}" \
    "${PDV3_HIGHDATA_STACK_VAL_SIZE}" \
    "${PDV3_HIGHDATA_FINAL_TEST_SIZE}" \
    "${PDV3_HIGHDATA_STUDENT_EPOCHS}" \
    "${PDV3_HIGHDATA_EARLY_STOP_PATIENCE}" \
    "${highdata_dependency}"
fi

cat <<SUMMARY
pdv3_pilot_and_highdata_submission:
  pilot_root: ${PDV3_PILOT_ROOT}
  highdata_root: ${PDV3_HIGHDATA_ROOT}
  highdata_after_pilot: ${PDV3_HIGHDATA_AFTER_PILOT}
  pilot_report_job: ${pilot_report_job:-none}
  highdata_dependency: ${highdata_dependency:-none}
  note: this wrapper delegates sbatch submission to submit_pdv3_full_experiment.sh
SUMMARY
