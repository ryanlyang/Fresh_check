#!/usr/bin/env bash
# Promote the first-stage curriculum recipe to high data only after its pilot gate.

set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_HIGHDATA:=0}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_PILOT_REPORT_OK:=0}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_OVERRIDE_PILOT_GATE:=0}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_PILOT_REPORT_DIR:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/first_stage_pilot/final_report}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_HIGHDATA_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/highdata}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_STUDENT_UPLIFT_THRESHOLD:=0.003}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_FUSION_UPLIFT_THRESHOLD:=0.005}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_MINIMUM_VALIDATION_COVERAGE:=0.99}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_MAXIMUM_NONFINITE_FRACTION:=0.01}"

fresh_setup "$@"

fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_HIGHDATA}" || {
  echo "High-data is disabled by default; set LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_HIGHDATA=1" >&2
  exit 2
}
fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_CURRICULUM_PILOT_REPORT_OK}" || {
  echo "High-data requires LOCAL_RESIDUAL_FIELD_CURRICULUM_PILOT_REPORT_OK=1" >&2
  exit 2
}
if ! fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_CURRICULUM_OVERRIDE_PILOT_GATE}"; then
  "${PYTHON_BIN}" scripts/check_local_residual_curriculum_pilot_gate.py \
    --report-dir "${LOCAL_RESIDUAL_FIELD_CURRICULUM_PILOT_REPORT_DIR}" \
    --student-uplift-threshold "${LOCAL_RESIDUAL_FIELD_CURRICULUM_STUDENT_UPLIFT_THRESHOLD}" \
    --fusion-uplift-threshold "${LOCAL_RESIDUAL_FIELD_CURRICULUM_FUSION_UPLIFT_THRESHOLD}" \
    --minimum-validation-coverage "${LOCAL_RESIDUAL_FIELD_CURRICULUM_MINIMUM_VALIDATION_COVERAGE}" \
    --maximum-nonfinite-fraction "${LOCAL_RESIDUAL_FIELD_CURRICULUM_MAXIMUM_NONFINITE_FRACTION}"
else
  echo "WARNING: explicit pilot uplift override is enabled" >&2
fi
export LOCAL_RESIDUAL_FIELD_ROOT="${LOCAL_RESIDUAL_FIELD_CURRICULUM_HIGHDATA_ROOT}"
export LOCAL_RESIDUAL_FIELD_CURRICULUM_MODE=first_stage_pilot
export LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE="${LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE:-full_first_stage}"
bash "${PROJECT_DIR}/sbatch/submit_lprf_curriculum_pilot.sh"
