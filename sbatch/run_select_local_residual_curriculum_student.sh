#!/usr/bin/env bash
# Select the best P2/P4/P7a/P7b deployable student for G0.

#SBATCH --job-name=lprf_bestp
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/first_stage_pilot}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/curriculum}"
: "${LOCAL_RESIDUAL_FIELD_SELECTED_STUDENT_JSON:=${LOCAL_RESIDUAL_FIELD_ROOT}/selected_curriculum_student.json}"

fresh_setup "$@"
cmd=("${PYTHON_BIN}" -u scripts/select_local_residual_curriculum_student.py --output "${LOCAL_RESIDUAL_FIELD_SELECTED_STUDENT_JSON}")
for run_id in P2 P4 P7a P7b; do
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT}/${run_id}/run_report.json"
  cmd+=(--report "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT}/${run_id}/run_report.json")
done
if [[ -e "${LOCAL_RESIDUAL_FIELD_SELECTED_STUDENT_JSON}" ]] && ! fresh_bool_enabled "${OVERWRITE}"; then
  echo "selected curriculum student artifact already exists; set OVERWRITE=1 only after inspection" >&2
  exit 2
fi
fresh_run "${cmd[@]}"
if ! fresh_is_dry_run; then fresh_require_file "${LOCAL_RESIDUAL_FIELD_SELECTED_STUDENT_JSON}"; fi
