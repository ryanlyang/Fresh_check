#!/usr/bin/env bash
# Cache final-test logits only for the already selected deployable P student.

#SBATCH --job-name=lprf_psel
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=08:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/first_stage_pilot}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/curriculum}"
: "${LOCAL_RESIDUAL_FIELD_SELECTED_STUDENT_JSON:=${LOCAL_RESIDUAL_FIELD_ROOT}/selected_curriculum_student.json}"
fresh_setup "$@"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_SELECTED_STUDENT_JSON}"
selected_p="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_run_id"])' "${LOCAL_RESIDUAL_FIELD_SELECTED_STUDENT_JSON}")"
case "${selected_p}" in P2|P4|P7a|P7b) ;; *) echo "invalid selected curriculum student ${selected_p}" >&2; exit 2 ;; esac
export LOCAL_RESIDUAL_FIELD_PREDICT_MODEL_ROOT="${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT}"
export LOCAL_RESIDUAL_FIELD_PREDICT_SPLITS=final_test
export CONFIRM_FINAL_TEST=1
bash "${PROJECT_DIR}/sbatch/run_predict_local_residual_field_tagger.sh" "${selected_p}"
