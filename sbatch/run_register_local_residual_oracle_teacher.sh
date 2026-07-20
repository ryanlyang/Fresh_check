#!/usr/bin/env bash
# Register one provenance-compatible oracle teacher from an existing LPRF run.

#SBATCH --job-name=lprf_oreg
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

TEACHER_ID="${1:?Usage: sbatch run_register_local_residual_oracle_teacher.sh <O0|Ofull|Orobust_light> <source-run-dir>}"
SOURCE_RUN_DIR="${2:?Usage: sbatch run_register_local_residual_oracle_teacher.sh <O0|Ofull|Orobust_light> <source-run-dir>}"

: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field}"
: "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/split_manifest.json.gz}"
: "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/hlt_cache}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/targets}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/taggers}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_MODEL_SIZE:=base}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_ROBUST_LIGHT_NOISE_STD:=0.05}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_ROBUST_LIGHT_FIELD_DROPOUT:=0.10}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_ROBUST_LIGHT_GROUP_DROPOUT:=0.0}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_REUSE_CANDIDATE_DIRS:=}"

expected_source=""
expected_alpha="1.0"
expected_noise="0.0"
expected_dropout="0.0"
expected_group_dropout="0.0"
case "${TEACHER_ID}" in
  O0)
    expected_source="zero"
    ;;
  Ofull)
    expected_source="oracle_scaled"
    ;;
  Orobust_light)
    expected_source="oracle_noisy"
    expected_noise="${LOCAL_RESIDUAL_FIELD_ORACLE_ROBUST_LIGHT_NOISE_STD}"
    expected_dropout="${LOCAL_RESIDUAL_FIELD_ORACLE_ROBUST_LIGHT_FIELD_DROPOUT}"
    expected_group_dropout="${LOCAL_RESIDUAL_FIELD_ORACLE_ROBUST_LIGHT_GROUP_DROPOUT}"
    ;;
  *)
    echo "Step 2 first-stage registration allows only O0, Ofull, or Orobust_light; got ${TEACHER_ID}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${TEACHER_ID}"

fresh_setup "$@"
fresh_require_dir "${SOURCE_RUN_DIR}"
fresh_require_file "${SOURCE_RUN_DIR}/best_model_val.pt"
fresh_require_file "${SOURCE_RUN_DIR}/run_report.json"
fresh_require_file "${SOURCE_RUN_DIR}/training_curves.json"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/register_local_residual_oracle_teacher.py"
  --source-run-dir "${SOURCE_RUN_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --teacher-id "${TEACHER_ID}"
  --expected-field-source "${expected_source}"
  --expected-alpha "${expected_alpha}"
  --expected-noise-std "${expected_noise}"
  --expected-field-dropout "${expected_dropout}"
  --expected-group-dropout "${expected_group_dropout}"
  --expected-model-size "${LOCAL_RESIDUAL_FIELD_TAGGER_MODEL_SIZE}"
  --manifest-path "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
  --hlt-cache-dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
  --target-cache-dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
)
fresh_split_words reuse_candidate_dirs "${LOCAL_RESIDUAL_FIELD_ORACLE_REUSE_CANDIDATE_DIRS}"
for candidate_dir in "${reuse_candidate_dirs[@]}"; do
  fresh_require_dir "${candidate_dir}"
  cmd+=(--candidate-run-dir "${candidate_dir}")
done
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"

fresh_write_run_config "${OUTPUT_DIR}" "local_residual_oracle_teacher_registration_${TEACHER_ID}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/teacher_config.json"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/registration_report.json"
fi
