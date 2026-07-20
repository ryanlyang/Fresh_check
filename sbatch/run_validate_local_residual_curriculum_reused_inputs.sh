#!/usr/bin/env bash
# Metadata-only audit for every cache explicitly reused by the curriculum campaign.

#SBATCH --job-name=lprf_inputs
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/first_stage_pilot}"
: "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/split_manifest.json.gz}"
: "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/hlt_cache}"
: "${LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR:=}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/targets}"
: "${LOCAL_RESIDUAL_FIELD_OFFLINE_TEACHER_LOGITS_DIR:=}"
: "${LOCAL_RESIDUAL_FIELD_HLT_PROFILE:=fixed_hlt_v2_realistic}"
: "${LOCAL_RESIDUAL_FIELD_HLT_DEGRADATION_STRENGTH:=2.5}"
: "${LOCAL_RESIDUAL_FIELD_REUSE_SPLIT_MANIFEST:=1}"
: "${LOCAL_RESIDUAL_FIELD_REUSE_HLT_CACHE:=1}"
: "${LOCAL_RESIDUAL_FIELD_REUSE_OFFLINE_CACHE:=0}"
: "${LOCAL_RESIDUAL_FIELD_REUSE_TARGET_CACHE:=1}"
: "${LOCAL_RESIDUAL_FIELD_REUSE_OFFLINE_TEACHER_LOGITS:=0}"
: "${LOCAL_RESIDUAL_FIELD_REUSED_INPUT_AUDIT:=${LOCAL_RESIDUAL_FIELD_ROOT}/input_audit/reused_inputs_report.json}"

fresh_setup "$@"
if ! fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_REUSE_SPLIT_MANIFEST}"; then
  echo "Step 10 submitters currently require an existing split manifest; build it before launching the pilot" >&2
  exit 2
fi
fresh_require_file "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
cmd=(
  "${PYTHON_BIN}" -u scripts/validate_local_residual_curriculum_reused_inputs.py
  --manifest-path "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
  --expected-hlt-profile "${LOCAL_RESIDUAL_FIELD_HLT_PROFILE}"
  --expected-hlt-degradation-strength "${LOCAL_RESIDUAL_FIELD_HLT_DEGRADATION_STRENGTH}"
  --output "${LOCAL_RESIDUAL_FIELD_REUSED_INPUT_AUDIT}"
)
if fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_REUSE_HLT_CACHE}"; then
  fresh_require_dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
  cmd+=(--hlt-cache-dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}")
fi
if fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_REUSE_OFFLINE_CACHE}"; then
  fresh_require_dir "${LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR}"
  cmd+=(--offline-cache-dir "${LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR}")
fi
if fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_REUSE_TARGET_CACHE}"; then
  fresh_require_dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
  cmd+=(--target-cache-dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}")
fi
if fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_REUSE_OFFLINE_TEACHER_LOGITS}"; then
  fresh_require_dir "${LOCAL_RESIDUAL_FIELD_OFFLINE_TEACHER_LOGITS_DIR}"
  cmd+=(--offline-teacher-logits-dir "${LOCAL_RESIDUAL_FIELD_OFFLINE_TEACHER_LOGITS_DIR}")
fi
mkdir -p "$(dirname "${LOCAL_RESIDUAL_FIELD_REUSED_INPUT_AUDIT}")"
fresh_run "${cmd[@]}"
if ! fresh_is_dry_run; then fresh_require_file "${LOCAL_RESIDUAL_FIELD_REUSED_INPUT_AUDIT}"; fi
