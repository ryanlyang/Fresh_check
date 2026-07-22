#!/usr/bin/env bash
# CPU-only immutable source preflight for the A0/P7b fusion campaign.

#SBATCH --job-name=lprf_fuse_src
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"

: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/first_stage_pilot}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_fusion/p7b_seed_control}"
: "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/split_manifest/split_manifest.json.gz}"
: "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_MANIFEST:=${LOCAL_RESIDUAL_FIELD_ROOT}/input_audit/reused_inputs_report.json}"
: "${LOCAL_RESIDUAL_FIELD_A0_CHECKPOINT:=${LOCAL_RESIDUAL_FIELD_ROOT}/taggers/A0/best_model_val.pt}"
: "${LOCAL_RESIDUAL_FIELD_A0_REPORT:=${LOCAL_RESIDUAL_FIELD_ROOT}/taggers/A0/run_report.json}"
: "${LOCAL_RESIDUAL_FIELD_P7B_CHECKPOINT:=${LOCAL_RESIDUAL_FIELD_ROOT}/curriculum/P7b/best_model_val.pt}"
: "${LOCAL_RESIDUAL_FIELD_P7B_REPORT:=${LOCAL_RESIDUAL_FIELD_ROOT}/curriculum/P7b/run_report.json}"
: "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON:=${LOCAL_RESIDUAL_FIELD_ROOT}/selected_consumer.json}"
: "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/predictions}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT:=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/source_artifact_audit.json}"

fresh_setup "$@"
for path in \
  "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}" \
  "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_MANIFEST}" \
  "${LOCAL_RESIDUAL_FIELD_A0_CHECKPOINT}" \
  "${LOCAL_RESIDUAL_FIELD_A0_REPORT}" \
  "${LOCAL_RESIDUAL_FIELD_P7B_CHECKPOINT}" \
  "${LOCAL_RESIDUAL_FIELD_P7B_REPORT}" \
  "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON}"; do
  fresh_require_file "${path}"
done
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR}/A0"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR}/P7b"
mkdir -p "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}"

cmd=(
  "${PYTHON_BIN}" -u scripts/audit_local_residual_field_fusion_sources.py
  --output "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}"
  --a0-checkpoint "${LOCAL_RESIDUAL_FIELD_A0_CHECKPOINT}"
  --a0-report "${LOCAL_RESIDUAL_FIELD_A0_REPORT}"
  --p7b-checkpoint "${LOCAL_RESIDUAL_FIELD_P7B_CHECKPOINT}"
  --p7b-report "${LOCAL_RESIDUAL_FIELD_P7B_REPORT}"
  --selected-consumer-json "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON}"
  --manifest-path "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
  --hlt-cache-manifest "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_MANIFEST}"
  --a0-prediction-dir "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR}"
  --p7b-prediction-dir "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR}"
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_run "${cmd[@]}"
if ! fresh_is_dry_run; then
  fresh_assert_json_ok "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}"
fi
