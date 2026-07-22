#!/usr/bin/env bash
# Cache strict stack-only ParT representations for A0, A0_seed1, or P7b.

#SBATCH --job-name=lprf_fuse_feat
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=12:00:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"

MEMBER_ID="${1:?Usage: sbatch run_cache_local_residual_field_fusion_features.sh <A0|A0_seed1|P7b>}"
case "${MEMBER_ID}" in A0|A0_seed1|P7b) ;; *) echo "invalid fusion member ${MEMBER_ID}" >&2; exit 2 ;; esac

: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/first_stage_pilot}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_fusion/p7b_seed_control}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT:=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/source_artifact_audit.json}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_SOURCES:=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/predictions/development_prediction_sources.json}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_DIR:=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/representations}"
: "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/split_manifest.json.gz}"
: "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/hlt_cache}"
: "${LOCAL_RESIDUAL_FIELD_FEATURE_BATCH_SIZE:=128}"
: "${LOCAL_RESIDUAL_FIELD_FEATURE_NUM_WORKERS:=4}"
: "${LOCAL_RESIDUAL_FIELD_FEATURE_STORAGE_DTYPE:=float16}"
: "${LOCAL_RESIDUAL_FIELD_FEATURE_DISABLE_AMP:=0}"

case "${MEMBER_ID}" in
  A0) CHECKPOINT="${LOCAL_RESIDUAL_FIELD_ROOT}/taggers/A0/best_model_val.pt" ;;
  A0_seed1) CHECKPOINT="${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/taggers/A0_seed1/best_model_val.pt" ;;
  P7b) CHECKPOINT="${LOCAL_RESIDUAL_FIELD_ROOT}/curriculum/P7b/best_model_val.pt" ;;
esac

fresh_setup "$@"
fresh_require_file "${CHECKPOINT}"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_SOURCES}"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
mkdir -p "${LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_DIR}"
if ! fresh_is_dry_run && [[ ! -f "${LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_DIR}/${MEMBER_ID}/representation_manifest.json" && -d "${LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_DIR}/${MEMBER_ID}" ]]; then
  partial_dir="${LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_DIR}/${MEMBER_ID}.partial_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-manual}"
  echo "Quarantining interrupted representation cache: ${partial_dir}"
  mv -- "${LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_DIR}/${MEMBER_ID}" "${partial_dir}"
fi

gate_cmd=("${PYTHON_BIN}" -u scripts/require_local_residual_field_fusion_source_audit.py --audit "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}")
cmd=(
  "${PYTHON_BIN}" -u scripts/cache_local_residual_field_fusion_features.py
  --checkpoint "${CHECKPOINT}"
  --member-id "${MEMBER_ID}"
  --output-dir "${LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_DIR}"
  --prediction-sources "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_SOURCES}"
  --source-artifact-audit "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}"
  --hlt-cache-dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
  --manifest-path "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
  --batch-size "${LOCAL_RESIDUAL_FIELD_FEATURE_BATCH_SIZE}"
  --num-workers "${LOCAL_RESIDUAL_FIELD_FEATURE_NUM_WORKERS}"
  --storage-dtype "${LOCAL_RESIDUAL_FIELD_FEATURE_STORAGE_DTYPE}"
  --device "${DEVICE}"
)
fresh_append_flag_if_enabled cmd --disable-amp "${LOCAL_RESIDUAL_FIELD_FEATURE_DISABLE_AMP}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_write_run_config "${LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_DIR}/${MEMBER_ID}" "local_residual_field_fusion_features_${MEMBER_ID}" "${gate_cmd[@]}" "${cmd[@]}"
fresh_run "${gate_cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  for split in stack_train stack_val; do
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_DIR}/${MEMBER_ID}/${split}_representations.npz"
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_DIR}/${MEMBER_ID}/${split}_representations_metadata.json"
  done
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_DIR}/${MEMBER_ID}/representation_manifest.json"
fi
