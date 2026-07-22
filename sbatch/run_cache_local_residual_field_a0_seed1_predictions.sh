#!/usr/bin/env bash
# Cache stack-only A0_seed1 predictions after revalidating frozen A0/P7b sources.

#SBATCH --job-name=lprf_A0s1_pred
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

: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/first_stage_pilot}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_fusion/p7b_seed_control}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT:=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/source_artifact_audit.json}"
: "${LOCAL_RESIDUAL_FIELD_A0_SEED1_CHECKPOINT:=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/taggers/A0_seed1/best_model_val.pt}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_DIR:=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/predictions}"
: "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/split_manifest/split_manifest.json.gz}"
: "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/hlt_cache}"
: "${LOCAL_RESIDUAL_FIELD_PREDICT_BATCH_SIZE:=128}"
: "${LOCAL_RESIDUAL_FIELD_PREDICT_NUM_WORKERS:=4}"
: "${LOCAL_RESIDUAL_FIELD_PREDICT_MAX_JETS:=}"
: "${LOCAL_RESIDUAL_FIELD_PREDICT_DISABLE_AMP:=0}"

fresh_setup "$@"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_A0_SEED1_CHECKPOINT}"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
mkdir -p "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_DIR}"
if ! fresh_is_dry_run && [[ ! -f "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_DIR}/development_prediction_sources.json" && -d "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_DIR}/A0_seed1" ]]; then
  partial_dir="${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_DIR}/A0_seed1.partial_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-manual}"
  echo "Quarantining interrupted A0_seed1 prediction cache: ${partial_dir}"
  mv -- "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_DIR}/A0_seed1" "${partial_dir}"
fi

gate_cmd=(
  "${PYTHON_BIN}" -u scripts/require_local_residual_field_fusion_source_audit.py
  --audit "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}"
)
cmd=(
  "${PYTHON_BIN}" -u scripts/cache_local_residual_field_a0_seed1_predictions.py
  --source-artifact-audit "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}"
  --checkpoint "${LOCAL_RESIDUAL_FIELD_A0_SEED1_CHECKPOINT}"
  --prediction-dir "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_DIR}"
  --hlt-cache-dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
  --manifest-path "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
  --batch-size "${LOCAL_RESIDUAL_FIELD_PREDICT_BATCH_SIZE}"
  --num-workers "${LOCAL_RESIDUAL_FIELD_PREDICT_NUM_WORKERS}"
  --device "${DEVICE}"
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --disable-amp "${LOCAL_RESIDUAL_FIELD_PREDICT_DISABLE_AMP}"
if [[ -n "${LOCAL_RESIDUAL_FIELD_PREDICT_MAX_JETS}" ]]; then cmd+=(--max-jets "${LOCAL_RESIDUAL_FIELD_PREDICT_MAX_JETS}"); fi

fresh_write_run_config \
  "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_DIR}/A0_seed1" \
  "local_residual_field_A0_seed1_development_predictions" \
  "${gate_cmd[@]}" \
  "${cmd[@]}"
fresh_run "${gate_cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  for split in stack_train stack_val; do
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_DIR}/A0_seed1/${split}_predictions.npz"
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_DIR}/A0_seed1/${split}_predictions_metadata.json"
  done
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_DIR}/development_prediction_sources.json"
fi
