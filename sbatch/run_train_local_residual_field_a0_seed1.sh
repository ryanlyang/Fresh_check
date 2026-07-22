#!/usr/bin/env bash
# Train the audited independent HLT-only seed control for the P7b fusion campaign.

#SBATCH --job-name=lprf_A0_seed1
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=3-00:00:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"

: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/first_stage_pilot}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_fusion/p7b_seed_control}"
: "${LOCAL_RESIDUAL_FIELD_A0_DIR:=${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT}/taggers/A0}"
: "${LOCAL_RESIDUAL_FIELD_A0_SOURCE_METADATA:=${LOCAL_RESIDUAL_FIELD_A0_DIR}/source_metadata.json}"
: "${LOCAL_RESIDUAL_FIELD_A0_CHECKPOINT:=${LOCAL_RESIDUAL_FIELD_A0_DIR}/best_model_val.pt}"
: "${LOCAL_RESIDUAL_FIELD_A0_RUN_CONFIG:=${LOCAL_RESIDUAL_FIELD_A0_DIR}/slurm_run_config.json}"
: "${LOCAL_RESIDUAL_FIELD_A0_SEED1_DIR:=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/taggers/A0_seed1}"
: "${LOCAL_RESIDUAL_FIELD_A0_SEED1_RECIPE:=${LOCAL_RESIDUAL_FIELD_A0_SEED1_DIR}/a0_seed1_train_config.json}"
: "${LOCAL_RESIDUAL_FIELD_A0_SEED1_AUDIT:=${LOCAL_RESIDUAL_FIELD_A0_SEED1_DIR}/a0_seed_recipe_audit.json}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT:=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/source_artifact_audit.json}"

fresh_setup "$@"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_A0_SOURCE_METADATA}"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_A0_CHECKPOINT}"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_A0_RUN_CONFIG}"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}"
if ! fresh_is_dry_run && [[ -d "${LOCAL_RESIDUAL_FIELD_A0_SEED1_DIR}" ]]; then
  if [[ -f "${LOCAL_RESIDUAL_FIELD_A0_SEED1_DIR}/seed_control_completion.json" ]]; then
    echo "Refusing to rerun a completed immutable A0_seed1 directory: ${LOCAL_RESIDUAL_FIELD_A0_SEED1_DIR}" >&2
    exit 2
  fi
  partial_dir="${LOCAL_RESIDUAL_FIELD_A0_SEED1_DIR}.partial_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-manual}"
  echo "Quarantining interrupted A0_seed1 directory: ${LOCAL_RESIDUAL_FIELD_A0_SEED1_DIR} -> ${partial_dir}"
  mv -- "${LOCAL_RESIDUAL_FIELD_A0_SEED1_DIR}" "${partial_dir}"
fi
fresh_claim_new_dir "${LOCAL_RESIDUAL_FIELD_A0_SEED1_DIR}"

source_gate_cmd=(
  "${PYTHON_BIN}" -u scripts/require_local_residual_field_fusion_source_audit.py
  --audit "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}"
)

prepare_cmd=(
  "${PYTHON_BIN}" -u scripts/prepare_local_residual_field_a0_seed1.py
  --a0-source-metadata "${LOCAL_RESIDUAL_FIELD_A0_SOURCE_METADATA}"
  --a0-checkpoint "${LOCAL_RESIDUAL_FIELD_A0_CHECKPOINT}"
  --a0-run-config "${LOCAL_RESIDUAL_FIELD_A0_RUN_CONFIG}"
  --output-dir "${LOCAL_RESIDUAL_FIELD_A0_SEED1_DIR}"
  --recipe-output "${LOCAL_RESIDUAL_FIELD_A0_SEED1_RECIPE}"
  --audit-output "${LOCAL_RESIDUAL_FIELD_A0_SEED1_AUDIT}"
)

train_cmd=(
  "${PYTHON_BIN}" -u scripts/train_local_residual_field_a0_seed1.py
  --recipe-json "${LOCAL_RESIDUAL_FIELD_A0_SEED1_RECIPE}"
  --audit-json "${LOCAL_RESIDUAL_FIELD_A0_SEED1_AUDIT}"
  --source-artifact-audit "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}"
)

fresh_write_run_config \
  "${LOCAL_RESIDUAL_FIELD_A0_SEED1_DIR}" \
  "local_residual_field_A0_seed1" \
  "${source_gate_cmd[@]}" \
  "${prepare_cmd[@]}" \
  "${train_cmd[@]}"
fresh_run "${source_gate_cmd[@]}"
fresh_run "${prepare_cmd[@]}"
fresh_run "${train_cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_A0_SEED1_AUDIT}"
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_A0_SEED1_RECIPE}"
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_A0_SEED1_DIR}/best_model_val.pt"
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_A0_SEED1_DIR}/run_report.json"
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_A0_SEED1_DIR}/seed_control_recipe_binding.json"
  fresh_assert_json_ok "${LOCAL_RESIDUAL_FIELD_A0_SEED1_DIR}/seed_control_completion.json"
fi
