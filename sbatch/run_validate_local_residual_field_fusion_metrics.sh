#!/usr/bin/env bash
# CPU reproduction audit for the already-opened A0/P7b final-test metrics.

#SBATCH --job-name=lprf_fuse_metrics
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
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_fusion/p7b_seed_control}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT:=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/source_artifact_audit.json}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_METRIC_AUDIT:=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/metric_reproduction_audit.json}"

fresh_setup "$@"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}"
if ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Set CONFIRM_FINAL_TEST=1: this audit intentionally reads the already-opened exploratory final_test" >&2
  exit 2
fi
gate_cmd=("${PYTHON_BIN}" -u scripts/require_local_residual_field_fusion_source_audit.py --audit "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}")
cmd=(
  "${PYTHON_BIN}" -u scripts/validate_local_residual_field_fusion_metrics.py
  --source-artifact-audit "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}"
  --output "${LOCAL_RESIDUAL_FIELD_FUSION_METRIC_AUDIT}"
  --confirm-exploratory-final-test
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_run "${gate_cmd[@]}"
fresh_run "${cmd[@]}"
if ! fresh_is_dry_run; then fresh_assert_json_ok "${LOCAL_RESIDUAL_FIELD_FUSION_METRIC_AUDIT}"; fi
