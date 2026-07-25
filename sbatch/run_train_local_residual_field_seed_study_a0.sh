#!/usr/bin/env bash
# Train one missing from-scratch A0 replicate for the matched seed study.

#SBATCH --job-name=lprf_seed_A0
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

SEED="${1:?Usage: sbatch run_train_local_residual_field_seed_study_a0.sh <seed>}"
shift
case "${SEED}" in 20623|20724|20825) ;; *) echo "unsupported new A0 seed ${SEED}" >&2; exit 2 ;; esac
: "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT:?seed-study campaign root is required}"
: "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_MANIFEST:=${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}/study_manifest.json}"
OUTPUT_DIR="${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}/runs/seed_${SEED}/A0"

fresh_setup "$@"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_MANIFEST}"
if ! fresh_is_dry_run && [[ -d "${OUTPUT_DIR}" ]]; then
  partial_dir="${OUTPUT_DIR}.partial_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-manual}"
  echo "Quarantining incomplete A0 seed directory: ${OUTPUT_DIR} -> ${partial_dir}"
  mv -- "${OUTPUT_DIR}" "${partial_dir}"
fi
cmd=(
  "${PYTHON_BIN}" -u scripts/train_local_residual_field_seed_study_a0.py
  --study-manifest "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_MANIFEST}"
  --seed "${SEED}"
  --output-dir "${OUTPUT_DIR}"
)
fresh_write_run_config "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}/run_configs/A0_seed_${SEED}" \
  "local_residual_field_seed_study_A0_${SEED}" "${cmd[@]}"
fresh_run "${cmd[@]}"
if ! fresh_is_dry_run; then
  fresh_assert_json_ok "${OUTPUT_DIR}/seed_study_completion.json"
fi
