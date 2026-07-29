#!/usr/bin/env bash
#SBATCH --job-name=retb_export
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_FINAL_CONSUMER_RUN:?RETB_FINAL_CONSUMER_RUN is required}"
: "${RETB_CONSUMER_REGISTRATION:?RETB_CONSUMER_REGISTRATION is required}"
: "${RETB_CONSUMER_CHECKPOINT:=}"
: "${RETB_PREPARED_EXPORT:?RETB_PREPARED_EXPORT is required}"
: "${RETB_PREPARED_EXPORT_SHA256:?RETB_PREPARED_EXPORT_SHA256 is required}"
: "${RETB_DEPLOYABLE_PARENTS:?RETB_DEPLOYABLE_PARENTS is required}"
: "${RETB_DEPLOYABLE_OUTPUT:?RETB_DEPLOYABLE_OUTPUT is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
checkpoint_args=()
if [[ -n "${RETB_CONSUMER_CHECKPOINT}" ]]; then
  checkpoint_args=(
    --consumer-checkpoint "${RETB_CONSUMER_CHECKPOINT}"
  )
fi
python scripts/export_retb_deployable_graph.py \
  --campaign-root "${CAMPAIGN_ROOT}" --run "${RETB_FINAL_CONSUMER_RUN}" \
  --consumer-registration "${RETB_CONSUMER_REGISTRATION}" \
  "${checkpoint_args[@]}" \
  --prepared-export "${RETB_PREPARED_EXPORT}" \
  --prepared-export-sha256 "${RETB_PREPARED_EXPORT_SHA256}" \
  --parent-hashes "${RETB_DEPLOYABLE_PARENTS}" \
  --output-dir "${RETB_DEPLOYABLE_OUTPUT}"
