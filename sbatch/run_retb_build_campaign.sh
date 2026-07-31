#!/usr/bin/env bash
#SBATCH --job-name=retb_campaign
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
# shellcheck source=retb_common.sh
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_require_campaign_root
retb_activate
: "${CAMPAIGN_ID:?CAMPAIGN_ID is required}"
arguments=(
  --parent-manifest "${CAMPAIGN_ROOT}/bootstrap/split_manifest.json.gz"
  --output-dir "${CAMPAIGN_ROOT}"
  --campaign-id "${CAMPAIGN_ID}"
)
if [[ "${RETB_MINIATURE:-0}" == "1" ]]; then
  arguments+=(--miniature)
else
  : "${RETB_STORAGE_MEASUREMENTS:?RETB_STORAGE_MEASUREMENTS is required}"
  arguments+=(--storage-measurements "${RETB_STORAGE_MEASUREMENTS}")
fi
python scripts/build_retb_campaign.py "${arguments[@]}"
python scripts/bootstrap_retb_input_tasks.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --production-graph "${CAMPAIGN_ROOT}/job_ledgers/production_graph.json" \
  --data-dir "${DATA_DIR}"
retb_materialize_downstream "campaign_bootstrap"
