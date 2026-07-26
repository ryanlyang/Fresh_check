#!/usr/bin/env bash
#SBATCH --job-name=rpt_campaign
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=01:00:00

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

if [[ -z "${CAMPAIGN_ID:-}" || -z "${RPT_STORAGE_MEASUREMENTS:-}" ]]; then
  echo "CAMPAIGN_ID and RPT_STORAGE_MEASUREMENTS are required" >&2
  exit 2
fi
miniature=()
if [[ "${RPT_MINIATURE:-0}" == "1" ]]; then
  miniature=(--miniature)
fi
python scripts/build_relational_part_campaign.py \
  --parent-manifest "${CAMPAIGN_ROOT}/bootstrap/split_manifest.json.gz" \
  --output-dir "${CAMPAIGN_ROOT}" \
  --campaign-id "${CAMPAIGN_ID}" \
  --storage-measurements "${RPT_STORAGE_MEASUREMENTS}" \
  "${miniature[@]}"
