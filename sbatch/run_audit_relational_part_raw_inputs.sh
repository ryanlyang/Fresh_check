#!/usr/bin/env bash
#SBATCH --job-name=rpt_raw_audit
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=06:00:00

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

miniature=()
if [[ "${RPT_MINIATURE:-0}" == "1" ]]; then
  miniature=(--miniature)
fi
python scripts/audit_relational_part_raw_inputs.py \
  --manifest "${CAMPAIGN_ROOT}/inputs/split_manifest.json.gz" \
  --raw-input-schema "${CAMPAIGN_ROOT}/inputs/raw_input_schema.json" \
  --data-dir "${DATA_DIR}" \
  --tree-name tree \
  --output "${CAMPAIGN_ROOT}/inputs/preconstruction_raw_input_audit.json" \
  "${miniature[@]}"
