#!/usr/bin/env bash
#SBATCH --job-name=rpt_tree_probe
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

binary_name="$(rpt_field "${CAMPAIGN_ROOT}/backend/backend_manifest.json" binary_filename)"
miniature=()
if [[ "${RPT_MINIATURE:-0}" == "1" ]]; then
  miniature=(--miniature)
fi
override=()
if [[ -n "${RPT_TREE_OPERATIONAL_OVERRIDE:-}" ]]; then
  override=(--operational-override-json "${RPT_TREE_OPERATIONAL_OVERRIDE}")
fi
python scripts/probe_relational_part_tree_backend.py \
  --cache-dir "${CAMPAIGN_ROOT}/inputs/hlt_cache" \
  --hlt-binding "${CAMPAIGN_ROOT}/inputs/hlt_cache_audit.json" \
  --tree-resource "${CAMPAIGN_ROOT}/inputs/angular_tree_resource_contract.json" \
  --backend-manifest "${CAMPAIGN_ROOT}/backend/backend_manifest.json" \
  --backend-binary "${CAMPAIGN_ROOT}/backend/${binary_name}" \
  --storage-projection "${CAMPAIGN_ROOT}/storage_projection.json" \
  --output "${CAMPAIGN_ROOT}/backend/throughput_probe.json" \
  "${miniature[@]}" \
  "${override[@]}"
