#!/usr/bin/env bash
#SBATCH --job-name=rpt_tree_finalize
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=03:00:00

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

if [[ -z "${RPT_TREE_SPLIT:-}" ]]; then
  echo "RPT_TREE_SPLIT is required" >&2
  exit 2
fi
count="$(rpt_field "${CAMPAIGN_ROOT}/inputs/split_audit.json" "split_sizes.${RPT_TREE_SPLIT}")"
hlt_sha="$(rpt_field "${CAMPAIGN_ROOT}/inputs/hlt_cache_audit.json" "split_reports.${RPT_TREE_SPLIT}.hlt_content_hash")"
resource_sha="$(rpt_field "${CAMPAIGN_ROOT}/inputs/angular_tree_resource_contract.json" content_hash)"
backend_sha="$(rpt_field "${CAMPAIGN_ROOT}/backend/backend_manifest.json" content_hash)"
tree_dir="${CAMPAIGN_ROOT}/inputs/relation_tree_cache/${RPT_TREE_SPLIT}_exclusive_ca_v1"
python scripts/finalize_relational_part_angular_tree_cache.py \
  --tree-dir "${tree_dir}" \
  --split "${RPT_TREE_SPLIT}" \
  --expected-jet-count "${count}" \
  --hlt-content-sha256 "${hlt_sha}" \
  --tree-resource-sha256 "${resource_sha}" \
  --backend-manifest-sha256 "${backend_sha}"
