#!/usr/bin/env bash
#SBATCH --job-name=rpt_tree_shard
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=02:00:00

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

if [[ -z "${RPT_TREE_SPLIT:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "RPT_TREE_SPLIT and SLURM_ARRAY_TASK_ID are required" >&2
  exit 2
fi
count="$(rpt_field "${CAMPAIGN_ROOT}/inputs/split_audit.json" "split_sizes.${RPT_TREE_SPLIT}")"
start="$((SLURM_ARRAY_TASK_ID * RPT_TREE_SHARD_SIZE))"
stop="$((start + RPT_TREE_SHARD_SIZE))"
if (( stop > count )); then
  stop="${count}"
fi
if (( start >= stop )); then
  echo "Tree shard range is empty: ${start}:${stop}" >&2
  exit 2
fi
binary_name="$(rpt_field "${CAMPAIGN_ROOT}/backend/backend_manifest.json" binary_filename)"
tree_dir="${CAMPAIGN_ROOT}/inputs/relation_tree_cache/${RPT_TREE_SPLIT}_exclusive_ca_v1"
python scripts/build_relational_part_angular_tree_cache.py \
  --cache-dir "${CAMPAIGN_ROOT}/inputs/hlt_cache" \
  --split "${RPT_TREE_SPLIT}" \
  --start "${start}" \
  --stop "${stop}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --tree-resource "${CAMPAIGN_ROOT}/inputs/angular_tree_resource_contract.json" \
  --backend-manifest "${CAMPAIGN_ROOT}/backend/backend_manifest.json" \
  --backend-binary "${CAMPAIGN_ROOT}/backend/${binary_name}" \
  --output-dir "${tree_dir}"
