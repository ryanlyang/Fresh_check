#!/usr/bin/env bash
#SBATCH --job-name=rpt_off_tree
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
set -euo pipefail
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_offline_setup
: "${RPT_TREE_SPLIT:?RPT_TREE_SPLIT is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
case "${RPT_TREE_SPLIT}" in
  model_train) count=1000000 ;;
  model_val|stack_val) count=125000 ;;
  final_test) count=500000 ;;
  *) echo "unknown offline tree split" >&2; exit 2 ;;
esac
start="$((SLURM_ARRAY_TASK_ID * RPT_TREE_SHARD_SIZE))"
stop="$((start + RPT_TREE_SHARD_SIZE))"
if (( stop > count )); then
  stop="${count}"
fi
binary_name="$(rpt_field "${CAMPAIGN_ROOT}/backend/backend_manifest.json" binary_filename)"
python scripts/build_relational_part_offline_tree_shard.py \
  --cache-dir "${CAMPAIGN_ROOT}/inputs/offline_cache" \
  --split "${RPT_TREE_SPLIT}" --start "${start}" --stop "${stop}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --tree-resource "${CAMPAIGN_ROOT}/inputs/angular_tree_resource_contract.json" \
  --backend-manifest "${CAMPAIGN_ROOT}/backend/backend_manifest.json" \
  --backend-binary "${CAMPAIGN_ROOT}/backend/${binary_name}" \
  --output-dir "${CAMPAIGN_ROOT}/inputs/relation_tree_cache/${RPT_TREE_SPLIT}_exclusive_ca_v1"
