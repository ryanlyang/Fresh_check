#!/usr/bin/env bash
#SBATCH --job-name=rpt_off_tree_fin
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
set -euo pipefail
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_offline_setup
: "${RPT_TREE_SPLIT:?RPT_TREE_SPLIT is required}"
case "${RPT_TREE_SPLIT}" in
  model_train) count=1000000 ;;
  model_val|stack_val) count=125000 ;;
  final_test) count=500000 ;;
  *) exit 2 ;;
esac
python scripts/finalize_relational_part_offline_tree_split.py \
  --cache-dir "${CAMPAIGN_ROOT}/inputs/offline_cache" \
  --tree-dir "${CAMPAIGN_ROOT}/inputs/relation_tree_cache/${RPT_TREE_SPLIT}_exclusive_ca_v1" \
  --split "${RPT_TREE_SPLIT}" --expected-jet-count "${count}" \
  --tree-resource "${CAMPAIGN_ROOT}/inputs/angular_tree_resource_contract.json" \
  --backend-manifest "${CAMPAIGN_ROOT}/backend/backend_manifest.json"
