#!/usr/bin/env bash
#SBATCH --job-name=rpt_off_regshard
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
set -euo pipefail
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_offline_setup
python scripts/fit_relational_part_region_normalization_shard.py \
  --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --plan "${CAMPAIGN_ROOT}/inputs/region_normalization_map/plan.json" \
  --tree-dir "${CAMPAIGN_ROOT}/inputs/relation_tree_cache/model_train_exclusive_ca_v1" \
  --output-dir "${CAMPAIGN_ROOT}/inputs/region_normalization_map/partials"
