#!/usr/bin/env bash
#SBATCH --job-name=rpt_off_regplan
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
set -euo pipefail
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_offline_setup
python scripts/prepare_relational_part_region_normalization_map.py \
  --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --cache-dir "${CAMPAIGN_ROOT}/inputs/offline_cache" \
  --tree-dir "${CAMPAIGN_ROOT}/inputs/relation_tree_cache/model_train_exclusive_ca_v1" \
  --relation-normalization "${CAMPAIGN_ROOT}/inputs/relation_normalization.json" \
  --output-dir "${CAMPAIGN_ROOT}/inputs/region_normalization_map" \
  --input-view offline
