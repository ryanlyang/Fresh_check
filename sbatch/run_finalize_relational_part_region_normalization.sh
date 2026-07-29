#!/usr/bin/env bash
#SBATCH --job-name=rpt_region_reduce
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=8:00:00

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

python scripts/finalize_relational_part_region_normalization.py \
  --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --plan "${CAMPAIGN_ROOT}/inputs/region_normalization_map/plan.json" \
  --partials-dir "${CAMPAIGN_ROOT}/inputs/region_normalization_map/partials" \
  --tree-dir "${CAMPAIGN_ROOT}/inputs/relation_tree_cache/model_train_exclusive_ca_v1" \
  --relation-normalization "${CAMPAIGN_ROOT}/inputs/relation_normalization.json" \
  --output "${CAMPAIGN_ROOT}/inputs/region_normalization.json"
