#!/usr/bin/env bash
#SBATCH --job-name=rpt_full_audit
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

tree_root="${CAMPAIGN_ROOT}/inputs/relation_tree_cache"
python scripts/audit_relational_part_inputs.py \
  --manifest "${CAMPAIGN_ROOT}/inputs/split_manifest.json.gz" \
  --preconstruction-audit "${CAMPAIGN_ROOT}/inputs/preconstruction_raw_input_audit.json" \
  --hlt-binding "${CAMPAIGN_ROOT}/inputs/hlt_cache_audit.json" \
  --relation-normalization "${CAMPAIGN_ROOT}/inputs/relation_normalization.json" \
  --region-normalization "${CAMPAIGN_ROOT}/inputs/region_normalization.json" \
  --backend-manifest "${CAMPAIGN_ROOT}/backend/backend_manifest.json" \
  --throughput-probe "${CAMPAIGN_ROOT}/backend/throughput_probe.json" \
  --tree-split "model_train=${tree_root}/model_train_exclusive_ca_v1" \
  --tree-split "model_val=${tree_root}/model_val_exclusive_ca_v1" \
  --tree-split "stack_val=${tree_root}/stack_val_exclusive_ca_v1" \
  --tree-split "final_test=${tree_root}/final_test_exclusive_ca_v1" \
  --storage-projection "${CAMPAIGN_ROOT}/storage_projection.json" \
  --output "${CAMPAIGN_ROOT}/inputs/postconstruction_input_audit.json"
