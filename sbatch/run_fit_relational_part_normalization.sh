#!/usr/bin/env bash
#SBATCH --job-name=rpt_normalize
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

python scripts/fit_relational_part_normalization.py \
  --cache-dir "${CAMPAIGN_ROOT}/inputs/hlt_cache" \
  --normalization-contract "${CAMPAIGN_ROOT}/inputs/normalization_contract.json" \
  --relation-registry "${CAMPAIGN_ROOT}/registry/relation_family_registry.json" \
  --raw-input-schema "${CAMPAIGN_ROOT}/inputs/raw_input_schema.json" \
  --hlt-binding "${CAMPAIGN_ROOT}/inputs/hlt_cache_audit.json" \
  --output "${CAMPAIGN_ROOT}/inputs/relation_normalization.json"
