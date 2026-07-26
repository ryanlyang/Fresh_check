#!/usr/bin/env bash
#SBATCH --job-name=rpt_hlt_cache
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

python scripts/build_fixed_hlt_cache.py \
  --manifest "${CAMPAIGN_ROOT}/inputs/split_manifest.json.gz" \
  --data-dir "${DATA_DIR}" \
  --cache-dir "${CAMPAIGN_ROOT}/inputs/hlt_cache" \
  --splits model_train model_val stack_val final_test \
  --hlt-profile fixed_hlt_v1 \
  --hlt-degradation-strength 0.6 \
  --verify-label-branches
python scripts/bind_relational_part_hlt_cache.py \
  --manifest "${CAMPAIGN_ROOT}/inputs/split_manifest.json.gz" \
  --cache-dir "${CAMPAIGN_ROOT}/inputs/hlt_cache" \
  --split-binding "${CAMPAIGN_ROOT}/inputs/split_audit.json" \
  --hlt-expectation "${CAMPAIGN_ROOT}/inputs/hlt_expectation.json" \
  --output "${CAMPAIGN_ROOT}/inputs/hlt_cache_audit.json"
