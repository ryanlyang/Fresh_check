#!/usr/bin/env bash
#SBATCH --job-name=rpt_off_cache
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
set -euo pipefail
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_offline_setup
python scripts/cache_architecture_view_offline_inputs.py \
  --manifest-path "${CAMPAIGN_ROOT}/inputs/split_manifest.json.gz" \
  --output-dir "${CAMPAIGN_ROOT}/inputs/offline_cache" \
  --splits model_train model_val stack_val final_test \
  --data-dir "${DATA_DIR}" \
  --read-chunk-size 50000
