#!/usr/bin/env bash
#SBATCH --job-name=rpt_off_bind
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_offline_setup
python scripts/bind_relational_part_offline_cache.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --parent-hlt-cache "${RPT_OFFLINE_PARENT_ROOT}/inputs/hlt_cache"
