#!/usr/bin/env bash
#SBATCH --job-name=rpt_weaver_parity
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=01:00:00

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

python scripts/validate_relational_part_weaver_parity.py \
  --device cpu \
  --output "${CAMPAIGN_ROOT}/backend/weaver_parity.json"
