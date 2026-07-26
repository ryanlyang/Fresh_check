#!/usr/bin/env bash
#SBATCH --job-name=rpt_model_contracts
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

python scripts/build_relational_part_model_contracts.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --parity-report "${CAMPAIGN_ROOT}/backend/weaver_parity.json"
