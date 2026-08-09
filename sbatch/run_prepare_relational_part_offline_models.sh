#!/usr/bin/env bash
#SBATCH --job-name=rpt_off_models
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_offline_setup
python scripts/prepare_relational_part_offline_model_contracts.py --campaign-root "${CAMPAIGN_ROOT}"
