#!/usr/bin/env bash
#SBATCH --job-name=rpt_off_norm
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
set -euo pipefail
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_offline_setup
python scripts/fit_relational_part_offline_normalization.py --campaign-root "${CAMPAIGN_ROOT}"
