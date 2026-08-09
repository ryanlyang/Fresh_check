#!/usr/bin/env bash
#SBATCH --job-name=rpt_off_report
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
set -euo pipefail
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_offline_setup
python scripts/write_relational_part_offline_report.py --campaign-root "${CAMPAIGN_ROOT}"
