#!/usr/bin/env bash
#SBATCH --job-name=rpt_off_final
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=08:00:00
set -euo pipefail
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_offline_setup
python scripts/run_relational_part_offline_final_task.py --campaign-root "${CAMPAIGN_ROOT}" --device "${RPT_DEVICE}"
