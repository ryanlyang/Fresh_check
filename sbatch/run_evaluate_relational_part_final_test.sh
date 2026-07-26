#!/usr/bin/env bash
#SBATCH --job-name=rpt_final_eval
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

if [[ -z "${RPT_FINAL_TASK_REGISTRY:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "RPT_FINAL_TASK_REGISTRY and SLURM_ARRAY_TASK_ID are required" >&2
  exit 2
fi
python scripts/run_relational_part_final_task.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --task-registry "${RPT_FINAL_TASK_REGISTRY}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --device "${RPT_DEVICE}"
