#!/usr/bin/env bash
#SBATCH --job-name=rpt_train
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is required" >&2
  exit 2
fi
mode="${RPT_TRAIN_MODE:-screening}"
extra=()
if [[ "${mode}" == "confirmation" ]]; then
  if [[ -z "${RPT_TASK_REGISTRY:-}" ]]; then
    echo "RPT_TASK_REGISTRY is required for confirmation" >&2
    exit 2
  fi
  extra=(--task-registry "${RPT_TASK_REGISTRY}")
fi
python scripts/run_relational_part_task.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --cache-dir "${CAMPAIGN_ROOT}/inputs/hlt_cache" \
  --tree-root "${CAMPAIGN_ROOT}/inputs/relation_tree_cache" \
  --mode "${mode}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --device "${RPT_DEVICE}" \
  "${extra[@]}"
