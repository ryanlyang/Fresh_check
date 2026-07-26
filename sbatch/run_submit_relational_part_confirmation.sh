#!/usr/bin/env bash
#SBATCH --job-name=rpt_confirm_submit
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

: "${RPT_CONFIRMATION_CONCURRENCY:=4}"
python scripts/prepare_relational_part_confirmation.py \
  --campaign-root "${CAMPAIGN_ROOT}"
tasks="${CAMPAIGN_ROOT}/selection/confirmation_tasks.json"
count="$(rpt_field "${tasks}" task_count)"
if (( count <= 0 )); then
  echo "Confirmation task registry is empty" >&2
  exit 2
fi
last="$((count - 1))"
train_job="$(rpt_submit_dynamic_once confirmation_training \
  "afterok:${SLURM_JOB_ID}" \
  --account="${SBATCH_ACCOUNT}" \
  --partition="${SBATCH_PARTITION}" \
  --gres="${GPU_GRES}" \
  --cpus-per-task="${GPU_CPUS_PER_TASK}" \
  --mem="${GPU_MEM}" \
  --output="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.out" \
  --error="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.err" \
  --dependency="afterok:${SLURM_JOB_ID}" \
  --array="0-${last}%${RPT_CONFIRMATION_CONCURRENCY}" \
  --export="ALL,RPT_TRAIN_MODE=confirmation,RPT_TASK_REGISTRY=${tasks}" \
  "${SCRIPT_DIR}/run_train_relational_part.sh")"
summary_job="$(rpt_submit_dynamic_once confirmation_summary \
  "afterok:${train_job}" \
  --account="${SBATCH_ACCOUNT}" \
  --partition="${SBATCH_PARTITION}" \
  --cpus-per-task="${CPU_CPUS_PER_TASK}" \
  --mem="${CPU_MEM}" \
  --output="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.out" \
  --error="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.err" \
  --dependency="afterok:${train_job}" \
  --export="ALL,RPT_AGGREGATE_MODE=summary" \
  "${SCRIPT_DIR}/run_aggregate_relational_part_confirmation.sh")"
printf 'confirmation training array: %s\nconfirmation summary: %s\n' \
  "${train_job}" "${summary_job}"
