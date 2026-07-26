#!/usr/bin/env bash
#SBATCH --job-name=rpt_final_submit
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=01:00:00

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

: "${RPT_FINAL_CONCURRENCY:=3}"
python scripts/prepare_relational_part_final_tasks.py \
  --campaign-root "${CAMPAIGN_ROOT}"
tasks="${CAMPAIGN_ROOT}/final_test/task_registry.json"
count="$(rpt_field "${tasks}" task_count)"
if (( count <= 0 )); then
  echo "Final-test task registry is empty" >&2
  exit 2
fi
last="$((count - 1))"
eval_job="$(sbatch --parsable \
  --account="${SBATCH_ACCOUNT}" \
  --partition="${SBATCH_PARTITION}" \
  --gres="${GPU_GRES}" \
  --cpus-per-task="${GPU_CPUS_PER_TASK}" \
  --mem="${GPU_MEM}" \
  --output="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.out" \
  --error="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.err" \
  --dependency="afterok:${SLURM_JOB_ID}" \
  --array="0-${last}%${RPT_FINAL_CONCURRENCY}" \
  --export="ALL,RPT_FINAL_TASK_REGISTRY=${tasks}" \
  "${SCRIPT_DIR}/run_evaluate_relational_part_final_test.sh")"
rpt_record_dynamic_job final_test_evaluation "${eval_job}" "afterok:${SLURM_JOB_ID}"
report_job="$(sbatch --parsable \
  --account="${SBATCH_ACCOUNT}" \
  --partition="${SBATCH_PARTITION}" \
  --cpus-per-task="${CPU_CPUS_PER_TASK}" \
  --mem="${CPU_MEM}" \
  --output="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.out" \
  --error="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.err" \
  --dependency="afterok:${eval_job}" \
  "${SCRIPT_DIR}/run_write_relational_part_report.sh")"
rpt_record_dynamic_job final_report "${report_job}" "afterok:${eval_job}"
printf 'final-test array: %s\nreport: %s\n' "${eval_job}" "${report_job}"
