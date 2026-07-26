#!/usr/bin/env bash
#SBATCH --job-name=rpt_report
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

tasks="${CAMPAIGN_ROOT}/final_test/task_registry.json"
count="$(rpt_field "${tasks}" task_count)"
evaluation_args=()
for ((index=0; index<count; index++)); do
  output_dir="$(rpt_field "${tasks}" "tasks.${index}.output_dir")"
  evaluation_args+=(--final-evaluation "${output_dir}/metrics.json")
done
python scripts/write_relational_part_report.py \
  --locked-finalists "${CAMPAIGN_ROOT}/selection/locked_finalists.json" \
  --confirmation-summary "${CAMPAIGN_ROOT}/selection/confirmation_summary.json" \
  "${evaluation_args[@]}" \
  --json-output "${CAMPAIGN_ROOT}/reports/relational_part_report.json" \
  --markdown-output "${CAMPAIGN_ROOT}/reports/relational_part_report.md"
python scripts/write_relational_part_job_ledger.py \
  --production-graph "${CAMPAIGN_ROOT}/job_ledgers/production_graph.json" \
  --initial-ledger "${CAMPAIGN_ROOT}/job_ledgers/initial_submission_ledger.json" \
  --dynamic-ledger "${CAMPAIGN_ROOT}/job_ledgers/dynamic_jobs.tsv" \
  --report-json "${CAMPAIGN_ROOT}/reports/relational_part_report.json" \
  --output "${CAMPAIGN_ROOT}/job_ledgers/completed_ledger.json" \
  --final
