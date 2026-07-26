#!/usr/bin/env bash
#SBATCH --job-name=rpt_confirm_aggregate
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=02:00:00

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

mode="${RPT_AGGREGATE_MODE:-summary}"
confirmation_results="${CAMPAIGN_ROOT}/selection/confirmation_results.json"
python scripts/collect_relational_part_results.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --registry "${CAMPAIGN_ROOT}/selection/confirmation_registry.json" \
  --mode confirmation \
  --output "${confirmation_results}"
split_sha="$(rpt_field "${CAMPAIGN_ROOT}/campaign_spec.json" split_manifest_hash)"
mapfile -t hashes < <(rpt_hlt_hash_args)
hash_args=()
for value in "${hashes[@]}"; do
  hash_args+=(--hlt-cache-hash "${value}")
done

if [[ "${mode}" == "summary" ]]; then
  python scripts/aggregate_relational_part_confirmation.py \
    --confirmation-registry "${CAMPAIGN_ROOT}/selection/confirmation_registry.json" \
    --results "${confirmation_results}" \
    --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
    --split-manifest-sha256 "${split_sha}" \
    "${hash_args[@]}" \
    --summary-output "${CAMPAIGN_ROOT}/selection/confirmation_summary.json" \
    --summary-only
  python scripts/prepare_relational_part_semantic_controls.py \
    --campaign-root "${CAMPAIGN_ROOT}"
  semantic_job="$(rpt_submit_dynamic_once semantic_controls \
    "afterok:${SLURM_JOB_ID}" \
    --account="${SBATCH_ACCOUNT}" \
    --partition="${SBATCH_PARTITION}" \
    --gres="${GPU_GRES}" \
    --cpus-per-task="${GPU_CPUS_PER_TASK}" \
    --mem="${GPU_MEM}" \
    --output="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.out" \
    --error="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.err" \
    --dependency="afterok:${SLURM_JOB_ID}" \
    "${SCRIPT_DIR}/run_evaluate_relational_part_semantic_controls.sh")"
  unary_job="$(rpt_submit_dynamic_once unary_training \
    "afterok:${SLURM_JOB_ID}" \
    --account="${SBATCH_ACCOUNT}" \
    --partition="${SBATCH_PARTITION}" \
    --gres="${GPU_GRES}" \
    --cpus-per-task="${GPU_CPUS_PER_TASK}" \
    --mem="${GPU_MEM}" \
    --output="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.out" \
    --error="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.err" \
    --dependency="afterok:${SLURM_JOB_ID}" \
    --array="0-2%3" \
    --export="ALL,RPT_TRAIN_MODE=unary" \
    "${SCRIPT_DIR}/run_train_relational_part.sh")"
  lock_job="$(rpt_submit_dynamic_once finalist_lock \
    "afterok:${semantic_job}:${unary_job}" \
    --account="${SBATCH_ACCOUNT}" \
    --partition="${SBATCH_PARTITION}" \
    --cpus-per-task="${CPU_CPUS_PER_TASK}" \
    --mem="${CPU_MEM}" \
    --output="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.out" \
    --error="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.err" \
    --dependency="afterok:${semantic_job}:${unary_job}" \
    --export="ALL,RPT_AGGREGATE_MODE=lock" \
    "${SCRIPT_DIR}/run_aggregate_relational_part_confirmation.sh")"
  printf 'semantic controls: %s\nunary training: %s\nfinalist lock: %s\n' \
    "${semantic_job}" "${unary_job}" "${lock_job}"
elif [[ "${mode}" == "lock" ]]; then
  unary_results="${CAMPAIGN_ROOT}/selection/semantic_controls/unary_results.json"
  python scripts/collect_relational_part_results.py \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --registry "${CAMPAIGN_ROOT}/selection/semantic_controls/unary_control_registry.json" \
    --mode unary \
    --output "${unary_results}"
  python scripts/aggregate_relational_part_confirmation.py \
    --confirmation-registry "${CAMPAIGN_ROOT}/selection/confirmation_registry.json" \
    --results "${confirmation_results}" \
    --unary-results "${unary_results}" \
    --semantic-perturbations "${CAMPAIGN_ROOT}/selection/semantic_controls/perturbation_metrics.json" \
    --unary-registry "${CAMPAIGN_ROOT}/selection/semantic_controls/unary_control_registry.json" \
    --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
    --split-manifest-sha256 "${split_sha}" \
    "${hash_args[@]}" \
    --summary-output "${CAMPAIGN_ROOT}/selection/confirmation_summary.json" \
    --lock-output "${CAMPAIGN_ROOT}/selection/locked_finalists.json"
  final_submit="$(rpt_submit_dynamic_once final_test_submit \
    "afterok:${SLURM_JOB_ID}" \
    --account="${SBATCH_ACCOUNT}" \
    --partition="${SBATCH_PARTITION}" \
    --cpus-per-task="${CPU_CPUS_PER_TASK}" \
    --mem="${CPU_MEM}" \
    --output="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.out" \
    --error="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.err" \
    --dependency="afterok:${SLURM_JOB_ID}" \
    "${SCRIPT_DIR}/run_submit_relational_part_final_test.sh")"
  printf 'final-test continuation: %s\n' "${final_submit}"
else
  echo "Unknown RPT_AGGREGATE_MODE=${mode}" >&2
  exit 2
fi
