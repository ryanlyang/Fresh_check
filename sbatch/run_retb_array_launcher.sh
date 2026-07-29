#!/usr/bin/env bash
#SBATCH --job-name=retb_array_launch
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_setup
: "${RETB_NODE_ID:?RETB_NODE_ID is required}"
manifest="$(retb_task_manifest_path "${RETB_NODE_ID}")"
if [[ ! -f "${manifest}" ]]; then
  echo "Dynamic array manifest was not published by its dependency: ${manifest}" >&2
  exit 2
fi
readarray -t resolved < <(
  python -c \
    'import json,sys; from teacher_logit_reco.relation_expert_token_bridge.production import validate_task_manifest; p=json.load(open(sys.argv[1])); validate_task_manifest(p); print(p["slurm_array"]); print(p["content_hash"])' \
    "${manifest}"
)
array_spec="${resolved[0]}"
manifest_sha="${resolved[1]}"
dependency="${SLURM_JOB_ID:-}"
resource_arguments=(
  --account="${SBATCH_ACCOUNT}"
  --partition="${SBATCH_PARTITION}"
  --cpus-per-task="${CPU_CPUS_PER_TASK}"
  --mem="${CPU_MEM}"
)
if [[ "${RETB_NODE_RESOURCE:-cpu}" == "gpu" ]]; then
  resource_arguments+=(
    --gres="${GPU_GRES}"
    --cpus-per-task="${GPU_CPUS_PER_TASK}"
    --mem="${GPU_MEM}"
  )
fi
job_id="$(sbatch --parsable --wait \
  "${resource_arguments[@]}" \
  --array="${array_spec}" \
  --job-name="${CAMPAIGN_ID}_${RETB_NODE_ID}_rows" \
  --output="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.out" \
  --error="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.err" \
  --export="ALL,CAMPAIGN_ROOT=${CAMPAIGN_ROOT},CAMPAIGN_ID=${CAMPAIGN_ID},RETB_NODE_ID=${RETB_NODE_ID},RETB_TASK_MANIFEST=${manifest}" \
  "${PROJECT_DIR}/sbatch/run_retb_production_task.sh")"
retb_record_dynamic_job \
  "${RETB_NODE_ID}" "${job_id}" "${dependency}" "${manifest_sha}"
