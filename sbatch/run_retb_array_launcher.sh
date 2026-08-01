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
if python -c \
  'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if any("continuation_intent" in row["input_artifact_hashes"] for row in p["rows"]) else 1)' \
  "${manifest}"; then
  python scripts/validate_retb_dynamic_continuation.py \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --task-manifest "${manifest}" >/dev/null
fi
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
  resolved_gpu_mem="${GPU_MEM}"
  if [[ "${RETB_SUBMISSION_SCOPE:-complete}" == "offline_abc_streamed" && "${RETB_NODE_ID}" == "offline_fusion_cache" ]]; then
    resolved_gpu_mem="${RETB_STREAMED_GPU_MEM}"
  fi
  resource_arguments+=(
    --gres="${GPU_GRES}"
    --cpus-per-task="${GPU_CPUS_PER_TASK}"
    --mem="${resolved_gpu_mem}"
  )
fi
job_id="$(sbatch --parsable --wait \
  "${resource_arguments[@]}" \
  --array="${array_spec}" \
    --job-name="${CAMPAIGN_ID}_${RETB_NODE_ID}_rows" \
  --output="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.out" \
  --error="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.err" \
    --export="ALL,CAMPAIGN_ROOT=${CAMPAIGN_ROOT},CAMPAIGN_ID=${CAMPAIGN_ID},RETB_NODE_ID=${RETB_NODE_ID},RETB_TASK_MANIFEST=${manifest},RETB_DEFER_MANIFEST_MATERIALIZATION=1" \
  "${PROJECT_DIR}/sbatch/run_retb_production_task.sh")"
python scripts/attest_retb_task_manifest_completion.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --task-manifest "${manifest}" >/dev/null
retb_materialize_downstream "${RETB_NODE_ID}"
retb_record_dynamic_job \
  "${RETB_NODE_ID}" "${job_id}" "${dependency}" "${manifest_sha}"
