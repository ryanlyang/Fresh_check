#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CAMPAIGN_ROOT" >&2
  exit 2
fi
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_setup
CAMPAIGN_ROOT="$(cd -- "$1" && pwd)"
export CAMPAIGN_ROOT
CAMPAIGN_ID="$(basename "${CAMPAIGN_ROOT}")"
export CAMPAIGN_ID
python -c 'from pathlib import Path; from teacher_logit_reco.relation_expert_token_bridge.workflow import load_and_validate_campaign_source; load_and_validate_campaign_source(Path("'"${CAMPAIGN_ROOT}"'"), repo_root=Path("'"${PROJECT_DIR}"'"))'
previous_job=""
bindings=()
while IFS='|' read -r phase_id stage resource kind; do
  phase_artifact="${CAMPAIGN_ROOT}/evaluations/streamed_smoke/phases/${phase_id}.json"
  if [[ -z "${previous_job}" && -f "${phase_artifact}" ]]; then
    python scripts/run_retb_streamed_smoke_phase.py \
      --campaign-root "${CAMPAIGN_ROOT}" --phase-id "${phase_id}" >/dev/null
    printf 'reused locally       Stage %s %-24s authenticated\n' "${stage}" "${phase_id}"
    continue
  fi
  dependency=()
  [[ -n "${previous_job}" ]] && dependency=(--dependency="afterok:${previous_job}")
  resources=(--account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" --cpus-per-task="${CPU_CPUS_PER_TASK}" --mem="${CPU_MEM}")
  if [[ "${resource}" == "gpu" ]]; then
    resources+=(--gres="${GPU_GRES}" --cpus-per-task="${GPU_CPUS_PER_TASK}" --mem="${GPU_MEM}")
  fi
  previous_job="$(sbatch --parsable "${resources[@]}" "${dependency[@]}" \
    --job-name="${CAMPAIGN_ID}_streamed_smoke_${phase_id}" \
    --output="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.out" \
    --error="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.err" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CAMPAIGN_ROOT=${CAMPAIGN_ROOT},CAMPAIGN_ID=${CAMPAIGN_ID},RETB_SMOKE_PHASE_ID=${phase_id},RETB_NODE_RESOURCE=${resource},RETB_SUBMISSION_SCOPE=streamed_smoke" \
    "${PROJECT_DIR}/sbatch/run_retb_streamed_smoke_phase.sh")"
  bindings+=("${phase_id}=${previous_job}")
  printf 'resubmitted/reusable Stage %s %-24s %s (%s)\n' "${stage}" "${phase_id}" "${previous_job}" "${kind}"
done < <(python scripts/print_retb_streamed_smoke_phases.py)
if [[ -z "${previous_job}" ]]; then
  printf 'Compact streamed smoke is already complete; no jobs submitted.\n'
else
  printf 'Only the first incomplete suffix was submitted.\n'
fi
