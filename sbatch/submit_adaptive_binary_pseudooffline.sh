#!/usr/bin/env bash
# Canonical ABPH entry point. Approval checks remain in the Python submitter.
set -euo pipefail

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${OUTPUT_ROOT:=${PROJECT_DIR}/checkpoints}"
: "${ABPH_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data/jetclass_part1}"
: "${ABPH_CAMPAIGN_MODE:=pilot}"
: "${ABPH_STAGE_MODE:=full}"
: "${ABPH_CLUSTER:=tigris}"
: "${ABPH_RECONSTRUCTOR_PARALLELISM:=ddp4}"
: "${ABPH_STORAGE_PROFILE:=cache_heavy_v1}"
: "${PYTHONNOUSERSITE:=1}"
: "${CONDA_ENV:=atlas_kd_tigris}"

if [[ -z "${ABPH_ROOT:-}" ]]; then
  stamp="${ABPH_SUBMISSION_STAMP:-$(date +%Y%m%d_%H%M%S)}"
  ABPH_ROOT="${OUTPUT_ROOT}/adaptive_binary_pseudooffline_hltv2_s2p5_${ABPH_CAMPAIGN_MODE}_${stamp}"
fi

if [[ -n "${CONDA_BASE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

cmd=(python -u "${PROJECT_DIR}/scripts/submit_adaptive_binary_pseudooffline.py"
  --campaign-root "${ABPH_ROOT}"
  --data-dir "${ABPH_DATA_DIR}"
  --campaign-mode "${ABPH_CAMPAIGN_MODE}"
  --stage-mode "${ABPH_STAGE_MODE}"
  --cluster "${ABPH_CLUSTER}"
  --reconstructor-parallelism "${ABPH_RECONSTRUCTOR_PARALLELISM}"
  --storage-profile "${ABPH_STORAGE_PROFILE}"
  --project-dir "${PROJECT_DIR}")

if [[ -n "${ABPH_SBATCH_ACCOUNT:-}" ]]; then cmd+=(--account "${ABPH_SBATCH_ACCOUNT}"); fi
if [[ "${ABPH_APPROVE_HIGHDATA:-0}" == "1" ]]; then cmd+=(--approve-highdata); fi
if [[ -n "${ABPH_PILOT_REPORT_PATH:-}" ]]; then cmd+=(--pilot-report "${ABPH_PILOT_REPORT_PATH}"); fi
if [[ "${ABPH_APPROVE_FINAL_TEST:-0}" == "1" ]]; then cmd+=(--approve-final-test); fi
if [[ "${CONFIRM_FINAL_TEST:-0}" == "1" ]]; then cmd+=(--confirm-final-test); fi
if [[ -n "${ABPH_SELECTION_REPORT_PATH:-}" ]]; then cmd+=(--selection-report "${ABPH_SELECTION_REPORT_PATH}"); fi
if [[ -n "${ABPH_FINAL_CLAIM_CONTRACT:-}" ]]; then cmd+=(--final-claim-contract "${ABPH_FINAL_CLAIM_CONTRACT}"); fi
if [[ -n "${ABPH_RUNTIME_ACCEPTANCE_PATH:-}" ]]; then cmd+=(--runtime-acceptance "${ABPH_RUNTIME_ACCEPTANCE_PATH}"); fi
if [[ -n "${ABPH_TAGGER_DDP_ACCEPTANCE_PATH:-}" ]]; then cmd+=(--tagger-ddp-acceptance "${ABPH_TAGGER_DDP_ACCEPTANCE_PATH}"); fi
if [[ -n "${ABPH_STORAGE_PROJECTION_PATH:-}" ]]; then cmd+=(--storage-projection "${ABPH_STORAGE_PROJECTION_PATH}"); fi
if [[ "${ABPH_ALLOW_DEBUG_SINGLE_RECONSTRUCTOR:-0}" == "1" ]]; then cmd+=(--allow-debug-single-reconstructor); fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then cmd+=(--dry-run); fi
if [[ -n "${ABPH_GPU_MEMORY:-}" ]]; then cmd+=(--gpu-memory "${ABPH_GPU_MEMORY}"); fi
if [[ -n "${ABPH_CPU_MEMORY:-}" ]]; then cmd+=(--cpu-memory "${ABPH_CPU_MEMORY}"); fi

printf 'ABPH command:' >&2
printf ' %q' "${cmd[@]}" >&2
printf '\n' >&2
"${cmd[@]}"
