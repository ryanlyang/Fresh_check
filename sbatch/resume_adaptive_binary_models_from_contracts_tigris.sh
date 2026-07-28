#!/usr/bin/env bash
# Resume the ABPH model wave from already-validated DDP runtime contracts.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${ABPH_ROOT:?Set ABPH_ROOT to the prepared streaming campaign root}"
: "${ABPH_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data/jetclass_part1}"
: "${ABPH_SBATCH_ACCOUNT:=reu-aisocial}"
: "${ABPH_SBATCH_PARTITION:=tigris}"

[[ "${ABPH_CONFIRM_MODELS_RESUME:-0}" == "1" ]] || {
  echo "Set ABPH_CONFIRM_MODELS_RESUME=1 to replace active abph_* setup jobs." >&2
  exit 2
}

export PROJECT_DIR CONDA_BASE CONDA_ENV PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"
fresh_prepare_submitter
fresh_activate_env

project_root="$(readlink -f "${PROJECT_DIR}")"
campaign_root="$(readlink -f "${ABPH_ROOT}")"
case "${campaign_root}" in
  "${project_root}"/checkpoints/*) ;;
  *)
    echo "ABPH_ROOT must be under ${project_root}/checkpoints: ${campaign_root}" >&2
    exit 2
    ;;
esac

manifest="${campaign_root}/submission_logs/abph_full_submission.json"
fresh_require_file "${manifest}"
fresh_require_file "${campaign_root}/storage/storage_acceptance.json"
fresh_require_file "${campaign_root}/audits/step1_input_audit.json"

mapfile -t provenance < <(
  "${PYTHON_BIN}" - "${manifest}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
runtime = payload.get("runtime_acceptance") or {}
projection = payload.get("storage_projection") or {}
tagger = payload.get("tagger_ddp_acceptance") or {}
print(runtime.get("path", ""))
print(projection.get("path", ""))
print(tagger.get("path", ""))
PY
)
runtime_acceptance="${provenance[0]:-}"
storage_projection="${provenance[1]:-}"
tagger_acceptance="${provenance[2]:-}"
fresh_require_file "${runtime_acceptance}"
fresh_require_file "${storage_projection}"
if [[ -n "${tagger_acceptance}" ]]; then
  fresh_require_file "${tagger_acceptance}"
fi

export ABPH_ROOT="${campaign_root}"
export ABPH_CAMPAIGN_MODE=pilot
export ABPH_STAGE_MODE=models
export ABPH_RECONSTRUCTOR_PARALLELISM=ddp8
export ABPH_RECONSTRUCTOR_SCHEDULE_POLICY=accelerated_screening_v2_7day
export ABPH_STORAGE_PROFILE=streaming_30gb_v1
export ABPH_STORAGE_PROJECTION_PATH="${storage_projection}"
export ABPH_RUNTIME_ACCEPTANCE_PATH="${runtime_acceptance}"
export ABPH_TAGGER_DDP_ACCEPTANCE_PATH="${tagger_acceptance}"
export ABPH_SBATCH_ACCOUNT ABPH_SBATCH_PARTITION ABPH_DATA_DIR
export OVERWRITE=1
export CONFIRM_FINAL_TEST=0

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
preflight_output="/tmp/abph_models_resume_preflight_${USER}_${stamp}.json"
echo "Validating retained inputs and all required trained-model contracts..."
DRY_RUN=1 bash \
  "${PROJECT_DIR}/sbatch/submit_adaptive_binary_pseudooffline_streaming30gb_tigris.sh" \
  > "${preflight_output}"
echo "Models preflight passed: ${preflight_output}"

mapfile -t stale_jobs < <(
  squeue --me -h -o "%i|%j" |
    awk -F'|' '$2 ~ /^abph_/ {print $1}'
)
if ((${#stale_jobs[@]})); then
  echo "Cancelling stale ABPH jobs: ${stale_jobs[*]}"
  scancel "${stale_jobs[@]}"
fi

for _ in $(seq 1 90); do
  if ! squeue --me -h -o "%j" | grep -q '^abph_'; then
    break
  fi
  sleep 2
done
if squeue --me -h -o "%j" | grep -q '^abph_'; then
  echo "Timed out waiting for stale ABPH jobs to leave the queue." >&2
  exit 2
fi

DRY_RUN=0 bash \
  "${PROJECT_DIR}/sbatch/submit_adaptive_binary_pseudooffline_streaming30gb_tigris.sh"

echo "adaptive_binary_models_resume_queued:"
echo "  campaign_root: ${campaign_root}"
echo "  contracts: trained DDP8 variants only"
echo "  oracle_references: B4 and D6 use direct single-GPU diagnostics"
echo "  retained: inputs, baselines, targets, storage/runtime acceptance"
