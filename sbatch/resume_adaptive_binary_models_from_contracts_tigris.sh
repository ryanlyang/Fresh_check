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
if [[ -n "${tagger_acceptance}" ]]; then
  export ABPH_TAGGER_DDP_ACCEPTANCE_PATH="${tagger_acceptance}"
else
  unset ABPH_TAGGER_DDP_ACCEPTANCE_PATH
fi
export ABPH_SBATCH_ACCOUNT ABPH_SBATCH_PARTITION ABPH_DATA_DIR
export OVERWRITE=1
export CONFIRM_FINAL_TEST=0

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
mapfile -t missing_contract_variants < <(
  "${PYTHON_BIN}" - "${campaign_root}" <<'PY'
import sys
from pathlib import Path

from teacher_logit_reco.adaptive_binary_pseudooffline.orchestration import (
    ABPH_TRAINED_RECONSTRUCTOR_VARIANTS,
)

root = Path(sys.argv[1])
for variant in ABPH_TRAINED_RECONSTRUCTOR_VARIANTS:
    contract = (
        root
        / "runtime_batch_contracts"
        / variant
        / "runtime_batch_contract.json"
    )
    if not contract.is_file():
        print(variant)
PY
)
if ((${#missing_contract_variants[@]})); then
  latest_probe_manifest="$(
    find "${campaign_root}/submission_logs" -maxdepth 1 -type f \
      -name 'abph_clean_model_recovery_contracts_*.tsv' \
      -printf '%T@ %p\n' |
      sort -n |
      tail -1 |
      cut -d' ' -f2-
  )"
  [[ -f "${latest_probe_manifest}" ]] || {
    echo "Required runtime contracts are missing and no recovery manifest exists: ${missing_contract_variants[*]}" >&2
    exit 2
  }

  producer_jobs=()
  for variant in "${missing_contract_variants[@]}"; do
    producer="$(
      awk -F $'\t' -v variant="${variant}" \
        '$1 == variant && $2 == "compile" {job=$4} END {print job}' \
        "${latest_probe_manifest}"
    )"
    [[ "${producer}" =~ ^[0-9]+$ ]] || {
      echo "No compiler job was recorded for missing contract ${variant}." >&2
      exit 2
    }
    active_row="$(
      (squeue -h -j "${producer}" -o "%T|%R" 2>/dev/null || true) |
        tail -1
    )"
    active_state="${active_row%%|*}"
    active_reason="${active_row#*|}"
    if [[ -n "${active_state}" ]] &&
       [[ "${active_reason}" != *DependencyNeverSatisfied* ]]; then
      echo "Waiting for ${variant} contract producer ${producer} (${active_state}; ${active_reason})."
      producer_jobs+=("${producer}")
      continue
    fi
    if [[ "${active_reason}" == *DependencyNeverSatisfied* ]]; then
      echo "Missing ${variant} contract producer ${producer} cannot run because one of its probes failed." >&2
      exit 2
    fi
    terminal_state="$(
      sacct -X -n -j "${producer}" --format=State -P |
        head -1 |
        cut -d'|' -f1
    )"
    echo "Missing ${variant} contract producer ${producer} is ${terminal_state:-UNKNOWN}." >&2
    exit 2
  done

  dependency="afterok:$(IFS=:; echo "${producer_jobs[*]}")"
  continuation="$(
    sbatch --parsable \
      --account="${ABPH_SBATCH_ACCOUNT}" \
      --partition="${ABPH_SBATCH_PARTITION}" \
      --job-name=fresh_abph_models_resume \
      --output=/dev/null \
      --error=/dev/null \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=2 \
      --mem=16G \
      --time=00:30:00 \
      --dependency="${dependency}" \
      --export=ALL,ABPH_CONFIRM_MODELS_RESUME=1 \
      --chdir="${PROJECT_DIR}" \
      "${PROJECT_DIR}/sbatch/resume_adaptive_binary_models_from_contracts_tigris.sh"
  )"
  continuation_job="${continuation%%;*}"
  [[ "${continuation_job}" =~ ^[0-9]+$ ]] || {
    echo "Invalid models-resume continuation response: ${continuation}" >&2
    exit 2
  }
  echo "Models resume will continue automatically as job ${continuation_job} after ${producer_jobs[*]}."
  exit 0
fi

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
