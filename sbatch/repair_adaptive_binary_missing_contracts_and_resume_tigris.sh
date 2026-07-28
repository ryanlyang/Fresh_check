#!/usr/bin/env bash
# Rebuild selected trained-model contracts, then resume the ABPH model wave.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${ABPH_ROOT:?Set ABPH_ROOT to the prepared streaming campaign root}"
: "${ABPH_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data/jetclass_part1}"
: "${ABPH_SBATCH_ACCOUNT:=reu-aisocial}"
: "${ABPH_SBATCH_PARTITION:=tigris}"

[[ "${ABPH_CONFIRM_CONTRACT_REPAIR:-0}" == "1" ]] || {
  echo "Set ABPH_CONFIRM_CONTRACT_REPAIR=1 to replace stale ABPH recovery jobs." >&2
  exit 2
}
 (($#)) || {
  echo "Pass at least one trained variant whose contract must be rebuilt." >&2
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

"${PYTHON_BIN}" - "$@" <<'PY'
import sys

from teacher_logit_reco.adaptive_binary_pseudooffline.orchestration import (
    ABPH_TRAINED_RECONSTRUCTOR_VARIANTS,
)

allowed = set(ABPH_TRAINED_RECONSTRUCTOR_VARIANTS)
unknown = [value for value in sys.argv[1:] if value not in allowed]
if unknown:
    raise SystemExit(
        "contract repair accepts trained DDP variants only: " + ", ".join(unknown)
    )
PY

mapfile -t stale_jobs < <(
  squeue --me -h -o "%i|%j" |
    awk -F'|' \
      '$2 ~ /^abph_/ || $2 == "fresh_abph_models_resume" {print $1}'
)
if ((${#stale_jobs[@]})); then
  echo "Cancelling stale ABPH recovery jobs: ${stale_jobs[*]}"
  scancel "${stale_jobs[@]}"
fi
for _ in $(seq 1 90); do
  if ! squeue --me -h -o "%j" |
    grep -Eq '^(abph_|fresh_abph_models_resume$)'; then
    break
  fi
  sleep 2
done
if squeue --me -h -o "%j" |
  grep -Eq '^(abph_|fresh_abph_models_resume$)'; then
  echo "Timed out waiting for stale ABPH jobs to leave the queue." >&2
  exit 2
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive_root="${campaign_root}/archives/runtime_contract_repair_${stamp}"
archive_created=0
for variant in "$@"; do
  for artifact_kind in runtime_batch_measurements runtime_batch_contracts; do
    source_path="${campaign_root}/${artifact_kind}/${variant}"
    [[ -e "${source_path}" ]] || continue
    [[ -d "${source_path}" && ! -L "${source_path}" ]] || {
      echo "Refusing to archive unexpected repair source: ${source_path}" >&2
      exit 2
    }

    destination_parent="${archive_root}/${artifact_kind}"
    destination_path="${destination_parent}/${variant}"
    [[ ! -e "${destination_path}" ]] || {
      echo "Repair archive destination already exists: ${destination_path}" >&2
      exit 2
    }
    mkdir -p "${destination_parent}"
    mv -- "${source_path}" "${destination_path}"
    archive_created=1
    echo "Archived stale ${artifact_kind} for ${variant}: ${destination_path}"
  done
done
if ((archive_created == 0)); then
  echo "No stale runtime measurements or contracts required archival."
fi

export ABPH_ROOT="${campaign_root}"
export ABPH_STORAGE_PROFILE=streaming_30gb_v1
export ABPH_TARGET_MODE_REPORT="${campaign_root}/audits/target_mode_selection.json"
export ABPH_RUNTIME_BATCH_WORLD_SIZE=8
export ABPH_RUNTIME_BATCH_MEASUREMENT_ROOT="${campaign_root}/runtime_batch_measurements"
export ABPH_RUNTIME_BATCH_CONTRACT_ROOT="${campaign_root}/runtime_batch_contracts"
export ABPH_RUNTIME_BATCH_PROBE_MANIFEST="${campaign_root}/submission_logs/abph_contract_repair_${stamp}.tsv"
export ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY=""
export ABPH_SBATCH_ACCOUNT ABPH_SBATCH_PARTITION ABPH_DATA_DIR
export OVERWRITE=1

bash "${PROJECT_DIR}/sbatch/submit_adaptive_binary_runtime_batch_probes_tigris.sh" "$@"

mapfile -t contract_jobs < <(
  awk -F $'\t' '$2 == "compile" {print $4}' \
    "${ABPH_RUNTIME_BATCH_PROBE_MANIFEST}"
)
if ((${#contract_jobs[@]} != $#)); then
  echo "Expected $# compiler jobs, found ${#contract_jobs[@]}." >&2
  exit 2
fi
for job_id in "${contract_jobs[@]}"; do
  [[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "Invalid contract compiler job id: ${job_id}" >&2
    exit 2
  }
done

dependency="afterok:$(IFS=:; echo "${contract_jobs[*]}")"
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

echo "adaptive_binary_contract_repair_queued:"
echo "  campaign_root: ${campaign_root}"
echo "  variants: $*"
if ((archive_created)); then
  echo "  preserved_stale_evidence: ${archive_root}"
fi
echo "  contract_jobs: ${contract_jobs[*]}"
echo "  models_resume_job: ${continuation_job}"
