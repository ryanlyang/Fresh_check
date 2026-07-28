#!/usr/bin/env bash
# Replace a stale ABPH dependency graph while retaining prepared campaign data.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${ABPH_ROOT:?Set ABPH_ROOT to the existing streaming campaign root}"
: "${ABPH_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data/jetclass_part1}"
: "${ABPH_SBATCH_ACCOUNT:=reu-aisocial}"
: "${ABPH_SBATCH_PARTITION:=tigris}"

[[ "${ABPH_CONFIRM_CLEAN_MODEL_RECOVERY:-0}" == "1" ]] || {
  echo "Set ABPH_CONFIRM_CLEAN_MODEL_RECOVERY=1 to cancel the user's active abph_* jobs." >&2
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

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive="${campaign_root}/archives/clean_model_recovery_${stamp}"
mkdir -p "${archive}"
for name in runtime_batch_measurements runtime_batch_contracts; do
  source_path="${campaign_root}/${name}"
  if [[ -e "${source_path}" ]]; then
    mv "${source_path}" "${archive}/${name}"
  fi
done

export ABPH_ROOT="${campaign_root}"
export ABPH_CAMPAIGN_MODE=pilot
export ABPH_STORAGE_PROFILE=streaming_30gb_v1
export ABPH_STORAGE_PROJECTION_PATH="${storage_projection}"
export ABPH_RAM_STAGE_RESERVATION_BYTES="$(
  "${PYTHON_BIN}" -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["projected_peak_persistent_bytes"]))' \
    "${storage_projection}"
)"
export OVERWRITE=1

target_metadata="${campaign_root}/targets/model_train_exclusive_kt_adaptive_binary_targets_metadata.json"
export ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY=""
if [[ ! -f "${target_metadata}" ]]; then
  for required_dir in \
    "${campaign_root}/inputs/hlt_cache" \
    "${campaign_root}/inputs/offline_cache"; do
    [[ -d "${required_dir}" ]] || {
      echo "Cannot rebuild targets; prepared input cache is absent: ${required_dir}" >&2
      exit 2
    }
  done

  target_cache_submitted="$(
    sbatch --parsable \
      --account="${ABPH_SBATCH_ACCOUNT}" \
      --partition="${ABPH_SBATCH_PARTITION}" \
      --job-name=abph_clean_targets \
      --output="${PROJECT_DIR}/fresh_check_logs/abph_clean_targets_%j.out" \
      --error="${PROJECT_DIR}/fresh_check_logs/abph_clean_targets_%j.err" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=16 \
      --mem=220G \
      --time=2-00:00:00 \
      "${PROJECT_DIR}/sbatch/run_adaptive_binary_targets.sh" cache
  )"
  target_cache_job="${target_cache_submitted%%;*}"
  [[ "${target_cache_job}" =~ ^[0-9]+$ ]] || {
    echo "Invalid target-cache job response: ${target_cache_submitted}" >&2
    exit 2
  }

  target_preflight_submitted="$(
    sbatch --parsable \
      --account="${ABPH_SBATCH_ACCOUNT}" \
      --partition="${ABPH_SBATCH_PARTITION}" \
      --job-name=abph_clean_target_preflight \
      --output="${PROJECT_DIR}/fresh_check_logs/abph_clean_target_preflight_%j.out" \
      --error="${PROJECT_DIR}/fresh_check_logs/abph_clean_target_preflight_%j.err" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=16 \
      --mem=220G \
      --time=1-00:00:00 \
      --dependency="afterok:${target_cache_job}" \
      "${PROJECT_DIR}/sbatch/run_adaptive_binary_targets.sh" preflight
  )"
  target_preflight_job="${target_preflight_submitted%%;*}"
  [[ "${target_preflight_job}" =~ ^[0-9]+$ ]] || {
    echo "Invalid target-preflight response: ${target_preflight_submitted}" >&2
    exit 2
  }
  export ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY="afterok:${target_preflight_job}"
  echo "Replacement target jobs: cache=${target_cache_job} preflight=${target_preflight_job}"
fi

export ABPH_RUNTIME_BATCH_WORLD_SIZE=8
export ABPH_RUNTIME_BATCH_MEASUREMENT_ROOT="${campaign_root}/runtime_batch_measurements"
export ABPH_RUNTIME_BATCH_CONTRACT_ROOT="${campaign_root}/runtime_batch_contracts"
export ABPH_RUNTIME_BATCH_PROBE_MANIFEST="${campaign_root}/submission_logs/abph_clean_model_recovery_contracts_${stamp}.tsv"
export ABPH_SBATCH_ACCOUNT ABPH_SBATCH_PARTITION ABPH_DATA_DIR

bash "${PROJECT_DIR}/sbatch/submit_adaptive_binary_runtime_batch_probes_tigris.sh"

mapfile -t contract_jobs < <(
  awk -F $'\t' '$2 == "compile" {print $4}' \
    "${ABPH_RUNTIME_BATCH_PROBE_MANIFEST}"
)
if ((${#contract_jobs[@]} == 0)); then
  echo "No replacement runtime-contract jobs were submitted." >&2
  exit 2
fi
for job_id in "${contract_jobs[@]}"; do
  [[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "Invalid runtime-contract job id: ${job_id}" >&2
    exit 2
  }
done

export ABPH_STAGE_MODE=models
export ABPH_RECONSTRUCTOR_PARALLELISM=ddp8
export ABPH_RECONSTRUCTOR_SCHEDULE_POLICY=accelerated_screening_v2_7day
export ABPH_RUNTIME_ACCEPTANCE_PATH="${runtime_acceptance}"
export ABPH_TAGGER_DDP_ACCEPTANCE_PATH="${tagger_acceptance}"
export CONFIRM_FINAL_TEST=0

dependency="afterok:$(IFS=:; echo "${contract_jobs[*]}")"
controller="$(
  sbatch --parsable \
    --account="${ABPH_SBATCH_ACCOUNT}" \
    --partition="${ABPH_SBATCH_PARTITION}" \
    --job-name=abph_clean_models_submit \
    --output="${PROJECT_DIR}/fresh_check_logs/abph_clean_models_submit_%j.out" \
    --error="${PROJECT_DIR}/fresh_check_logs/abph_clean_models_submit_%j.err" \
    --time=00:30:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --dependency="${dependency}" \
    "${PROJECT_DIR}/sbatch/submit_adaptive_binary_pseudooffline_streaming30gb_tigris.sh"
)"
controller_job="${controller%%;*}"
[[ "${controller_job}" =~ ^[0-9]+$ ]] || {
  echo "Invalid clean-model controller response: ${controller}" >&2
  exit 2
}

echo "adaptive_binary_clean_model_recovery_queued:"
echo "  campaign_root: ${campaign_root}"
echo "  archived_runtime_evidence: ${archive}"
echo "  runtime_probe_upstream: ${ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY:-already-prepared}"
echo "  runtime_contract_jobs: ${contract_jobs[*]}"
echo "  clean_models_submit_job: ${controller_job}"
echo "  retained: inputs, baselines, storage acceptance"
