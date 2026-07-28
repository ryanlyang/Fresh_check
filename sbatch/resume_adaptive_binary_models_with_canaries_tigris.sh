#!/usr/bin/env bash
# Replace a broken ABPH model graph only after real DDP8 and oracle launch canaries.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${ABPH_ROOT:?Set ABPH_ROOT to the prepared streaming campaign root}"
: "${ABPH_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data/jetclass_part1}"
: "${ABPH_SBATCH_ACCOUNT:=reu-aisocial}"
: "${ABPH_SBATCH_PARTITION:=tigris}"

[[ "${ABPH_CONFIRM_CANARY_MODELS_RESUME:-0}" == "1" ]] || {
  echo "Set ABPH_CONFIRM_CANARY_MODELS_RESUME=1 to replace active ABPH jobs." >&2
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

fresh_require_file "${campaign_root}/submission_logs/abph_full_submission.json"
fresh_require_file "${campaign_root}/runtime_batch_contracts/B1_semantic_query_root/runtime_batch_contract.json"
fresh_require_file "${campaign_root}/audits/actual_target_feasibility.json"
fresh_require_file "${campaign_root}/audits/target_mode_selection.json"
fresh_require_file "${campaign_root}/inputs/split_manifest/split_manifest.json.gz"
for split in model_train model_val; do
  fresh_require_file "${campaign_root}/inputs/hlt_cache/${split}_fixed_hlt_metadata.json"
  fresh_require_file "${campaign_root}/inputs/offline_cache/${split}_offline_metadata.json"
done

mapfile -t stale_jobs < <(
  squeue --me -h -o "%i|%j" |
    awk -F'|' '$2 ~ /^(abph_|fresh_abph_)/ {print $1}'
)
if ((${#stale_jobs[@]})); then
  echo "Cancelling broken ABPH graph: ${stale_jobs[*]}"
  scancel "${stale_jobs[@]}"
fi
for _ in $(seq 1 90); do
  if ! squeue --me -h -o "%j" | grep -Eq '^(abph_|fresh_abph_)'; then
    break
  fi
  sleep 2
done
if squeue --me -h -o "%j" | grep -Eq '^(abph_|fresh_abph_)'; then
  echo "Timed out waiting for the broken ABPH graph to leave the queue." >&2
  exit 2
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
worker="${PROJECT_DIR}/sbatch/run_adaptive_binary_variant.sh"
common_export="ALL,PROJECT_DIR=${PROJECT_DIR},CONDA_BASE=${CONDA_BASE},CONDA_ENV=${CONDA_ENV},PYTHONNOUSERSITE=1,ABPH_ROOT=${campaign_root},ABPH_DATA_DIR=${ABPH_DATA_DIR},DATA_DIR=${ABPH_DATA_DIR},ABPH_STORAGE_PROFILE=streaming_30gb_v1,ABPH_TARGET_MODE_REPORT=${campaign_root}/audits/target_mode_selection.json,ABPH_RECONSTRUCTOR_SCHEDULE_POLICY=accelerated_screening_v2_7day,ABPH_MAXIMUM_UPDATES=1,ABPH_MAX_VAL_BATCHES=1,OVERWRITE=1,DEVICE=cuda"

b1_output="/tmp/abph_models_canary_B1_${USER}_${stamp}"
b1_response="$(
  sbatch --parsable \
    --account="${ABPH_SBATCH_ACCOUNT}" \
    --partition="${ABPH_SBATCH_PARTITION}" \
    --job-name=abph_canary_ddp8_B1 \
    --output="${PROJECT_DIR}/fresh_check_logs/abph_canary_ddp8_B1_%j.out" \
    --error="${PROJECT_DIR}/fresh_check_logs/abph_canary_ddp8_B1_%j.err" \
    --nodes=8 \
    --ntasks=8 \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --mem=220G \
    --time=02:00:00 \
    --gres=gpu:gh200:1 \
    --export="${common_export},ABPH_RECONSTRUCTOR_PARALLELISM=ddp8,ABPH_JOB_LAUNCHER=srun,ABPH_DISTRIBUTED_NODES=8,ABPH_DISTRIBUTED_NTASKS=8,ABPH_DISTRIBUTED_NTASKS_PER_NODE=1,ABPH_DISTRIBUTED_GPUS_PER_NODE=1,ABPH_DISTRIBUTED_WORLD_SIZE=8" \
    --chdir="${PROJECT_DIR}" \
    "${worker}" B1_semantic_query_root 1 "${b1_output}"
)"
b1_job="${b1_response%%;*}"

b4_output="/tmp/abph_models_canary_B4_${USER}_${stamp}"
b4_response="$(
  sbatch --parsable \
    --account="${ABPH_SBATCH_ACCOUNT}" \
    --partition="${ABPH_SBATCH_PARTITION}" \
    --job-name=abph_canary_single_B4 \
    --output="${PROJECT_DIR}/fresh_check_logs/abph_canary_single_B4_%j.out" \
    --error="${PROJECT_DIR}/fresh_check_logs/abph_canary_single_B4_%j.err" \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --mem=220G \
    --time=02:00:00 \
    --gres=gpu:gh200:1 \
    --export="${common_export},ABPH_RECONSTRUCTOR_PARALLELISM=single,ABPH_JOB_LAUNCHER=direct,ABPH_DISTRIBUTED_NODES=1,ABPH_DISTRIBUTED_NTASKS=1,ABPH_DISTRIBUTED_NTASKS_PER_NODE=1,ABPH_DISTRIBUTED_GPUS_PER_NODE=1,ABPH_DISTRIBUTED_WORLD_SIZE=1" \
    --chdir="${PROJECT_DIR}" \
    "${worker}" B4_oracle_root_diagnostic 1 "${b4_output}"
)"
b4_job="${b4_response%%;*}"

[[ "${b1_job}" =~ ^[0-9]+$ && "${b4_job}" =~ ^[0-9]+$ ]] || {
  echo "Invalid canary submission responses: B1=${b1_response}; B4=${b4_response}" >&2
  exit 2
}

continuation_response="$(
  sbatch --parsable \
    --account="${ABPH_SBATCH_ACCOUNT}" \
    --partition="${ABPH_SBATCH_PARTITION}" \
    --job-name=fresh_abph_models_resume \
    --output="${PROJECT_DIR}/fresh_check_logs/fresh_abph_models_resume_%j.out" \
    --error="${PROJECT_DIR}/fresh_check_logs/fresh_abph_models_resume_%j.err" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=16G \
    --time=00:30:00 \
    --dependency="afterok:${b1_job}:${b4_job}" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CONDA_BASE=${CONDA_BASE},CONDA_ENV=${CONDA_ENV},PYTHONNOUSERSITE=1,ABPH_ROOT=${campaign_root},ABPH_DATA_DIR=${ABPH_DATA_DIR},ABPH_SBATCH_ACCOUNT=${ABPH_SBATCH_ACCOUNT},ABPH_SBATCH_PARTITION=${ABPH_SBATCH_PARTITION},ABPH_CONFIRM_MODELS_RESUME=1" \
    --chdir="${PROJECT_DIR}" \
    "${PROJECT_DIR}/sbatch/resume_adaptive_binary_models_from_contracts_tigris.sh"
)"
continuation_job="${continuation_response%%;*}"
[[ "${continuation_job}" =~ ^[0-9]+$ ]] || {
  echo "Invalid models continuation response: ${continuation_response}" >&2
  exit 2
}

echo "adaptive_binary_canary_models_resume_queued:"
echo "  ddp8_canary: ${b1_job}"
echo "  single_oracle_canary: ${b4_job}"
echo "  models_continuation: ${continuation_job}"
echo "  continuation_dependency: afterok:${b1_job}:${b4_job}"
