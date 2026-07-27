#!/usr/bin/env bash
# One-command bootstrap and gated submission for a fresh 30 GB ABPH pilot.

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
export PROJECT_DIR CONDA_BASE CONDA_ENV
source "${PROJECT_DIR}/sbatch/common.sh"
fresh_prepare_submitter
fresh_activate_env

: "${ABPH_PREPARED_ROOT:?Set ABPH_PREPARED_ROOT to the prepared 22 GB pilot root}"
: "${ABPH_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data/jetclass_part1}"
: "${ABPH_SBATCH_ACCOUNT:=reu-aisocial}"
: "${ABPH_SBATCH_PARTITION:=tigris}"
: "${ABPH_BOOTSTRAP_STAMP:=$(date +%Y%m%d_%H%M%S)}"
: "${ABPH_BOOTSTRAP_CAMPAIGN_ROOT:=${PROJECT_DIR}/checkpoints/adaptive_binary_pseudooffline_hltv2_s2p5_pilot_streaming_${ABPH_BOOTSTRAP_STAMP}}"
: "${ABPH_BOOTSTRAP_EVIDENCE_ROOT:=${ABPH_PREPARED_ROOT}/audits/bootstrap_${ABPH_BOOTSTRAP_STAMP}}"
[[ "${ABPH_APPROVE_PREPARED_ROOT_PRUNE:-0}" == "1" ]] || {
  echo "Set ABPH_APPROVE_PREPARED_ROOT_PRUNE=1 to authorize removal of the prepared root's rebuildable archives, targets, and inputs after runtime acceptance." >&2
  exit 2
}
[[ "${ABPH_CONFIRM_PREPARED_ROOT_IDLE:-0}" == "1" ]] || {
  echo "Set ABPH_CONFIRM_PREPARED_ROOT_IDLE=1 only after confirming that no old job still reads the prepared root." >&2
  exit 2
}
export ABPH_DATA_DIR ABPH_SBATCH_ACCOUNT ABPH_SBATCH_PARTITION
export ABPH_APPROVE_PREPARED_ROOT_PRUNE ABPH_CONFIRM_PREPARED_ROOT_IDLE
export PYTHONNOUSERSITE=1

fresh_require_file "${ABPH_PREPARED_ROOT}/audits/actual_target_feasibility.json"
fresh_require_file "${ABPH_PREPARED_ROOT}/runs/A0_hlt_part/best_model_val.pt"
[[ ! -e "${ABPH_BOOTSTRAP_CAMPAIGN_ROOT}" ]] || {
  echo "Fresh bootstrap campaign root already exists: ${ABPH_BOOTSTRAP_CAMPAIGN_ROOT}" >&2
  exit 2
}
mkdir -p "${ABPH_BOOTSTRAP_EVIDENCE_ROOT}"
projection="${ABPH_BOOTSTRAP_EVIDENCE_ROOT}/storage_projection.json"
ddp4_measurements="${ABPH_BOOTSTRAP_EVIDENCE_ROOT}/runtime_batch_measurements_ddp4"
ddp4_contracts="${ABPH_BOOTSTRAP_EVIDENCE_ROOT}/runtime_batch_contracts_ddp4"
ddp8_measurements="${ABPH_BOOTSTRAP_EVIDENCE_ROOT}/runtime_batch_measurements_ddp8"
ddp8_contracts="${ABPH_BOOTSTRAP_EVIDENCE_ROOT}/runtime_batch_contracts_ddp8"
export ABPH_SINGLE_PATH_ACCEPTANCE_PATH="${ABPH_BOOTSTRAP_EVIDENCE_ROOT}/runtime_acceptance/single_path_acceptance.json"

fresh_run "${PYTHON_BIN}" scripts/build_adaptive_binary_bootstrap_storage_projection.py \
  --prepared-root "${ABPH_PREPARED_ROOT}" \
  --campaign-root "${ABPH_BOOTSTRAP_CAMPAIGN_ROOT}" \
  --campaign-mode pilot \
  --output "${projection}"

export ABPH_ROOT="${ABPH_PREPARED_ROOT}"
export ABPH_STORAGE_PROFILE=cache_heavy_v1
export ABPH_RUNTIME_ACCEPTANCE_ROOT="${ABPH_BOOTSTRAP_EVIDENCE_ROOT}/runtime_acceptance"

export ABPH_RUNTIME_BATCH_WORLD_SIZE=4
export ABPH_RUNTIME_BATCH_MEASUREMENT_ROOT="${ddp4_measurements}"
export ABPH_RUNTIME_BATCH_CONTRACT_ROOT="${ddp4_contracts}"
export ABPH_RUNTIME_BATCH_PROBE_MANIFEST="${ABPH_BOOTSTRAP_EVIDENCE_ROOT}/runtime_batch_probes_ddp4.tsv"
bash "${PROJECT_DIR}/sbatch/submit_adaptive_binary_runtime_batch_probes_tigris.sh" \
  B1_semantic_query_root D1_kt32_mh4_particles
mapfile -t ddp4_contract_job_ids < <(
  awk -F $'\t' '$2 == "compile" {print $4}' "${ABPH_RUNTIME_BATCH_PROBE_MANIFEST}"
)

export ABPH_RUNTIME_BATCH_WORLD_SIZE=8
export ABPH_RUNTIME_BATCH_MEASUREMENT_ROOT="${ddp8_measurements}"
export ABPH_RUNTIME_BATCH_CONTRACT_ROOT="${ddp8_contracts}"
export ABPH_RUNTIME_BATCH_PROBE_MANIFEST="${ABPH_BOOTSTRAP_EVIDENCE_ROOT}/runtime_batch_probes_ddp8.tsv"
bash "${PROJECT_DIR}/sbatch/submit_adaptive_binary_runtime_batch_probes_tigris.sh" \
  B1_semantic_query_root D1_kt32_mh4_particles
mapfile -t ddp8_contract_job_ids < <(
  awk -F $'\t' '$2 == "compile" {print $4}' "${ABPH_RUNTIME_BATCH_PROBE_MANIFEST}"
)
contract_job_ids=("${ddp4_contract_job_ids[@]}" "${ddp8_contract_job_ids[@]}")
[[ "${#contract_job_ids[@]}" -eq 4 ]] || {
  echo "Expected exactly four DDP4/DDP8 runtime batch contract jobs" >&2
  exit 2
}
export ABPH_RUNTIME_BATCH_DEPENDENCY="afterok:$(IFS=:; echo "${contract_job_ids[*]}")"
export ABPH_RUNTIME_BATCH_CONTRACT_ROOT="${ddp4_contracts}"
export ABPH_RUNTIME_BATCH_CONTRACT_ROOT_DDP8="${ddp8_contracts}"

bash "${PROJECT_DIR}/sbatch/submit_adaptive_binary_runtime_acceptance_tigris.sh"
if fresh_is_dry_run; then
  fresh_print_shell_command sbatch --parsable --account="${ABPH_SBATCH_ACCOUNT}" \
    --partition="${ABPH_SBATCH_PARTITION}" --dependency=afterok:DRYRUN_RUNTIME_GATE \
    "${PROJECT_DIR}/sbatch/run_prune_adaptive_binary_prepared_root.sh"
  fresh_print_shell_command sbatch --parsable --account="${ABPH_SBATCH_ACCOUNT}" \
    --partition="${ABPH_SBATCH_PARTITION}" --dependency=afterok:DRYRUN_PRUNE \
    "${PROJECT_DIR}/sbatch/run_submit_adaptive_binary_streaming_campaign.sh"
  exit 0
fi
acceptance_manifest="${ABPH_RUNTIME_ACCEPTANCE_ROOT}/submission.tsv"
report_job_id="$(awk -F $'\t' '$1 == "report" {print $4}' "${acceptance_manifest}")"
[[ "${report_job_id}" =~ ^[0-9]+$ ]] || {
  echo "Unable to recover runtime acceptance report job id" >&2
  exit 2
}

runtime_acceptance="${ABPH_RUNTIME_ACCEPTANCE_ROOT}/runtime_acceptance.json"
export ABPH_BOOTSTRAP_CAMPAIGN_ROOT
export ABPH_BOOTSTRAP_STORAGE_PROJECTION="${projection}"
export ABPH_BOOTSTRAP_RUNTIME_ACCEPTANCE="${runtime_acceptance}"
export ABPH_BOOTSTRAP_RUNTIME_SCOPE=ddp8_runtime
export ABPH_BOOTSTRAP_EVIDENCE_ROOT ABPH_PREPARED_ROOT
continuation="${PROJECT_DIR}/sbatch/run_submit_adaptive_binary_streaming_campaign.sh"
prune_submitted="$(sbatch --parsable --account="${ABPH_SBATCH_ACCOUNT}" \
  --partition="${ABPH_SBATCH_PARTITION}" --dependency="afterok:${report_job_id}" \
  "${PROJECT_DIR}/sbatch/run_prune_adaptive_binary_prepared_root.sh")"
prune_job_id="${prune_submitted%%;*}"
continuation_submitted="$(sbatch --parsable --account="${ABPH_SBATCH_ACCOUNT}" \
  --partition="${ABPH_SBATCH_PARTITION}" --dependency="afterok:${prune_job_id}" \
  "${continuation}")"

echo "adaptive_binary_streaming_bootstrap_submission_complete:"
echo "  prepared_root: ${ABPH_PREPARED_ROOT}"
echo "  fresh_campaign_root: ${ABPH_BOOTSTRAP_CAMPAIGN_ROOT}"
echo "  storage_projection: ${projection}"
echo "  runtime_acceptance: ${runtime_acceptance}"
echo "  contract_jobs: ${contract_job_ids[*]}"
echo "  runtime_report_job: ${report_job_id}"
echo "  prepared_root_prune_job: ${prune_job_id}"
echo "  gated_campaign_submit_job: ${continuation_submitted%%;*}"
