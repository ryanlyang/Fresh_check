#!/usr/bin/env bash
# Rerun E7/F0 with four-rank RAM streaming and compile their promotion gate.
set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${ABPH_ROOT:?Set ABPH_ROOT to a completed single-rank streaming campaign}"
: "${ABPH_SBATCH_ACCOUNT:=reu-aisocial}"
: "${ABPH_SBATCH_PARTITION:=tigris}"
: "${ABPH_TAGGER_DDP_EVIDENCE_ROOT:=${ABPH_ROOT}/tagger_ddp_acceptance/ddp4}"
: "${ABPH_TAGGER_DDP_ACCEPTANCE_PATH:=${ABPH_ROOT}/tagger_ddp_acceptance/tagger_ddp_acceptance.json}"
: "${DRY_RUN:=0}"

for variant in E7_dual_hierarchy_dualcross F0_ce_reco_primary; do
  [[ -f "${ABPH_ROOT}/runs/${variant}/run_report.json" ]] || {
    echo "missing single-rank evidence ${ABPH_ROOT}/runs/${variant}/run_report.json" >&2
    exit 2
  }
done

submit_ddp() {
  local variant="$1"
  local output_dir="${ABPH_TAGGER_DDP_EVIDENCE_ROOT}/runs/${variant}"
  local command=(sbatch --parsable
    --account="${ABPH_SBATCH_ACCOUNT}"
    --partition="${ABPH_SBATCH_PARTITION}"
    --nodes=4 --ntasks=4 --ntasks-per-node=1
    --gres=gpu:1 --cpus-per-task=16 --mem=300G
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},ABPH_ROOT=${ABPH_ROOT},ABPH_STORAGE_PROFILE=streaming_30gb_v1,ABPH_TAGGER_PARALLELISM=ddp4,ABPH_TAGGER_DISTRIBUTED_WORLD_SIZE=4,ABPH_JOB_LAUNCHER=srun,ABPH_DISTRIBUTED_NODES=4,ABPH_DISTRIBUTED_NTASKS=4,ABPH_DISTRIBUTED_NTASKS_PER_NODE=1,ABPH_DISTRIBUTED_GPUS_PER_NODE=1,ABPH_DISTRIBUTED_WORLD_SIZE=4,PYTHONNOUSERSITE=1"
    "${PROJECT_DIR}/sbatch/run_adaptive_binary_variant.sh"
    "${variant}" 1 "${output_dir}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:' >&2
    printf ' %q' "${command[@]}" >&2
    printf '\n' >&2
    printf 'DRY_%s\n' "${variant}"
  else
    "${command[@]}"
  fi
}

e7_job="$(submit_ddp E7_dual_hierarchy_dualcross)"
f0_job="$(submit_ddp F0_ce_reco_primary)"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN: compile gate after ${e7_job},${f0_job}"
else
  sbatch --parsable \
    --account="${ABPH_SBATCH_ACCOUNT}" \
    --partition="${ABPH_SBATCH_PARTITION}" \
    --dependency="afterok:${e7_job}:${f0_job}" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},ABPH_ROOT=${ABPH_ROOT},ABPH_TAGGER_DDP_EVIDENCE_ROOT=${ABPH_TAGGER_DDP_EVIDENCE_ROOT},ABPH_TAGGER_DDP_ACCEPTANCE_PATH=${ABPH_TAGGER_DDP_ACCEPTANCE_PATH},PYTHONNOUSERSITE=1" \
    "${PROJECT_DIR}/sbatch/run_compile_adaptive_binary_tagger_ddp_acceptance.sh"
fi
