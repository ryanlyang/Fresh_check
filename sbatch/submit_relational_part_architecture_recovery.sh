#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${PROJECT_DIR:=$(pwd -P)}"
: "${CAMPAIGN_ID:=$(basename "${CAMPAIGN_ROOT}")}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${RPT_CONFIRMATION_CONCURRENCY:=4}"

export CAMPAIGN_ROOT CAMPAIGN_ID PROJECT_DIR CONDA_BASE CONDA_ENV
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

python scripts/prepare_relational_part_architecture_recovery.py \
  --campaign-root "${CAMPAIGN_ROOT}"

recovery_root="${CAMPAIGN_ROOT}/selection/architecture_recovery_v1"
tasks="${recovery_root}/architecture_tasks.json"
authorization="${recovery_root}/source_recovery_authorization.json"
export RPT_SOURCE_RECOVERY_AUTHORIZATION="${authorization}"

# shellcheck source=relational_part_common.sh
source "${PROJECT_DIR}/sbatch/relational_part_common.sh"
rpt_setup

count="$(rpt_field "${tasks}" task_count)"
if [[ "${count}" != "12" ]]; then
  echo "Architecture recovery must contain exactly 12 tasks; found=${count}" >&2
  exit 2
fi
last="$((count - 1))"

train_job="$(rpt_submit_dynamic_once architecture_recovery_training_v1 \
  "recovery_of:failed_architecture_tasks_only" \
  --account="${SBATCH_ACCOUNT}" \
  --partition="${SBATCH_PARTITION}" \
  --gres="${GPU_GRES}" \
  --cpus-per-task="${GPU_CPUS_PER_TASK}" \
  --mem="${GPU_MEM}" \
  --time=2-00:00:00 \
  --output="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.out" \
  --error="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.err" \
  --array="0-${last}%${RPT_CONFIRMATION_CONCURRENCY}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},CAMPAIGN_ROOT=${CAMPAIGN_ROOT},RPT_SOURCE_RECOVERY_AUTHORIZATION=${authorization},RPT_TRAIN_MODE=confirmation,RPT_TASK_REGISTRY=${tasks}" \
  "${PROJECT_DIR}/sbatch/run_train_relational_part.sh")"

summary_job="$(rpt_submit_dynamic_once confirmation_summary_recovery_v1 \
  "afterok:${train_job}" \
  --account="${SBATCH_ACCOUNT}" \
  --partition="${SBATCH_PARTITION}" \
  --cpus-per-task="${CPU_CPUS_PER_TASK}" \
  --mem="${CPU_MEM}" \
  --time=02:00:00 \
  --output="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.out" \
  --error="${CAMPAIGN_ROOT}/job_ledgers/slurm/%x_%A_%a.err" \
  --dependency="afterok:${train_job}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},CAMPAIGN_ROOT=${CAMPAIGN_ROOT},RPT_SOURCE_RECOVERY_AUTHORIZATION=${authorization},RPT_AGGREGATE_MODE=summary" \
  "${PROJECT_DIR}/sbatch/run_aggregate_relational_part_confirmation.sh")"

printf 'architecture recovery training: %s\nconfirmation continuation: %s\n' \
  "${train_job}" "${summary_job}"
