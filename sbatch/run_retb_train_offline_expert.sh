#!/usr/bin/env bash
#SBATCH --job-name=retb_offline_expert
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_RUN_ID:?RETB_RUN_ID is required}"
: "${RETB_TRAIN_NPZ:?RETB_TRAIN_NPZ is required}"
: "${RETB_VAL_STOP_NPZ:?RETB_VAL_STOP_NPZ is required}"
: "${RETB_MICROBATCH_SIZE:=64}"
: "${RETB_GRADIENT_ACCUMULATION_STEPS:=2}"
: "${RETB_DEVICE:=auto}"
: "${RETB_DRY_RUN:=0}"

conda_hook="${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ ! -f "${conda_hook}" ]]; then
  echo "Conda activation hook is absent: ${conda_hook}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${conda_hook}"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

arguments=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --run-id "${RETB_RUN_ID}"
  --train-npz "${RETB_TRAIN_NPZ}"
  --val-stop-npz "${RETB_VAL_STOP_NPZ}"
  --microbatch-size "${RETB_MICROBATCH_SIZE}"
  --gradient-accumulation-steps "${RETB_GRADIENT_ACCUMULATION_STEPS}"
  --device "${RETB_DEVICE}"
)

optional_path() {
  local environment_name="$1"
  local argument_name="$2"
  local value="${!environment_name:-}"
  if [[ -n "${value}" ]]; then
    arguments+=("${argument_name}" "${value}")
  fi
}

optional_path RETB_RELATION_NORMALIZATION --relation-normalization
optional_path RETB_REGION_NORMALIZATION --region-normalization
optional_path RETB_REGION_TREE_ROOT --region-tree-root
optional_path RETB_INITIALIZATION_CHECKPOINT --initialization-checkpoint
optional_path RETB_ATTACHMENT_PRETRAINING_RECORD --attachment-pretraining-record
optional_path RETB_RESOURCE_PROFILE --resource-profile
optional_path RETB_TEACHER_LOGITS_MANIFEST --teacher-logits-manifest
optional_path RETB_OUTPUT_DIR --output-dir

for teacher_name in O_BASE O_FULLREL SELECTED_STRONGEST; do
  variable_name="RETB_TEACHER_${teacher_name}"
  teacher_path="${!variable_name:-}"
  if [[ -n "${teacher_path}" ]]; then
    arguments+=(--teacher-checkpoint "${teacher_name}=${teacher_path}")
  fi
done

if [[ "${RETB_DRY_RUN}" == "1" ]]; then
  arguments+=(--dry-run)
elif [[ "${RETB_DRY_RUN}" != "0" ]]; then
  echo "RETB_DRY_RUN must be 0 or 1" >&2
  exit 2
fi

python scripts/train_retb_offline_expert.py "${arguments[@]}"
