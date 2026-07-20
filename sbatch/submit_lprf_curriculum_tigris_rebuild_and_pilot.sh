#!/usr/bin/env bash
# Rebuild pilot caches/A0/C0, then queue the full curriculum first stage afterok.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${PD10_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data}"
: "${OUTPUT_ROOT:=${PROJECT_DIR}/checkpoints}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${PYTHONNOUSERSITE:=1}"
: "${DEVICE:=cuda}"
: "${LOCAL_RESIDUAL_FIELD_SBATCH_ACCOUNT:=reu-aisocial}"
: "${LOCAL_RESIDUAL_FIELD_SBATCH_PARTITION:=tigris}"
: "${LOCAL_RESIDUAL_FIELD_GPU_GRES:=gpu:gh200:1}"
: "${LOCAL_RESIDUAL_FIELD_GPU_CPUS_PER_TASK:=16}"
: "${LOCAL_RESIDUAL_FIELD_GPU_MEM:=500G}"
: "${LOCAL_RESIDUAL_FIELD_CPU_CPUS_PER_TASK:=16}"
: "${LOCAL_RESIDUAL_FIELD_CPU_MEM:=500G}"
: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/rebuild_and_pilot_$(date +%Y%m%d_%H%M%S)}"

export PROJECT_DIR PD10_DATA_DIR OUTPUT_ROOT CONDA_BASE CONDA_ENV PYTHONNOUSERSITE DEVICE
export LOCAL_RESIDUAL_FIELD_SBATCH_ACCOUNT LOCAL_RESIDUAL_FIELD_SBATCH_PARTITION
export LOCAL_RESIDUAL_FIELD_GPU_GRES LOCAL_RESIDUAL_FIELD_GPU_CPUS_PER_TASK LOCAL_RESIDUAL_FIELD_GPU_MEM
export LOCAL_RESIDUAL_FIELD_CPU_CPUS_PER_TASK LOCAL_RESIDUAL_FIELD_CPU_MEM
export LOCAL_RESIDUAL_FIELD_ROOT

export LOCAL_RESIDUAL_FIELD_DATA_DIR="${PD10_DATA_DIR}"
export LOCAL_RESIDUAL_FIELD_INPUTS_DIR="${LOCAL_RESIDUAL_FIELD_ROOT}/inputs"
export LOCAL_RESIDUAL_FIELD_SPLIT_MANIFEST_DIR="${LOCAL_RESIDUAL_FIELD_INPUTS_DIR}/split_manifest"
export LOCAL_RESIDUAL_FIELD_MANIFEST_PATH="${LOCAL_RESIDUAL_FIELD_SPLIT_MANIFEST_DIR}/split_manifest.json.gz"
export LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR="${LOCAL_RESIDUAL_FIELD_INPUTS_DIR}/hlt_cache"
export LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR="${LOCAL_RESIDUAL_FIELD_INPUTS_DIR}/offline_cache"
export LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR="${LOCAL_RESIDUAL_FIELD_ROOT}/targets"
export LOCAL_RESIDUAL_FIELD_RECON_ROOT="${LOCAL_RESIDUAL_FIELD_ROOT}/reconstructors"
export LOCAL_RESIDUAL_FIELD_TAGGER_ROOT="${LOCAL_RESIDUAL_FIELD_ROOT}/taggers"

# Minimal prerequisite rebuild: no legacy KD teacher, predictions, fusion, report,
# high-data campaign, or nonessential C/A/B/D/E/F variants.
export LOCAL_RESIDUAL_FIELD_CAMPAIGN_MODE=pilot
export LOCAL_RESIDUAL_FIELD_SUBMIT_SPLITS=1
export LOCAL_RESIDUAL_FIELD_SUBMIT_HLT_CACHE=1
export LOCAL_RESIDUAL_FIELD_SUBMIT_OFFLINE_CACHE=1
export LOCAL_RESIDUAL_FIELD_SUBMIT_TARGETS=1
export LOCAL_RESIDUAL_FIELD_SUBMIT_TEACHER_LOGITS=0
export LOCAL_RESIDUAL_FIELD_SUBMIT_RECONSTRUCTORS=1
export LOCAL_RESIDUAL_FIELD_RECON_RUN_IDS=C0
export LOCAL_RESIDUAL_FIELD_SUBMIT_TAGGERS=1
export LOCAL_RESIDUAL_FIELD_TAGGER_RUN_IDS=A0
export LOCAL_RESIDUAL_FIELD_SUBMIT_PREDICTIONS=0
export LOCAL_RESIDUAL_FIELD_SUBMIT_FUSION=0
export LOCAL_RESIDUAL_FIELD_SUBMIT_REPORT=0
export SKIP_EXISTING=0
export OVERWRITE=0

dependency_file="$(mktemp "${TMPDIR:-/tmp}/lprf_bootstrap_dependency.XXXXXX")"
cleanup() { rm -f -- "${dependency_file}"; }
trap cleanup EXIT
export LOCAL_RESIDUAL_FIELD_BOOTSTRAP_DEPENDENCY_FILE="${dependency_file}"

echo "Queueing minimal local residual-field rebuild"
echo "  root: ${LOCAL_RESIDUAL_FIELD_ROOT}"
bash "${PROJECT_DIR}/sbatch/submit_local_particle_residual_field_experiment.sh"

bootstrap_dependency="$(tr -d '\r\n' < "${dependency_file}")"
if [[ ! "${bootstrap_dependency}" =~ ^([0-9]+|DRYRUN_[A-Za-z0-9_]+)(:([0-9]+|DRYRUN_[A-Za-z0-9_]+))*$ ]]; then
  echo "Invalid bootstrap dependency receipt: ${bootstrap_dependency:-empty}" >&2
  exit 2
fi

export LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE=full_first_stage
export LOCAL_RESIDUAL_FIELD_CURRICULUM_MODE=first_stage_pilot
export LOCAL_RESIDUAL_FIELD_CURRICULUM_UPSTREAM_DEPENDENCY="${bootstrap_dependency}"
export LOCAL_RESIDUAL_FIELD_REUSE_SPLIT_MANIFEST=1
export LOCAL_RESIDUAL_FIELD_REUSE_HLT_CACHE=1
export LOCAL_RESIDUAL_FIELD_REUSE_OFFLINE_CACHE=0
export LOCAL_RESIDUAL_FIELD_REUSE_TARGET_CACHE=1
export LOCAL_RESIDUAL_FIELD_REUSE_OFFLINE_TEACHER_LOGITS=0
export CONFIRM_FINAL_TEST=1

echo "Queueing full_first_stage after bootstrap dependency ${bootstrap_dependency}"
bash "${PROJECT_DIR}/sbatch/submit_lprf_curriculum_tigris_pilot.sh"

echo "rebuild_and_curriculum_submission_complete:"
echo "  root: ${LOCAL_RESIDUAL_FIELD_ROOT}"
echo "  bootstrap_dependency: ${bootstrap_dependency}"
