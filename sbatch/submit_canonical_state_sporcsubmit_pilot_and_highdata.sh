#!/usr/bin/env bash
# Submit both Canonical Multi-Scale Jet State campaigns on the sporcsubmit tier3 cluster.
#
# This wrapper deliberately resets the Tigris-only launch knobs and uses the
# corrected high-data split sizes that fit the available 10M JetClass rows.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"

: "${PD10_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data}"
: "${OUTPUT_ROOT:=${PROJECT_DIR}/checkpoints}"
: "${DEVICE:=cuda}"
: "${CONDA_ENV:=atlas_kd}"
: "${CONFIRM_FINAL_TEST:=1}"
: "${SKIP_EXISTING:=0}"
: "${OVERWRITE:=0}"

: "${CANONICAL_STATE_PAIR_STAMP:=$(date +%Y%m%d_%H%M%S)}"
: "${CANONICAL_STATE_PILOT_ROOT:=${OUTPUT_ROOT}/canonical_multi_scale_jet_state_hltv2_s2p5_pilot_${CANONICAL_STATE_PAIR_STAMP}}"
: "${CANONICAL_STATE_HIGHDATA_ROOT:=${OUTPUT_ROOT}/canonical_multi_scale_jet_state_hltv2_s2p5_highdata_${CANONICAL_STATE_PAIR_STAMP}}"

# Sporcsubmit/tier3 resource shape. These override any lingering Tigris exports
# such as gpu:gh200:1 or partition=tigris in the current shell.
export CANONICAL_STATE_SBATCH_PARTITION="${CANONICAL_STATE_SPORC_PARTITION:-tier3}"
export CANONICAL_STATE_GPU_GRES="${CANONICAL_STATE_SPORC_GPU_GRES:-gpu:1}"
export CANONICAL_STATE_GPU_CPUS_PER_TASK="${CANONICAL_STATE_SPORC_GPU_CPUS_PER_TASK:-8}"
export CANONICAL_STATE_GPU_MEM="${CANONICAL_STATE_SPORC_GPU_MEM:-300G}"
export CANONICAL_STATE_CPU_CPUS_PER_TASK="${CANONICAL_STATE_SPORC_CPU_CPUS_PER_TASK:-8}"
export CANONICAL_STATE_CPU_MEM="${CANONICAL_STATE_SPORC_CPU_MEM:-300G}"
export CANONICAL_STATE_PHI_CHUNK_SIZE="${CANONICAL_STATE_SPORC_PHI_CHUNK_SIZE:-32768}"
export CANONICAL_STATE_DATA_DIR="${CANONICAL_STATE_SPORC_DATA_DIR:-${PD10_DATA_DIR}}"
export CANONICAL_STATE_SUBMIT_SPLITS=1
export CANONICAL_STATE_SUBMIT_HLT_CACHE=1
export CANONICAL_STATE_SUBMIT_OFFLINE_CACHE=1
export CANONICAL_STATE_SUBMIT_AUDIT=1
export CANONICAL_STATE_SUBMIT_PHI_HLT=1
export CANONICAL_STATE_SUBMIT_PHI_OFFLINE=1
export CANONICAL_STATE_SUBMIT_SINGLE_MODELS=1
export CANONICAL_STATE_SUBMIT_FUSION=1
export CANONICAL_STATE_SUBMIT_ORACLE_DIAGNOSTICS=1
export CANONICAL_STATE_SUBMIT_REPORT=1
export CANONICAL_STATE_PREQUEUE_VALIDATE_INPUTS=1

if [[ "${CONDA_BASE:-}" == *miniforge3-aarch64* ]]; then
  unset CONDA_BASE
fi
if [[ "${LD_LIBRARY_PATH:-}" == *miniforge3-aarch64* ]]; then
  unset LD_LIBRARY_PATH
fi

export PROJECT_DIR
export PD10_DATA_DIR
export OUTPUT_ROOT
export DEVICE
export CONDA_ENV
export CONFIRM_FINAL_TEST
export SKIP_EXISTING
export OVERWRITE

echo "canonical_state_sporcsubmit_pilot_and_highdata_submission_start:"
echo "  pilot_root: ${CANONICAL_STATE_PILOT_ROOT}"
echo "  highdata_root: ${CANONICAL_STATE_HIGHDATA_ROOT}"
echo "  partition: ${CANONICAL_STATE_SBATCH_PARTITION}"
echo "  gpu_gres: ${CANONICAL_STATE_GPU_GRES}"
echo "  gpu_mem: ${CANONICAL_STATE_GPU_MEM}"
echo "  cpu_mem: ${CANONICAL_STATE_CPU_MEM}"

CANONICAL_STATE_CAMPAIGN_MODE=pilot \
CANONICAL_STATE_ROOT="${CANONICAL_STATE_PILOT_ROOT}" \
CANONICAL_STATE_MODEL_TRAIN_SIZE=500000 \
CANONICAL_STATE_MODEL_VAL_SIZE=150000 \
CANONICAL_STATE_STACK_TRAIN_SIZE=300000 \
CANONICAL_STATE_STACK_VAL_SIZE=150000 \
CANONICAL_STATE_FINAL_TEST_SIZE=150000 \
  bash "${SCRIPT_DIR}/submit_canonical_state_experiment.sh"

CANONICAL_STATE_CAMPAIGN_MODE=highdata \
CANONICAL_STATE_ROOT="${CANONICAL_STATE_HIGHDATA_ROOT}" \
CANONICAL_STATE_MODEL_TRAIN_SIZE=5000000 \
CANONICAL_STATE_MODEL_VAL_SIZE=1000000 \
CANONICAL_STATE_STACK_TRAIN_SIZE=2000000 \
CANONICAL_STATE_STACK_VAL_SIZE=1000000 \
CANONICAL_STATE_FINAL_TEST_SIZE=1000000 \
  bash "${SCRIPT_DIR}/submit_canonical_state_experiment.sh"

cat <<SUMMARY
canonical_state_sporcsubmit_pilot_and_highdata_submission_complete:
  pilot_root: ${CANONICAL_STATE_PILOT_ROOT}
  highdata_root: ${CANONICAL_STATE_HIGHDATA_ROOT}
SUMMARY
