#!/usr/bin/env bash
# Render by default; actual Slurm submission requires PREDICTION_ANCHORED_EXECUTE=1.

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${PREDICTION_ANCHORED_GRAPH:?Set PREDICTION_ANCHORED_GRAPH to the reviewed immutable graph}"
: "${PREDICTION_ANCHORED_ARTIFACT_ROOT:=${PROJECT_DIR}/fresh_check_outputs/prediction_anchored_bridge}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"
fresh_prepare_submitter

args=("${PYTHON_BIN}" -u scripts/submit_prediction_anchored_bridge_graph.py
  --graph "${PREDICTION_ANCHORED_GRAPH}"
  --ledger-output "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/job_ledgers/submission.json")
if [[ "${PREDICTION_ANCHORED_EXECUTE:-0}" == "1" ]]; then
  args+=(--execute)
else
  args+=(--dry-run)
fi
if [[ "${PREDICTION_ANCHORED_INCLUDE_FINAL_TEST:-0}" == "1" ]]; then
  args+=(--include-final-test)
  [[ "${PREDICTION_ANCHORED_APPROVE_FINAL_TEST:-0}" == "1" ]] && args+=(--approve-final-test)
fi
fresh_run "${args[@]}"
