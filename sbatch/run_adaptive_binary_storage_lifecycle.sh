#!/usr/bin/env bash
# Execute one hash-bound ABPH storage lifecycle operation.
#SBATCH --job-name=abph_storage_wave
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
: "${ABPH_ROOT:=${OUTPUT_ROOT}/adaptive_binary_pseudooffline}"
: "${ABPH_STORAGE_PROFILE:=streaming_30gb_v1}"
: "${ABPH_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data/jetclass_part1}"
: "${ABPH_STORAGE_LIFECYCLE_EXECUTOR:=${PROJECT_DIR}/scripts/manage_adaptive_binary_storage_lifecycle.py}"

action="${1:?Usage: run_adaptive_binary_storage_lifecycle.sh <manifest|consumer_receipt|cleanup_privileged|cleanup_deployable|wave_receipt> [args...]}"
shift
export PYTHONNOUSERSITE=1
fresh_setup
fresh_require_file "${ABPH_STORAGE_LIFECYCLE_EXECUTOR}"
base=("${PYTHON_BIN}" -u "${ABPH_STORAGE_LIFECYCLE_EXECUTOR}"
  --campaign-root "${ABPH_ROOT}" --storage-profile "${ABPH_STORAGE_PROFILE}")

case "${action}" in
  manifest)
    cmd=("${base[@]}" manifest --data-dir "${ABPH_DATA_DIR}")
    ;;
  consumer_receipt)
    consumer="${1:?consumer_receipt requires a consumer name}"
    report="${2:?consumer_receipt requires a run-report path}"
    kind="${3:-target_consumer}"
    cmd=("${base[@]}" consumer-receipt --consumer "${consumer}"
      --run-report "${report}" --consumer-kind "${kind}")
    ;;
  cleanup_privileged)
    cmd=("${base[@]}" cleanup --barrier privileged)
    for consumer in "$@"; do cmd+=(--expected-consumer "${consumer}"); done
    ;;
  cleanup_deployable)
    cmd=("${base[@]}" cleanup --barrier deployable)
    for member in "$@"; do cmd+=(--scoring-member "${member}"); done
    ;;
  wave_receipt)
    wave="${1:?wave_receipt requires a wave number}"
    cmd=("${base[@]}" wave-receipt --wave "${wave}")
    ;;
  *) echo "Unknown ABPH storage lifecycle action ${action}" >&2; exit 2 ;;
esac
fresh_run "${cmd[@]}"
