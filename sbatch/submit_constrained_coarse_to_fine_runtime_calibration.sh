#!/usr/bin/env bash
# Submit the complete manifest-bound C2F runtime-calibration artifact build.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_prepare_submitter
fresh_activate_env

: "${CONSTRAINED_C2F_PARENT_MANIFEST_PATH:?Set CONSTRAINED_C2F_PARENT_MANIFEST_PATH to the pilot manifest}"
: "${CONSTRAINED_C2F_CALIBRATION_ROOT:=${OUTPUT_ROOT}/constrained_c2f_runtime_calibration_$(date +%Y%m%d_%H%M%S)}"
: "${CONSTRAINED_C2F_SBATCH_ACCOUNT:=}"
: "${CONSTRAINED_C2F_SBATCH_PARTITION:=}"
: "${CONSTRAINED_C2F_CALIBRATION_CPUS:=16}"
: "${CONSTRAINED_C2F_CALIBRATION_MEM:=300G}"

fresh_require_file "${CONSTRAINED_C2F_PARENT_MANIFEST_PATH}"
if ! fresh_is_dry_run && [[ -e "${CONSTRAINED_C2F_CALIBRATION_ROOT}" ]]; then
  echo "Refusing to reuse an existing calibration root: ${CONSTRAINED_C2F_CALIBRATION_ROOT}" >&2
  exit 2
fi

export CONSTRAINED_C2F_CALIBRATION_ROOT CONSTRAINED_C2F_PARENT_MANIFEST_PATH
args=(--cpus-per-task="${CONSTRAINED_C2F_CALIBRATION_CPUS}" --mem="${CONSTRAINED_C2F_CALIBRATION_MEM}")
[[ -n "${CONSTRAINED_C2F_SBATCH_ACCOUNT}" ]] && args+=(--account="${CONSTRAINED_C2F_SBATCH_ACCOUNT}")
[[ -n "${CONSTRAINED_C2F_SBATCH_PARTITION}" ]] && args+=(--partition="${CONSTRAINED_C2F_SBATCH_PARTITION}")

if fresh_is_dry_run; then
  fresh_print_shell_command sbatch "${args[@]}" "${SCRIPT_DIR}/run_build_constrained_coarse_to_fine_calibration_artifacts.sh"
  exit 0
fi

output="$(sbatch "${args[@]}" "${SCRIPT_DIR}/run_build_constrained_coarse_to_fine_calibration_artifacts.sh")"
echo "${output}"
echo "constrained_c2f_calibration_root=${CONSTRAINED_C2F_CALIBRATION_ROOT}"
