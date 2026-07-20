#!/usr/bin/env bash
# Select the Stage 1b consumer from both completed alpha diagnostics.

#SBATCH --job-name=lprf_select
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"

: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/first_stage_pilot}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_DIAGNOSTICS_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/oracle_diagnostics}"
: "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON:=${LOCAL_RESIDUAL_FIELD_ROOT}/selected_consumer.json}"
: "${LOCAL_RESIDUAL_FIELD_SELECTOR_MINIMUM_GAIN:=0.002}"
: "${LOCAL_RESIDUAL_FIELD_SELECTOR_CLOSE_TOLERANCE:=0.001}"
: "${LOCAL_RESIDUAL_FIELD_SELECTOR_DROP_TOLERANCE:=0.002}"
: "${LOCAL_RESIDUAL_FIELD_SELECTOR_STACK_BRITTLENESS:=0.003}"

OFULL_REPORT="${LOCAL_RESIDUAL_FIELD_ORACLE_DIAGNOSTICS_ROOT}/D_alpha_eval_Ofull/run_report.json"
OROBUST_REPORT="${LOCAL_RESIDUAL_FIELD_ORACLE_DIAGNOSTICS_ROOT}/D_alpha_eval_Orobust/run_report.json"
fresh_setup "$@"
fresh_require_file "${OFULL_REPORT}"
fresh_require_file "${OROBUST_REPORT}"
if [[ -e "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON}" ]] && ! fresh_bool_enabled "${OVERWRITE}"; then
  echo "selected_consumer.json already exists; set OVERWRITE=1 only after inspecting both alpha curves" >&2
  exit 2
fi
mkdir -p "$(dirname "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON}")"
cmd=(
  "${PYTHON_BIN}" -u scripts/select_local_residual_curriculum_consumer.py
  --ofull-report "${OFULL_REPORT}"
  --orobust-report "${OROBUST_REPORT}"
  --output "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON}"
  --minimum-gain "${LOCAL_RESIDUAL_FIELD_SELECTOR_MINIMUM_GAIN}"
  --close-accuracy-tolerance "${LOCAL_RESIDUAL_FIELD_SELECTOR_CLOSE_TOLERANCE}"
  --drop-tolerance "${LOCAL_RESIDUAL_FIELD_SELECTOR_DROP_TOLERANCE}"
  --stack-brittleness-tolerance "${LOCAL_RESIDUAL_FIELD_SELECTOR_STACK_BRITTLENESS}"
)
fresh_run "${cmd[@]}"
if ! fresh_is_dry_run; then fresh_require_file "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON}"; fi
