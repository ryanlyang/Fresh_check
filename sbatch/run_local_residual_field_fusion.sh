#!/usr/bin/env bash
# Run local residual-field late-logit fusion groups.

#SBATCH --job-name=lprf_fuse
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field}"
: "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/predictions}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/fusion}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_GROUPS:=G0:A0,D5 G1:D5,D5_seed1,D5_seed2,D5_seed3 G2:D5,D6 G3:E6,E5,E3}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_SPLITS:=stack_train stack_val final_test}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_MODES:=uniform_logit_mean scalar_weighted_logit_mean}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_FIT_SPLIT:=stack_train}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_SCALAR_WEIGHT_TRIALS:=128}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CONTROL_SEED:=4079}"

fresh_setup "$@"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR}"
fresh_claim_new_dir "${LOCAL_RESIDUAL_FIELD_FUSION_DIR}"
fresh_split_words group_args "${LOCAL_RESIDUAL_FIELD_FUSION_GROUPS}"
fresh_split_words split_args "${LOCAL_RESIDUAL_FIELD_FUSION_SPLITS}"
fresh_split_words mode_args "${LOCAL_RESIDUAL_FIELD_FUSION_MODES}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_local_residual_field_fusion.py"
  --prediction-dir "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR}"
  --output-dir "${LOCAL_RESIDUAL_FIELD_FUSION_DIR}"
  --splits "${split_args[@]}"
  --fusion-modes "${mode_args[@]}"
  --fit-split "${LOCAL_RESIDUAL_FIELD_FUSION_FIT_SPLIT}"
  --scalar-weight-trials "${LOCAL_RESIDUAL_FIELD_FUSION_SCALAR_WEIGHT_TRIALS}"
  --control-seed "${LOCAL_RESIDUAL_FIELD_FUSION_CONTROL_SEED}"
)
for group in "${group_args[@]}"; do
  cmd+=(--group "${group}")
done
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"

fresh_write_run_config "${LOCAL_RESIDUAL_FIELD_FUSION_DIR}" "local_residual_field_fusion" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_FUSION_DIR}/fusion_report.json"
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_FUSION_DIR}/fusion_metrics.csv"
fi
