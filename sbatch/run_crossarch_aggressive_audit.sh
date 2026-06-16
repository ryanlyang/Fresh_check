#!/usr/bin/env bash
# Audit aggressive crossarch reconstructor diagnostics, split use, and fusion membership.

#SBATCH --job-name=crossarch_aggr_audit
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_setup "$@"
fresh_require_file "scripts/audit_crossarch_aggressive_experiment.py"
fresh_require_file "${CROSSARCH_AGGRESSIVE_FUSION_DIR}/fusion_report.json"

fresh_split_words reco_args "${CROSSARCH_AGGRESSIVE_RECO_ARCHITECTURES}"
fresh_split_words teacher_args "${CROSSARCH_AGGRESSIVE_RECO_TEACHERS}"
fresh_split_words split_args "${CROSSARCH_AGGRESSIVE_RECO_PREDICT_SPLITS}"
fresh_split_words fusion_group_args "${CROSSARCH_AGGRESSIVE_AUDIT_GROUPS}"

fresh_claim_new_dir "${CROSSARCH_AGGRESSIVE_AUDIT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/audit_crossarch_aggressive_experiment.py"
  --prediction-dir "${CROSSARCH_AGGRESSIVE_PREDICTION_DIR}"
  --reco-model-dir "${CROSSARCH_AGGRESSIVE_RECO_MODEL_DIR}"
  --adapted-tagger-dir "${CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_DIR}"
  --output-dir "${CROSSARCH_AGGRESSIVE_AUDIT_DIR}"
  --fusion-report "${CROSSARCH_AGGRESSIVE_FUSION_DIR}/fusion_report.json"
  --reconstructors "${reco_args[@]}"
  --teachers "${teacher_args[@]}"
  --splits "${split_args[@]}"
  --fusion-groups "${fusion_group_args[@]}"
  --stack-train-size "${CROSSARCH_STACK_TRAIN_SIZE}"
  --stack-val-size "${CROSSARCH_STACK_VAL_SIZE}"
  --final-test-size "${CROSSARCH_FINAL_TEST_SIZE}"
)
fresh_append_flag_if_enabled cmd --require-ok "${CROSSARCH_AGGRESSIVE_AUDIT_REQUIRE_OK}"
fresh_append_flag_if_enabled cmd --check-prediction-arrays "${CROSSARCH_AGGRESSIVE_AUDIT_CHECK_PREDICTION_ARRAYS}"

fresh_write_run_config "${CROSSARCH_AGGRESSIVE_AUDIT_DIR}" "crossarch_aggressive_audit" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${CROSSARCH_AGGRESSIVE_AUDIT_DIR}/aggressive_audit_report.json"
  fresh_require_file "${CROSSARCH_AGGRESSIVE_AUDIT_DIR}/aggressive_audit_summary.md"
  if fresh_bool_enabled "${CROSSARCH_AGGRESSIVE_AUDIT_REQUIRE_OK}"; then
    fresh_assert_json_ok "${CROSSARCH_AGGRESSIVE_AUDIT_DIR}/aggressive_audit_report.json"
  fi
fi
