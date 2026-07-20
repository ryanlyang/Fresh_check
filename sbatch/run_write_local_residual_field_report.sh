#!/usr/bin/env bash
# Write the final local residual-field report.

#SBATCH --job-name=lprf_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field}"
: "${LOCAL_RESIDUAL_FIELD_REPORT_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/final_report}"
: "${LOCAL_RESIDUAL_FIELD_RUN_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/taggers}"
: "${LOCAL_RESIDUAL_FIELD_RECON_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/reconstructors}"
: "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/predictions}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/fusion}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/targets}"
: "${LOCAL_RESIDUAL_FIELD_REQUIRED_TAGGER_RUN_IDS:=A0 A1 A2 B0 B1 B2 B3 B4 D0 D1 D2 D3 D4 D5 D6 E0 E1 E2 E3 E4 E5 E6 F0 F1 F2 F3 F4 F5}"
: "${LOCAL_RESIDUAL_FIELD_REQUIRED_RECON_RUN_IDS:=C0 C1 C2 C3 C4 C5 C6}"
: "${LOCAL_RESIDUAL_FIELD_REQUIRED_FUSION_GROUPS:=G0 G1 G2 G3}"
: "${LOCAL_RESIDUAL_FIELD_REQUIRE_FUSION:=1}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/curriculum}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_DIAGNOSTICS_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/oracle_diagnostics}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_DIAGNOSTICS_ROOT:=${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT}}"
: "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON:=${LOCAL_RESIDUAL_FIELD_ROOT}/selected_consumer.json}"
: "${LOCAL_RESIDUAL_FIELD_REQUIRED_CURRICULUM_RUN_IDS:=P0 P2 P4 P7a P7b Q0 Q3}"
: "${LOCAL_RESIDUAL_FIELD_REQUIRE_CURRICULUM:=0}"
: "${LOCAL_RESIDUAL_FIELD_PAIRED_CONSUMER_MODE:=0}"

fresh_setup "$@"
if [[ ! -f "scripts/write_local_residual_field_report.py" ]]; then
  echo "missing scripts/write_local_residual_field_report.py" >&2
  exit 2
fi
fresh_claim_new_dir "${LOCAL_RESIDUAL_FIELD_REPORT_DIR}"

fresh_split_words required_tagger_ids "${LOCAL_RESIDUAL_FIELD_REQUIRED_TAGGER_RUN_IDS}"
fresh_split_words required_recon_ids "${LOCAL_RESIDUAL_FIELD_REQUIRED_RECON_RUN_IDS}"
fresh_split_words required_fusion_groups "${LOCAL_RESIDUAL_FIELD_REQUIRED_FUSION_GROUPS}"
cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_local_residual_field_report.py"
  --output-dir "${LOCAL_RESIDUAL_FIELD_REPORT_DIR}"
  --tagger-root "${LOCAL_RESIDUAL_FIELD_RUN_ROOT}"
  --reconstructor-root "${LOCAL_RESIDUAL_FIELD_RECON_ROOT}"
  --prediction-dir "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR}"
  --fusion-dir "${LOCAL_RESIDUAL_FIELD_FUSION_DIR}"
  --target-cache-dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
  --required-tagger-run-ids "${required_tagger_ids[@]}"
  --required-reconstructor-run-ids "${required_recon_ids[@]}"
  --required-fusion-groups "${required_fusion_groups[@]}"
)
if fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_REQUIRE_FUSION}"; then
  cmd+=(--require-fusion)
fi
if fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_REQUIRE_CURRICULUM}"; then
  fresh_require_dir "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT}"
  fresh_require_dir "${LOCAL_RESIDUAL_FIELD_ORACLE_DIAGNOSTICS_ROOT}"
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON}"
  fresh_split_words required_curriculum_ids "${LOCAL_RESIDUAL_FIELD_REQUIRED_CURRICULUM_RUN_IDS}"
  cmd+=(
    --curriculum-root "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT}"
    --oracle-diagnostics-root "${LOCAL_RESIDUAL_FIELD_ORACLE_DIAGNOSTICS_ROOT}"
    --curriculum-diagnostics-root "${LOCAL_RESIDUAL_FIELD_CURRICULUM_DIAGNOSTICS_ROOT}"
    --selected-consumer-json "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON}"
    --required-curriculum-run-ids "${required_curriculum_ids[@]}"
    --require-curriculum
  )
fi
fresh_append_flag_if_enabled cmd --paired-consumer-mode "${LOCAL_RESIDUAL_FIELD_PAIRED_CONSUMER_MODE}"
if fresh_bool_enabled "${CONFIRM_FINAL_TEST:-0}"; then
  cmd+=(--confirm-final-test --require-final-test-provenance)
fi

fresh_write_run_config "${LOCAL_RESIDUAL_FIELD_REPORT_DIR}" "local_residual_field_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_REPORT_DIR}/summary.md"
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_REPORT_DIR}/provenance_audit.json"
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_REPORT_DIR}/tagger_metrics.csv"
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_REPORT_DIR}/reconstructor_metrics.csv"
  if fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_REQUIRE_CURRICULUM}"; then
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_REPORT_DIR}/deployable_leaderboard.csv"
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_REPORT_DIR}/consumer_selection.csv"
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_REPORT_DIR}/curriculum_student_metrics.csv"
  fi
fi
