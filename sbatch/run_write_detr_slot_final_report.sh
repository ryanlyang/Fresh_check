#!/usr/bin/env bash
# Write DETR/free-slot audit tables and final report.

#SBATCH --job-name=detrslot_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${DETR_SLOT_REQUIRE_FIVE_VIEW_AUDIT:=1}"
: "${DETR_SLOT_REQUIRE_OFFLINE_REFERENCE:=0}"
: "${DETR_SLOT_HLT_REFERENCE_REPORT:=}"

fresh_setup "$@"
fresh_require_file "scripts/write_detr_slot_final_report.py"
fresh_split_words reco_args "${DETR_SLOT_ARCHITECTURES}"
fresh_split_words tagger_args "${DETR_SLOT_TAGGER_VARIANTS}"
for architecture in "${reco_args[@]}"; do
  fresh_require_file "${DETR_SLOT_RECONSTRUCTOR_DIR}/${architecture}/run_report.json"
  fresh_require_file "${DETR_SLOT_RECONSTRUCTED_VIEW_DIR}/${architecture}/cache_report.json"
done
for variant in "${tagger_args[@]}"; do
  fresh_require_file "${DETR_SLOT_TAGGER_ROOT}/${variant}/run_report.json"
done
if fresh_bool_enabled "${DETR_SLOT_REQUIRE_FIVE_VIEW_AUDIT}"; then
  fresh_require_file "${DETR_SLOT_AUDIT_DIR}/run_report.json"
  fresh_require_file "${DETR_SLOT_AUDIT_DIR}/summary.csv"
fi
fresh_claim_new_dir "${DETR_SLOT_FINAL_REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_detr_slot_final_report.py"
  --output-dir "${DETR_SLOT_FINAL_REPORT_DIR}"
  --experiment-dir "${DETR_SLOT_ROOT}"
  --reconstructor-dir "${DETR_SLOT_RECONSTRUCTOR_DIR}"
  --reconstructed-view-dir "${DETR_SLOT_RECONSTRUCTED_VIEW_DIR}"
  --tagger-root "${DETR_SLOT_TAGGER_ROOT}"
  --architectures "${reco_args[@]}"
  --tagger-variants "${tagger_args[@]}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${DETR_SLOT_CONFIRM_FINAL_TEST}"
fresh_append_optional_arg cmd --hlt-reference-report "${DETR_SLOT_HLT_REFERENCE_REPORT}"
if [[ -d "${DETR_SLOT_AUDIT_DIR}" ]] || fresh_bool_enabled "${DETR_SLOT_REQUIRE_FIVE_VIEW_AUDIT}"; then
  cmd+=(--five-view-audit-dir "${DETR_SLOT_AUDIT_DIR}")
fi
if [[ -d "${DETR_SLOT_OFFLINE_REFERENCE_DIR}" ]] || fresh_bool_enabled "${DETR_SLOT_REQUIRE_OFFLINE_REFERENCE}"; then
  cmd+=(--offline-reference-dir "${DETR_SLOT_OFFLINE_REFERENCE_DIR}")
fi

fresh_write_run_config "${DETR_SLOT_FINAL_REPORT_DIR}" "detr_slot_final_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${DETR_SLOT_FINAL_REPORT_DIR}/detr_slot_final_report.json"
  fresh_require_file "${DETR_SLOT_FINAL_REPORT_DIR}/detr_slot_final_report.md"
  fresh_require_file "${DETR_SLOT_FINAL_REPORT_DIR}/reconstructor_summary.csv"
  fresh_require_file "${DETR_SLOT_FINAL_REPORT_DIR}/cache_export_summary.csv"
  fresh_require_file "${DETR_SLOT_FINAL_REPORT_DIR}/tagger_summary.csv"
  fresh_require_file "${DETR_SLOT_FINAL_REPORT_DIR}/binary_operating_points.csv"
fi
