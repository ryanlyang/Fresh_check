#!/usr/bin/env bash
#SBATCH --job-name=fresh_audit_samehlt
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=06:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${AUDIT_MAX_ITER:=500}"
: "${AUDIT_FEATURE_MODE:=logits_probs}"
: "${AUDIT_SEED:=1701}"
: "${ALLOW_FILE_OVERLAP:=1}"
: "${RUN_HLT5_AUDIT:=1}"
: "${REQUIRE_HLT5_AUDIT:=1}"
: "${VERIFY_HLT_CACHE_ARRAYS:=0}"

fresh_setup "$@"
fresh_require_file "${MANIFEST_PATH}"
fresh_require_file "${RECO7_FUSION_DIR}/fusion_report.json"

audit_common=(
  --manifest "${MANIFEST_PATH}"
  --hlt-cache-dir "${HLT_CACHE_DIR}"
  --feature-mode "${AUDIT_FEATURE_MODE}"
  --max-iter "${AUDIT_MAX_ITER}"
  --seed "${AUDIT_SEED}"
  --fail-on-audit-failure
)
if fresh_bool_enabled "${ALLOW_FILE_OVERLAP}"; then
  audit_common+=(--allow-file-overlap)
fi
if fresh_bool_enabled "${VERIFY_HLT_CACHE_ARRAYS}"; then
  audit_common+=(--verify-hlt-cache-arrays)
fi

cmd=(
  "${PYTHON_BIN}" "scripts/run_leakage_audits.py"
  "${audit_common[@]}"
  --fusion-dir "${RECO7_FUSION_DIR}"
  --output-dir "${RECO7_AUDIT_DIR}"
)
fresh_write_run_config "${RECO7_AUDIT_DIR}" "audit_reco7_plus_hlt" "${cmd[@]}"
fresh_run "${cmd[@]}"

if fresh_bool_enabled "${RUN_HLT5_AUDIT}"; then
  if [[ ! -f "${HLT5_FUSION_DIR}/fusion_report.json" ]]; then
    if fresh_bool_enabled "${REQUIRE_HLT5_AUDIT}"; then
      echo "HLT5 fusion report missing and REQUIRE_HLT5_AUDIT=1: ${HLT5_FUSION_DIR}/fusion_report.json" >&2
      exit 2
    fi
  else
    hlt5_cmd=(
      "${PYTHON_BIN}" "scripts/run_leakage_audits.py"
      "${audit_common[@]}"
      --fusion-dir "${HLT5_FUSION_DIR}"
      --output-dir "${HLT5_AUDIT_DIR}"
    )
    fresh_write_run_config "${HLT5_AUDIT_DIR}" "audit_hlt5_seed_control" "${hlt5_cmd[@]}"
    fresh_run "${hlt5_cmd[@]}"
  fi
fi

if ! fresh_is_dry_run; then
  fresh_assert_json_ok "${RECO7_AUDIT_DIR}/audit_report.json"
  fresh_write_audit_summary "${RECO7_AUDIT_DIR}/audit_report.json" "${RECO7_AUDIT_DIR}/audit_summary.txt"
  if [[ -f "${HLT5_AUDIT_DIR}/audit_report.json" ]]; then
    fresh_assert_json_ok "${HLT5_AUDIT_DIR}/audit_report.json"
    fresh_write_audit_summary "${HLT5_AUDIT_DIR}/audit_report.json" "${HLT5_AUDIT_DIR}/audit_summary.txt"
  fi
fi
