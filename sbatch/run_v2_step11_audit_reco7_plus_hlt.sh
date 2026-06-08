#!/usr/bin/env bash
# V2 Step 11: leakage and stacker audits for Reco7 + HLT baseline.

#SBATCH --job-name=v2_step11_audit7
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
: "${VERIFY_HLT_CACHE_ARRAYS:=0}"

fresh_setup "$@"
fresh_require_file "${MANIFEST_PATH}"
fresh_require_file "${V2_STEP7_FUSION_DIR}/fusion_report.json"
fresh_refuse_existing_dir "${V2_STEP7_AUDIT_DIR}"

cmd=(
  "${PYTHON_BIN}" "scripts/run_leakage_audits.py"
  --manifest "${MANIFEST_PATH}"
  --hlt-cache-dir "${HLT_CACHE_DIR}"
  --fusion-dir "${V2_STEP7_FUSION_DIR}"
  --output-dir "${V2_STEP7_AUDIT_DIR}"
  --feature-mode "${AUDIT_FEATURE_MODE}"
  --max-iter "${AUDIT_MAX_ITER}"
  --seed "${AUDIT_SEED}"
  --fail-on-audit-failure
)
if fresh_bool_enabled "${ALLOW_FILE_OVERLAP}"; then
  cmd+=(--allow-file-overlap)
fi
if fresh_bool_enabled "${VERIFY_HLT_CACHE_ARRAYS}"; then
  cmd+=(--verify-hlt-cache-arrays)
fi

fresh_write_run_config "${V2_STEP7_AUDIT_DIR}" "v2_step11_audit_reco7_plus_hlt" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_assert_json_ok "${V2_STEP7_AUDIT_DIR}/audit_report.json"
  fresh_write_audit_summary "${V2_STEP7_AUDIT_DIR}/audit_report.json" "${V2_STEP7_AUDIT_DIR}/audit_summary.txt"
fi
