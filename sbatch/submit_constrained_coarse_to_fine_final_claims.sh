#!/usr/bin/env bash
# Append one immutable final-test claim to a completed high-data campaign.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONSTRAINED_C2F_ROOT:?Set CONSTRAINED_C2F_ROOT to the selected completed high-data campaign}"
: "${CONSTRAINED_C2F_APPROVE_FINAL_TEST:=0}"

if [[ "${CONSTRAINED_C2F_APPROVE_FINAL_TEST}" != "1" ]]; then
  echo "Refusing final test: set CONSTRAINED_C2F_APPROVE_FINAL_TEST=1 after freezing model selection." >&2
  exit 2
fi

selection_report="${CONSTRAINED_C2F_ROOT}/final_report/final_report.json"
python_bin="${PYTHON_BIN:-python}"
"${python_bin}" - "${selection_report}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing frozen selection report: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("ok") is not True:
    raise SystemExit("selection report is not successful")
if report.get("final_test_policy", {}).get("confirmed") is True:
    raise SystemExit("selection report already consumed final_test")
PY

: "${CONSTRAINED_C2F_TAGGER_RUN_IDS:=A0 D0 D1 D2 D3 D4 D5 D5-B1 D5-B2 D5-B3 D6 D7 D8 D0-seed1 D0-seed2 D1-seed1 D1-seed2 D2-seed1 D2-seed2 D3-seed1 D3-seed2 D4-seed1 D4-seed2 D5-seed1 D5-seed2 D5-B1-seed1 D5-B1-seed2 D5-B2-seed1 D5-B2-seed2 D6-seed1 D6-seed2 D7-seed1 D7-seed2 D8-seed1 D8-seed2 E0 E1 E2 E3 E4 E5 E6}"
: "${CONSTRAINED_C2F_PREDICT_RUN_IDS:=${CONSTRAINED_C2F_TAGGER_RUN_IDS}}"

IFS=' ' read -r -a predict_ids <<< "${CONSTRAINED_C2F_PREDICT_RUN_IDS}"
claim_ids=("${predict_ids[@]}" F0 F1 F2 F3 F4 F5)
for run_id in "${claim_ids[@]}"; do
  model_dir="${CONSTRAINED_C2F_ROOT}/predictions/${run_id}"
  if [[ -e "${model_dir}/final_test_predictions.npz" || -e "${model_dir}/final_test_claim_receipt.json" ]]; then
    echo "Refusing duplicate final claim; ${run_id} already has final-test artifacts." >&2
    exit 2
  fi
done

export CONSTRAINED_C2F_STAGE_MODE=final_claims
export CONSTRAINED_C2F_PREDICT_RUN_IDS
export CONSTRAINED_C2F_PREDICT_SPLITS=final_test
export CONSTRAINED_C2F_FUSION_DIR="${CONSTRAINED_C2F_ROOT}/fusion_final_claim"
export CONSTRAINED_C2F_REPORT_DIR="${CONSTRAINED_C2F_ROOT}/final_claim_report"
export CONFIRM_FINAL_TEST=1
export SKIP_EXISTING=0
export OVERWRITE=0

exec bash "${PROJECT_DIR}/sbatch/submit_constrained_coarse_to_fine_experiment.sh"
