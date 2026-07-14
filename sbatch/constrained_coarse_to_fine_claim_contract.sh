#!/usr/bin/env bash
# Immutable model/fusion membership and one-shot checks for final-test claims.

set -euo pipefail

readonly C2F_FROZEN_RECON_RUN_IDS="B0 B1 B2 B3 B4 B5 B6 B7 C0 C1 C2 C3 C4 C5 C6 C5-B1 C5-B2 C5-B3 C5-no-slot Cdirect-unconstrained"
readonly C2F_FROZEN_TAGGER_RUN_IDS="A0 A1 A2 A4 D0 D1 D2 D3 D4 D5 D5-B1 D5-B2 D5-B3 D6 D7 D8 D0-seed1 D0-seed2 D1-seed1 D1-seed2 D2-seed1 D2-seed2 D3-seed1 D3-seed2 D4-seed1 D4-seed2 D5-seed1 D5-seed2 D5-B1-seed1 D5-B1-seed2 D5-B2-seed1 D5-B2-seed2 D6-seed1 D6-seed2 D7-seed1 D7-seed2 D8-seed1 D8-seed2 E0 E1 E2 E3 E4 E5 E6"
readonly C2F_FROZEN_FUSION_GROUPS="F0:mean_logits:A0,BEST_D F1:simplex_logits:A0,BEST_D F2:representation_stacker:D3,D4,D5,D6,D8 F3:simplex_logits:A0,BEST_D F4:mean_logits:BEST_D,BEST_D_SEED1,BEST_D_SEED2 F5:linear_stacker:D8,D6,BEST_D,BEST_D_SEED1,BEST_D_SEED2"
readonly C2F_FROZEN_REQUIRED_FUSION_GROUPS="F0 F1 F2 F3 F4 F5"

c2f_require_frozen_value() {
  local name="$1" observed="$2" expected="$3"
  if [[ "${observed}" != "${expected}" ]]; then
    echo "Final-claim contract forbids overriding ${name}." >&2
    return 2
  fi
}

c2f_validate_final_claim_contract() {
  local selection_report="$1"
  local prediction_dir="$2"
  local fusion_dir="$3"
  local report_dir="$4"

  [[ "${CONSTRAINED_C2F_CAMPAIGN_MODE}" == "highdata" ]] || {
    echo "final_claims requires CONSTRAINED_C2F_CAMPAIGN_MODE=highdata" >&2
    return 2
  }
  [[ "${CONSTRAINED_C2F_APPROVE_FINAL_TEST:-0}" == "1" ]] || {
    echo "Refusing final test: set CONSTRAINED_C2F_APPROVE_FINAL_TEST=1 after freezing model selection." >&2
    return 2
  }
  [[ "${CONFIRM_FINAL_TEST:-0}" == "1" ]] || {
    echo "final_claims requires CONFIRM_FINAL_TEST=1" >&2
    return 2
  }
  [[ "${SKIP_EXISTING:-0}" == "0" && "${OVERWRITE:-0}" == "0" ]] || {
    echo "Final claims require SKIP_EXISTING=0 and OVERWRITE=0." >&2
    return 2
  }

  c2f_require_frozen_value CONSTRAINED_C2F_RECON_RUN_IDS "${CONSTRAINED_C2F_RECON_RUN_IDS}" "${C2F_FROZEN_RECON_RUN_IDS}"
  c2f_require_frozen_value CONSTRAINED_C2F_TAGGER_RUN_IDS "${CONSTRAINED_C2F_TAGGER_RUN_IDS}" "${C2F_FROZEN_TAGGER_RUN_IDS}"
  c2f_require_frozen_value CONSTRAINED_C2F_PREDICT_RUN_IDS "${CONSTRAINED_C2F_PREDICT_RUN_IDS}" "${C2F_FROZEN_TAGGER_RUN_IDS}"
  c2f_require_frozen_value CONSTRAINED_C2F_REPORT_RECON_RUN_IDS "${CONSTRAINED_C2F_REPORT_RECON_RUN_IDS}" "${C2F_FROZEN_RECON_RUN_IDS}"
  c2f_require_frozen_value CONSTRAINED_C2F_REPORT_TAGGER_RUN_IDS "${CONSTRAINED_C2F_REPORT_TAGGER_RUN_IDS}" "${C2F_FROZEN_TAGGER_RUN_IDS}"
  c2f_require_frozen_value CONSTRAINED_C2F_FUSION_GROUPS "${CONSTRAINED_C2F_FUSION_GROUPS}" "${C2F_FROZEN_FUSION_GROUPS}"
  c2f_require_frozen_value CONSTRAINED_C2F_REQUIRED_FUSION_GROUPS "${CONSTRAINED_C2F_REQUIRED_FUSION_GROUPS}" "${C2F_FROZEN_REQUIRED_FUSION_GROUPS}"

  local selection_sha
  selection_sha="$("${PYTHON_BIN}" - "${selection_report}" "${C2F_FROZEN_TAGGER_RUN_IDS}" "${C2F_FROZEN_REQUIRED_FUSION_GROUPS}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_taggers = sys.argv[2].split()
expected_fusions = sys.argv[3].split()
if not path.is_file():
    raise SystemExit(f"missing frozen selection report: {path}")
raw = path.read_bytes()
report = json.loads(raw.decode("utf-8"))
if report.get("ok") is not True:
    raise SystemExit("selection report is not successful")
if report.get("contract") != "constrained_coarse_to_fine_step9_campaign_report_v1":
    raise SystemExit("selection report has the wrong contract")
if report.get("final_test_policy", {}).get("confirmed") is True:
    raise SystemExit("selection report already consumed final_test")
if report.get("required_tagger_runs") != expected_taggers:
    raise SystemExit("selection report tagger membership differs from the frozen final-claim contract")
if report.get("required_fusion_groups") != expected_fusions:
    raise SystemExit("selection report fusion membership differs from the frozen final-claim contract")
if report.get("config", {}).get("require_all_runs") is not True:
    raise SystemExit("selection report did not require the complete frozen campaign")
print(hashlib.sha256(raw).hexdigest())
PY
)" || return 2

  local run_id
  local claim_ids=()
  IFS=' ' read -r -a claim_ids <<< "${C2F_FROZEN_TAGGER_RUN_IDS} ${C2F_FROZEN_REQUIRED_FUSION_GROUPS}"
  for run_id in "${claim_ids[@]}"; do
    if [[ -e "${prediction_dir}/${run_id}/final_test_predictions.npz" \
      || -e "${prediction_dir}/${run_id}/final_test_predictions_metadata.json" \
      || -e "${prediction_dir}/${run_id}/final_test_claim_receipt.json" ]]; then
      echo "Refusing duplicate final claim; ${run_id} already has final-test artifacts." >&2
      return 2
    fi
  done
  if [[ -e "${fusion_dir}" || -e "${report_dir}" ]]; then
    echo "Refusing final claim because its immutable fusion/report output path already exists." >&2
    return 2
  fi

  printf '%s\n' "${selection_sha}"
}
