#!/usr/bin/env bash
# Submit a tiny same-HLT smoke test for pipeline correctness only.

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

export MODEL_TRAIN_SIZE="${MODEL_TRAIN_SIZE:-10000}"
export MODEL_VAL_SIZE="${MODEL_VAL_SIZE:-3000}"
export STACK_TRAIN_SIZE="${STACK_TRAIN_SIZE:-5000}"
export STACK_VAL_SIZE="${STACK_VAL_SIZE:-2000}"
export FINAL_TEST_SIZE="${FINAL_TEST_SIZE:-10000}"
export OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-${OUTPUT_ROOT}/jetclass_fresh_smoke}"
export MANIFEST_PATH="${OUTPUT_ROOT}/splits/split_manifest.json.gz"
export HLT_CACHE_DIR="${OUTPUT_ROOT}/hlt_cache"
export HLT_BASELINE_DIR="${OUTPUT_ROOT}/hlt_baseline_seed101"
export RECO7_ROOT="${OUTPUT_ROOT}/reco7"
export RECO7_FUSION_DIR="${OUTPUT_ROOT}/fusion/reco7_plus_hlt_m2_base_only"
export RECO7_AUDIT_DIR="${OUTPUT_ROOT}/audits/reco7_plus_hlt_m2_base_only"
export FINAL_REPORT_DIR="${OUTPUT_ROOT}/final_report_draft"
export RECO7_VARIANTS="m2_base"
export MODEL_SIZE="${MODEL_SIZE:-tiny}"
export EPOCHS="${EPOCHS:-1}"
export MAX_TRAIN_JETS="${MAX_TRAIN_JETS:-1000}"
export MAX_VAL_JETS="${MAX_VAL_JETS:-500}"
export FUSION_MAX_JETS_PER_SPLIT="${FUSION_MAX_JETS_PER_SPLIT:-500}"
export RUN_HLT5_AUDIT=0
export REQUIRE_HLT5_AUDIT=0
export ALLOW_MISSING_FINAL_INPUTS=1

fresh_prepare_submitter

submit_count=0
submit_job() {
  local label="$1"
  shift
  submit_count=$((submit_count + 1))
  if fresh_is_dry_run; then
    printf 'DRY_RUN sbatch %s: ' "${label}" >&2
    fresh_print_shell_command sbatch "$@" >&2
    printf '\n' >&2
    local clean_label="${label//[^A-Za-z0-9_]/_}"
    printf 'DRYRUN_%s\n' "${clean_label}"
    return 0
  fi
  local output
  output="$(sbatch "$@")"
  echo "${output}" >&2
  echo "${output}" | awk '{print $NF}'
}

fresh_refuse_existing_dir "${OUTPUT_ROOT}"

split_jid="$(submit_job "smoke_splits" --partition=debug --time=04:00:00 --mem=32G "${SCRIPT_DIR}/run_build_fresh_splits.sh")"
cache_jid="$(submit_job "smoke_hlt_cache" --partition=debug --time=12:00:00 --mem=64G --dependency="afterok:${split_jid}" "${SCRIPT_DIR}/run_build_fresh_hlt_cache.sh")"
baseline_jid="$(submit_job "smoke_hlt_baseline" --partition=debug --time=12:00:00 --mem=64G --dependency="afterok:${cache_jid}" "${SCRIPT_DIR}/run_train_fresh_hlt_baseline.sh")"
reco_jid="$(submit_job "smoke_reco_m2_base" --partition=debug --time=12:00:00 --mem=64G --dependency="afterok:${baseline_jid}" "${SCRIPT_DIR}/run_train_fresh_reco7_variant.sh" "m2_base")"
fusion_dep="$(fresh_join_by_colon "${baseline_jid}" "${reco_jid}")"
fusion_jid="$(submit_job "smoke_fusion" --partition=debug --time=12:00:00 --mem=64G --dependency="afterok:${fusion_dep}" "${SCRIPT_DIR}/run_fuse_fresh_samehlt7_plus_hlt.sh")"
audit_jid="$(submit_job "smoke_audit" --partition=debug --time=12:00:00 --mem=64G --dependency="afterok:${fusion_jid}" "${SCRIPT_DIR}/run_audit_fresh_samehlt7_plus_hlt.sh")"
report_jid="$(submit_job "smoke_report" --partition=debug --time=04:00:00 --mem=32G --dependency="afterok:${audit_jid}" "${SCRIPT_DIR}/run_write_fresh_final_report.sh")"

cat <<SUMMARY
fresh_smoke_test_submission:
  warning: smoke metrics are for pipeline correctness only, not physics interpretation
  split_job_id: ${split_jid}
  hlt_cache_job_id: ${cache_jid}
  hlt_baseline_job_id: ${baseline_jid}
  reco_m2_base_job_id: ${reco_jid}
  fusion_job_id: ${fusion_jid}
  audit_job_id: ${audit_jid}
  report_job_id: ${report_jid}
  output_root: ${OUTPUT_ROOT}
  logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
