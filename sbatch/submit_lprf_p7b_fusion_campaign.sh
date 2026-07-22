#!/usr/bin/env bash
# Dependency-safe, resumable submitter for the complete A0/P7b fusion campaign.
set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"

STAGE="${1:-full_campaign}"
CAMPAIGN_ID="${2:-${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ID:-p7b_fusion_$(date +%Y%m%d_%H%M%S)}}"
case "${STAGE}" in
  preflight|train_seed_control|cache_stack|fit_candidates|select|evaluate_final|report|full_campaign) ;;
  *) echo "Unsupported stage: ${STAGE}" >&2; exit 2 ;;
esac

: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/first_stage_pilot}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_fusion/p7b_seed_control_${CAMPAIGN_ID}}"
: "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH:=${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT}/inputs/split_manifest/split_manifest.json.gz}"
: "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT}/inputs/hlt_cache}"
: "${LPRF_FUSION_SBATCH_ACCOUNT:=reu-aisocial}"
: "${LPRF_FUSION_SBATCH_PARTITION:=tigris}"
export LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ID="${CAMPAIGN_ID}"
export LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT
export LOCAL_RESIDUAL_FIELD_ROOT="${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT}"
export LOCAL_RESIDUAL_FIELD_MANIFEST_PATH LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR
export LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT="${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/source_artifact_audit.json"
export LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_SOURCES="${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/predictions/development_prediction_sources.json"
export LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_DIR="${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/representations"
export LOCAL_RESIDUAL_FIELD_FUSION_STABILITY_PLAN="${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/selection/stability_plan.json"
export LOCAL_RESIDUAL_FIELD_FUSION_METRIC_AUDIT="${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/metric_reproduction_audit.json"
export CONFIRM_FINAL_TEST=1

fresh_prepare_submitter
SUBMISSION_MANIFEST="${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/submission_jobs.tsv"
if ! fresh_is_dry_run; then
  mkdir -p "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}"
  exec 9>>"${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/.submission.lock"
  if ! flock -n 9; then
    echo "Another submitter currently owns this campaign: ${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}" >&2
    exit 3
  fi
  touch "${SUBMISSION_MANIFEST}"
fi
echo "stage=${STAGE}"
echo "campaign_id=${CAMPAIGN_ID}"
echo "campaign_root=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}"
echo "curriculum_root=${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT}"
echo "manifest_path=${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
echo "hlt_cache_dir=${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
echo "account=${LPRF_FUSION_SBATCH_ACCOUNT}"
echo "partition=${LPRF_FUSION_SBATCH_PARTITION}"
echo "PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"
echo "PRINT_ONLY=${PRINT_ONLY}"

declare -A JOBS=()

completion_contract() {
  local label="$1"
  case "${label}" in
    preflight) echo "local_residual_field_fusion_source_artifact_audit_v1" ;;
    metric_audit) echo "local_residual_field_fusion_metric_reproduction_v1" ;;
    seed_train) echo "local_residual_field_a0_seed1_completion_v1" ;;
    predictions) echo "local_residual_field_fusion_prediction_sources_v1" ;;
    feature_*) echo "local_residual_field_fusion_feature_manifest_v1" ;;
    screen_*|stability_*) echo "local_residual_field_fusion_candidate_report_v1" ;;
    stability_plan) echo "local_residual_field_fusion_stability_plan_v1" ;;
    selector) echo "local_residual_field_selected_fusion_set_v1" ;;
    replay) echo "local_residual_field_fusion_recipe_replay_v1" ;;
    final) echo "local_residual_field_fusion_final_evaluation_v1" ;;
    runtime) echo "local_residual_field_fusion_runtime_v1" ;;
    bootstrap) echo "local_residual_field_fusion_bootstrap_audit_v1" ;;
    report) echo "local_residual_field_fusion_campaign_report_v1" ;;
    *) echo "Unknown completion label: ${label}" >&2; return 2 ;;
  esac
}

artifact_done() {
  local label="$1" path="$2" expected_contract
  [[ -f "${path}" ]] || return 1
  if [[ "${path}" == *.json ]]; then
    expected_contract="$(completion_contract "${label}")"
    "${PYTHON_BIN}" scripts/validate_local_residual_field_fusion_completion.py \
      --path "${path}" --expected-contract "${expected_contract}" >/dev/null
  fi
}

afterok_arg() {
  local values=() value
  for value in "$@"; do [[ -n "${value}" ]] && values+=("${value}"); done
  if [[ "${#values[@]}" -gt 0 ]]; then
    local joined
    joined="$(IFS=:; echo "${values[*]}")"
    echo "--dependency=afterok:${joined}"
  fi
}

submit_job() {
  local label="$1" dependency="$2" script="$3"
  shift 3
  local command=(sbatch --parsable --partition="${LPRF_FUSION_SBATCH_PARTITION}" --account="${LPRF_FUSION_SBATCH_ACCOUNT}")
  [[ -n "${dependency}" ]] && command+=("${dependency}")
  command+=("${script}" "$@")
  echo "submit_label=${label} dependency=${dependency:-none} script=${script} args=$*" >&2
  if fresh_is_dry_run; then
    echo "DRYRUN_${label//[^A-Za-z0-9_]/_}"
  else
    "${command[@]}"
  fi
}

normalize_dependency() {
  local dependency="$1" ids
  [[ -n "${dependency}" ]] || { echo ""; return 0; }
  dependency="${dependency#--dependency=afterok:}"
  ids="$(printf '%s\n' "${dependency//:/$'\n'}" | sed '/^$/d' | sort -n | paste -sd: -)"
  [[ -n "${ids}" ]] && echo "--dependency=afterok:${ids}" || echo ""
}

active_submission_job() {
  local label="$1" completion="$2" expected_dependency="$3"
  local record job_id recorded_dependency status state reason
  fresh_is_dry_run && return 1
  [[ -f "${SUBMISSION_MANIFEST}" ]] || return 1
  record="$(awk -F '\t' -v label="${label}" -v completion="${completion}" '
    $2 == label && $3 == completion { job_id = $4; dependency = $5 }
    END { if (job_id != "") printf "%s\t%s", job_id, dependency }
  ' "${SUBMISSION_MANIFEST}")"
  job_id="${record%%$'\t'*}"
  recorded_dependency=""
  [[ "${record}" == *$'\t'* ]] && recorded_dependency="${record#*$'\t'}"
  [[ -n "${job_id}" ]] || return 1
  status="$(squeue -h -j "${job_id}" -o '%T|%r' 2>/dev/null | head -n 1 || true)"
  [[ -n "${status}" ]] || return 1
  state="${status%%|*}"
  reason="${status#*|}"
  expected_dependency="$(normalize_dependency "${expected_dependency}")"
  recorded_dependency="$(normalize_dependency "${recorded_dependency}")"
  if [[ "${recorded_dependency}" != "${expected_dependency}" ]]; then
    echo "resume_label=${label} stale_job=${job_id} reason=dependency_chain_changed recorded=${recorded_dependency:-none} expected=${expected_dependency:-none} action=cancel_and_resubmit" >&2
    scancel "${job_id}" 2>/dev/null || true
    return 1
  fi
  case "${state}" in
    BOOT_FAIL|CANCELLED|COMPLETED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|TIMEOUT)
      echo "resume_label=${label} stale_job=${job_id} state=${state} reason=${reason} action=resubmit" >&2
      return 1
      ;;
  esac
  if [[ "${reason}" == *DependencyNeverSatisfied* ]]; then
    echo "resume_label=${label} stale_job=${job_id} state=${state} reason=${reason} action=cancel_and_resubmit" >&2
    scancel "${job_id}" 2>/dev/null || true
    return 1
  fi
  echo "resume_label=${label} active_job=${job_id} state=${state} reason=${reason} completion=${completion}" >&2
  echo "${job_id}"
}

resume_or_submit() {
  local label="$1" completion="$2" dependency="$3" script="$4"
  shift 4
  if [[ -e "${completion}" ]]; then
    if artifact_done "${label}" "${completion}"; then
      echo "resume_label=${label} completed_artifact=${completion}" >&2
      echo ""
      return 0
    fi
    local quarantine="${completion}.invalid_$(date -u +%Y%m%dT%H%M%SZ)"
    if fresh_is_dry_run; then
      echo "resume_label=${label} would_quarantine_invalid_completion=${completion}" >&2
    else
      echo "Quarantining invalid completion artifact: ${completion} -> ${quarantine}" >&2
      mv -- "${completion}" "${quarantine}"
    fi
  fi
  local active_job submitted_job
  if active_job="$(active_submission_job "${label}" "${completion}" "${dependency}")"; then
    echo "${active_job}"
    return 0
  fi
  submitted_job="$(submit_job "${label}" "${dependency}" "${script}" "$@")"
  if ! fresh_is_dry_run; then
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${label}" "${completion}" "${submitted_job}" \
      "$(normalize_dependency "${dependency}")" >>"${SUBMISSION_MANIFEST}"
  fi
  echo "${submitted_job}"
}

require_or_dependency() {
  local path="$1" dependency="$2"
  if [[ -z "${dependency}" ]] && ! fresh_is_dry_run && [[ ! -e "${path}" ]]; then
    echo "Missing prerequisite and no queued parent: ${path}" >&2
    exit 2
  fi
}

submit_preflight() {
  JOBS[preflight]="$(resume_or_submit preflight "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}" "" sbatch/run_audit_local_residual_field_fusion_sources.sh)"
}

submit_metric_audit() {
  local parent="${1:-}"
  require_or_dependency "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}" "${parent}"
  JOBS[metric_audit]="$(resume_or_submit metric_audit "${LOCAL_RESIDUAL_FIELD_FUSION_METRIC_AUDIT}" "$(afterok_arg "${parent}")" sbatch/run_validate_local_residual_field_fusion_metrics.sh)"
}

submit_seed_training() {
  local parent="${1:-}"
  require_or_dependency "${LOCAL_RESIDUAL_FIELD_FUSION_SOURCE_AUDIT}" "${parent}"
  JOBS[seed_train]="$(resume_or_submit seed_train "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/taggers/A0_seed1/seed_control_completion.json" "$(afterok_arg "${parent}")" sbatch/run_train_local_residual_field_a0_seed1.sh)"
}

submit_stack_caches() {
  local parent="${1:-}"
  require_or_dependency "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/taggers/A0_seed1/seed_control_completion.json" "${parent}"
  JOBS[predictions]="$(resume_or_submit predictions "${LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_SOURCES}" "$(afterok_arg "${parent}")" sbatch/run_cache_local_residual_field_a0_seed1_predictions.sh)"
  local feature_parent="$(afterok_arg "${JOBS[predictions]}")" member
  for member in A0 A0_seed1 P7b; do
    JOBS["feature_${member}"]="$(resume_or_submit "feature_${member}" "${LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_DIR}/${member}/representation_manifest.json" "${feature_parent}" sbatch/run_cache_local_residual_field_fusion_features.sh "${member}")"
  done
}

submit_screening() {
  local parents=("$@") dep group candidate metric_parent="${JOBS[metric_audit]:-}"
  if [[ -n "${metric_parent}" ]]; then parents+=("${metric_parent}"); fi
  dep="$(afterok_arg "${parents[@]}")"
  require_or_dependency "${LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_DIR}/P7b/representation_manifest.json" "${dep}"
  if [[ -z "${metric_parent}" ]] && ! fresh_is_dry_run; then
    if ! artifact_done metric_audit "${LOCAL_RESIDUAL_FIELD_FUSION_METRIC_AUDIT}"; then
      echo "fit_candidates requires a valid completed raw-metric reproduction audit: ${LOCAL_RESIDUAL_FIELD_FUSION_METRIC_AUDIT}" >&2
      exit 2
    fi
  fi
  local candidates=(L0_mean_logits L1_mean_probs L2_temp_mean_logits L3_scalar_simplex_logits L4_classwise_simplex_logits L5_linear_stacker R0_linear_embeddings R1_mlp_embeddings_logits R2_scalar_event_gate R3_classwise_event_gate R4_A0_anchored_residual)
  for group in F_method F_seed; do
    for candidate in "${candidates[@]}"; do
      JOBS["screen_${group}_${candidate}"]="$(resume_or_submit "screen_${group}_${candidate}" "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/candidates/${group}/${candidate}/candidate_report.json" "${dep}" sbatch/run_fit_local_residual_field_fusion_candidate.sh "${group}" "${candidate}" screening)"
    done
  done
}

submit_selection() {
  local screening_jobs=("$@") screening_dep group candidate
  screening_dep="$(afterok_arg "${screening_jobs[@]}")"
  require_or_dependency "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/candidates/F_method/L0_mean_logits/candidate_report.json" "${screening_dep}"
  JOBS[stability_plan]="$(resume_or_submit stability_plan "${LOCAL_RESIDUAL_FIELD_FUSION_STABILITY_PLAN}" "${screening_dep}" sbatch/run_plan_local_residual_field_fusion_stability.sh)"
  local stability_dep="$(afterok_arg "${JOBS[stability_plan]}")"
  local representation=(R0_linear_embeddings R1_mlp_embeddings_logits R2_scalar_event_gate R3_classwise_event_gate R4_A0_anchored_residual)
  local stability_jobs=()
  for group in F_method F_seed; do
    for candidate in "${representation[@]}"; do
      local completion="${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/candidates/${group}/${candidate}/candidate_stability_report.json"
      if [[ -f "${LOCAL_RESIDUAL_FIELD_FUSION_STABILITY_PLAN}" ]] && ! fresh_is_dry_run; then
        if ! STABILITY_PLAN="${LOCAL_RESIDUAL_FIELD_FUSION_STABILITY_PLAN}" STABILITY_CANDIDATE="${candidate}" python -c 'import json,os; p=json.load(open(os.environ["STABILITY_PLAN"])); raise SystemExit(0 if os.environ["STABILITY_CANDIDATE"] in p["required_candidate_ids"] else 1)'; then
          echo "resume_label=stability_${group}_${candidate} reason=not_in_frozen_union" >&2
          continue
        fi
      fi
      JOBS["stability_${group}_${candidate}"]="$(resume_or_submit "stability_${group}_${candidate}" "${completion}" "${stability_dep}" sbatch/run_fit_local_residual_field_fusion_candidate.sh "${group}" "${candidate}" stability)"
      [[ -n "${JOBS[stability_${group}_${candidate}]}" ]] && stability_jobs+=("${JOBS[stability_${group}_${candidate}]}")
    done
  done
  local selector_dep="$(afterok_arg "${stability_jobs[@]}")"
  [[ -z "${selector_dep}" ]] && selector_dep="${stability_dep}"
  JOBS[selector]="$(resume_or_submit selector "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/selection/selected_fusion.json" "${selector_dep}" sbatch/run_select_local_residual_field_fusion.sh)"
}

final_result_path() {
  local selected="${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/selection/selected_fusion.json"
  if [[ -f "${selected}" ]]; then
    SELECTED_PATH="${selected}" python -c 'import json,os; p=json.load(open(os.environ["SELECTED_PATH"])); print(os.path.join(os.path.dirname(os.path.dirname(os.environ["SELECTED_PATH"])),"final_evaluation",p["artifact_hash"][:16],"final_evaluation.json"))'
  else
    echo "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/final_evaluation/PENDING_SELECTION/final_evaluation.json"
  fi
}

submit_final() {
  local parent="${1:-}" selected="${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/selection/selected_fusion.json"
  require_or_dependency "${selected}" "${parent}"
  local dep="$(afterok_arg "${parent}")"
  JOBS[replay]="$(resume_or_submit replay "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/selection/recipe_replay/recipe_replay.json" "${dep}" sbatch/run_replay_selected_local_residual_field_fusion_recipe.sh)"
  JOBS[final]="$(resume_or_submit final "$(final_result_path)" "${dep}" sbatch/run_evaluate_selected_local_residual_field_fusion.sh)"
}

submit_report() {
  local final_parent="${1:-}" replay_parent="${2:-}"
  local final_path="$(final_result_path)"
  require_or_dependency "${final_path}" "${final_parent}"
  require_or_dependency "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/selection/recipe_replay/recipe_replay.json" "${replay_parent}"
  local final_root="$(dirname "${final_path}")"
  JOBS[runtime]="$(resume_or_submit runtime "${final_root}/runtime_metrics.json" "$(afterok_arg "${final_parent}")" sbatch/run_benchmark_selected_local_residual_field_fusion.sh)"
  JOBS[bootstrap]="$(resume_or_submit bootstrap "${final_root}/bootstrap_audit.json" "$(afterok_arg "${final_parent}")" sbatch/run_audit_selected_local_residual_field_fusion_bootstraps.sh)"
  local report_dep="$(afterok_arg "${replay_parent}" "${final_parent}" "${JOBS[runtime]}" "${JOBS[bootstrap]}")"
  JOBS[report]="$(resume_or_submit report "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/final_report/run_report.json" "${report_dep}" sbatch/run_write_local_residual_field_fusion_campaign_report.sh)"
}

case "${STAGE}" in
  preflight) submit_preflight; submit_metric_audit "${JOBS[preflight]}" ;;
  train_seed_control) submit_seed_training ;;
  cache_stack) submit_stack_caches ;;
  fit_candidates) submit_screening ;;
  select) submit_selection ;;
  evaluate_final) submit_final ;;
  report) submit_report ;;
  full_campaign)
    submit_preflight
    submit_metric_audit "${JOBS[preflight]}"
    submit_seed_training "${JOBS[preflight]}"
    submit_stack_caches "${JOBS[seed_train]}"
    feature_jobs=("${JOBS[feature_A0]}" "${JOBS[feature_A0_seed1]}" "${JOBS[feature_P7b]}")
    submit_screening "${feature_jobs[@]}"
    screening_jobs=()
    for key in "${!JOBS[@]}"; do [[ "${key}" == screen_* && -n "${JOBS[$key]}" ]] && screening_jobs+=("${JOBS[$key]}"); done
    submit_selection "${screening_jobs[@]}"
    submit_final "${JOBS[selector]}"
    submit_report "${JOBS[final]}" "${JOBS[replay]}"
    ;;
esac

echo "submission_summary_begin"
for key in "${!JOBS[@]}"; do echo "job_${key}=${JOBS[$key]:-completed}"; done | sort
echo "submission_summary_end"
echo "campaign_root=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}"
