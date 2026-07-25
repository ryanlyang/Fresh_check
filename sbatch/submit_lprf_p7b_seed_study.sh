#!/usr/bin/env bash
# Dependency-safe submitter for five matched A0/P7b training seeds.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"

STAGE="${1:-full_campaign}"
CAMPAIGN_ID="${2:-${LOCAL_RESIDUAL_FIELD_SEED_STUDY_CAMPAIGN_ID:-p7b_seed_study_$(date +%Y%m%d_%H%M%S)}}"
case "${STAGE}" in preflight|train|report|full_campaign) ;; *) echo "unsupported stage ${STAGE}" >&2; exit 2 ;; esac

: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/first_stage_pilot}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_fusion/p7b_seed_control}"
: "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_seed_study/${CAMPAIGN_ID}}"
: "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_SBATCH_ACCOUNT:=reu-aisocial}"
: "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_SBATCH_PARTITION:=tigris}"
export LOCAL_RESIDUAL_FIELD_SEED_STUDY_CAMPAIGN_ID="${CAMPAIGN_ID}"
export LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT
export LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT
export LOCAL_RESIDUAL_FIELD_SEED_STUDY_MANIFEST="${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}/study_manifest.json"

fresh_prepare_submitter
mkdir -p "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}"
exec 9>>"${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}/.submission.lock"
if ! flock -n 9; then
  echo "another submitter owns ${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}" >&2
  exit 3
fi
SUBMISSION_MANIFEST="${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}/submission_jobs.tsv"
touch "${SUBMISSION_MANIFEST}"

echo "stage=${STAGE}"
echo "campaign_id=${CAMPAIGN_ID}"
echo "campaign_root=${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}"
echo "curriculum_root=${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT}"
echo "fusion_root=${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}"
echo "matched_seeds=20421,20522,20623,20724,20825"
echo "reused_A0_seeds=20421,20522"
echo "new_A0_seeds=20623,20724,20825"
echo "new_P7b_seeds=20421,20522,20623,20724,20825"
echo "final_test=forbidden"
echo "account=${LOCAL_RESIDUAL_FIELD_SEED_STUDY_SBATCH_ACCOUNT}"
echo "partition=${LOCAL_RESIDUAL_FIELD_SEED_STUDY_SBATCH_PARTITION}"

afterok_arg() {
  local ids=() value
  for value in "$@"; do [[ -n "${value}" ]] && ids+=("${value}"); done
  if ((${#ids[@]})); then
    local joined
    joined="$(IFS=:; echo "${ids[*]}")"
    echo "--dependency=afterok:${joined}"
  fi
}

normalize_dependency() {
  local dependency="$1" ids
  dependency="${dependency#--dependency=afterok:}"
  [[ -n "${dependency}" ]] || { echo ""; return 0; }
  ids="$(printf '%s\n' "${dependency//:/$'\n'}" | sed '/^$/d' | sort | paste -sd: -)"
  echo "--dependency=afterok:${ids}"
}

active_job() {
  local label="$1" completion="$2" expected_dependency="$3"
  local record job_id recorded_dependency status state reason
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
  if [[ "$(normalize_dependency "${recorded_dependency}")" != "$(normalize_dependency "${expected_dependency}")" ]]; then
    echo "stale_label=${label} job=${job_id} reason=dependency_chain_changed action=cancel_and_resubmit" >&2
    scancel "${job_id}" 2>/dev/null || true
    return 1
  fi
  if [[ "${reason}" == *DependencyNeverSatisfied* ]]; then
    echo "stale_label=${label} job=${job_id} state=${state} reason=${reason} action=cancel_and_resubmit" >&2
    scancel "${job_id}" 2>/dev/null || true
    return 1
  fi
  echo "${job_id}"
}

submit_or_reuse() {
  local label="$1" completion="$2" dependency="$3" script="$4"
  shift 4
  if [[ -f "${completion}" ]]; then
    echo "reuse_label=${label} completion=${completion}" >&2
    echo ""
    return 0
  fi
  local existing command job_id
  if existing="$(active_job "${label}" "${completion}" "${dependency}")"; then
    echo "reuse_label=${label} active_job=${existing}" >&2
    echo "${existing}"
    return 0
  fi
  command=(sbatch --parsable
    --partition="${LOCAL_RESIDUAL_FIELD_SEED_STUDY_SBATCH_PARTITION}"
    --account="${LOCAL_RESIDUAL_FIELD_SEED_STUDY_SBATCH_ACCOUNT}")
  [[ -n "${dependency}" ]] && command+=("${dependency}")
  command+=("${script}" "$@")
  if fresh_is_dry_run; then
    job_id="DRYRUN_${label}"
    echo "${command[*]}" >&2
  else
    job_id="$("${command[@]}")"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "${label}" "${completion}" "${job_id}" "$(normalize_dependency "${dependency}")" \
      >>"${SUBMISSION_MANIFEST}"
  fi
  echo "${job_id}"
}

preflight_job=""
training_jobs=()
submit_preflight() {
  preflight_job="$(submit_or_reuse preflight "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_MANIFEST}" "" \
    sbatch/run_prepare_local_residual_field_seed_study.sh)"
}
submit_training() {
  local dependency seed job
  dependency="$(afterok_arg "${preflight_job}")"
  if [[ -z "${dependency}" && ! -f "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_MANIFEST}" && "${DRY_RUN:-0}" != 1 && "${PRINT_ONLY:-0}" != 1 ]]; then
    echo "training requires a completed preflight manifest or queued preflight job" >&2
    exit 2
  fi
  for seed in 20623 20724 20825; do
    job="$(submit_or_reuse "A0_${seed}" \
      "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}/runs/seed_${seed}/A0/seed_study_completion.json" \
      "${dependency}" sbatch/run_train_local_residual_field_seed_study_a0.sh "${seed}")"
    [[ -n "${job}" ]] && training_jobs+=("${job}")
  done
  for seed in 20421 20522 20623 20724 20825; do
    job="$(submit_or_reuse "P7b_${seed}" \
      "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}/runs/seed_${seed}/P7b/seed_study_completion.json" \
      "${dependency}" sbatch/run_train_local_residual_field_seed_study_p7b.sh "${seed}")"
    [[ -n "${job}" ]] && training_jobs+=("${job}")
  done
}
submit_report() {
  local dependency seed completion
  dependency="$(afterok_arg "${training_jobs[@]}")"
  if [[ -z "${dependency}" && ! fresh_is_dry_run ]]; then
    for seed in 20623 20724 20825; do
      completion="${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}/runs/seed_${seed}/A0/seed_study_completion.json"
      [[ -f "${completion}" ]] || { echo "missing A0 completion: ${completion}" >&2; exit 2; }
    done
    for seed in 20421 20522 20623 20724 20825; do
      completion="${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}/runs/seed_${seed}/P7b/seed_study_completion.json"
      [[ -f "${completion}" ]] || { echo "missing P7b completion: ${completion}" >&2; exit 2; }
    done
  fi
  report_job="$(submit_or_reuse report \
    "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}/final_report/run_report.json" \
    "${dependency}" sbatch/run_write_local_residual_field_seed_study_report.sh)"
}

case "${STAGE}" in
  preflight) submit_preflight ;;
  train) submit_training ;;
  report) submit_report ;;
  full_campaign) submit_preflight; submit_training; submit_report ;;
esac

echo "submission_summary_begin"
echo "preflight_job=${preflight_job:-completed_or_not_requested}"
printf 'training_job=%s\n' "${training_jobs[@]:-}"
echo "report_job=${report_job:-not_requested}"
echo "submission_summary_end"
echo "campaign_root=${LOCAL_RESIDUAL_FIELD_SEED_STUDY_ROOT}"
