#!/usr/bin/env bash
# Submit cross-architecture prediction-block collection jobs.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${RECO_UPSTREAM_DEPENDENCY:=${UPSTREAM_DEPENDENCY}}"
: "${HLT_UPSTREAM_DEPENDENCY:=${UPSTREAM_DEPENDENCY}}"

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

dep_args() {
  local dependency="$1"
  shift
  if [[ -n "${dependency}" ]]; then
    printf '%s\n' --dependency="afterok:${dependency}"
  fi
  printf '%s\n' "$@"
}

maybe_skip_existing_prediction() {
  local source_dir="$1"
  local label="$2"
  if fresh_is_dry_run; then
    return 1
  fi
  if fresh_bool_enabled "${CROSSARCH_STEP6_SKIP_EXISTING_PREDICTIONS}" && [[ -d "${source_dir}" ]]; then
    echo "skipping existing prediction source ${label}: ${source_dir}" >&2
    return 0
  fi
  return 1
}

fresh_split_words reco_args "${CROSSARCH_RECO_ARCHITECTURES}"
fresh_split_words teacher_args "${CROSSARCH_RECO_TEACHERS}"
fresh_split_words hlt_arch_args "${CROSSARCH_HLT_BASELINE_ARCHITECTURES}"
fresh_split_words split_args "${CROSSARCH_RECO_PREDICT_SPLITS}"

submitter_lock_dir="${CROSSARCH_ROOT}/.step6_prediction_submission_lock"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "reconstructors=$(fresh_join_by_space "${reco_args[@]}")"
    echo "teachers=$(fresh_join_by_space "${teacher_args[@]}")"
    echo "hlt_architectures=$(fresh_join_by_space "${hlt_arch_args[@]}")"
    echo "splits=$(fresh_join_by_space "${split_args[@]}")"
    echo "hlt_cache_dir=${CROSSARCH_HLT_CACHE_DIR}"
    echo "prediction_dir=${CROSSARCH_PREDICTION_DIR}"
    echo "reco_prediction_run_dir=${CROSSARCH_RECO_PREDICTION_RUN_DIR}"
    echo "hlt_prediction_run_dir=${CROSSARCH_HLT_PREDICTION_RUN_DIR}"
    echo "submit_hlt_predictions=${CROSSARCH_STEP6_SUBMIT_HLT_PREDICTIONS}"
    echo "skip_existing_predictions=${CROSSARCH_STEP6_SKIP_EXISTING_PREDICTIONS}"
    echo "reco_upstream_dependency=${RECO_UPSTREAM_DEPENDENCY:-none}"
    echo "hlt_upstream_dependency=${HLT_UPSTREAM_DEPENDENCY:-none}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

hlt_predict_job_ids=()
hlt_model_names=()
if fresh_bool_enabled "${CROSSARCH_STEP6_SUBMIT_HLT_PREDICTIONS}"; then
  for architecture in "${hlt_arch_args[@]}"; do
    model_name="$(fresh_crossarch_hlt_model_name "${architecture}")"
    source_dir="${CROSSARCH_PREDICTION_DIR}/${model_name}"
    if maybe_skip_existing_prediction "${source_dir}" "${model_name}"; then
      continue
    fi
    fresh_refuse_existing_dir "${source_dir}"
    mapfile -t hlt_args < <(
      dep_args \
        "${HLT_UPSTREAM_DEPENDENCY}" \
        "${SCRIPT_DIR}/run_crossarch_predict_hlt_baseline.sh" \
        "${architecture}"
    )
    predict_jid="$(submit_job "crossarch_predict_${model_name}" "${hlt_args[@]}")"
    hlt_predict_job_ids+=("${predict_jid}")
    hlt_model_names+=("${model_name}")
    echo "submitted crossarch_predict_${model_name}=${predict_jid}"
  done
fi

reco_predict_job_ids=()
reco_model_names=()
for reco_architecture in "${reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    model_name="$(fresh_crossarch_reco_model_name "${reco_architecture}" "${teacher_architecture}")"
    source_dir="${CROSSARCH_PREDICTION_DIR}/${model_name}"
    if maybe_skip_existing_prediction "${source_dir}" "${model_name}"; then
      continue
    fi
    fresh_refuse_existing_dir "${source_dir}"
    mapfile -t reco_submit_args < <(
      dep_args \
        "${RECO_UPSTREAM_DEPENDENCY}" \
        "${SCRIPT_DIR}/run_crossarch_predict_reconstructor.sh" \
        "${reco_architecture}" \
        "${teacher_architecture}"
    )
    predict_jid="$(submit_job "crossarch_predict_${model_name}" "${reco_submit_args[@]}")"
    reco_predict_job_ids+=("${predict_jid}")
    reco_model_names+=("${model_name}")
    echo "submitted crossarch_predict_${model_name}=${predict_jid}"
  done
done

cat <<SUMMARY
crossarch_step6_predictions_submission:
  hlt_predict_job_ids: $(fresh_join_by_space "${hlt_predict_job_ids[@]}")
  reco_predict_job_ids: $(fresh_join_by_space "${reco_predict_job_ids[@]}")
  dependency_summary:
    hlt_afterok: ${HLT_UPSTREAM_DEPENDENCY:-none}
    reco_afterok: ${RECO_UPSTREAM_DEPENDENCY:-none}
  submitted_model_names:
    hlt: $(fresh_join_by_space "${hlt_model_names[@]}")
    reco: $(fresh_join_by_space "${reco_model_names[@]}")
  expected_sources:
    hlt: 4
    reco: 16
    total: 20
  submit_hlt_predictions: ${CROSSARCH_STEP6_SUBMIT_HLT_PREDICTIONS}
  skip_existing_predictions: ${CROSSARCH_STEP6_SKIP_EXISTING_PREDICTIONS}
  splits: $(fresh_join_by_space "${split_args[@]}")
  output_dirs:
    root: ${CROSSARCH_ROOT}
    predictions: ${CROSSARCH_PREDICTION_DIR}
    reco_prediction_runs: ${CROSSARCH_RECO_PREDICTION_RUN_DIR}
    hlt_prediction_runs: ${CROSSARCH_HLT_PREDICTION_RUN_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
