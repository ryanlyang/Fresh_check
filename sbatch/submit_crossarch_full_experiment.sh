#!/usr/bin/env bash
# Submit the full cross-architecture 16x4 teacher-logit experiment graph.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

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

fresh_split_words offline_teacher_args "${CROSSARCH_OFFLINE_TEACHER_ARCHITECTURES}"
fresh_split_words hlt_arch_args "${CROSSARCH_HLT_BASELINE_ARCHITECTURES}"
fresh_split_words reco_args "${CROSSARCH_RECO_ARCHITECTURES}"
fresh_split_words teacher_args "${CROSSARCH_RECO_TEACHERS}"
fresh_split_words split_args "${CROSSARCH_RECO_PREDICT_SPLITS}"

submitter_lock_dir="${CROSSARCH_ROOT}/.full_experiment_submission_lock"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "source_status_hash=$(fresh_source_status_hash)"
    echo "root=${CROSSARCH_ROOT}"
    echo "offline_teachers=$(fresh_join_by_space "${offline_teacher_args[@]}")"
    echo "hlt_architectures=$(fresh_join_by_space "${hlt_arch_args[@]}")"
    echo "reconstructors=$(fresh_join_by_space "${reco_args[@]}")"
    echo "reco_teachers=$(fresh_join_by_space "${teacher_args[@]}")"
    echo "prediction_splits=$(fresh_join_by_space "${split_args[@]}")"
    echo "fusion_include_optional_groups=${CROSSARCH_FUSION_INCLUDE_OPTIONAL_GROUPS}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

fresh_refuse_existing_path "${CROSSARCH_MANIFEST_PATH}"
if ! fresh_is_dry_run && [[ -d "${CROSSARCH_HLT_CACHE_DIR}" ]] && ! fresh_bool_enabled "${OVERWRITE}"; then
  echo "Refusing existing crossarch HLT cache without OVERWRITE=1: ${CROSSARCH_HLT_CACHE_DIR}" >&2
  exit 2
fi

split_jid="$(submit_job "crossarch_build_splits" "${SCRIPT_DIR}/run_crossarch_build_splits.sh")"
cache_jid="$(submit_job "crossarch_build_hlt_cache" \
  --dependency="afterok:${split_jid}" \
  "${SCRIPT_DIR}/run_crossarch_build_hlt_cache.sh")"
audit_jid="$(submit_job "crossarch_audit_splits_hlt_cache" \
  --dependency="afterok:${cache_jid}" \
  "${SCRIPT_DIR}/run_crossarch_audit_splits_hlt_cache.sh")"

teacher_job_ids=()
for architecture in "${offline_teacher_args[@]}"; do
  fresh_refuse_existing_dir "${CROSSARCH_OFFLINE_TEACHER_DIR}/${architecture}"
  teacher_jid="$(submit_job "crossarch_teacher_${architecture}" \
    --dependency="afterok:${audit_jid}" \
    "${SCRIPT_DIR}/run_crossarch_train_offline_teacher.sh" \
    "${architecture}")"
  teacher_job_ids+=("${teacher_jid}")
  echo "submitted crossarch_teacher_${architecture}=${teacher_jid}"
done
teacher_dep="$(fresh_join_by_colon "${teacher_job_ids[@]}")"

hlt_train_job_ids=()
hlt_predict_job_ids=()
hlt_model_names=()
for architecture in "${hlt_arch_args[@]}"; do
  model_name="$(fresh_crossarch_hlt_model_name "${architecture}")"
  fresh_refuse_existing_dir "${CROSSARCH_HLT_BASELINE_DIR}/${architecture}"
  hlt_train_jid="$(submit_job "crossarch_hlt_train_${architecture}" \
    --dependency="afterok:${audit_jid}" \
    "${SCRIPT_DIR}/run_crossarch_train_hlt_baseline.sh" \
    "${architecture}")"
  hlt_train_job_ids+=("${hlt_train_jid}")
  echo "submitted crossarch_hlt_train_${architecture}=${hlt_train_jid}"

  fresh_refuse_existing_dir "${CROSSARCH_PREDICTION_DIR}/${model_name}"
  hlt_predict_jid="$(submit_job "crossarch_hlt_predict_${architecture}" \
    --dependency="afterok:${hlt_train_jid}" \
    "${SCRIPT_DIR}/run_crossarch_predict_hlt_baseline.sh" \
    "${architecture}")"
  hlt_predict_job_ids+=("${hlt_predict_jid}")
  hlt_model_names+=("${model_name}")
  echo "submitted crossarch_hlt_predict_${architecture}=${hlt_predict_jid}"
done

reco_train_job_ids=()
reco_model_names=()
for reco_architecture in "${reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    model_name="$(fresh_crossarch_reco_model_name "${reco_architecture}" "${teacher_architecture}")"
    output_dir="${CROSSARCH_RECO_MODEL_DIR}/${reco_architecture}/${teacher_architecture}"
    fresh_refuse_existing_dir "${output_dir}"
    reco_train_jid="$(submit_job "crossarch_reco_train_${model_name}" \
      --dependency="afterok:${teacher_dep}" \
      "${SCRIPT_DIR}/run_crossarch_train_reconstructor.sh" \
      "${reco_architecture}" \
      "${teacher_architecture}")"
    reco_train_job_ids+=("${reco_train_jid}")
    reco_model_names+=("${model_name}")
    echo "submitted crossarch_reco_train_${model_name}=${reco_train_jid}"
  done
done
reco_train_dep="$(fresh_join_by_colon "${reco_train_job_ids[@]}")"

reco_predict_job_ids=()
for reco_architecture in "${reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    model_name="$(fresh_crossarch_reco_model_name "${reco_architecture}" "${teacher_architecture}")"
    source_dir="${CROSSARCH_PREDICTION_DIR}/${model_name}"
    fresh_refuse_existing_dir "${source_dir}"
    reco_predict_jid="$(submit_job "crossarch_reco_predict_${model_name}" \
      --dependency="afterok:${reco_train_dep}" \
      "${SCRIPT_DIR}/run_crossarch_predict_reconstructor.sh" \
      "${reco_architecture}" \
      "${teacher_architecture}")"
    reco_predict_job_ids+=("${reco_predict_jid}")
    echo "submitted crossarch_reco_predict_${model_name}=${reco_predict_jid}"
  done
done

prediction_job_ids=("${hlt_predict_job_ids[@]}" "${reco_predict_job_ids[@]}")
prediction_dep="$(fresh_join_by_colon "${prediction_job_ids[@]}")"

fresh_refuse_existing_dir "${CROSSARCH_FUSION_DIR}"
fusion_jid="$(submit_job "crossarch_fusion" \
  --dependency="afterok:${prediction_dep}" \
  "${SCRIPT_DIR}/run_crossarch_fusion.sh")"

fresh_refuse_existing_dir "${CROSSARCH_FINAL_REPORT_DIR}"
final_report_jid="$(submit_job "crossarch_final_report" \
  --dependency="afterok:${fusion_jid}" \
  "${SCRIPT_DIR}/run_crossarch_write_final_report.sh")"

cat <<SUMMARY
crossarch_full_experiment_submission:
  split_job_id: ${split_jid}
  hlt_cache_job_id: ${cache_jid}
  step2_audit_job_id: ${audit_jid}
  offline_teacher_job_ids: $(fresh_join_by_space "${teacher_job_ids[@]}")
  hlt_train_job_ids: $(fresh_join_by_space "${hlt_train_job_ids[@]}")
  hlt_predict_job_ids: $(fresh_join_by_space "${hlt_predict_job_ids[@]}")
  reco_train_job_ids: $(fresh_join_by_space "${reco_train_job_ids[@]}")
  reco_predict_job_ids: $(fresh_join_by_space "${reco_predict_job_ids[@]}")
  fusion_job_id: ${fusion_jid}
  final_report_job_id: ${final_report_jid}
  dependency_summary:
    hlt_cache_afterok: ${split_jid}
    step2_audit_afterok: ${cache_jid}
    offline_teachers_afterok: ${audit_jid}
    hlt_train_afterok: ${audit_jid}
    each_hlt_predict_after_its_train: true
    reco_train_afterok: ${teacher_dep}
    reco_predict_afterok: ${reco_train_dep}
    fusion_afterok: ${prediction_dep}
    final_report_afterok: ${fusion_jid}
  expected_jobs:
    offline_teachers: 4
    hlt_train: 4
    hlt_predict: 4
    reco_train: 16
    reco_predict: 16
    total_submitted: ${submit_count}
  expected_sources:
    hlt: $(fresh_join_by_space "${hlt_model_names[@]}")
    reco: $(fresh_join_by_space "${reco_model_names[@]}")
    total: 20
  fusion:
    include_optional_groups: ${CROSSARCH_FUSION_INCLUDE_OPTIONAL_GROUPS}
    fusers: ${CROSSARCH_FUSERS}
    controls_disabled: ${CROSSARCH_FUSION_SKIP_CONTROLS}
  output_dirs:
    root: ${CROSSARCH_ROOT}
    manifest: ${CROSSARCH_MANIFEST_PATH}
    hlt_cache: ${CROSSARCH_HLT_CACHE_DIR}
    step2_audits: ${CROSSARCH_STEP2_AUDIT_DIR}
    offline_teachers: ${CROSSARCH_OFFLINE_TEACHER_DIR}
    hlt_baselines: ${CROSSARCH_HLT_BASELINE_DIR}
    reco_models: ${CROSSARCH_RECO_MODEL_DIR}
    predictions: ${CROSSARCH_PREDICTION_DIR}
    fusion: ${CROSSARCH_FUSION_DIR}
    final_report: ${CROSSARCH_FINAL_REPORT_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
