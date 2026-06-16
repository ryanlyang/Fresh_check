#!/usr/bin/env bash
# Submit the aggressive crossarch reconstructor, adapted-tagger, and fusion graph.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${FUSION_UPSTREAM_DEPENDENCY:=}"

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

afterok_args() {
  local dependency="$1"
  shift
  if [[ -n "${dependency}" ]]; then
    printf '%s\n' --dependency="afterok:${dependency}"
  fi
  printf '%s\n' "$@"
}

fresh_split_words reco_args "${CROSSARCH_AGGRESSIVE_RECO_ARCHITECTURES}"
fresh_split_words teacher_args "${CROSSARCH_AGGRESSIVE_RECO_TEACHERS}"
fresh_split_words hlt_arch_args "${CROSSARCH_HLT_BASELINE_ARCHITECTURES}"
fresh_split_words split_args "${CROSSARCH_AGGRESSIVE_RECO_PREDICT_SPLITS}"
fresh_split_words fusion_group_args "${CROSSARCH_AGGRESSIVE_FUSION_GROUPS}"

submitter_lock_dir="${CROSSARCH_AGGRESSIVE_ROOT}/.submission_lock"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "aggressive_reconstructors=$(fresh_join_by_space "${reco_args[@]}")"
    echo "teachers=$(fresh_join_by_space "${teacher_args[@]}")"
    echo "hlt_architectures=$(fresh_join_by_space "${hlt_arch_args[@]}")"
    echo "splits=$(fresh_join_by_space "${split_args[@]}")"
    echo "fusion_groups=$(fresh_join_by_space "${fusion_group_args[@]}")"
    echo "manifest=${CROSSARCH_MANIFEST_PATH}"
    echo "hlt_cache_dir=${CROSSARCH_HLT_CACHE_DIR}"
    echo "offline_teacher_dir=${CROSSARCH_OFFLINE_TEACHER_DIR}"
    echo "aggressive_root=${CROSSARCH_AGGRESSIVE_ROOT}"
    echo "aggressive_reco_model_dir=${CROSSARCH_AGGRESSIVE_RECO_MODEL_DIR}"
    echo "prediction_dir=${CROSSARCH_AGGRESSIVE_PREDICTION_DIR}"
    echo "fusion_dir=${CROSSARCH_AGGRESSIVE_FUSION_DIR}"
    echo "audit_dir=${CROSSARCH_AGGRESSIVE_AUDIT_DIR}"
    echo "upstream_dependency=${UPSTREAM_DEPENDENCY:-none}"
    echo "fusion_upstream_dependency=${FUSION_UPSTREAM_DEPENDENCY:-none}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

if ! fresh_is_dry_run; then
  fresh_require_file "${CROSSARCH_MANIFEST_PATH}"
  fresh_require_dir "${CROSSARCH_HLT_CACHE_DIR}"
  for teacher_architecture in "${teacher_args[@]}"; do
    fresh_require_file "${CROSSARCH_OFFLINE_TEACHER_DIR}/${teacher_architecture}/best_model_val.pt"
  done
  if fresh_bool_enabled "${CROSSARCH_AGGRESSIVE_REQUIRE_HLT_PREDICTIONS_AT_SUBMIT}"; then
    for architecture in "${hlt_arch_args[@]}"; do
      hlt_name="$(fresh_crossarch_hlt_model_name "${architecture}")"
      for split in "${split_args[@]}"; do
        fresh_require_file "${CROSSARCH_AGGRESSIVE_PREDICTION_DIR}/${hlt_name}/${split}_predictions.npz"
        fresh_require_file "${CROSSARCH_AGGRESSIVE_PREDICTION_DIR}/${hlt_name}/${split}_predictions_metadata.json"
      done
    done
  else
    echo "Skipping submit-time HLT prediction preflight; HLT predictions must be produced before fusion runs." >&2
  fi
fi

reco_model_names=()
adapted_model_names=()
for reco_architecture in "${reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    reco_model_name="$(fresh_crossarch_aggressive_reco_model_name "${reco_architecture}" "${teacher_architecture}")"
    adapted_model_name="$(fresh_crossarch_aggressive_reco_domain_tagger_model_name "${reco_architecture}" "${teacher_architecture}")"
    reco_model_names+=("${reco_model_name}")
    adapted_model_names+=("${adapted_model_name}")
    fresh_refuse_existing_dir "${CROSSARCH_AGGRESSIVE_RECO_MODEL_DIR}/${reco_architecture}/${teacher_architecture}"
    fresh_refuse_existing_dir "${CROSSARCH_AGGRESSIVE_RECO_PREDICTION_RUN_DIR}/${reco_model_name}"
    fresh_refuse_existing_dir "${CROSSARCH_AGGRESSIVE_PREDICTION_DIR}/${reco_model_name}"
    fresh_refuse_existing_dir "${CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_DIR}/${reco_architecture}/${teacher_architecture}"
    fresh_refuse_existing_dir "${CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_PREDICTION_RUN_DIR}/${adapted_model_name}"
    fresh_refuse_existing_dir "${CROSSARCH_AGGRESSIVE_PREDICTION_DIR}/${adapted_model_name}"
  done
done
fresh_refuse_existing_dir "${CROSSARCH_AGGRESSIVE_FUSION_DIR}"
fresh_refuse_existing_dir "${CROSSARCH_AGGRESSIVE_AUDIT_DIR}"

train_job_ids=()
reco_predict_job_ids=()
adapt_train_job_ids=()
adapt_predict_job_ids=()
declare -A train_job_id_by_model=()
declare -A adapt_train_job_id_by_model=()

for reco_architecture in "${reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    model_name="$(fresh_crossarch_aggressive_reco_model_name "${reco_architecture}" "${teacher_architecture}")"
    mapfile -t train_args < <(
      afterok_args \
        "${UPSTREAM_DEPENDENCY}" \
        "${SCRIPT_DIR}/run_crossarch_aggressive_train_reconstructor.sh" \
        "${reco_architecture}" \
        "${teacher_architecture}"
    )
    train_jid="$(submit_job "crossarch_aggressive_train_${model_name}" "${train_args[@]}")"
    train_job_ids+=("${train_jid}")
    train_job_id_by_model["${model_name}"]="${train_jid}"
    echo "submitted crossarch_aggressive_train_${model_name}=${train_jid}"
  done
done

for reco_architecture in "${reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    model_name="$(fresh_crossarch_aggressive_reco_model_name "${reco_architecture}" "${teacher_architecture}")"
    train_jid="${train_job_id_by_model[$model_name]}"
    predict_jid="$(submit_job "crossarch_aggressive_predict_${model_name}" \
      --dependency="afterok:${train_jid}" \
      "${SCRIPT_DIR}/run_crossarch_aggressive_predict_reconstructor.sh" \
      "${reco_architecture}" \
      "${teacher_architecture}")"
    reco_predict_job_ids+=("${predict_jid}")
    echo "submitted crossarch_aggressive_predict_${model_name}=${predict_jid}"
  done
done

for reco_architecture in "${reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    reco_model_name="$(fresh_crossarch_aggressive_reco_model_name "${reco_architecture}" "${teacher_architecture}")"
    adapted_model_name="$(fresh_crossarch_aggressive_reco_domain_tagger_model_name "${reco_architecture}" "${teacher_architecture}")"
    train_jid="${train_job_id_by_model[$reco_model_name]}"
    adapt_train_jid="$(submit_job "crossarch_aggressive_adapt_train_${adapted_model_name}" \
      --dependency="afterok:${train_jid}" \
      "${SCRIPT_DIR}/run_crossarch_aggressive_train_reco_domain_tagger.sh" \
      "${reco_architecture}" \
      "${teacher_architecture}")"
    adapt_train_job_ids+=("${adapt_train_jid}")
    adapt_train_job_id_by_model["${adapted_model_name}"]="${adapt_train_jid}"
    echo "submitted crossarch_aggressive_adapt_train_${adapted_model_name}=${adapt_train_jid}"
  done
done

for reco_architecture in "${reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    adapted_model_name="$(fresh_crossarch_aggressive_reco_domain_tagger_model_name "${reco_architecture}" "${teacher_architecture}")"
    adapt_train_jid="${adapt_train_job_id_by_model[$adapted_model_name]}"
    adapt_predict_jid="$(submit_job "crossarch_aggressive_adapt_predict_${adapted_model_name}" \
      --dependency="afterok:${adapt_train_jid}" \
      "${SCRIPT_DIR}/run_crossarch_aggressive_predict_reco_domain_tagger.sh" \
      "${reco_architecture}" \
      "${teacher_architecture}")"
    adapt_predict_job_ids+=("${adapt_predict_jid}")
    echo "submitted crossarch_aggressive_adapt_predict_${adapted_model_name}=${adapt_predict_jid}"
  done
done

fusion_dependencies=("${reco_predict_job_ids[@]}" "${adapt_predict_job_ids[@]}")
if [[ -n "${FUSION_UPSTREAM_DEPENDENCY}" ]]; then
  fusion_dependencies+=("${FUSION_UPSTREAM_DEPENDENCY}")
fi
fusion_dep="$(fresh_join_by_colon "${fusion_dependencies[@]}")"
fusion_jid="$(submit_job "crossarch_aggressive_fusion" \
  --dependency="afterok:${fusion_dep}" \
  "${SCRIPT_DIR}/run_crossarch_aggressive_fusion.sh")"
audit_jid="$(submit_job "crossarch_aggressive_audit" \
  --dependency="afterok:${fusion_jid}" \
  "${SCRIPT_DIR}/run_crossarch_aggressive_audit.sh")"

cat <<SUMMARY
crossarch_aggressive_experiment_submission:
  train_job_ids: $(fresh_join_by_space "${train_job_ids[@]}")
  frozen_teacher_predict_job_ids: $(fresh_join_by_space "${reco_predict_job_ids[@]}")
  adapted_train_job_ids: $(fresh_join_by_space "${adapt_train_job_ids[@]}")
  adapted_predict_job_ids: $(fresh_join_by_space "${adapt_predict_job_ids[@]}")
  fusion_job_id: ${fusion_jid}
  audit_job_id: ${audit_jid}
  dependency_summary:
    train_afterok_extra: ${UPSTREAM_DEPENDENCY:-none}
    each_frozen_teacher_prediction_after_its_reco_train: true
    each_adapted_train_after_its_reco_train: true
    each_adapted_prediction_after_its_adapted_train: true
    fusion_afterok: ${fusion_dep}
    audit_afterok: ${fusion_jid}
  expected_jobs:
    aggressive_reco_train: 16
    aggressive_frozen_teacher_predict: 16
    aggressive_adapted_tagger_train: 16
    aggressive_adapted_tagger_predict: 16
    fusion: 1
    audit: 1
    total_submitted: ${submit_count}
  expected_sources:
    aggressive_frozen_teacher: $(fresh_join_by_space "${reco_model_names[@]}")
    aggressive_adapted_taggers: $(fresh_join_by_space "${adapted_model_names[@]}")
    hlt4_reused: hlt_part hlt_pn hlt_pfn hlt_pcnn
  fusion_groups: $(fresh_join_by_space "${fusion_group_args[@]}")
  split_sizes:
    model_train: ${CROSSARCH_MODEL_TRAIN_SIZE}
    model_val: ${CROSSARCH_MODEL_VAL_SIZE}
    stack_train: ${CROSSARCH_STACK_TRAIN_SIZE}
    stack_val: ${CROSSARCH_STACK_VAL_SIZE}
    final_test: ${CROSSARCH_FINAL_TEST_SIZE}
  output_dirs:
    aggressive_root: ${CROSSARCH_AGGRESSIVE_ROOT}
    aggressive_reco_models: ${CROSSARCH_AGGRESSIVE_RECO_MODEL_DIR}
    aggressive_reco_prediction_runs: ${CROSSARCH_AGGRESSIVE_RECO_PREDICTION_RUN_DIR}
    aggressive_adapted_taggers: ${CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_DIR}
    aggressive_adapted_prediction_runs: ${CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_PREDICTION_RUN_DIR}
    prediction_sources: ${CROSSARCH_AGGRESSIVE_PREDICTION_DIR}
    fusion: ${CROSSARCH_AGGRESSIVE_FUSION_DIR}
    audits: ${CROSSARCH_AGGRESSIVE_AUDIT_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
