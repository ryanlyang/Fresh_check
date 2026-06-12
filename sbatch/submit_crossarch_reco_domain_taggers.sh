#!/usr/bin/env bash
# Submit frozen-reco -> adapted-tagger jobs and their separate fusion branch.

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

fresh_split_words reco_args "${CROSSARCH_RECO_ARCHITECTURES}"
fresh_split_words teacher_args "${CROSSARCH_RECO_TEACHERS}"
fresh_split_words hlt_arch_args "${CROSSARCH_HLT_BASELINE_ARCHITECTURES}"
fresh_split_words split_args "${CROSSARCH_RECO_DOMAIN_TAGGER_PREDICT_SPLITS}"

if ! fresh_is_dry_run; then
  fresh_require_dir "${CROSSARCH_HLT_CACHE_DIR}"
  for architecture in "${hlt_arch_args[@]}"; do
    hlt_name="$(fresh_crossarch_hlt_model_name "${architecture}")"
    for split in "${split_args[@]}"; do
      fresh_require_file "${CROSSARCH_PREDICTION_DIR}/${hlt_name}/${split}_predictions.npz"
      fresh_require_file "${CROSSARCH_PREDICTION_DIR}/${hlt_name}/${split}_predictions_metadata.json"
    done
  done
fi

model_names=()
if ! fresh_bool_enabled "${CROSSARCH_RECO_DOMAIN_SKIP_RECONSTRUCTOR_PREFLIGHT}"; then
  for reco_architecture in "${reco_args[@]}"; do
    for teacher_architecture in "${teacher_args[@]}"; do
      fresh_require_file "${CROSSARCH_RECO_MODEL_DIR}/${reco_architecture}/${teacher_architecture}/best_model_val.pt"
    done
  done
fi

for reco_architecture in "${reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    model_name="$(fresh_crossarch_reco_domain_tagger_model_name "${reco_architecture}" "${teacher_architecture}")"
    model_names+=("${model_name}")
    fresh_refuse_existing_dir "${CROSSARCH_RECO_DOMAIN_TAGGER_DIR}/${reco_architecture}/${teacher_architecture}"
    fresh_refuse_existing_dir "${CROSSARCH_PREDICTION_DIR}/${model_name}"
    fresh_refuse_existing_dir "${CROSSARCH_RECO_DOMAIN_TAGGER_PREDICTION_RUN_DIR}/${model_name}"
  done
done
fresh_refuse_existing_dir "${CROSSARCH_RECO_DOMAIN_FUSION_DIR}"

train_prefix_args=()
if [[ -n "${CROSSARCH_RECO_DOMAIN_TAGGER_TRAIN_DEPENDENCY}" ]]; then
  train_prefix_args=(--dependency="afterok:${CROSSARCH_RECO_DOMAIN_TAGGER_TRAIN_DEPENDENCY}")
fi

train_job_ids=()
predict_job_ids=()
declare -A train_job_id_by_model=()
for reco_architecture in "${reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    model_name="$(fresh_crossarch_reco_domain_tagger_model_name "${reco_architecture}" "${teacher_architecture}")"
    train_jid="$(submit_job "crossarch_adapt_train_${model_name}" \
      "${train_prefix_args[@]}" \
      "${SCRIPT_DIR}/run_crossarch_train_reco_domain_tagger.sh" \
      "${reco_architecture}" \
      "${teacher_architecture}")"
    train_job_ids+=("${train_jid}")
    train_job_id_by_model["${model_name}"]="${train_jid}"
    echo "submitted crossarch_adapt_train_${model_name}=${train_jid}"
  done
done

for reco_architecture in "${reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    model_name="$(fresh_crossarch_reco_domain_tagger_model_name "${reco_architecture}" "${teacher_architecture}")"
    train_jid="${train_job_id_by_model[$model_name]}"
    predict_jid="$(submit_job "crossarch_adapt_predict_${model_name}" \
      --dependency="afterok:${train_jid}" \
      "${SCRIPT_DIR}/run_crossarch_predict_reco_domain_tagger.sh" \
      "${reco_architecture}" \
      "${teacher_architecture}")"
    predict_job_ids+=("${predict_jid}")
    echo "submitted crossarch_adapt_predict_${model_name}=${predict_jid}"
  done
done

prediction_dep="$(fresh_join_by_colon "${predict_job_ids[@]}")"
fusion_jid="$(submit_job "crossarch_adapt_fusion" \
  --dependency="afterok:${prediction_dep}" \
  "${SCRIPT_DIR}/run_crossarch_fusion_reco_domain_taggers.sh")"

cat <<SUMMARY
crossarch_reco_domain_tagger_submission:
  train_job_ids: $(fresh_join_by_space "${train_job_ids[@]}")
  predict_job_ids: $(fresh_join_by_space "${predict_job_ids[@]}")
  fusion_job_id: ${fusion_jid}
  dependency_summary:
    train_afterok_extra: ${CROSSARCH_RECO_DOMAIN_TAGGER_TRAIN_DEPENDENCY:-none}
    each_predict_after_its_train: true
    fusion_afterok: ${prediction_dep}
  expected_jobs:
    train: 16
    predict: 16
    fusion: 1
    total_submitted: ${submit_count}
  expected_sources:
    adapted_taggers: $(fresh_join_by_space "${model_names[@]}")
    hlt4_added_in_fusion: hlt_part hlt_pn hlt_pfn hlt_pcnn
    largest_fusion_group: adapted_all16_plus_hlt4
    largest_fusion_group_models: 20
  output_dirs:
    root: ${CROSSARCH_ROOT}
    adapted_taggers: ${CROSSARCH_RECO_DOMAIN_TAGGER_DIR}
    predictions: ${CROSSARCH_PREDICTION_DIR}
    prediction_runs: ${CROSSARCH_RECO_DOMAIN_TAGGER_PREDICTION_RUN_DIR}
    fusion: ${CROSSARCH_RECO_DOMAIN_FUSION_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
