#!/usr/bin/env bash
# Submit split cross-architecture fusion jobs for frozen-teacher and adapted-tagger branches.

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

fresh_split_words family_args "${CROSSARCH_SPLIT_FUSION_FAMILIES}"
fresh_split_words group_args "${CROSSARCH_SPLIT_FUSION_GROUPS}"
fresh_split_words bundle_args "${CROSSARCH_SPLIT_FUSION_BUNDLES}"

submitter_lock_dir="${CROSSARCH_SPLIT_FUSION_ROOT}/.split_fusion_submission_lock"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "source_status_hash=$(fresh_source_status_hash)"
    echo "split_fusion_root=${CROSSARCH_SPLIT_FUSION_ROOT}"
    echo "families=$(fresh_join_by_space "${family_args[@]}")"
    echo "groups=$(fresh_join_by_space "${group_args[@]}")"
    echo "bundles=$(fresh_join_by_space "${bundle_args[@]}")"
    echo "upstream_dependency=${CROSSARCH_SPLIT_FUSION_DEPENDENCY:-none}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

for family in "${family_args[@]}"; do
  case "${family}" in
    frozen|adapted) ;;
    *)
      echo "Unknown split fusion family in CROSSARCH_SPLIT_FUSION_FAMILIES: ${family}" >&2
      exit 2
      ;;
  esac
done
for bundle in "${bundle_args[@]}"; do
  case "${bundle}" in
    main|gated|controls) ;;
    *)
      echo "Unknown split fusion bundle in CROSSARCH_SPLIT_FUSION_BUNDLES: ${bundle}" >&2
      exit 2
      ;;
  esac
done

job_prefix_args=()
if [[ -n "${CROSSARCH_SPLIT_FUSION_DEPENDENCY}" ]]; then
  job_prefix_args=(--dependency="afterok:${CROSSARCH_SPLIT_FUSION_DEPENDENCY}")
fi

fusion_job_ids=()
for family in "${family_args[@]}"; do
  for group in "${group_args[@]}"; do
    for bundle in "${bundle_args[@]}"; do
      output_dir="${CROSSARCH_SPLIT_FUSION_ROOT}/${family}/${group}/${bundle}"
      fresh_refuse_existing_dir "${output_dir}"
      jid="$(submit_job "crossarch_split_${family}_${group}_${bundle}" \
        "${job_prefix_args[@]}" \
        "${SCRIPT_DIR}/run_crossarch_split_fusion.sh" \
        "${family}" \
        "${group}" \
        "${bundle}")"
      fusion_job_ids+=("${jid}")
      echo "submitted crossarch_split_${family}_${group}_${bundle}=${jid}"
    done
  done
done

summary_jid="skipped"
summary_count=0
if fresh_bool_enabled "${CROSSARCH_SPLIT_FUSION_SUBMIT_SUMMARY}"; then
  fresh_refuse_existing_dir "${CROSSARCH_SPLIT_FUSION_ROOT}/summary"
  fusion_dep="$(fresh_join_by_colon "${fusion_job_ids[@]}")"
  summary_jid="$(submit_job "crossarch_split_fusion_summary" \
    --dependency="afterok:${fusion_dep}" \
    "${SCRIPT_DIR}/run_crossarch_split_fusion_summary.sh")"
  summary_count=1
fi

cat <<SUMMARY
crossarch_split_fusion_submission:
  fusion_job_ids: $(fresh_join_by_space "${fusion_job_ids[@]}")
  summary_job_id: ${summary_jid}
  dependency_summary:
    split_jobs_afterok: ${CROSSARCH_SPLIT_FUSION_DEPENDENCY:-none}
    summary_afterok: $(fresh_join_by_colon "${fusion_job_ids[@]}")
  expected_jobs:
    families: ${#family_args[@]}
    groups: ${#group_args[@]}
    bundles: ${#bundle_args[@]}
    split_fusions: ${#fusion_job_ids[@]}
    summary: ${CROSSARCH_SPLIT_FUSION_SUBMIT_SUMMARY}
    total_submitted: $((${#fusion_job_ids[@]} + summary_count))
  configuration:
    families: $(fresh_join_by_space "${family_args[@]}")
    groups: $(fresh_join_by_space "${group_args[@]}")
    bundles: $(fresh_join_by_space "${bundle_args[@]}")
    main_fusers: ${CROSSARCH_SPLIT_FUSION_MAIN_FUSERS}
    gated_fusers: ${CROSSARCH_SPLIT_FUSION_GATED_FUSERS}
    control_fusers: ${CROSSARCH_SPLIT_FUSION_CONTROL_FUSERS}
    controls_enabled_only_for_bundle: controls
  output_dirs:
    root: ${CROSSARCH_SPLIT_FUSION_ROOT}
    summary: ${CROSSARCH_SPLIT_FUSION_ROOT}/summary
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
