#!/usr/bin/env bash
# Submit Version A PN/PFN/PCNN residual experts on the 500k QCD/Hgg HLT0.6 split.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${ARCH_RESIDUAL_PART_QCD_HGG_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${ARCH_RESIDUAL_PART_QCD_HGG_ROOT:=${OUTPUT_ROOT}/arch_residual_part_qcd_hgg_binary_hlt0p6_500k_${ARCH_RESIDUAL_PART_QCD_HGG_TAG}}"
: "${ARCH_RESIDUAL_PART_QCD_HGG_INPUT_ROOT:=/home/ryreu/atlas/Fresh_check/checkpoints/multiscale_subjet_part_qcd_hgg_binary_hlt0p6_qcd_hgg_hlt06_500k_full_20260628_194154}"
: "${ARCH_RESIDUAL_PART_QCD_HGG_MANIFEST_PATH:=${ARCH_RESIDUAL_PART_QCD_HGG_INPUT_ROOT}/binary_inputs/split_manifest.json.gz}"
: "${ARCH_RESIDUAL_PART_QCD_HGG_HLT_CACHE_DIR:=${ARCH_RESIDUAL_PART_QCD_HGG_INPUT_ROOT}/binary_inputs/hlt_cache}"
: "${ARCH_RESIDUAL_PART_QCD_HGG_BASELINE_CHECKPOINT:=/home/ryreu/atlas/Fresh_check/checkpoints/multiscale_subjet_part_qcd_hgg_binary_hlt0p6_qcd_hgg_hlt06_500k_full_gradfix2_20260629_031038/taggers/hlt_part_baseline/best_model_val.pt}"
: "${ARCH_RESIDUAL_PART_QCD_HGG_ARCHITECTURES:=pfn pcnn pn}"
: "${ARCH_RESIDUAL_PART_QCD_HGG_MODEL_TRAIN_SIZE:=500000}"
: "${ARCH_RESIDUAL_PART_QCD_HGG_MODEL_VAL_SIZE:=150000}"
: "${ARCH_RESIDUAL_PART_QCD_HGG_STACK_VAL_SIZE:=150000}"
: "${ARCH_RESIDUAL_PART_QCD_HGG_FINAL_TEST_SIZE:=500000}"
: "${ARCH_RESIDUAL_PART_QCD_HGG_EPOCHS:=30}"
: "${ARCH_RESIDUAL_PART_QCD_HGG_TRAIN_TIME:=2-12:00:00}"
: "${ARCH_RESIDUAL_PART_QCD_HGG_TRAIN_MEM:=160G}"
: "${ARCH_RESIDUAL_PART_QCD_HGG_TRAIN_CPUS:=8}"

export ARCH_RESIDUAL_PART_ROOT="${ARCH_RESIDUAL_PART_QCD_HGG_ROOT}"
export ARCH_RESIDUAL_PART_TAGGER_ROOT="${ARCH_RESIDUAL_PART_ROOT}/taggers"
export ARCH_RESIDUAL_PART_MANIFEST_PATH="${ARCH_RESIDUAL_PART_QCD_HGG_MANIFEST_PATH}"
export ARCH_RESIDUAL_PART_HLT_CACHE_DIR="${ARCH_RESIDUAL_PART_QCD_HGG_HLT_CACHE_DIR}"
export ARCH_RESIDUAL_PART_BASELINE_CHECKPOINT="${ARCH_RESIDUAL_PART_QCD_HGG_BASELINE_CHECKPOINT}"
export ARCH_RESIDUAL_PART_MODEL_TRAIN_SIZE="${ARCH_RESIDUAL_PART_QCD_HGG_MODEL_TRAIN_SIZE}"
export ARCH_RESIDUAL_PART_MODEL_VAL_SIZE="${ARCH_RESIDUAL_PART_QCD_HGG_MODEL_VAL_SIZE}"
export ARCH_RESIDUAL_PART_STACK_VAL_SIZE="${ARCH_RESIDUAL_PART_QCD_HGG_STACK_VAL_SIZE}"
export ARCH_RESIDUAL_PART_FINAL_TEST_SIZE="${ARCH_RESIDUAL_PART_QCD_HGG_FINAL_TEST_SIZE}"
export ARCH_RESIDUAL_PART_EPOCHS="${ARCH_RESIDUAL_PART_QCD_HGG_EPOCHS}"
export ARCH_RESIDUAL_PART_EXPECTED_HLT_DEGRADATION_STRENGTH=0.6
export ARCH_RESIDUAL_PART_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH=1
export ARCH_RESIDUAL_PART_CONFIRM_FINAL_TEST=1

fresh_prepare_submitter

submit_job() {
  local label="$1"
  shift
  if fresh_is_dry_run; then
    printf 'DRY_RUN sbatch %s: ' "${label}" >&2
    fresh_print_shell_command sbatch "$@" >&2
    printf '\n' >&2
    printf 'DRYRUN_%s\n' "${label//[^A-Za-z0-9_]/_}"
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

fresh_split_words arch_args "${ARCH_RESIDUAL_PART_QCD_HGG_ARCHITECTURES}"

submitter_lock_dir="${ARCH_RESIDUAL_PART_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${ARCH_RESIDUAL_PART_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  fresh_require_file "${ARCH_RESIDUAL_PART_QCD_HGG_MANIFEST_PATH}"
  fresh_require_file "${ARCH_RESIDUAL_PART_QCD_HGG_BASELINE_CHECKPOINT}"
  for split in model_train model_val stack_val final_test; do
    fresh_require_file "${ARCH_RESIDUAL_PART_QCD_HGG_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
    fresh_require_file "${ARCH_RESIDUAL_PART_QCD_HGG_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
  done
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=QCD_vs_Hgg_arch_residual_part_version_A"
    echo "root=${ARCH_RESIDUAL_PART_ROOT}"
    echo "reused_input_root=${ARCH_RESIDUAL_PART_QCD_HGG_INPUT_ROOT}"
    echo "manifest=${ARCH_RESIDUAL_PART_QCD_HGG_MANIFEST_PATH}"
    echo "hlt_cache=${ARCH_RESIDUAL_PART_QCD_HGG_HLT_CACHE_DIR}"
    echo "baseline_checkpoint=${ARCH_RESIDUAL_PART_QCD_HGG_BASELINE_CHECKPOINT}"
    echo "architectures=$(fresh_join_by_space "${arch_args[@]}")"
    echo "epochs=${ARCH_RESIDUAL_PART_EPOCHS}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

job_ids=()
for arch in "${arch_args[@]}"; do
  mapfile -t train_args < <(
    afterok_args \
      "${UPSTREAM_DEPENDENCY}" \
      --time="${ARCH_RESIDUAL_PART_QCD_HGG_TRAIN_TIME}" \
      --cpus-per-task="${ARCH_RESIDUAL_PART_QCD_HGG_TRAIN_CPUS}" \
      --mem="${ARCH_RESIDUAL_PART_QCD_HGG_TRAIN_MEM}" \
      "${SCRIPT_DIR}/run_train_arch_residual_part.sh" \
      "${arch}"
  )
  jid="$(submit_job "archres_part_${arch}" "${train_args[@]}")"
  job_ids+=("${jid}")
  echo "submitted archres_part_${arch}=${jid}"
done

cat <<SUMMARY
arch_residual_part_qcd_hgg_hlt0p6_500k_submission:
  task: QCD_vs_Hgg_arch_residual_part_version_A
  root: ${ARCH_RESIDUAL_PART_ROOT}
  reused_input_root: ${ARCH_RESIDUAL_PART_QCD_HGG_INPUT_ROOT}
  manifest: ${ARCH_RESIDUAL_PART_QCD_HGG_MANIFEST_PATH}
  hlt_cache: ${ARCH_RESIDUAL_PART_QCD_HGG_HLT_CACHE_DIR}
  baseline_checkpoint: ${ARCH_RESIDUAL_PART_QCD_HGG_BASELINE_CHECKPOINT}
  architectures: $(fresh_join_by_space "${arch_args[@]}")
  train_job_ids: $(fresh_join_by_space "${job_ids[@]}")
SUMMARY
