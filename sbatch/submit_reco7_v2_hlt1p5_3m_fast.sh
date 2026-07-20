#!/usr/bin/env bash
# Queue a faster high-data reco7 rerun on HLT v2 realistic degradation strength 1.5.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${DEVICE:=cuda}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_prepare_submitter

: "${RECO7_V2_FAST_ROOT:=${OUTPUT_ROOT}/reco7_v2_hlt1p5_3m_fast_$(date +%Y%m%d_%H%M%S)}"
: "${RECO7_V2_FAST_MODEL_TRAIN_SIZE:=3000000}"
: "${RECO7_V2_FAST_MODEL_VAL_SIZE:=500000}"
: "${RECO7_V2_FAST_STACK_TRAIN_SIZE:=500000}"
: "${RECO7_V2_FAST_STACK_VAL_SIZE:=150000}"
: "${RECO7_V2_FAST_FINAL_TEST_SIZE:=500000}"
: "${RECO7_V2_FAST_HLT_PROFILE:=fixed_hlt_v2_realistic}"
: "${RECO7_V2_FAST_HLT_DEGRADATION_STRENGTH:=1.5}"
: "${RECO7_V2_FAST_VARIANTS:=${RECO7_VARIANTS}}"
: "${RECO7_V2_FAST_SBATCH_ACCOUNT:=reu-aisocial}"
: "${RECO7_V2_FAST_SBATCH_PARTITION:=tigris}"
: "${RECO7_V2_FAST_GPU_GRES:=gpu:gh200:1}"

# Faster defaults for the old reco7 path. Stage A is usually much lighter than
# the frozen-reconstructor dual-view Particle Transformer, so it gets the larger
# physical batch. Tigris/GH200 showed fp16 AMP overflows in this older
# reconstructor path, so this runner defaults to fp32 unless explicitly changed.
: "${RECO7_V2_FAST_BASELINE_BATCH_SIZE:=192}"
: "${RECO7_V2_FAST_RECO_BATCH_SIZE:=128}"
: "${RECO7_V2_FAST_STAGE_A_BATCH_SIZE:=256}"
: "${RECO7_V2_FAST_STAGE2_BATCH_SIZE:=128}"
: "${RECO7_V2_FAST_BASELINE_EPOCHS:=16}"
: "${RECO7_V2_FAST_STAGE_A_EPOCHS:=12}"
: "${RECO7_V2_FAST_STAGE2_EPOCHS:=16}"
: "${RECO7_V2_FAST_LR:=0.001}"
: "${RECO7_V2_FAST_STAGE_A_LR:=0.0003}"
: "${RECO7_V2_FAST_STAGE2_LR:=}"
: "${RECO7_V2_FAST_WEIGHT_DECAY:=0.0001}"
: "${RECO7_V2_FAST_EARLY_STOP_PATIENCE:=4}"
: "${RECO7_V2_FAST_NUM_WORKERS:=8}"
: "${RECO7_V2_FAST_NO_AMP:=1}"
: "${RECO7_V2_FAST_COMPILE_MODEL:=0}"
: "${RECO7_V2_FAST_MODEL_SIZE:=base}"
: "${RECO7_V2_FAST_STAGE2_ARCHITECTURE:=cross_attention_fusion}"
: "${RECO7_V2_FAST_READ_CHUNK_SIZE:=50000}"
: "${RECO7_V2_FAST_HLT_SPLITS:=model_train model_val stack_train stack_val final_test}"
: "${RECO7_V2_FAST_TORCH_NATIVE_TRITON:=disable}"
: "${RECO7_V2_FAST_TORCH_NATIVE_TRITON_PROBE:=1}"

: "${RECO7_V2_FAST_SPLIT_TIME:=08:00:00}"
: "${RECO7_V2_FAST_SPLIT_MEM:=64G}"
: "${RECO7_V2_FAST_CACHE_TIME:=2-00:00:00}"
: "${RECO7_V2_FAST_CACHE_MEM:=220G}"
: "${RECO7_V2_FAST_BASELINE_TIME:=3-00:00:00}"
: "${RECO7_V2_FAST_BASELINE_MEM:=128G}"
: "${RECO7_V2_FAST_RECO_TIME:=3-00:00:00}"
: "${RECO7_V2_FAST_RECO_MEM:=160G}"
: "${RECO7_V2_FAST_FUSION_TIME:=1-00:00:00}"
: "${RECO7_V2_FAST_FUSION_MEM:=220G}"
: "${RECO7_V2_FAST_AUDIT_TIME:=08:00:00}"
: "${RECO7_V2_FAST_AUDIT_MEM:=128G}"

manifest_dir="${RECO7_V2_FAST_ROOT}/splits"
manifest_path="${manifest_dir}/split_manifest.json.gz"
hlt_cache_dir="${RECO7_V2_FAST_ROOT}/hlt_cache"
hlt_baseline_dir="${RECO7_V2_FAST_ROOT}/hlt_baseline_seed${HLT_BASELINE_SEED}"
v2_root="${RECO7_V2_FAST_ROOT}/v2_original_mechanism_step7"
reco_root="${v2_root}/reco7"
fusion_dir="${v2_root}/fusion/reco7_plus_hlt"
audit_dir="${v2_root}/audits/reco7_plus_hlt"

submitter_lock_dir="${RECO7_V2_FAST_ROOT}/submission_logs/reco7_v2_fast_$(date +%Y%m%d_%H%M%S)"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "root=${RECO7_V2_FAST_ROOT}"
    echo "hlt_profile=${RECO7_V2_FAST_HLT_PROFILE}"
    echo "hlt_degradation_strength=${RECO7_V2_FAST_HLT_DEGRADATION_STRENGTH}"
    echo "model_train_size=${RECO7_V2_FAST_MODEL_TRAIN_SIZE}"
    echo "model_val_size=${RECO7_V2_FAST_MODEL_VAL_SIZE}"
    echo "stack_train_size=${RECO7_V2_FAST_STACK_TRAIN_SIZE}"
    echo "stack_val_size=${RECO7_V2_FAST_STACK_VAL_SIZE}"
    echo "final_test_size=${RECO7_V2_FAST_FINAL_TEST_SIZE}"
    echo "variants=${RECO7_V2_FAST_VARIANTS}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

print_env_args() {
  local arg
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
}

SUBMITTED_JID=""
submit_sbatch() {
  local label="$1"
  shift
  local -a env_args=()
  while (($#)); do
    if [[ "$1" == "--" ]]; then
      shift
      break
    fi
    env_args+=("$1")
    shift
  done
  local -a cmd=(sbatch --parsable)
  if [[ -n "${RECO7_V2_FAST_SBATCH_ACCOUNT}" ]]; then
    cmd+=(--account="${RECO7_V2_FAST_SBATCH_ACCOUNT}")
  fi
  if [[ -n "${RECO7_V2_FAST_SBATCH_PARTITION}" ]]; then
    cmd+=(--partition="${RECO7_V2_FAST_SBATCH_PARTITION}")
  fi
  cmd+=("$@")
  if fresh_is_dry_run; then
    SUBMITTED_JID="DRYRUN_${label//[^A-Za-z0-9_]/_}"
    printf 'DRY_RUN %s:' "${label}"
    printf ' env'
    print_env_args "${env_args[@]}"
    printf ' '
    fresh_print_shell_command "${cmd[@]}"
    printf '\n'
  else
    SUBMITTED_JID="$(env "${env_args[@]}" "${cmd[@]}")"
  fi
  echo "${label}=${SUBMITTED_JID}"
}

common_env=(
  "PROJECT_DIR=${PROJECT_DIR}"
  "DATA_DIR=${DATA_DIR}"
  "DATA_DIRS=${DATA_DIRS:-}"
  "OUTPUT_ROOT=${OUTPUT_ROOT}"
  "DIAGNOSTICS_ROOT=${DIAGNOSTICS_ROOT}"
  "LOG_DIR=${LOG_DIR}"
  "MIRROR_DIAGNOSTICS=${MIRROR_DIAGNOSTICS}"
  "DIAGNOSTICS_MAX_FILE_MB=${DIAGNOSTICS_MAX_FILE_MB}"
  "CONDA_ENV=${CONDA_ENV}"
  "CONDA_BASE=${CONDA_BASE:-}"
  "PYTHON_BIN=${PYTHON_BIN}"
  "DEVICE=${DEVICE}"
  "OVERWRITE=${OVERWRITE}"
  "CONSTRAINED_C2F_TORCH_NATIVE_TRITON=${RECO7_V2_FAST_TORCH_NATIVE_TRITON}"
  "CONSTRAINED_C2F_TORCH_NATIVE_TRITON_PROBE=${RECO7_V2_FAST_TORCH_NATIVE_TRITON_PROBE}"
  "MANIFEST_PATH=${manifest_path}"
  "HLT_CACHE_DIR=${hlt_cache_dir}"
  "HLT_BASELINE_DIR=${hlt_baseline_dir}"
  "HLT_BASELINE_REPORT=${hlt_baseline_dir}/model_val_report.json"
  "V2_STEP7_ROOT=${v2_root}"
  "V2_STEP7_RECO_ROOT=${reco_root}"
  "V2_STEP7_FUSION_DIR=${fusion_dir}"
  "V2_STEP7_AUDIT_DIR=${audit_dir}"
  "V2_STEP7_VARIANTS=${RECO7_V2_FAST_VARIANTS}"
)

submit_sbatch "reco7v2_splits" \
  "${common_env[@]}" \
  "MODEL_TRAIN_SIZE=${RECO7_V2_FAST_MODEL_TRAIN_SIZE}" \
  "MODEL_VAL_SIZE=${RECO7_V2_FAST_MODEL_VAL_SIZE}" \
  "STACK_TRAIN_SIZE=${RECO7_V2_FAST_STACK_TRAIN_SIZE}" \
  "STACK_VAL_SIZE=${RECO7_V2_FAST_STACK_VAL_SIZE}" \
  "FINAL_TEST_SIZE=${RECO7_V2_FAST_FINAL_TEST_SIZE}" \
  -- --time="${RECO7_V2_FAST_SPLIT_TIME}" --mem="${RECO7_V2_FAST_SPLIT_MEM}" --cpus-per-task=4 \
  "${SCRIPT_DIR}/run_build_fresh_splits.sh"
split_jid="${SUBMITTED_JID}"

submit_sbatch "reco7v2_hlt_cache" \
  "${common_env[@]}" \
  "HLT_PROFILE=${RECO7_V2_FAST_HLT_PROFILE}" \
  "HLT_DEGRADATION_STRENGTH=${RECO7_V2_FAST_HLT_DEGRADATION_STRENGTH}" \
  "HLT_SPLITS=${RECO7_V2_FAST_HLT_SPLITS}" \
  "READ_CHUNK_SIZE=${RECO7_V2_FAST_READ_CHUNK_SIZE}" \
  -- --dependency="afterok:${split_jid}" --time="${RECO7_V2_FAST_CACHE_TIME}" \
  --mem="${RECO7_V2_FAST_CACHE_MEM}" --cpus-per-task="${RECO7_V2_FAST_NUM_WORKERS}" \
  "${SCRIPT_DIR}/run_build_fresh_hlt_cache.sh"
cache_jid="${SUBMITTED_JID}"

submit_sbatch "reco7v2_hlt_baseline" \
  "${common_env[@]}" \
  "BATCH_SIZE=${RECO7_V2_FAST_BASELINE_BATCH_SIZE}" \
  "EPOCHS=${RECO7_V2_FAST_BASELINE_EPOCHS}" \
  "LR=${RECO7_V2_FAST_LR}" \
  "WEIGHT_DECAY=${RECO7_V2_FAST_WEIGHT_DECAY}" \
  "EARLY_STOP_PATIENCE=${RECO7_V2_FAST_EARLY_STOP_PATIENCE}" \
  "NUM_WORKERS=${RECO7_V2_FAST_NUM_WORKERS}" \
  "NO_AMP=${RECO7_V2_FAST_NO_AMP}" \
  "COMPILE_MODEL=${RECO7_V2_FAST_COMPILE_MODEL}" \
  "MODEL_SIZE=${RECO7_V2_FAST_MODEL_SIZE}" \
  -- --dependency="afterok:${cache_jid}" --time="${RECO7_V2_FAST_BASELINE_TIME}" \
  --gres="${RECO7_V2_FAST_GPU_GRES}" \
  --mem="${RECO7_V2_FAST_BASELINE_MEM}" --cpus-per-task="${RECO7_V2_FAST_NUM_WORKERS}" \
  "${SCRIPT_DIR}/run_train_fresh_hlt_baseline.sh"
baseline_jid="${SUBMITTED_JID}"

fresh_split_words variant_args "${RECO7_V2_FAST_VARIANTS}"
reco_jids=()
for variant in "${variant_args[@]}"; do
  submit_sbatch "reco7v2_${variant}" \
    "${common_env[@]}" \
    "BATCH_SIZE=${RECO7_V2_FAST_RECO_BATCH_SIZE}" \
    "EPOCHS=${RECO7_V2_FAST_STAGE2_EPOCHS}" \
    "LR=${RECO7_V2_FAST_LR}" \
    "STAGE_A_LR=${RECO7_V2_FAST_STAGE_A_LR}" \
    "STAGE2_LR=${RECO7_V2_FAST_STAGE2_LR}" \
    "STAGE_A_BATCH_SIZE=${RECO7_V2_FAST_STAGE_A_BATCH_SIZE}" \
    "STAGE2_BATCH_SIZE=${RECO7_V2_FAST_STAGE2_BATCH_SIZE}" \
    "STAGE_A_EPOCHS=${RECO7_V2_FAST_STAGE_A_EPOCHS}" \
    "STAGE2_EPOCHS=${RECO7_V2_FAST_STAGE2_EPOCHS}" \
    "WEIGHT_DECAY=${RECO7_V2_FAST_WEIGHT_DECAY}" \
    "EARLY_STOP_PATIENCE=${RECO7_V2_FAST_EARLY_STOP_PATIENCE}" \
    "NUM_WORKERS=${RECO7_V2_FAST_NUM_WORKERS}" \
    "NO_AMP=${RECO7_V2_FAST_NO_AMP}" \
    "MODEL_SIZE=${RECO7_V2_FAST_MODEL_SIZE}" \
    "STAGE2_ARCHITECTURE=${RECO7_V2_FAST_STAGE2_ARCHITECTURE}" \
    -- --dependency="afterok:${baseline_jid}" --time="${RECO7_V2_FAST_RECO_TIME}" \
    --gres="${RECO7_V2_FAST_GPU_GRES}" \
    --mem="${RECO7_V2_FAST_RECO_MEM}" --cpus-per-task="${RECO7_V2_FAST_NUM_WORKERS}" \
    "${SCRIPT_DIR}/run_v2_step7_train_variant.sh" "${variant}"
  reco_jids+=("${SUBMITTED_JID}")
done

fusion_dep="$(fresh_join_by_colon "${baseline_jid}" "${reco_jids[@]}")"
submit_sbatch "reco7v2_fusion" \
  "${common_env[@]}" \
  "FUSION_BATCH_SIZE=192" \
  "FUSION_NUM_WORKERS=${RECO7_V2_FAST_NUM_WORKERS}" \
  "FUSION_DEVICE=${DEVICE}" \
  "CONFIRM_FINAL_TEST=1" \
  -- --dependency="afterok:${fusion_dep}" --time="${RECO7_V2_FAST_FUSION_TIME}" \
  --gres="${RECO7_V2_FAST_GPU_GRES}" \
  --mem="${RECO7_V2_FAST_FUSION_MEM}" --cpus-per-task="${RECO7_V2_FAST_NUM_WORKERS}" \
  "${SCRIPT_DIR}/run_v2_step10_fuse_reco7_plus_hlt.sh"
fusion_jid="${SUBMITTED_JID}"

submit_sbatch "reco7v2_audit" \
  "${common_env[@]}" \
  -- --dependency="afterok:${fusion_jid}" --time="${RECO7_V2_FAST_AUDIT_TIME}" \
  --mem="${RECO7_V2_FAST_AUDIT_MEM}" --cpus-per-task="${RECO7_V2_FAST_NUM_WORKERS}" \
  "${SCRIPT_DIR}/run_v2_step11_audit_reco7_plus_hlt.sh"
audit_jid="${SUBMITTED_JID}"

cat <<SUMMARY
reco7_v2_hlt1p5_3m_fast_submission:
  root: ${RECO7_V2_FAST_ROOT}
  hlt_profile: ${RECO7_V2_FAST_HLT_PROFILE}
  hlt_degradation_strength: ${RECO7_V2_FAST_HLT_DEGRADATION_STRENGTH}
  sbatch_account: ${RECO7_V2_FAST_SBATCH_ACCOUNT}
  sbatch_partition: ${RECO7_V2_FAST_SBATCH_PARTITION}
  gpu_gres: ${RECO7_V2_FAST_GPU_GRES}
  split_job: ${split_jid}
  hlt_cache_job: ${cache_jid}
  hlt_baseline_job: ${baseline_jid}
  reco7_variant_jobs: $(fresh_join_by_space "${reco_jids[@]}")
  fusion_job: ${fusion_jid}
  audit_job: ${audit_jid}
  sizes:
    model_train: ${RECO7_V2_FAST_MODEL_TRAIN_SIZE}
    model_val: ${RECO7_V2_FAST_MODEL_VAL_SIZE}
    stack_train: ${RECO7_V2_FAST_STACK_TRAIN_SIZE}
    stack_val: ${RECO7_V2_FAST_STACK_VAL_SIZE}
    final_test: ${RECO7_V2_FAST_FINAL_TEST_SIZE}
  speed_knobs:
    no_amp: ${RECO7_V2_FAST_NO_AMP}
    stage2_architecture: ${RECO7_V2_FAST_STAGE2_ARCHITECTURE}
    torch_native_triton: ${RECO7_V2_FAST_TORCH_NATIVE_TRITON}
    torch_native_triton_probe: ${RECO7_V2_FAST_TORCH_NATIVE_TRITON_PROBE}
    baseline_batch_size: ${RECO7_V2_FAST_BASELINE_BATCH_SIZE}
    stage_a_batch_size: ${RECO7_V2_FAST_STAGE_A_BATCH_SIZE}
    stage2_batch_size: ${RECO7_V2_FAST_STAGE2_BATCH_SIZE}
    baseline_epochs: ${RECO7_V2_FAST_BASELINE_EPOCHS}
    stage_a_epochs: ${RECO7_V2_FAST_STAGE_A_EPOCHS}
    stage2_epochs: ${RECO7_V2_FAST_STAGE2_EPOCHS}
  output_dirs:
    manifest: ${manifest_path}
    hlt_cache: ${hlt_cache_dir}
    hlt_baseline: ${hlt_baseline_dir}
    reco7_root: ${reco_root}
    fusion: ${fusion_dir}
    audit: ${audit_dir}
SUMMARY
