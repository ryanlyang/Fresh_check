#!/usr/bin/env bash
# Cache logits for one local residual-field oracle teacher.

#SBATCH --job-name=lprf_olog
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

TEACHER_ID="${1:?Usage: sbatch run_cache_local_residual_oracle_teacher_logits.sh <O0|Ofull|Orobust_light>}"
case "${TEACHER_ID}" in
  O0|Ofull|Orobust_light) ;;
  *)
    echo "Step 2 first-stage logit caching allows only O0, Ofull, or Orobust_light; got ${TEACHER_ID}" >&2
    exit 2
    ;;
esac

: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field}"
: "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/split_manifest.json.gz}"
: "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/hlt_cache}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/targets}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/taggers}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGITS_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/oracle_teacher_logits}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGIT_SPLITS:=model_train model_val stack_train stack_val}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGIT_BATCH_SIZE:=128}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGIT_NUM_WORKERS:=4}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGIT_MAX_JETS:=}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGIT_DISABLE_AMP:=0}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGIT_ALLOW_PARTIAL_SPLITS:=0}"
: "${LOCAL_RESIDUAL_FIELD_NO_VERIFY_HASH:=0}"

CHECKPOINT="${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${TEACHER_ID}/best_model_val.pt"

fresh_setup "$@"
fresh_require_file "${CHECKPOINT}"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${TEACHER_ID}/teacher_config.json"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
mkdir -p "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGITS_DIR}"
fresh_split_words split_args "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGIT_SPLITS}"

required_splits=(model_train model_val stack_train stack_val)
declare -A allowed_splits=()
declare -A requested_splits=()
for split in "${required_splits[@]}"; do allowed_splits["${split}"]=1; done
for split in "${split_args[@]}"; do
  if [[ "${split}" == "final_test" ]]; then
    echo "Refusing to cache final_test oracle teacher logits for ${TEACHER_ID} in the primary Step 2 path." >&2
    exit 2
  fi
  if [[ -z "${allowed_splits[${split}]:-}" ]]; then
    echo "Unsupported oracle teacher logit split ${split}; allowed: ${required_splits[*]}" >&2
    exit 2
  fi
  if [[ -n "${requested_splits[${split}]:-}" ]]; then
    echo "Duplicate oracle teacher logit split requested: ${split}" >&2
    exit 2
  fi
  requested_splits["${split}"]=1
done
if ! fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGIT_ALLOW_PARTIAL_SPLITS}"; then
  for split in "${required_splits[@]}"; do
    if [[ -z "${requested_splits[${split}]:-}" ]]; then
      echo "Step 2 requires oracle logits for ${split}; set ALLOW_PARTIAL_SPLITS=1 only for recovery jobs." >&2
      exit 2
    fi
  done
fi

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/predict_local_residual_field_tagger.py"
  --checkpoint "${CHECKPOINT}"
  --prediction-dir "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGITS_DIR}"
  --model-name "${TEACHER_ID}"
  --hlt-cache-dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
  --target-cache-dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
  --manifest-path "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
  --splits "${split_args[@]}"
  --batch-size "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGIT_BATCH_SIZE}"
  --num-workers "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGIT_NUM_WORKERS}"
  --device "${DEVICE}"
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --disable-amp "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGIT_DISABLE_AMP}"
fresh_append_flag_if_enabled cmd --no-verify-hash "${LOCAL_RESIDUAL_FIELD_NO_VERIFY_HASH}"
if [[ -n "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGIT_MAX_JETS}" ]]; then
  cmd+=(--max-jets "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGIT_MAX_JETS}")
fi

fresh_write_run_config "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGITS_DIR}/${TEACHER_ID}" "local_residual_oracle_teacher_logits_${TEACHER_ID}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  for split in "${split_args[@]}"; do
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGITS_DIR}/${TEACHER_ID}/${split}_predictions.npz"
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGITS_DIR}/${TEACHER_ID}/${split}_predictions_metadata.json"
    cp -f \
      "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGITS_DIR}/${TEACHER_ID}/${split}_predictions_metadata.json" \
      "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGITS_DIR}/${TEACHER_ID}/${split}_metadata.json"
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGITS_DIR}/${TEACHER_ID}/${split}_metadata.json"
  done
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGITS_DIR}/${TEACHER_ID}/prediction_manifest.json"
  validate_cmd=(
    "${PYTHON_BIN}" "-u" "scripts/validate_local_residual_oracle_teacher_logits.py"
    --teacher-dir "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${TEACHER_ID}"
    --prediction-dir "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGITS_DIR}"
    --teacher-id "${TEACHER_ID}"
    --splits "${split_args[@]}"
  )
  fresh_append_flag_if_enabled \
    validate_cmd --allow-partial-splits "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGIT_ALLOW_PARTIAL_SPLITS}"
  fresh_run "${validate_cmd[@]}"
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGITS_DIR}/${TEACHER_ID}/cache_validation_report.json"
fi
