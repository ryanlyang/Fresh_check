#!/usr/bin/env bash
# Train one named first-stage deployable curriculum student.

#SBATCH --job-name=lprf_curr
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=3-00:00:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"

RUN_ID="${1:?Usage: sbatch run_train_local_residual_field_curriculum_student.sh <P0|P2|P4|P7a|P7b|Q0|Q3>}"
case "${RUN_ID}" in P0|P2|P4|P7a|P7b|Q0|Q3) ;; *) echo "unsupported curriculum pilot run ${RUN_ID}" >&2; exit 2 ;; esac

: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/first_stage_pilot}"
: "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/split_manifest.json.gz}"
: "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/hlt_cache}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/targets}"
: "${LOCAL_RESIDUAL_FIELD_RECON_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/reconstructors}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/taggers}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/curriculum}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGITS_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/oracle_teacher_logits}"
: "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON:=${LOCAL_RESIDUAL_FIELD_ROOT}/selected_consumer.json}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_PREDICTOR_WARM_START:=${LOCAL_RESIDUAL_FIELD_RECON_ROOT}/C0/best_model_val.pt}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_A0_CHECKPOINT:=${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/A0/best_model_val.pt}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_EPOCHS:=12}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_BATCH_SIZE:=24}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_EVAL_BATCH_SIZE:=64}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_NUM_WORKERS:=0}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_GRADIENT_ACCUMULATION_STEPS:=1}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_MAX_TRAIN_JETS:=}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_MAX_VAL_JETS:=}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_MAX_STACK_VAL_JETS:=}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_DISABLE_AMP:=0}"
: "${LOCAL_RESIDUAL_FIELD_NO_VERIFY_HASH:=0}"

OUTPUT_DIR="${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT}/${RUN_ID}"
fresh_setup "$@"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_CURRICULUM_PREDICTOR_WARM_START}"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_CURRICULUM_A0_CHECKPOINT}"
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" -u scripts/train_local_residual_field_curriculum_student.py
  --run-id "${RUN_ID}"
  --output-dir "${OUTPUT_DIR}"
  --hlt-cache-dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
  --target-cache-dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
  --manifest-path "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
  --predictor-warm-start-checkpoint "${LOCAL_RESIDUAL_FIELD_CURRICULUM_PREDICTOR_WARM_START}"
  --student-warm-start-checkpoint "${LOCAL_RESIDUAL_FIELD_CURRICULUM_A0_CHECKPOINT}"
  --epochs "${LOCAL_RESIDUAL_FIELD_CURRICULUM_EPOCHS}"
  --batch-size "${LOCAL_RESIDUAL_FIELD_CURRICULUM_BATCH_SIZE}"
  --eval-batch-size "${LOCAL_RESIDUAL_FIELD_CURRICULUM_EVAL_BATCH_SIZE}"
  --num-workers "${LOCAL_RESIDUAL_FIELD_CURRICULUM_NUM_WORKERS}"
  --gradient-accumulation-steps "${LOCAL_RESIDUAL_FIELD_CURRICULUM_GRADIENT_ACCUMULATION_STEPS}"
  --device "${DEVICE}"
)

if [[ "${RUN_ID}" != P0 ]]; then
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON}"
  consumer_id="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_consumer_id"])' "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON}")"
  case "${consumer_id}" in Ofull|Orobust_light) ;; *) echo "selector returned invalid consumer ${consumer_id}" >&2; exit 2 ;; esac
  cmd+=(--selected-consumer-json "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON}")
  if [[ "${RUN_ID}" == P7b ]]; then
    cmd+=(--student-warm-start-checkpoint "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${consumer_id}/best_model_val.pt")
  fi
  if [[ "${RUN_ID}" != Q0 ]]; then
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${consumer_id}/best_model_val.pt"
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${consumer_id}/teacher_config.json"
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${consumer_id}/run_report.json"
    cmd+=(
      --oracle-teacher-checkpoint "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${consumer_id}/best_model_val.pt"
      --oracle-teacher-config-path "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${consumer_id}/teacher_config.json"
      --oracle-run-report-path "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${consumer_id}/run_report.json"
      --oracle-teacher-logits-dir "${LOCAL_RESIDUAL_FIELD_ORACLE_TEACHER_LOGITS_DIR}/${consumer_id}"
    )
  fi
fi
fresh_append_flag_if_enabled cmd --disable-amp "${LOCAL_RESIDUAL_FIELD_CURRICULUM_DISABLE_AMP}"
fresh_append_flag_if_enabled cmd --no-verify-hash "${LOCAL_RESIDUAL_FIELD_NO_VERIFY_HASH}"
if [[ -n "${LOCAL_RESIDUAL_FIELD_CURRICULUM_MAX_TRAIN_JETS}" ]]; then cmd+=(--max-train-jets "${LOCAL_RESIDUAL_FIELD_CURRICULUM_MAX_TRAIN_JETS}"); fi
if [[ -n "${LOCAL_RESIDUAL_FIELD_CURRICULUM_MAX_VAL_JETS}" ]]; then cmd+=(--max-val-jets "${LOCAL_RESIDUAL_FIELD_CURRICULUM_MAX_VAL_JETS}"); fi
if [[ -n "${LOCAL_RESIDUAL_FIELD_CURRICULUM_MAX_STACK_VAL_JETS}" ]]; then cmd+=(--max-stack-val-jets "${LOCAL_RESIDUAL_FIELD_CURRICULUM_MAX_STACK_VAL_JETS}"); fi
fresh_write_run_config "${OUTPUT_DIR}" "local_residual_curriculum_${RUN_ID}" "${cmd[@]}"
fresh_run "${cmd[@]}"
if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/curriculum_schedule.json"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
fi
