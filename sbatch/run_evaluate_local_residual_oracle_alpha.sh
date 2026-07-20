#!/usr/bin/env bash
# Evaluate one fixed oracle consumer over the first-stage alpha ladder.

#SBATCH --job-name=lprf_alpha
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=08:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"

CONSUMER_ID="${1:?Usage: sbatch run_evaluate_local_residual_oracle_alpha.sh <Ofull|Orobust_light>}"
case "${CONSUMER_ID}" in
  Ofull) RUN_ID=D_alpha_eval_Ofull ;;
  Orobust_light) RUN_ID=D_alpha_eval_Orobust ;;
  *) echo "alpha evaluation allows only Ofull or Orobust_light" >&2; exit 2 ;;
esac

: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/first_stage_pilot}"
: "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/split_manifest.json.gz}"
: "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/hlt_cache}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/targets}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/taggers}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_DIAGNOSTICS_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/oracle_diagnostics}"
: "${LOCAL_RESIDUAL_FIELD_A0_REPORT:=${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/A0/run_report.json}"
: "${LOCAL_RESIDUAL_FIELD_ALPHA_VALUES:=0.0 0.10 0.25 0.50 0.75 1.0}"
: "${LOCAL_RESIDUAL_FIELD_ALPHA_EVAL_BATCH_SIZE:=128}"
: "${LOCAL_RESIDUAL_FIELD_ALPHA_EVAL_NUM_WORKERS:=0}"
: "${LOCAL_RESIDUAL_FIELD_ALPHA_EVAL_MAX_JETS:=}"
: "${LOCAL_RESIDUAL_FIELD_ALPHA_EVAL_DISABLE_AMP:=0}"
: "${LOCAL_RESIDUAL_FIELD_NO_VERIFY_HASH:=0}"

CHECKPOINT="${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${CONSUMER_ID}/best_model_val.pt"
OUTPUT_DIR="${LOCAL_RESIDUAL_FIELD_ORACLE_DIAGNOSTICS_ROOT}/${RUN_ID}"

fresh_setup "$@"
fresh_require_file "${CHECKPOINT}"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_A0_REPORT}"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
fresh_claim_new_dir "${OUTPUT_DIR}"
fresh_split_words alpha_args "${LOCAL_RESIDUAL_FIELD_ALPHA_VALUES}"

cmd=(
  "${PYTHON_BIN}" -u scripts/evaluate_local_residual_oracle_alpha_curve.py
  --checkpoint "${CHECKPOINT}"
  --consumer-id "${CONSUMER_ID}"
  --output-dir "${OUTPUT_DIR}"
  --hlt-cache-dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
  --target-cache-dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
  --manifest-path "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
  --baseline-report "${LOCAL_RESIDUAL_FIELD_A0_REPORT}"
  --alphas "${alpha_args[@]}"
  --splits model_val stack_val
  --batch-size "${LOCAL_RESIDUAL_FIELD_ALPHA_EVAL_BATCH_SIZE}"
  --num-workers "${LOCAL_RESIDUAL_FIELD_ALPHA_EVAL_NUM_WORKERS}"
  --device "${DEVICE}"
)
fresh_append_flag_if_enabled cmd --disable-amp "${LOCAL_RESIDUAL_FIELD_ALPHA_EVAL_DISABLE_AMP}"
fresh_append_flag_if_enabled cmd --no-verify-hash "${LOCAL_RESIDUAL_FIELD_NO_VERIFY_HASH}"
if [[ -n "${LOCAL_RESIDUAL_FIELD_ALPHA_EVAL_MAX_JETS}" ]]; then
  cmd+=(--max-jets "${LOCAL_RESIDUAL_FIELD_ALPHA_EVAL_MAX_JETS}")
fi
fresh_write_run_config "${OUTPUT_DIR}" "${RUN_ID}" "${cmd[@]}"
fresh_run "${cmd[@]}"
if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/alpha_curve.csv"
fi
