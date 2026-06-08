#!/usr/bin/env bash
# V2 Step 7: train one original-mechanism reco7 variant, Stage A + Stage B.

#SBATCH --job-name=v2_step7_reco
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=12:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${MODEL_SIZE:=base}"
: "${NO_AMP:=1}"
: "${STAGE_A_LR:=0.0003}"
: "${STAGE2_LR:=}"
: "${MAX_TRAIN_BATCHES:=}"
: "${MAX_VAL_BATCHES:=}"
: "${MAX_TRAIN_JETS:=}"
: "${MAX_VAL_JETS:=}"
: "${HLT_BASELINE_REPORT:=${HLT_BASELINE_DIR}/model_val_report.json}"

VARIANT="${1:?Usage: sbatch/run_v2_step7_train_variant.sh VARIANT}"
export VARIANT

case " ${RECO7_VARIANTS} " in
  *" ${VARIANT} "*) ;;
  *)
    echo "Unknown V2 Step 7 variant ${VARIANT}; expected one of: ${RECO7_VARIANTS}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${V2_STEP7_RECO_ROOT}/${VARIANT}"

fresh_setup "$@"
fresh_require_data_dir
fresh_require_file "${MANIFEST_PATH}"
fresh_require_file "${HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${HLT_BASELINE_DIR}/best_model_val.pt"
fresh_require_file "${HLT_BASELINE_REPORT}"
fresh_refuse_existing_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "scripts/train_reco7_variants.py"
  --manifest "${MANIFEST_PATH}"
  --hlt-cache-dir "${HLT_CACHE_DIR}"
  --data-dir "${DATA_DIR}"
  --output-root "${V2_STEP7_RECO_ROOT}"
  --hlt-baseline-report "${HLT_BASELINE_REPORT}"
  --variants "${VARIANT}"
  --stage both
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --stage-a-lr "${STAGE_A_LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --model-size "${MODEL_SIZE}"
)
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP}"
fresh_append_optional_arg cmd --stage2-lr "${STAGE2_LR}"
fresh_append_optional_arg cmd --max-train-batches "${MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-train-jets "${MAX_TRAIN_JETS}"
fresh_append_optional_arg cmd --max-val-jets "${MAX_VAL_JETS}"

fresh_write_run_config "${OUTPUT_DIR}" "v2_step7_train_${VARIANT}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/stage_a/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/stage_a/model_val_reconstruction_report.json"
  fresh_require_file "${OUTPUT_DIR}/stage2_dual_view/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/stage2_dual_view/model_val_report.json"
fi
