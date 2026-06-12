#!/usr/bin/env bash
# Train one reco-domain tagger behind a frozen crossarch reconstructor.

#SBATCH --job-name=crossarch_adapt_train
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=2-00:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

RECO_ARCHITECTURE="${1:?Usage: sbatch run_crossarch_train_reco_domain_tagger.sh <gt|pn|pfn|pcnn> <part|pn|pfn|pcnn>}"
TEACHER_ARCHITECTURE="${2:?Usage: sbatch run_crossarch_train_reco_domain_tagger.sh <gt|pn|pfn|pcnn> <part|pn|pfn|pcnn>}"

MODEL_NAME="$(fresh_crossarch_reco_domain_tagger_model_name "${RECO_ARCHITECTURE}" "${TEACHER_ARCHITECTURE}")"
RECONSTRUCTOR_CHECKPOINT="${CROSSARCH_RECO_MODEL_DIR}/${RECO_ARCHITECTURE}/${TEACHER_ARCHITECTURE}/best_model_val.pt"
OUTPUT_DIR="${CROSSARCH_RECO_DOMAIN_TAGGER_DIR}/${RECO_ARCHITECTURE}/${TEACHER_ARCHITECTURE}"

: "${NO_AMP:=0}"
: "${COMPILE_MODEL:=0}"
: "${MAX_CONSTITS:=128}"
: "${TEACHER_WEIGHT_THRESHOLD:=0.0}"
: "${NON_STRICT_RECONSTRUCTOR_CHECKPOINT:=0}"

fresh_setup "$@"
fresh_require_file "scripts/train_crossarch_reco_domain_tagger.py"
fresh_require_file "${CROSSARCH_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${CROSSARCH_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${RECONSTRUCTOR_CHECKPOINT}"
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_crossarch_reco_domain_tagger.py"
  --reco-architecture "${RECO_ARCHITECTURE}"
  --teacher-architecture "${TEACHER_ARCHITECTURE}"
  --reconstructor-checkpoint "${RECONSTRUCTOR_CHECKPOINT}"
  --cache-dir "${CROSSARCH_HLT_CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${CROSSARCH_RECO_DOMAIN_TAGGER_SEED}"
  --batch-size "${CROSSARCH_RECO_DOMAIN_TAGGER_BATCH_SIZE}"
  --epochs "${CROSSARCH_RECO_DOMAIN_TAGGER_EPOCHS}"
  --lr "${CROSSARCH_RECO_DOMAIN_TAGGER_LR}"
  --weight-decay "${CROSSARCH_RECO_DOMAIN_TAGGER_WEIGHT_DECAY}"
  --num-workers "${CROSSARCH_RECO_DOMAIN_TAGGER_NUM_WORKERS}"
  --device "${CROSSARCH_RECO_DOMAIN_TAGGER_DEVICE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --early-stop-patience "${CROSSARCH_RECO_DOMAIN_TAGGER_EARLY_STOP_PATIENCE}"
  --max-train-jets "${CROSSARCH_RECO_DOMAIN_TAGGER_MAX_TRAIN_JETS}"
  --max-val-jets "${CROSSARCH_RECO_DOMAIN_TAGGER_MAX_VAL_JETS}"
  --model-size "${CROSSARCH_RECO_DOMAIN_TAGGER_MODEL_SIZE}"
  --max-constits "${MAX_CONSTITS}"
  --teacher-weight-threshold "${TEACHER_WEIGHT_THRESHOLD}"
)
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --non-strict-reconstructor-checkpoint "${NON_STRICT_RECONSTRUCTOR_CHECKPOINT}"
fresh_append_optional_arg cmd --max-train-batches "${CROSSARCH_RECO_DOMAIN_TAGGER_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${CROSSARCH_RECO_DOMAIN_TAGGER_MAX_VAL_BATCHES}"

fresh_write_run_config "${OUTPUT_DIR}" "crossarch_reco_domain_tagger_train_${MODEL_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
fi
