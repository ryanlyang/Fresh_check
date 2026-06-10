#!/usr/bin/env bash
# Step 8 helper: train one teacher-logit ParticleNet reconstructor.

#SBATCH --job-name=tlogit_pn_train
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ARCHITECTURE="${1:?Usage: sbatch/run_train_teacher_logit_pn_reco.sh <part|pn|pfn|pcnn>}"
case "${ARCHITECTURE}" in
  part|pn|pfn|pcnn) ;;
  *)
    echo "Unknown teacher-logit PN architecture ${ARCHITECTURE}; expected part pn pfn pcnn" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${TEACHER_LOGIT_PN_RECO_ROOT}/${ARCHITECTURE}"
TEACHER_CHECKPOINT="$(fresh_teacher_logit_pn_teacher_checkpoint "${ARCHITECTURE}")"

fresh_setup "$@"
fresh_require_data_dir
fresh_require_file "${MANIFEST_PATH}"
fresh_require_file "${HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${TEACHER_CHECKPOINT}"
fresh_claim_new_dir "${OUTPUT_DIR}"

fresh_split_words edgeconv_dim_args "${TEACHER_LOGIT_PN_EDGECONV_DIMS}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_teacher_logit_particle_net_reco.py"
  --output-dir "${OUTPUT_DIR}"
  --manifest-path "${MANIFEST_PATH}"
  --hlt-cache-dir "${HLT_CACHE_DIR}"
  --data-dir "${DATA_DIR}"
  --teacher-checkpoint "${TEACHER_CHECKPOINT}"
  --teacher-architecture "${ARCHITECTURE}"
  --seed "${TRAIN_SEED:-1305}"
  --batch-size "${TEACHER_LOGIT_PN_BATCH_SIZE}"
  --epochs "${TEACHER_LOGIT_PN_EPOCHS}"
  --lr "${TEACHER_LOGIT_PN_LR}"
  --weight-decay "${TEACHER_LOGIT_PN_WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --early-stop-patience "${TEACHER_LOGIT_PN_EARLY_STOP_PATIENCE}"
  --edgeconv-dims "${edgeconv_dim_args[@]}"
  --k "${TEACHER_LOGIT_PN_K}"
  --num-extra-candidates "${TEACHER_LOGIT_PN_NUM_EXTRA_CANDIDATES}"
  --dropout "${TEACHER_LOGIT_PN_DROPOUT}"
)
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP:-0}"
fresh_append_optional_arg cmd --max-train-batches "${TEACHER_LOGIT_PN_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${TEACHER_LOGIT_PN_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-train-jets "${TEACHER_LOGIT_PN_MAX_TRAIN_JETS}"
fresh_append_optional_arg cmd --max-val-jets "${TEACHER_LOGIT_PN_MAX_VAL_JETS}"

fresh_write_run_config "${OUTPUT_DIR}" "train_teacher_logit_pn_${ARCHITECTURE}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
fi
