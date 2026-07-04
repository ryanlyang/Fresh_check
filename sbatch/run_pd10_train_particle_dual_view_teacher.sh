#!/usr/bin/env bash
# Train the PD10-V2 particle-level HLT+offline dual-view teacher.

#SBATCH --job-name=pd10_pdv_teacher
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=4-00:00:00
#SBATCH --mem=220G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

OUTPUT_DIR="${PD10_V2_PARTICLE_DUAL_VIEW_TEACHER_DIR}"
HLT_TEACHER_CHECKPOINT="${PD10_TEACHERS_DIR}/hlt_part_teacher_10class/best_model_val.pt"
OFFLINE_TEACHER_CHECKPOINT="${PD10_TEACHERS_DIR}/offline_part_teacher_10class/best_model_val.pt"

fresh_setup "$@"
fresh_require_file "${PD10_MANIFEST_PATH}"
fresh_require_dir "${PD10_HLT_CACHE_DIR}"
fresh_require_file "${PD10_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${PD10_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${HLT_TEACHER_CHECKPOINT}"
fresh_require_file "${OFFLINE_TEACHER_CHECKPOINT}"
fresh_require_data_dir
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_pd10_particle_dual_view_teacher.py"
  --manifest "${PD10_MANIFEST_PATH}"
  --hlt-cache-dir "${PD10_HLT_CACHE_DIR}"
  --data-dir "${PD10_DATA_DIR}"
  --hlt-teacher-checkpoint "${HLT_TEACHER_CHECKPOINT}"
  --offline-teacher-checkpoint "${OFFLINE_TEACHER_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${PD10_V2_PARTICLE_DUAL_VIEW_TEACHER_SEED}"
  --batch-size "${PD10_V2_PARTICLE_DUAL_VIEW_BATCH_SIZE}"
  --eval-batch-size "${PD10_V2_PARTICLE_DUAL_VIEW_EVAL_BATCH_SIZE}"
  --epochs "${PD10_V2_PARTICLE_DUAL_VIEW_EPOCHS}"
  --head-warmup-epochs "${PD10_V2_PARTICLE_DUAL_VIEW_HEAD_WARMUP_EPOCHS}"
  --head-warmup-lr "${PD10_V2_PARTICLE_DUAL_VIEW_HEAD_WARMUP_LR}"
  --branch-lr "${PD10_V2_PARTICLE_DUAL_VIEW_BRANCH_LR}"
  --head-lr "${PD10_V2_PARTICLE_DUAL_VIEW_HEAD_LR}"
  --weight-decay "${PD10_V2_PARTICLE_DUAL_VIEW_WEIGHT_DECAY}"
  --num-workers "${PD10_V2_PARTICLE_DUAL_VIEW_NUM_WORKERS}"
  --device "${PD10_V2_PARTICLE_DUAL_VIEW_DEVICE}"
  --grad-clip-norm "${PD10_V2_PARTICLE_DUAL_VIEW_GRAD_CLIP_NORM}"
  --early-stop-patience "${PD10_V2_PARTICLE_DUAL_VIEW_EARLY_STOP_PATIENCE}"
  --max-train-jets "${PD10_MODEL_TRAIN_SIZE}"
  --max-val-jets "${PD10_MODEL_VAL_SIZE}"
  --model-size "${PD10_V2_PARTICLE_DUAL_VIEW_MODEL_SIZE}"
  --fusion-hidden-dim "${PD10_V2_PARTICLE_DUAL_VIEW_FUSION_HIDDEN_DIM}"
  --representation-dim "${PD10_V2_PARTICLE_DUAL_VIEW_REPRESENTATION_DIM}"
  --dropout "${PD10_V2_PARTICLE_DUAL_VIEW_DROPOUT}"
  --read-chunk-size "${PD10_V2_PARTICLE_DUAL_VIEW_READ_CHUNK_SIZE}"
)
fresh_append_flag_if_enabled cmd --no-amp "${PD10_V2_PARTICLE_DUAL_VIEW_NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${PD10_V2_PARTICLE_DUAL_VIEW_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${PD10_V2_PARTICLE_DUAL_VIEW_VERIFY_LABEL_BRANCHES}"
fresh_append_flag_if_enabled cmd --no-verify-hlt-hash "${PD10_V2_PARTICLE_DUAL_VIEW_NO_VERIFY_HLT_HASH}"
fresh_append_flag_if_enabled cmd --no-branch-init "${PD10_V2_PARTICLE_DUAL_VIEW_NO_BRANCH_INIT}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_optional_arg cmd --max-train-batches "${PD10_V2_PARTICLE_DUAL_VIEW_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${PD10_V2_PARTICLE_DUAL_VIEW_MAX_VAL_BATCHES}"

fresh_write_run_config "${OUTPUT_DIR}" "pd10_v2_particle_dual_view_teacher" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_assert_json_ok "${OUTPUT_DIR}/run_report.json"
fi
