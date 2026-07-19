#!/usr/bin/env bash
# Train one local residual-field augmented ParT tagger/control variant.

#SBATCH --job-name=lprf_tag
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
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

RUN_ID="${1:?Usage: sbatch run_train_local_residual_field_tagger.sh <A/B/D/E/F run_id>}"

: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field}"
: "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/split_manifest.json.gz}"
: "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/hlt_cache}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/targets}"
: "${LOCAL_RESIDUAL_FIELD_RECON_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/reconstructors}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/taggers}"
: "${LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT:=}"
: "${LOCAL_RESIDUAL_FIELD_TEACHER_LOGITS_DIR:=}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_SEED:=20421}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_BATCH_SIZE:=64}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_EVAL_BATCH_SIZE:=128}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_EPOCHS:=45}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_PART_LR:=0.00003}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_RECONSTRUCTOR_LR:=0.0003}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_WEIGHT_DECAY:=0.0001}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_NUM_WORKERS:=4}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_GRAD_CLIP_NORM:=1.0}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_EARLY_STOP_PATIENCE:=6}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_MODEL_SIZE:=base}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_FIELD_DROPOUT:=0.0}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_RESIDUAL_SCALE:=1.0}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_RESIDUAL_CLIP_VALUE:=8.0}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_RECON_LOSS_WEIGHT:=0.10}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_KD_LOSS_WEIGHT:=0.25}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_KD_TEMPERATURE:=2.0}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_MAX_TRAIN_JETS:=}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_MAX_VAL_JETS:=}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_MAX_STACK_VAL_JETS:=}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_MIN_SELECTION_VALID_FRACTION:=0.99}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_SAVE_LAST_CHECKPOINT:=1}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_DISABLE_AMP:=0}"
: "${LOCAL_RESIDUAL_FIELD_CONTROL_SEED:=9173}"
: "${LOCAL_RESIDUAL_FIELD_D6_RECON_RUN_ID:=C5}"
: "${LOCAL_RESIDUAL_FIELD_NO_VERIFY_HASH:=0}"

field_source="zero"
model_size="${LOCAL_RESIDUAL_FIELD_TAGGER_MODEL_SIZE}"
reco_checkpoint=""
baseline_checkpoint=""
require_warm_start=0
reco_loss_weight="0.0"
kd_loss_weight="0.0"
field_subset=()
seed="${LOCAL_RESIDUAL_FIELD_TAGGER_SEED}"

use_c0_checkpoint() { reco_checkpoint="${LOCAL_RESIDUAL_FIELD_RECON_ROOT}/C0/best_model_val.pt"; }
use_recon_checkpoint() {
  local run_id="$1"
  if [[ -z "${run_id}" ]]; then
    echo "reconstructor run ID must be non-empty" >&2
    exit 2
  fi
  reco_checkpoint="${LOCAL_RESIDUAL_FIELD_RECON_ROOT}/${run_id}/best_model_val.pt"
}
use_d6_checkpoint() { use_recon_checkpoint "${LOCAL_RESIDUAL_FIELD_D6_RECON_RUN_ID}"; }
use_baseline_if_available() {
  if [[ -n "${LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT}" ]]; then
    baseline_checkpoint="${LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT}"
  fi
}
require_baseline_checkpoint() {
  if [[ -z "${LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT}" ]]; then
    echo "${RUN_ID} requires LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT for the planned warm-start recipe." >&2
    exit 2
  fi
  baseline_checkpoint="${LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT}"
  require_warm_start=1
}
enable_kd_if_available() {
  if [[ -n "${LOCAL_RESIDUAL_FIELD_TEACHER_LOGITS_DIR}" ]]; then
    kd_loss_weight="${LOCAL_RESIDUAL_FIELD_TAGGER_KD_LOSS_WEIGHT}"
  fi
}
require_kd_logits() {
  if [[ -z "${LOCAL_RESIDUAL_FIELD_TEACHER_LOGITS_DIR}" ]]; then
    echo "${RUN_ID} requires LOCAL_RESIDUAL_FIELD_TEACHER_LOGITS_DIR for the planned KD recipe." >&2
    exit 2
  fi
  kd_loss_weight="${LOCAL_RESIDUAL_FIELD_TAGGER_KD_LOSS_WEIGHT}"
}

case "${RUN_ID}" in
  A0)
    field_source="hlt_only"
    ;;
  A1)
    field_source="hlt_only"
    require_baseline_checkpoint
    ;;
  A2)
    field_source="hlt_only"
    model_size="${LOCAL_RESIDUAL_FIELD_A2_MODEL_SIZE:=large}"
    ;;
  B0)
    field_source="oracle"
    field_subset=("r0p02.*")
    ;;
  B1|B4)
    field_source="oracle"
    ;;
  B2)
    field_source="oracle"
    field_subset=(pt_density)
    ;;
  B3)
    field_source="oracle"
    field_subset=(pt_density multiplicity)
    ;;
  D0)
    field_source="frozen_reconstructor"
    use_c0_checkpoint
    ;;
  D1)
    field_source="frozen_reconstructor"
    use_c0_checkpoint
    require_baseline_checkpoint
    ;;
  D2)
    field_source="joint_reconstructor"
    use_c0_checkpoint
    require_baseline_checkpoint
    reco_loss_weight="${LOCAL_RESIDUAL_FIELD_TAGGER_RECON_LOSS_WEIGHT}"
    ;;
  D3)
    field_source="joint_reconstructor"
    ;;
  D4)
    field_source="joint_reconstructor"
    reco_loss_weight="${LOCAL_RESIDUAL_FIELD_TAGGER_RECON_LOSS_WEIGHT}"
    ;;
  D5|D5_seed*)
    field_source="joint_reconstructor"
    use_c0_checkpoint
    require_baseline_checkpoint
    reco_loss_weight="${LOCAL_RESIDUAL_FIELD_TAGGER_RECON_LOSS_WEIGHT}"
    require_kd_logits
    case "${RUN_ID}" in
      D5_seed1) seed="$((LOCAL_RESIDUAL_FIELD_TAGGER_SEED + 101))" ;;
      D5_seed2) seed="$((LOCAL_RESIDUAL_FIELD_TAGGER_SEED + 202))" ;;
      D5_seed3) seed="$((LOCAL_RESIDUAL_FIELD_TAGGER_SEED + 303))" ;;
    esac
    ;;
  D6)
    field_source="joint_reconstructor"
    use_d6_checkpoint
    require_baseline_checkpoint
    reco_loss_weight="${LOCAL_RESIDUAL_FIELD_TAGGER_RECON_LOSS_WEIGHT}"
    require_kd_logits
    ;;
  E0)
    field_source="joint_reconstructor"; use_c0_checkpoint; require_baseline_checkpoint; reco_loss_weight="${LOCAL_RESIDUAL_FIELD_TAGGER_RECON_LOSS_WEIGHT}"; require_kd_logits; field_subset=(pt_density) ;;
  E1)
    field_source="joint_reconstructor"; use_c0_checkpoint; require_baseline_checkpoint; reco_loss_weight="${LOCAL_RESIDUAL_FIELD_TAGGER_RECON_LOSS_WEIGHT}"; require_kd_logits; field_subset=(centroid) ;;
  E2)
    field_source="joint_reconstructor"; use_c0_checkpoint; require_baseline_checkpoint; reco_loss_weight="${LOCAL_RESIDUAL_FIELD_TAGGER_RECON_LOSS_WEIGHT}"; require_kd_logits; field_subset=(multiplicity) ;;
  E3)
    field_source="joint_reconstructor"; use_c0_checkpoint; require_baseline_checkpoint; reco_loss_weight="${LOCAL_RESIDUAL_FIELD_TAGGER_RECON_LOSS_WEIGHT}"; require_kd_logits; field_subset=(composition) ;;
  E4)
    field_source="joint_reconstructor"; use_c0_checkpoint; require_baseline_checkpoint; reco_loss_weight="${LOCAL_RESIDUAL_FIELD_TAGGER_RECON_LOSS_WEIGHT}"; require_kd_logits; field_subset=(reliability) ;;
  E5)
    field_source="joint_reconstructor"; use_c0_checkpoint; require_baseline_checkpoint; reco_loss_weight="${LOCAL_RESIDUAL_FIELD_TAGGER_RECON_LOSS_WEIGHT}"; require_kd_logits; field_subset=(pt_density multiplicity) ;;
  E6)
    field_source="joint_reconstructor"; use_c0_checkpoint; require_baseline_checkpoint; reco_loss_weight="${LOCAL_RESIDUAL_FIELD_TAGGER_RECON_LOSS_WEIGHT}"; require_kd_logits ;;
  F0) field_source="random"; use_baseline_if_available ;;
  F1) field_source="cross_jet_shuffle"; use_baseline_if_available ;;
  F2) field_source="within_jet_shuffle"; use_baseline_if_available ;;
  F3) field_source="radius_permuted"; use_baseline_if_available ;;
  F4) field_source="learned_no_target"; use_baseline_if_available ;;
  F5) field_source="zero"; use_baseline_if_available ;;
  *)
    echo "Unknown local residual-field tagger RUN_ID: ${RUN_ID}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${RUN_ID}"

fresh_setup "$@"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
if [[ "${field_source}" != "hlt_only" ]]; then
  fresh_require_dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
fi
if [[ -n "${reco_checkpoint}" ]]; then fresh_require_file "${reco_checkpoint}"; fi
if [[ -n "${baseline_checkpoint}" ]]; then fresh_require_file "${baseline_checkpoint}"; fi
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_local_residual_field_tagger.py"
  --output-dir "${OUTPUT_DIR}"
  --hlt-cache-dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
  --target-cache-dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
  --manifest-path "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
  --seed "${seed}"
  --batch-size "${LOCAL_RESIDUAL_FIELD_TAGGER_BATCH_SIZE}"
  --eval-batch-size "${LOCAL_RESIDUAL_FIELD_TAGGER_EVAL_BATCH_SIZE}"
  --epochs "${LOCAL_RESIDUAL_FIELD_TAGGER_EPOCHS}"
  --part-lr "${LOCAL_RESIDUAL_FIELD_TAGGER_PART_LR}"
  --reconstructor-lr "${LOCAL_RESIDUAL_FIELD_TAGGER_RECONSTRUCTOR_LR}"
  --weight-decay "${LOCAL_RESIDUAL_FIELD_TAGGER_WEIGHT_DECAY}"
  --num-workers "${LOCAL_RESIDUAL_FIELD_TAGGER_NUM_WORKERS}"
  --device "${DEVICE}"
  --grad-clip-norm "${LOCAL_RESIDUAL_FIELD_TAGGER_GRAD_CLIP_NORM}"
  --early-stop-patience "${LOCAL_RESIDUAL_FIELD_TAGGER_EARLY_STOP_PATIENCE}"
  --model-size "${model_size}"
  --field-source "${field_source}"
  --residual-field-scale "${LOCAL_RESIDUAL_FIELD_TAGGER_RESIDUAL_SCALE}"
  --residual-field-clip-value "${LOCAL_RESIDUAL_FIELD_TAGGER_RESIDUAL_CLIP_VALUE}"
  --field-dropout "${LOCAL_RESIDUAL_FIELD_TAGGER_FIELD_DROPOUT}"
  --control-seed "${LOCAL_RESIDUAL_FIELD_CONTROL_SEED}"
  --reconstructor-loss-weight "${reco_loss_weight}"
  --kd-loss-weight "${kd_loss_weight}"
  --kd-temperature "${LOCAL_RESIDUAL_FIELD_TAGGER_KD_TEMPERATURE}"
  --min-selection-valid-fraction "${LOCAL_RESIDUAL_FIELD_TAGGER_MIN_SELECTION_VALID_FRACTION}"
)
if [[ -n "${reco_checkpoint}" ]]; then cmd+=(--reconstructor-checkpoint "${reco_checkpoint}"); fi
if [[ -n "${baseline_checkpoint}" ]]; then cmd+=(--baseline-checkpoint "${baseline_checkpoint}"); fi
if [[ "${require_warm_start}" -eq 1 ]]; then cmd+=(--require-baseline-warm-start); fi
if [[ -n "${LOCAL_RESIDUAL_FIELD_TEACHER_LOGITS_DIR}" ]]; then cmd+=(--teacher-logits-dir "${LOCAL_RESIDUAL_FIELD_TEACHER_LOGITS_DIR}"); fi
if ((${#field_subset[@]})); then cmd+=(--field-subset "${field_subset[@]}"); fi
fresh_append_flag_if_enabled cmd --disable-amp "${LOCAL_RESIDUAL_FIELD_TAGGER_DISABLE_AMP}"
fresh_append_flag_if_enabled cmd --no-verify-hash "${LOCAL_RESIDUAL_FIELD_NO_VERIFY_HASH}"
if ! fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_TAGGER_SAVE_LAST_CHECKPOINT}"; then cmd+=(--no-save-last-checkpoint); fi
if [[ -n "${LOCAL_RESIDUAL_FIELD_TAGGER_MAX_TRAIN_JETS}" ]]; then cmd+=(--max-train-jets "${LOCAL_RESIDUAL_FIELD_TAGGER_MAX_TRAIN_JETS}"); fi
if [[ -n "${LOCAL_RESIDUAL_FIELD_TAGGER_MAX_VAL_JETS}" ]]; then cmd+=(--max-val-jets "${LOCAL_RESIDUAL_FIELD_TAGGER_MAX_VAL_JETS}"); fi
if [[ -n "${LOCAL_RESIDUAL_FIELD_TAGGER_MAX_STACK_VAL_JETS}" ]]; then cmd+=(--max-stack-val-jets "${LOCAL_RESIDUAL_FIELD_TAGGER_MAX_STACK_VAL_JETS}"); fi

fresh_write_run_config "${OUTPUT_DIR}" "local_residual_field_tagger_${RUN_ID}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
fi
