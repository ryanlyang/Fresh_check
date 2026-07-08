#!/usr/bin/env bash
# Run cached-logit fusion for one deployable HLT-MV model group.

#SBATCH --job-name=hlt_mv_fuse
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=04:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${HLT_MV_PDV3_EXPERIMENT_NAME:=privileged_distill_v3_av10_adapter_fixed_hlt_v2_realistic_s1p0_highdata_20260705_190747}"
: "${HLT_MV_PDV3_ROOT:=${OUTPUT_ROOT}/${HLT_MV_PDV3_EXPERIMENT_NAME}}"
: "${HLT_MV_ROOT:=${HLT_MV_PDV3_ROOT}/hlt_multiview_source_fusion}"
: "${HLT_MV_SOURCE_MODELS_DIR:=${HLT_MV_ROOT}/source_models}"
: "${HLT_MV_RANDOM_HLT_CONTROLS_DIR:=${HLT_MV_ROOT}/hlt_random_seed_controls}"
: "${HLT_MV_PRETRAINED_DUALVIEW_DIR:=${HLT_MV_ROOT}/particle_dualview_pretrained}"
: "${HLT_MV_SCRATCH_DUALVIEW_DIR:=${HLT_MV_ROOT}/particle_dualview_scratch}"
: "${HLT_MV_LOGIT_FUSIONS_DIR:=${HLT_MV_ROOT}/logit_fusions}"
: "${HLT_MV_LOGIT_FUSION_OUTPUT_DIR:=}"
: "${HLT_MV_LOGIT_FUSION_MODEL_SPECS:=}"
: "${HLT_MV_LOGIT_FUSION_SKIP_WEIGHTED_AVERAGE:=0}"
: "${HLT_MV_LOGIT_FUSION_MAX_WEIGHT_STEPS:=30}"
: "${HLT_MV_LOGIT_FUSION_SPLITS:=model_val final_test}"

FUSION_NAME="${1:?fusion name is required, e.g. source_5view}"
OUTPUT_DIR="${HLT_MV_LOGIT_FUSION_OUTPUT_DIR:-${HLT_MV_LOGIT_FUSIONS_DIR}/${FUSION_NAME}}"
model_specs=()

add_spec() {
  local name="$1"
  local prediction_dir="$2"
  model_specs+=("${name}=${prediction_dir}")
}

if [[ -n "${HLT_MV_LOGIT_FUSION_MODEL_SPECS}" ]]; then
  fresh_split_words model_specs "${HLT_MV_LOGIT_FUSION_MODEL_SPECS}"
else
  case "${FUSION_NAME}" in
    source_5view)
      add_spec hlt_part_seed8801 "${HLT_MV_SOURCE_MODELS_DIR}/hlt_part_seed8801/predictions"
      add_spec hlt2_part_s0p10_seed8811 "${HLT_MV_SOURCE_MODELS_DIR}/hlt2_part_s0p10_seed8811/predictions"
      add_spec hlt2_part_s0p20_seed8821 "${HLT_MV_SOURCE_MODELS_DIR}/hlt2_part_s0p20_seed8821/predictions"
      add_spec hlt2_part_s0p35_seed8831 "${HLT_MV_SOURCE_MODELS_DIR}/hlt2_part_s0p35_seed8831/predictions"
      add_spec hlt2_part_s1p00_seed8841 "${HLT_MV_SOURCE_MODELS_DIR}/hlt2_part_s1p00_seed8841/predictions"
      ;;
    hlt_random_4seed)
      add_spec hlt_part_seed9101 "${HLT_MV_RANDOM_HLT_CONTROLS_DIR}/hlt_part_seed9101/predictions"
      add_spec hlt_part_seed9102 "${HLT_MV_RANDOM_HLT_CONTROLS_DIR}/hlt_part_seed9102/predictions"
      add_spec hlt_part_seed9103 "${HLT_MV_RANDOM_HLT_CONTROLS_DIR}/hlt_part_seed9103/predictions"
      add_spec hlt_part_seed9104 "${HLT_MV_RANDOM_HLT_CONTROLS_DIR}/hlt_part_seed9104/predictions"
      ;;
    pretrained_dualview_4model)
      add_spec sdv_hlt_hlt2_s0p10 "${HLT_MV_PRETRAINED_DUALVIEW_DIR}/sdv_hlt_hlt2_s0p10/predictions"
      add_spec sdv_hlt_hlt2_s0p20 "${HLT_MV_PRETRAINED_DUALVIEW_DIR}/sdv_hlt_hlt2_s0p20/predictions"
      add_spec sdv_hlt_hlt2_s0p35 "${HLT_MV_PRETRAINED_DUALVIEW_DIR}/sdv_hlt_hlt2_s0p35/predictions"
      add_spec sdv_hlt_hlt2_s1p00 "${HLT_MV_PRETRAINED_DUALVIEW_DIR}/sdv_hlt_hlt2_s1p00/predictions"
      ;;
    scratch_dualview_4model)
      add_spec sdv_hlt_hlt2_s0p10_scratch "${HLT_MV_SCRATCH_DUALVIEW_DIR}/sdv_hlt_hlt2_s0p10_scratch/predictions"
      add_spec sdv_hlt_hlt2_s0p20_scratch "${HLT_MV_SCRATCH_DUALVIEW_DIR}/sdv_hlt_hlt2_s0p20_scratch/predictions"
      add_spec sdv_hlt_hlt2_s0p35_scratch "${HLT_MV_SCRATCH_DUALVIEW_DIR}/sdv_hlt_hlt2_s0p35_scratch/predictions"
      add_spec sdv_hlt_hlt2_s1p00_scratch "${HLT_MV_SCRATCH_DUALVIEW_DIR}/sdv_hlt_hlt2_s1p00_scratch/predictions"
      ;;
    *)
      echo "Unknown built-in HLT-MV logit fusion: ${FUSION_NAME}" >&2
      echo "Set HLT_MV_LOGIT_FUSION_MODEL_SPECS='model=/prediction_dir ...' for a custom fusion." >&2
      exit 2
      ;;
  esac
fi

fusion_splits=()
fresh_split_words fusion_splits "${HLT_MV_LOGIT_FUSION_SPLITS}"

fresh_setup "$@"
if [[ "${HLT_MV_LOGIT_FUSION_SPLITS}" == *"final_test"* ]] && ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing HLT-MV logit fusion final-test evaluation without CONFIRM_FINAL_TEST=1." >&2
  exit 2
fi
for spec in "${model_specs[@]}"; do
  if [[ "${spec}" != *"="* ]]; then
    echo "Invalid HLT-MV logit fusion model spec: ${spec}" >&2
    exit 2
  fi
  model_name="${spec%%=*}"
  prediction_dir="${spec#*=}"
  fresh_require_dir "${prediction_dir}"
  for split in "${fusion_splits[@]}"; do
    fresh_require_file "${prediction_dir}/${model_name}/${split}_predictions.npz"
    fresh_require_file "${prediction_dir}/${model_name}/${split}_predictions_metadata.json"
  done
done
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_hlt_mv_logit_fusion.py"
  --output-root "${OUTPUT_ROOT}"
  --pdv3-experiment-name "${HLT_MV_PDV3_EXPERIMENT_NAME}"
  --fusion-name "${FUSION_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --max-weight-steps "${HLT_MV_LOGIT_FUSION_MAX_WEIGHT_STEPS}"
  --splits
)
cmd+=("${fusion_splits[@]}")
for spec in "${model_specs[@]}"; do
  cmd+=(--model-spec "${spec}")
done
fresh_append_flag_if_enabled cmd --skip-weighted-average "${HLT_MV_LOGIT_FUSION_SKIP_WEIGHTED_AVERAGE}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"

fresh_write_run_config "${OUTPUT_DIR}" "hlt_mv_logit_fusion_${FUSION_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/fusion_report.json"
  fresh_require_file "${OUTPUT_DIR}/summary.json"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/metric_table.csv"
  fresh_assert_json_ok "${OUTPUT_DIR}/run_report.json"
  for split in "${fusion_splits[@]}"; do
    fresh_require_file "${OUTPUT_DIR}/predictions/${FUSION_NAME}_uniform_logit_average/${split}_predictions.npz"
    fresh_require_file "${OUTPUT_DIR}/predictions/${FUSION_NAME}_uniform_logit_average/${split}_predictions_metadata.json"
    if ! fresh_bool_enabled "${HLT_MV_LOGIT_FUSION_SKIP_WEIGHTED_AVERAGE}"; then
      fresh_require_file "${OUTPUT_DIR}/predictions/${FUSION_NAME}_weighted_logit_average/${split}_predictions.npz"
      fresh_require_file "${OUTPUT_DIR}/predictions/${FUSION_NAME}_weighted_logit_average/${split}_predictions_metadata.json"
    fi
  done
fi
