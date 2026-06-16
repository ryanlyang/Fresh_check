#!/usr/bin/env bash
# Fit aggressive crossarch fusion groups over frozen-teacher and adapted sources.

#SBATCH --job-name=crossarch_aggr_fusion
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${CROSSARCH_AGGRESSIVE_REQUIRE_FUSION_OK:=0}"

fresh_setup "$@"
fresh_require_file "scripts/run_crossarch_fusion.py"

fresh_split_words reco_args "${CROSSARCH_AGGRESSIVE_RECO_ARCHITECTURES}"
fresh_split_words teacher_args "${CROSSARCH_AGGRESSIVE_RECO_TEACHERS}"
fresh_split_words conservative_reco_args "${CROSSARCH_RECO_ARCHITECTURES}"
fresh_split_words hlt_arch_args "${CROSSARCH_HLT_BASELINE_ARCHITECTURES}"
fresh_split_words split_args "${CROSSARCH_AGGRESSIVE_RECO_PREDICT_SPLITS}"
fresh_split_words feature_mode_args "${CROSSARCH_FUSION_FEATURE_MODES}"
fresh_split_words fuser_args "${CROSSARCH_FUSERS}"
fresh_split_words control_feature_mode_args "${CROSSARCH_FUSION_CONTROL_FEATURE_MODES}"
fresh_split_words fusion_group_args "${CROSSARCH_AGGRESSIVE_FUSION_GROUPS}"

need_hlt=0
need_aggressive_reco=0
need_aggressive_adapted=0
need_conservative_adapted=0
for group_name in "${fusion_group_args[@]}"; do
  case "${group_name}" in
    hlt4)
      need_hlt=1
      ;;
    aggressive_all16)
      need_aggressive_reco=1
      ;;
    aggressive_all16_plus_hlt4|aggressive_cross12_plus_hlt4|aggressive_part_teacher4_plus_hlt4|aggressive_pn_teacher4_plus_hlt4|aggressive_mixed4_plus_hlt4)
      need_hlt=1
      need_aggressive_reco=1
      ;;
    aggressive_adapted_all16_plus_hlt4)
      need_hlt=1
      need_aggressive_adapted=1
      ;;
    conservative_adapted_all16_plus_hlt4)
      need_hlt=1
      need_conservative_adapted=1
      ;;
    conservative_plus_aggressive_adapted_all32_plus_hlt4)
      need_hlt=1
      need_aggressive_adapted=1
      need_conservative_adapted=1
      ;;
    *)
      echo "Unknown aggressive fusion group requested: ${group_name}" >&2
      exit 2
      ;;
  esac
done

hlt_names=()
aggressive_reco_names=()
aggressive_adapted_names=()
conservative_adapted_names=()

for architecture in "${hlt_arch_args[@]}"; do
  hlt_names+=("$(fresh_crossarch_hlt_model_name "${architecture}")")
done

for reco_architecture in "${reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    aggressive_reco_names+=("$(fresh_crossarch_aggressive_reco_model_name "${reco_architecture}" "${teacher_architecture}")")
    aggressive_adapted_names+=("$(fresh_crossarch_aggressive_reco_domain_tagger_model_name "${reco_architecture}" "${teacher_architecture}")")
  done
done

for reco_architecture in "${conservative_reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    conservative_adapted_names+=("$(fresh_crossarch_reco_domain_tagger_model_name "${reco_architecture}" "${teacher_architecture}")")
  done
done

require_prediction_source() {
  local model_name="$1"
  local split
  for split in "${split_args[@]}"; do
    fresh_require_file "${CROSSARCH_AGGRESSIVE_PREDICTION_DIR}/${model_name}/${split}_predictions.npz"
    fresh_require_file "${CROSSARCH_AGGRESSIVE_PREDICTION_DIR}/${model_name}/${split}_predictions_metadata.json"
  done
}

if ! fresh_is_dry_run; then
  if [[ "${need_hlt}" == "1" ]]; then
    for model_name in "${hlt_names[@]}"; do
      require_prediction_source "${model_name}"
    done
  fi
  if [[ "${need_aggressive_reco}" == "1" ]]; then
    for model_name in "${aggressive_reco_names[@]}"; do
      require_prediction_source "${model_name}"
    done
  fi
  if [[ "${need_aggressive_adapted}" == "1" ]]; then
    for model_name in "${aggressive_adapted_names[@]}"; do
      require_prediction_source "${model_name}"
    done
  fi
  if [[ "${need_conservative_adapted}" == "1" ]]; then
    if ! fresh_bool_enabled "${CROSSARCH_AGGRESSIVE_REQUIRE_CONSERVATIVE_ADAPTED_FOR_FUSION}"; then
      echo "Conservative adapted fusion group requested; preflighting existing conservative adapted predictions." >&2
    fi
    for model_name in "${conservative_adapted_names[@]}"; do
      require_prediction_source "${model_name}"
    done
  fi
fi

fresh_claim_new_dir "${CROSSARCH_AGGRESSIVE_FUSION_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_crossarch_fusion.py"
  --fit-fusers
  --prediction-dir "${CROSSARCH_AGGRESSIVE_PREDICTION_DIR}"
  --output-dir "${CROSSARCH_AGGRESSIVE_FUSION_DIR}"
  --splits "${split_args[@]}"
  --include-optional-groups
  --groups "${fusion_group_args[@]}"
  --feature-modes "${feature_mode_args[@]}"
  --fusers "${fuser_args[@]}"
  --max-iter "${CROSSARCH_FUSION_MAX_ITER}"
  --min-bin-train-rows "${CROSSARCH_FUSION_MIN_BIN_TRAIN_ROWS}"
  --control-seed "${CROSSARCH_FUSION_CONTROL_SEED}"
  --control-feature-modes "${control_feature_mode_args[@]}"
  --control-warning-min-accuracy "${CROSSARCH_FUSION_CONTROL_WARNING_MIN_ACCURACY}"
  --control-warning-chance-margin "${CROSSARCH_FUSION_CONTROL_WARNING_CHANCE_MARGIN}"
  --confirm-final-test
)
fresh_append_flag_if_enabled cmd --skip-controls "${CROSSARCH_FUSION_SKIP_CONTROLS}"
if [[ -n "${CROSSARCH_FUSION_C_GRID}" ]]; then
  fresh_split_words c_grid_args "${CROSSARCH_FUSION_C_GRID}"
  cmd+=(--c-grid "${c_grid_args[@]}")
fi

fresh_write_run_config "${CROSSARCH_AGGRESSIVE_FUSION_DIR}" "crossarch_aggressive_fusion" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${CROSSARCH_AGGRESSIVE_FUSION_DIR}/fusion_report.json"
  if fresh_bool_enabled "${CROSSARCH_AGGRESSIVE_REQUIRE_FUSION_OK}"; then
    fresh_assert_json_ok "${CROSSARCH_AGGRESSIVE_FUSION_DIR}/fusion_report.json"
  fi
fi
