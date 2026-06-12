#!/usr/bin/env bash
# Fit fusion groups for reco-domain adapted taggers plus direct HLT baselines.

#SBATCH --job-name=crossarch_adapt_fuse
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

: "${CROSSARCH_RECO_DOMAIN_REQUIRE_FUSION_OK:=0}"

fresh_setup "$@"
fresh_require_file "scripts/run_crossarch_fusion.py"

fresh_split_words reco_args "${CROSSARCH_RECO_ARCHITECTURES}"
fresh_split_words teacher_args "${CROSSARCH_RECO_TEACHERS}"
fresh_split_words hlt_arch_args "${CROSSARCH_HLT_BASELINE_ARCHITECTURES}"
fresh_split_words split_args "${CROSSARCH_RECO_DOMAIN_TAGGER_PREDICT_SPLITS}"
fresh_split_words feature_mode_args "${CROSSARCH_FUSION_FEATURE_MODES}"
fresh_split_words fuser_args "${CROSSARCH_FUSERS}"
fresh_split_words control_feature_mode_args "${CROSSARCH_FUSION_CONTROL_FEATURE_MODES}"

hlt_names=()
adapted_all16=()
adapted_cross12=()
adapted_part_teacher4=()
adapted_pn_teacher4=()
adapted_mixed4=()

for architecture in "${hlt_arch_args[@]}"; do
  hlt_names+=("$(fresh_crossarch_hlt_model_name "${architecture}")")
done

for reco_architecture in "${reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    model_name="$(fresh_crossarch_reco_domain_tagger_model_name "${reco_architecture}" "${teacher_architecture}")"
    adapted_all16+=("${model_name}")
    if ! {
      [[ "${reco_architecture}" == "gt" && "${teacher_architecture}" == "part" ]] ||
      [[ "${reco_architecture}" == "pn" && "${teacher_architecture}" == "pn" ]] ||
      [[ "${reco_architecture}" == "pfn" && "${teacher_architecture}" == "pfn" ]] ||
      [[ "${reco_architecture}" == "pcnn" && "${teacher_architecture}" == "pcnn" ]]
    }; then
      adapted_cross12+=("${model_name}")
    fi
    if [[ "${teacher_architecture}" == "part" ]]; then
      adapted_part_teacher4+=("${model_name}")
    fi
    if [[ "${teacher_architecture}" == "pn" ]]; then
      adapted_pn_teacher4+=("${model_name}")
    fi
  done
done
adapted_mixed4+=("$(fresh_crossarch_reco_domain_tagger_model_name gt pn)")
adapted_mixed4+=("$(fresh_crossarch_reco_domain_tagger_model_name pn pfn)")
adapted_mixed4+=("$(fresh_crossarch_reco_domain_tagger_model_name pfn pcnn)")
adapted_mixed4+=("$(fresh_crossarch_reco_domain_tagger_model_name pcnn part)")

adapted_all16_plus_hlt4=("${adapted_all16[@]}" "${hlt_names[@]}")
adapted_cross12_plus_hlt4=("${adapted_cross12[@]}" "${hlt_names[@]}")
adapted_part_teacher4_plus_hlt4=("${adapted_part_teacher4[@]}" "${hlt_names[@]}")
adapted_pn_teacher4_plus_hlt4=("${adapted_pn_teacher4[@]}" "${hlt_names[@]}")
adapted_mixed4_plus_hlt4=("${adapted_mixed4[@]}" "${hlt_names[@]}")

if ! fresh_is_dry_run; then
  for model_name in "${hlt_names[@]}" "${adapted_all16[@]}"; do
    for split in "${split_args[@]}"; do
      fresh_require_file "${CROSSARCH_PREDICTION_DIR}/${model_name}/${split}_predictions.npz"
      fresh_require_file "${CROSSARCH_PREDICTION_DIR}/${model_name}/${split}_predictions_metadata.json"
    done
  done
fi

fresh_claim_new_dir "${CROSSARCH_RECO_DOMAIN_FUSION_DIR}"

group_args=(
  --group "hlt4:$(fresh_join_by_comma "${hlt_names[@]}")"
  --group "adapted_all16:$(fresh_join_by_comma "${adapted_all16[@]}")"
  --group "adapted_all16_plus_hlt4:$(fresh_join_by_comma "${adapted_all16_plus_hlt4[@]}")"
  --group "adapted_cross12_plus_hlt4:$(fresh_join_by_comma "${adapted_cross12_plus_hlt4[@]}")"
  --group "adapted_part_teacher4_plus_hlt4:$(fresh_join_by_comma "${adapted_part_teacher4_plus_hlt4[@]}")"
  --group "adapted_pn_teacher4_plus_hlt4:$(fresh_join_by_comma "${adapted_pn_teacher4_plus_hlt4[@]}")"
  --group "adapted_mixed4_plus_hlt4:$(fresh_join_by_comma "${adapted_mixed4_plus_hlt4[@]}")"
)

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_crossarch_fusion.py"
  --fit-fusers
  --prediction-dir "${CROSSARCH_PREDICTION_DIR}"
  --output-dir "${CROSSARCH_RECO_DOMAIN_FUSION_DIR}"
  --splits stack_train stack_val final_test
  --feature-modes "${feature_mode_args[@]}"
  --fusers "${fuser_args[@]}"
  --max-iter "${CROSSARCH_FUSION_MAX_ITER}"
  --min-bin-train-rows "${CROSSARCH_FUSION_MIN_BIN_TRAIN_ROWS}"
  --control-seed "${CROSSARCH_FUSION_CONTROL_SEED}"
  --control-feature-modes "${control_feature_mode_args[@]}"
  --control-warning-min-accuracy "${CROSSARCH_FUSION_CONTROL_WARNING_MIN_ACCURACY}"
  --control-warning-chance-margin "${CROSSARCH_FUSION_CONTROL_WARNING_CHANCE_MARGIN}"
  --confirm-final-test
  "${group_args[@]}"
)
fresh_append_flag_if_enabled cmd --skip-controls "${CROSSARCH_FUSION_SKIP_CONTROLS}"
if [[ -n "${CROSSARCH_FUSION_C_GRID}" ]]; then
  fresh_split_words c_grid_args "${CROSSARCH_FUSION_C_GRID}"
  cmd+=(--c-grid "${c_grid_args[@]}")
fi

fresh_write_run_config "${CROSSARCH_RECO_DOMAIN_FUSION_DIR}" "crossarch_reco_domain_tagger_fusion" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${CROSSARCH_RECO_DOMAIN_FUSION_DIR}/fusion_report.json"
  if fresh_bool_enabled "${CROSSARCH_RECO_DOMAIN_REQUIRE_FUSION_OK}"; then
    fresh_assert_json_ok "${CROSSARCH_RECO_DOMAIN_FUSION_DIR}/fusion_report.json"
  fi
fi
