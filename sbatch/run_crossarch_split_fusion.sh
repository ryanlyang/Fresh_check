#!/usr/bin/env bash
# Fit one small cross-architecture fusion slice: family x group x fuser bundle.

#SBATCH --job-name=crossarch_split_fuse
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=12:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_setup "$@"
fresh_require_file "scripts/run_crossarch_fusion.py"

FAMILY="${1:?Usage: run_crossarch_split_fusion.sh <frozen|adapted> <group> <main|gated|controls>}"
GROUP="${2:?Usage: run_crossarch_split_fusion.sh <frozen|adapted> <group> <main|gated|controls>}"
BUNDLE="${3:?Usage: run_crossarch_split_fusion.sh <frozen|adapted> <group> <main|gated|controls>}"

fresh_split_words reco_args "${CROSSARCH_RECO_ARCHITECTURES}"
fresh_split_words teacher_args "${CROSSARCH_RECO_TEACHERS}"
fresh_split_words hlt_arch_args "${CROSSARCH_HLT_BASELINE_ARCHITECTURES}"
fresh_split_words split_args "${CROSSARCH_SPLIT_FUSION_SPLITS}"
fresh_split_words feature_mode_args "${CROSSARCH_FUSION_FEATURE_MODES}"
fresh_split_words control_feature_mode_args "${CROSSARCH_FUSION_CONTROL_FEATURE_MODES}"

case "${FAMILY}" in
  frozen|adapted) ;;
  *)
    echo "Unknown split fusion family: ${FAMILY}" >&2
    exit 2
    ;;
esac

case "${BUNDLE}" in
  main)
    fresh_split_words fuser_args "${CROSSARCH_SPLIT_FUSION_MAIN_FUSERS}"
    skip_controls=1
    ;;
  gated)
    fresh_split_words fuser_args "${CROSSARCH_SPLIT_FUSION_GATED_FUSERS}"
    skip_controls=1
    ;;
  controls)
    fresh_split_words fuser_args "${CROSSARCH_SPLIT_FUSION_CONTROL_FUSERS}"
    skip_controls=0
    ;;
  *)
    echo "Unknown split fusion bundle: ${BUNDLE}" >&2
    exit 2
    ;;
esac

hlt_names=()
for architecture in "${hlt_arch_args[@]}"; do
  hlt_names+=("$(fresh_crossarch_hlt_model_name "${architecture}")")
done

all16=()
cross12=()
part_teacher4=()
pn_teacher4=()
mixed4=()

for reco_architecture in "${reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    if [[ "${FAMILY}" == "adapted" ]]; then
      model_name="$(fresh_crossarch_reco_domain_tagger_model_name "${reco_architecture}" "${teacher_architecture}")"
    else
      model_name="$(fresh_crossarch_reco_model_name "${reco_architecture}" "${teacher_architecture}")"
    fi
    all16+=("${model_name}")

    if ! {
      [[ "${reco_architecture}" == "gt" && "${teacher_architecture}" == "part" ]] ||
      [[ "${reco_architecture}" == "pn" && "${teacher_architecture}" == "pn" ]] ||
      [[ "${reco_architecture}" == "pfn" && "${teacher_architecture}" == "pfn" ]] ||
      [[ "${reco_architecture}" == "pcnn" && "${teacher_architecture}" == "pcnn" ]]
    }; then
      cross12+=("${model_name}")
    fi
    if [[ "${teacher_architecture}" == "part" ]]; then
      part_teacher4+=("${model_name}")
    fi
    if [[ "${teacher_architecture}" == "pn" ]]; then
      pn_teacher4+=("${model_name}")
    fi
  done
done

if [[ "${FAMILY}" == "adapted" ]]; then
  mixed4+=("$(fresh_crossarch_reco_domain_tagger_model_name gt pn)")
  mixed4+=("$(fresh_crossarch_reco_domain_tagger_model_name pn pfn)")
  mixed4+=("$(fresh_crossarch_reco_domain_tagger_model_name pfn pcnn)")
  mixed4+=("$(fresh_crossarch_reco_domain_tagger_model_name pcnn part)")
else
  mixed4+=("$(fresh_crossarch_reco_model_name gt pn)")
  mixed4+=("$(fresh_crossarch_reco_model_name pn pfn)")
  mixed4+=("$(fresh_crossarch_reco_model_name pfn pcnn)")
  mixed4+=("$(fresh_crossarch_reco_model_name pcnn part)")
fi

group_models=()
case "${GROUP}" in
  hlt4)
    group_models=("${hlt_names[@]}")
    ;;
  all16)
    group_models=("${all16[@]}")
    ;;
  all16_plus_hlt4)
    group_models=("${all16[@]}" "${hlt_names[@]}")
    ;;
  cross12_plus_hlt4)
    group_models=("${cross12[@]}" "${hlt_names[@]}")
    ;;
  part_teacher4_plus_hlt4)
    group_models=("${part_teacher4[@]}" "${hlt_names[@]}")
    ;;
  pn_teacher4_plus_hlt4)
    group_models=("${pn_teacher4[@]}" "${hlt_names[@]}")
    ;;
  mixed4_plus_hlt4)
    group_models=("${mixed4[@]}" "${hlt_names[@]}")
    ;;
  *)
    echo "Unknown split fusion group: ${GROUP}" >&2
    exit 2
    ;;
esac

if [[ "${#group_models[@]}" -eq 0 ]]; then
  echo "Split fusion group ${FAMILY}/${GROUP} resolved to no models" >&2
  exit 2
fi

if ! fresh_is_dry_run; then
  for model_name in "${group_models[@]}"; do
    for split in "${split_args[@]}"; do
      fresh_require_file "${CROSSARCH_PREDICTION_DIR}/${model_name}/${split}_predictions.npz"
      fresh_require_file "${CROSSARCH_PREDICTION_DIR}/${model_name}/${split}_predictions_metadata.json"
    done
  done
fi

OUTPUT_DIR="${CROSSARCH_SPLIT_FUSION_ROOT}/${FAMILY}/${GROUP}/${BUNDLE}"
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_crossarch_fusion.py"
  --fit-fusers
  --prediction-dir "${CROSSARCH_PREDICTION_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --splits "${split_args[@]}"
  --feature-modes "${feature_mode_args[@]}"
  --fusers "${fuser_args[@]}"
  --max-iter "${CROSSARCH_FUSION_MAX_ITER}"
  --min-bin-train-rows "${CROSSARCH_FUSION_MIN_BIN_TRAIN_ROWS}"
  --control-seed "${CROSSARCH_FUSION_CONTROL_SEED}"
  --control-feature-modes "${control_feature_mode_args[@]}"
  --control-warning-min-accuracy "${CROSSARCH_FUSION_CONTROL_WARNING_MIN_ACCURACY}"
  --control-warning-chance-margin "${CROSSARCH_FUSION_CONTROL_WARNING_CHANCE_MARGIN}"
  --confirm-final-test
  --group "${FAMILY}_${GROUP}:$(fresh_join_by_comma "${group_models[@]}")"
)

if [[ "${skip_controls}" == "1" ]]; then
  cmd+=(--skip-controls)
fi
if [[ -n "${CROSSARCH_FUSION_C_GRID}" ]]; then
  fresh_split_words c_grid_args "${CROSSARCH_FUSION_C_GRID}"
  cmd+=(--c-grid "${c_grid_args[@]}")
fi

fresh_write_run_config "${OUTPUT_DIR}" "crossarch_split_fusion_${FAMILY}_${GROUP}_${BUNDLE}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/fusion_report.json"
  fresh_assert_json_ok "${OUTPUT_DIR}/fusion_report.json"
fi
