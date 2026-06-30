#!/usr/bin/env bash
# Run frozen-score fusion over completed local-graph and multi-scale subjet checkpoints.

#SBATCH --job-name=lg_ms_fuse
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=10:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${LOCAL_GRAPH_MULTI_FUSION_LOCAL_ROOT:=${OUTPUT_ROOT}/local_graph_part_qcd_hgg_hlt0p6_3m1m1m_20260629_015555}"
: "${LOCAL_GRAPH_MULTI_FUSION_LOCAL_TAGGER_ROOT:=${LOCAL_GRAPH_MULTI_FUSION_LOCAL_ROOT}/taggers}"
: "${LOCAL_GRAPH_MULTI_FUSION_MULTISCALE_ROOT:=${OUTPUT_ROOT}/multiscale_subjet_part_qcd_hgg_binary_hlt0p6_qcd_hgg_hlt06_3m1m1m_full_gradfix2_20260629_031038}"
: "${LOCAL_GRAPH_MULTI_FUSION_MULTISCALE_TAGGER_ROOT:=${LOCAL_GRAPH_MULTI_FUSION_MULTISCALE_ROOT}/taggers}"
: "${LOCAL_GRAPH_MULTI_FUSION_HLT_CACHE_DIR:=${OUTPUT_ROOT}/multiscale_subjet_part_qcd_hgg_binary_hlt0p6_qcd_hgg_hlt06_3m1m1m_full_20260628_194154/binary_inputs/hlt_cache}"
: "${LOCAL_GRAPH_MULTI_FUSION_OUTPUT_DIR:=${LOCAL_GRAPH_MULTI_FUSION_LOCAL_ROOT}/score_fusion_with_multiscale}"
: "${LOCAL_GRAPH_MULTI_FUSION_PREDICTION_DIR:=${LOCAL_GRAPH_MULTI_FUSION_OUTPUT_DIR}/predictions}"
: "${LOCAL_GRAPH_MULTI_FUSION_LOCAL_VARIANTS:=hlt_part_baseline local_edgeconv_adapter local_point_attention_adapter local_point_attention_adapter_warmstart}"
: "${LOCAL_GRAPH_MULTI_FUSION_MULTISCALE_VARIANTS:=no_scale_bias one_scale_medium few_subjets}"
: "${LOCAL_GRAPH_MULTI_FUSION_MULTISCALE_PREFIX:=ms_}"
: "${LOCAL_GRAPH_MULTI_FUSION_BASELINE_VARIANT:=hlt_part_baseline}"
: "${LOCAL_GRAPH_MULTI_FUSION_PRIMARY_METRIC:=fpr_at_signal_eff_0p50}"
: "${LOCAL_GRAPH_MULTI_FUSION_MAX_STACK_JETS:=1000000}"
: "${LOCAL_GRAPH_MULTI_FUSION_MAX_FINAL_TEST_JETS:=1000000}"
: "${LOCAL_GRAPH_MULTI_FUSION_BATCH_SIZE:=256}"
: "${LOCAL_GRAPH_MULTI_FUSION_NUM_WORKERS:=${NUM_WORKERS}}"
: "${LOCAL_GRAPH_MULTI_FUSION_DEVICE:=${DEVICE}}"
: "${LOCAL_GRAPH_MULTI_FUSION_C_GRID:=0.01 0.03 0.1 0.3 1.0 3.0 10.0}"
: "${LOCAL_GRAPH_MULTI_FUSION_MAX_ITER:=1000}"
: "${LOCAL_GRAPH_MULTI_FUSION_WEIGHT_GRID_STEP:=0.05}"
: "${LOCAL_GRAPH_MULTI_FUSION_CONTROL_SEED:=17717}"
: "${LOCAL_GRAPH_MULTI_FUSION_OVERWRITE_PREDICTIONS:=0}"
: "${LOCAL_GRAPH_MULTI_FUSION_REQUIRE_ALL_VARIANTS:=1}"
: "${LOCAL_GRAPH_MULTI_FUSION_ALL_MODEL_SUBSETS:=0}"
: "${LOCAL_GRAPH_MULTI_FUSION_MAX_MODEL_SET_SIZE:=5}"
: "${LOCAL_GRAPH_MULTI_FUSION_SKIP_HLT_HASH_CHECK:=0}"
: "${LOCAL_GRAPH_MULTI_FUSION_NO_SKLEARN:=0}"
: "${LOCAL_GRAPH_MULTI_FUSION_SKIP_CONTROLS:=0}"
: "${LOCAL_GRAPH_MULTI_FUSION_CONFIRM_FINAL_TEST:=1}"

fresh_setup "$@"
fresh_require_file "scripts/run_local_graph_multiscale_score_fusion.py"
fresh_split_words local_variant_args "${LOCAL_GRAPH_MULTI_FUSION_LOCAL_VARIANTS}"
fresh_split_words multiscale_variant_args "${LOCAL_GRAPH_MULTI_FUSION_MULTISCALE_VARIANTS}"
fresh_split_words c_grid_args "${LOCAL_GRAPH_MULTI_FUSION_C_GRID}"

for split in stack_val final_test; do
  fresh_require_file "${LOCAL_GRAPH_MULTI_FUSION_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${LOCAL_GRAPH_MULTI_FUSION_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done

missing=()
for variant in "${local_variant_args[@]}"; do
  checkpoint="${LOCAL_GRAPH_MULTI_FUSION_LOCAL_TAGGER_ROOT}/${variant}/best_model_val.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    missing+=("local_graph:${variant}")
  fi
done
for variant in "${multiscale_variant_args[@]}"; do
  checkpoint="${LOCAL_GRAPH_MULTI_FUSION_MULTISCALE_TAGGER_ROOT}/${variant}/best_model_val.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    missing+=("multiscale_subjet:${variant}")
  fi
done
if [[ "${#missing[@]}" -gt 0 ]]; then
  echo "Missing requested cross-family fusion checkpoints: $(fresh_join_by_space "${missing[@]}")" >&2
  exit 2
fi

if ! fresh_bool_enabled "${LOCAL_GRAPH_MULTI_FUSION_OVERWRITE_PREDICTIONS}"; then
  fresh_claim_new_dir "${LOCAL_GRAPH_MULTI_FUSION_OUTPUT_DIR}"
else
  mkdir -p "${LOCAL_GRAPH_MULTI_FUSION_OUTPUT_DIR}"
fi

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_local_graph_multiscale_score_fusion.py"
  --local-graph-root "${LOCAL_GRAPH_MULTI_FUSION_LOCAL_ROOT}"
  --local-graph-tagger-root "${LOCAL_GRAPH_MULTI_FUSION_LOCAL_TAGGER_ROOT}"
  --multiscale-root "${LOCAL_GRAPH_MULTI_FUSION_MULTISCALE_ROOT}"
  --multiscale-tagger-root "${LOCAL_GRAPH_MULTI_FUSION_MULTISCALE_TAGGER_ROOT}"
  --hlt-cache-dir "${LOCAL_GRAPH_MULTI_FUSION_HLT_CACHE_DIR}"
  --prediction-dir "${LOCAL_GRAPH_MULTI_FUSION_PREDICTION_DIR}"
  --output-dir "${LOCAL_GRAPH_MULTI_FUSION_OUTPUT_DIR}"
  --local-graph-variants "${local_variant_args[@]}"
  --multiscale-variants "${multiscale_variant_args[@]}"
  --multiscale-prefix "${LOCAL_GRAPH_MULTI_FUSION_MULTISCALE_PREFIX}"
  --baseline-variant "${LOCAL_GRAPH_MULTI_FUSION_BASELINE_VARIANT}"
  --primary-metric "${LOCAL_GRAPH_MULTI_FUSION_PRIMARY_METRIC}"
  --max-stack-jets "${LOCAL_GRAPH_MULTI_FUSION_MAX_STACK_JETS}"
  --max-final-test-jets "${LOCAL_GRAPH_MULTI_FUSION_MAX_FINAL_TEST_JETS}"
  --batch-size "${LOCAL_GRAPH_MULTI_FUSION_BATCH_SIZE}"
  --num-workers "${LOCAL_GRAPH_MULTI_FUSION_NUM_WORKERS}"
  --device "${LOCAL_GRAPH_MULTI_FUSION_DEVICE}"
  --c-grid "${c_grid_args[@]}"
  --max-iter "${LOCAL_GRAPH_MULTI_FUSION_MAX_ITER}"
  --weight-grid-step "${LOCAL_GRAPH_MULTI_FUSION_WEIGHT_GRID_STEP}"
  --control-seed "${LOCAL_GRAPH_MULTI_FUSION_CONTROL_SEED}"
  --max-model-set-size "${LOCAL_GRAPH_MULTI_FUSION_MAX_MODEL_SET_SIZE}"
)
fresh_append_flag_if_enabled cmd --overwrite-predictions "${LOCAL_GRAPH_MULTI_FUSION_OVERWRITE_PREDICTIONS}"
fresh_append_flag_if_enabled cmd --require-all-variants "${LOCAL_GRAPH_MULTI_FUSION_REQUIRE_ALL_VARIANTS}"
fresh_append_flag_if_enabled cmd --all-model-subsets "${LOCAL_GRAPH_MULTI_FUSION_ALL_MODEL_SUBSETS}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${LOCAL_GRAPH_MULTI_FUSION_SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --no-sklearn "${LOCAL_GRAPH_MULTI_FUSION_NO_SKLEARN}"
fresh_append_flag_if_enabled cmd --skip-controls "${LOCAL_GRAPH_MULTI_FUSION_SKIP_CONTROLS}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${LOCAL_GRAPH_MULTI_FUSION_CONFIRM_FINAL_TEST}"

fresh_write_run_config "${LOCAL_GRAPH_MULTI_FUSION_OUTPUT_DIR}" "local_graph_multiscale_score_fusion" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${LOCAL_GRAPH_MULTI_FUSION_OUTPUT_DIR}/fusion_report.json"
  fresh_require_file "${LOCAL_GRAPH_MULTI_FUSION_OUTPUT_DIR}/fusion_report.md"
  fresh_require_file "${LOCAL_GRAPH_MULTI_FUSION_OUTPUT_DIR}/fusion_metric_table.csv"
  fresh_require_file "${LOCAL_GRAPH_MULTI_FUSION_OUTPUT_DIR}/fusion_weights.csv"
  fresh_require_file "${LOCAL_GRAPH_MULTI_FUSION_OUTPUT_DIR}/fusion_controls.csv"
  fresh_require_file "${LOCAL_GRAPH_MULTI_FUSION_OUTPUT_DIR}/fusion_prediction_manifest.json"
  fresh_require_file "${LOCAL_GRAPH_MULTI_FUSION_OUTPUT_DIR}/run_report.json"
fi
