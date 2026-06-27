#!/usr/bin/env bash
# Run frozen-score fusion over completed local-graph HLT ParT checkpoints.

#SBATCH --job-name=localgraph_fuse
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${LOCAL_GRAPH_PART_ROOT:=${OUTPUT_ROOT}/local_graph_part_step10_qcd_hgg_binary_hlt0p6}"
: "${LOCAL_GRAPH_SCORE_FUSION_TAGGER_ROOT:=${LOCAL_GRAPH_PART_ROOT}/taggers}"
: "${LOCAL_GRAPH_SCORE_FUSION_HLT_CACHE_DIR:=${LOCAL_GRAPH_PART_ROOT}/binary_inputs/hlt_cache}"
: "${LOCAL_GRAPH_SCORE_FUSION_OUTPUT_DIR:=${LOCAL_GRAPH_PART_ROOT}/score_fusion}"
: "${LOCAL_GRAPH_SCORE_FUSION_PREDICTION_DIR:=${LOCAL_GRAPH_SCORE_FUSION_OUTPUT_DIR}/predictions}"
: "${LOCAL_GRAPH_SCORE_FUSION_VARIANTS:=hlt_part_baseline local_edgeconv_adapter local_point_attention_adapter local_point_attention_adapter_warmstart}"
: "${LOCAL_GRAPH_SCORE_FUSION_BASELINE_VARIANT:=hlt_part_baseline}"
: "${LOCAL_GRAPH_SCORE_FUSION_PRIMARY_METRIC:=fpr_at_signal_eff_0p50}"
: "${LOCAL_GRAPH_SCORE_FUSION_MAX_STACK_JETS:=150000}"
: "${LOCAL_GRAPH_SCORE_FUSION_MAX_FINAL_TEST_JETS:=500000}"
: "${LOCAL_GRAPH_SCORE_FUSION_BATCH_SIZE:=256}"
: "${LOCAL_GRAPH_SCORE_FUSION_NUM_WORKERS:=${NUM_WORKERS}}"
: "${LOCAL_GRAPH_SCORE_FUSION_DEVICE:=${DEVICE}}"
: "${LOCAL_GRAPH_SCORE_FUSION_C_GRID:=0.01 0.03 0.1 0.3 1.0 3.0 10.0}"
: "${LOCAL_GRAPH_SCORE_FUSION_MAX_ITER:=1000}"
: "${LOCAL_GRAPH_SCORE_FUSION_WEIGHT_GRID_STEP:=0.05}"
: "${LOCAL_GRAPH_SCORE_FUSION_CONTROL_SEED:=17717}"
: "${LOCAL_GRAPH_SCORE_FUSION_OVERWRITE_PREDICTIONS:=0}"
: "${LOCAL_GRAPH_SCORE_FUSION_REQUIRE_ALL_VARIANTS:=0}"
: "${LOCAL_GRAPH_SCORE_FUSION_SKIP_HLT_HASH_CHECK:=0}"
: "${LOCAL_GRAPH_SCORE_FUSION_NO_SKLEARN:=0}"
: "${LOCAL_GRAPH_SCORE_FUSION_SKIP_CONTROLS:=0}"
: "${LOCAL_GRAPH_SCORE_FUSION_CONFIRM_FINAL_TEST:=1}"

fresh_setup "$@"
fresh_require_file "scripts/run_local_graph_score_fusion.py"
fresh_split_words variant_args "${LOCAL_GRAPH_SCORE_FUSION_VARIANTS}"
fresh_split_words c_grid_args "${LOCAL_GRAPH_SCORE_FUSION_C_GRID}"

for split in stack_val final_test; do
  fresh_require_file "${LOCAL_GRAPH_SCORE_FUSION_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${LOCAL_GRAPH_SCORE_FUSION_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done

available_variants=()
missing_variants=()
for variant in "${variant_args[@]}"; do
  checkpoint="${LOCAL_GRAPH_SCORE_FUSION_TAGGER_ROOT}/${variant}/best_model_val.pt"
  if [[ -f "${checkpoint}" ]]; then
    available_variants+=("${variant}")
  else
    missing_variants+=("${variant}")
  fi
done
if [[ "${#missing_variants[@]}" -gt 0 ]]; then
  if fresh_bool_enabled "${LOCAL_GRAPH_SCORE_FUSION_REQUIRE_ALL_VARIANTS}"; then
    echo "Missing required local graph checkpoints: $(fresh_join_by_space "${missing_variants[@]}")" >&2
    exit 2
  fi
  echo "Skipping missing local graph checkpoints: $(fresh_join_by_space "${missing_variants[@]}")" >&2
fi
if [[ "${#available_variants[@]}" -lt 2 ]]; then
  echo "Need at least two available variants for score fusion, got: $(fresh_join_by_space "${available_variants[@]}")" >&2
  exit 2
fi
if [[ ! " ${available_variants[*]} " =~ [[:space:]]${LOCAL_GRAPH_SCORE_FUSION_BASELINE_VARIANT}[[:space:]] ]]; then
  echo "Baseline variant is not available: ${LOCAL_GRAPH_SCORE_FUSION_BASELINE_VARIANT}" >&2
  exit 2
fi

if ! fresh_bool_enabled "${LOCAL_GRAPH_SCORE_FUSION_OVERWRITE_PREDICTIONS}"; then
  fresh_claim_new_dir "${LOCAL_GRAPH_SCORE_FUSION_OUTPUT_DIR}"
else
  mkdir -p "${LOCAL_GRAPH_SCORE_FUSION_OUTPUT_DIR}"
fi

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_local_graph_score_fusion.py"
  --experiment-root "${LOCAL_GRAPH_PART_ROOT}"
  --tagger-root "${LOCAL_GRAPH_SCORE_FUSION_TAGGER_ROOT}"
  --hlt-cache-dir "${LOCAL_GRAPH_SCORE_FUSION_HLT_CACHE_DIR}"
  --prediction-dir "${LOCAL_GRAPH_SCORE_FUSION_PREDICTION_DIR}"
  --output-dir "${LOCAL_GRAPH_SCORE_FUSION_OUTPUT_DIR}"
  --variants "${available_variants[@]}"
  --baseline-variant "${LOCAL_GRAPH_SCORE_FUSION_BASELINE_VARIANT}"
  --primary-metric "${LOCAL_GRAPH_SCORE_FUSION_PRIMARY_METRIC}"
  --max-stack-jets "${LOCAL_GRAPH_SCORE_FUSION_MAX_STACK_JETS}"
  --max-final-test-jets "${LOCAL_GRAPH_SCORE_FUSION_MAX_FINAL_TEST_JETS}"
  --batch-size "${LOCAL_GRAPH_SCORE_FUSION_BATCH_SIZE}"
  --num-workers "${LOCAL_GRAPH_SCORE_FUSION_NUM_WORKERS}"
  --device "${LOCAL_GRAPH_SCORE_FUSION_DEVICE}"
  --c-grid "${c_grid_args[@]}"
  --max-iter "${LOCAL_GRAPH_SCORE_FUSION_MAX_ITER}"
  --weight-grid-step "${LOCAL_GRAPH_SCORE_FUSION_WEIGHT_GRID_STEP}"
  --control-seed "${LOCAL_GRAPH_SCORE_FUSION_CONTROL_SEED}"
)
fresh_append_flag_if_enabled cmd --overwrite-predictions "${LOCAL_GRAPH_SCORE_FUSION_OVERWRITE_PREDICTIONS}"
fresh_append_flag_if_enabled cmd --require-all-variants "${LOCAL_GRAPH_SCORE_FUSION_REQUIRE_ALL_VARIANTS}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${LOCAL_GRAPH_SCORE_FUSION_SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --no-sklearn "${LOCAL_GRAPH_SCORE_FUSION_NO_SKLEARN}"
fresh_append_flag_if_enabled cmd --skip-controls "${LOCAL_GRAPH_SCORE_FUSION_SKIP_CONTROLS}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${LOCAL_GRAPH_SCORE_FUSION_CONFIRM_FINAL_TEST}"

fresh_write_run_config "${LOCAL_GRAPH_SCORE_FUSION_OUTPUT_DIR}" "local_graph_score_fusion" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${LOCAL_GRAPH_SCORE_FUSION_OUTPUT_DIR}/fusion_report.json"
  fresh_require_file "${LOCAL_GRAPH_SCORE_FUSION_OUTPUT_DIR}/fusion_report.md"
  fresh_require_file "${LOCAL_GRAPH_SCORE_FUSION_OUTPUT_DIR}/fusion_metric_table.csv"
  fresh_require_file "${LOCAL_GRAPH_SCORE_FUSION_OUTPUT_DIR}/fusion_weights.csv"
  fresh_require_file "${LOCAL_GRAPH_SCORE_FUSION_OUTPUT_DIR}/fusion_controls.csv"
  fresh_require_file "${LOCAL_GRAPH_SCORE_FUSION_OUTPUT_DIR}/fusion_prediction_manifest.json"
  fresh_require_file "${LOCAL_GRAPH_SCORE_FUSION_OUTPUT_DIR}/run_report.json"
fi
