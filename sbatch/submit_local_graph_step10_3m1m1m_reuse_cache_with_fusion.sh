#!/usr/bin/env bash
# Submit the QCD-vs-Hgg local-graph comparison on the prebuilt 3M/1M/1M HLT0.6 cache.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${LOCAL_GRAPH_PART_3M_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${QCD_HGG_3M_ROOT:=/home/ryreu/atlas/Fresh_check/checkpoints/multiscale_subjet_part_qcd_hgg_binary_hlt0p6_qcd_hgg_hlt06_3m1m1m_full_20260628_194154}"
: "${QCD_HGG_3M_MANIFEST_PATH:=${QCD_HGG_3M_ROOT}/binary_inputs/split_manifest.json.gz}"
: "${QCD_HGG_3M_HLT_CACHE_DIR:=${QCD_HGG_3M_ROOT}/binary_inputs/hlt_cache}"
: "${LOCAL_GRAPH_PART_3M_ROOT:=${OUTPUT_ROOT}/local_graph_part_step10_qcd_hgg_binary_hlt0p6_3m1m1m_${LOCAL_GRAPH_PART_3M_TAG}}"

export LOCAL_GRAPH_PART_QCD_HGG_ROOT="${LOCAL_GRAPH_PART_3M_ROOT}"
export LOCAL_GRAPH_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH=0.6
export HLT_DEGRADATION_STRENGTH=0.6

# Reuse the existing binary manifest and fixed-HLT cache. Do not rebuild inputs.
export LOCAL_GRAPH_PART_QCD_HGG_BUILD_BINARY_INPUTS=0
export LOCAL_GRAPH_PART_QCD_HGG_BUILD_DIRECT_BINARY_SPLITS=0
export LOCAL_GRAPH_PART_QCD_HGG_SOURCE_MANIFEST_PATH="${QCD_HGG_3M_MANIFEST_PATH}"
export LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_PATH="${QCD_HGG_3M_MANIFEST_PATH}"
export LOCAL_GRAPH_PART_QCD_HGG_BINARY_HLT_CACHE_DIR="${QCD_HGG_3M_HLT_CACHE_DIR}"

export LOCAL_GRAPH_PART_QCD_HGG_VARIANTS="${LOCAL_GRAPH_PART_3M_VARIANTS:-hlt_part_baseline local_edgeconv_adapter local_point_attention_adapter local_point_attention_adapter_warmstart}"
export LOCAL_GRAPH_PART_QCD_HGG_MODEL_TRAIN_SIZE="${LOCAL_GRAPH_PART_3M_MODEL_TRAIN_SIZE:-3000000}"
export LOCAL_GRAPH_PART_QCD_HGG_MODEL_VAL_SIZE="${LOCAL_GRAPH_PART_3M_MODEL_VAL_SIZE:-1000000}"
export LOCAL_GRAPH_PART_QCD_HGG_STACK_TRAIN_SIZE="${LOCAL_GRAPH_PART_3M_STACK_TRAIN_SIZE:-3000000}"
export LOCAL_GRAPH_PART_QCD_HGG_STACK_VAL_SIZE="${LOCAL_GRAPH_PART_3M_STACK_VAL_SIZE:-1000000}"
export LOCAL_GRAPH_PART_QCD_HGG_FINAL_TEST_SIZE="${LOCAL_GRAPH_PART_3M_FINAL_TEST_SIZE:-1000000}"
export LOCAL_GRAPH_PART_QCD_HGG_EPOCHS="${LOCAL_GRAPH_PART_3M_EPOCHS:-45}"
export LOCAL_GRAPH_PART_QCD_HGG_SELECTION_METRIC="${LOCAL_GRAPH_PART_3M_SELECTION_METRIC:-fpr_at_signal_eff_0p50}"
export LOCAL_GRAPH_PART_QCD_HGG_K="${LOCAL_GRAPH_PART_3M_K:-16}"
export LOCAL_GRAPH_PART_QCD_HGG_WARM_START_FREEZE_EPOCHS="${LOCAL_GRAPH_PART_3M_WARM_START_FREEZE_EPOCHS:-0}"
export LOCAL_GRAPH_PART_QCD_HGG_WARM_START_RESIDUAL_GAMMA_INIT="${LOCAL_GRAPH_PART_3M_WARM_START_RESIDUAL_GAMMA_INIT:-0.01}"

# The previous 500k runs used about 2-2.5 hours per model. This larger run gets generous time.
export LOCAL_GRAPH_PART_QCD_HGG_TRAIN_TIME="${LOCAL_GRAPH_PART_3M_TRAIN_TIME:-3-00:00:00}"
export LOCAL_GRAPH_PART_QCD_HGG_TRAIN_MEM="${LOCAL_GRAPH_PART_3M_TRAIN_MEM:-220G}"
export LOCAL_GRAPH_PART_QCD_HGG_TRAIN_CPUS="${LOCAL_GRAPH_PART_3M_TRAIN_CPUS:-8}"
export LOCAL_GRAPH_PART_QCD_HGG_REPORT_TIME="${LOCAL_GRAPH_PART_3M_REPORT_TIME:-04:00:00}"
export LOCAL_GRAPH_PART_QCD_HGG_REPORT_MEM="${LOCAL_GRAPH_PART_3M_REPORT_MEM:-16G}"
export LOCAL_GRAPH_PART_QCD_HGG_REPORT_CPUS="${LOCAL_GRAPH_PART_3M_REPORT_CPUS:-2}"

export LOCAL_GRAPH_PART_QCD_HGG_SUBMIT_SCORE_FUSION=1
export LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_TIME="${LOCAL_GRAPH_PART_3M_SCORE_FUSION_TIME:-1-00:00:00}"
export LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_MEM="${LOCAL_GRAPH_PART_3M_SCORE_FUSION_MEM:-160G}"
export LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_CPUS="${LOCAL_GRAPH_PART_3M_SCORE_FUSION_CPUS:-4}"
export LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_REQUIRE_ALL_VARIANTS=1

fresh_prepare_submitter
if fresh_is_dry_run; then
  echo "DRY_RUN local_graph_part_step10_3m1m1m wrapper: delegated submitter will print sbatch commands." >&2
fi
for split in model_train model_val stack_train stack_val final_test; do
  fresh_require_file "${QCD_HGG_3M_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${QCD_HGG_3M_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
fresh_require_file "${QCD_HGG_3M_MANIFEST_PATH}"

cat <<SUMMARY
local_graph_part_step10_3m1m1m_reuse_cache_with_fusion:
  task: QCD_vs_Hgg_local_graph_part_step10_large_split
  root: ${LOCAL_GRAPH_PART_QCD_HGG_ROOT}
  reused_binary_root: ${QCD_HGG_3M_ROOT}
  manifest: ${QCD_HGG_3M_MANIFEST_PATH}
  hlt_cache: ${QCD_HGG_3M_HLT_CACHE_DIR}
  hlt_degradation_strength: 0.6
  split_caps:
    model_train: ${LOCAL_GRAPH_PART_QCD_HGG_MODEL_TRAIN_SIZE}
    model_val: ${LOCAL_GRAPH_PART_QCD_HGG_MODEL_VAL_SIZE}
    stack_train: ${LOCAL_GRAPH_PART_QCD_HGG_STACK_TRAIN_SIZE}
    stack_val: ${LOCAL_GRAPH_PART_QCD_HGG_STACK_VAL_SIZE}
    final_test: ${LOCAL_GRAPH_PART_QCD_HGG_FINAL_TEST_SIZE}
  model:
    variants: ${LOCAL_GRAPH_PART_QCD_HGG_VARIANTS}
    epochs: ${LOCAL_GRAPH_PART_QCD_HGG_EPOCHS}
    selection_metric: ${LOCAL_GRAPH_PART_QCD_HGG_SELECTION_METRIC}
  resources:
    train: time=${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_TIME} mem=${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_MEM} cpus=${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_CPUS}
    report: time=${LOCAL_GRAPH_PART_QCD_HGG_REPORT_TIME} mem=${LOCAL_GRAPH_PART_QCD_HGG_REPORT_MEM} cpus=${LOCAL_GRAPH_PART_QCD_HGG_REPORT_CPUS}
    score_fusion: time=${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_TIME} mem=${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_MEM} cpus=${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_CPUS}
SUMMARY

bash "${SCRIPT_DIR}/submit_local_graph_qcd_hgg_binary_experiment.sh"
