# Fresh JetClass Fusion Handoff: Model Loading and Evaluation Protocol

This document is for an independent fresh implementation of the JetClass fusion analysis. The goal is not to reuse the old fusion code. The goal is to load the already trained models, generate logits on cleanly separated fusion splits, and test whether a fusion/meta-classifier gives a real improvement.

Use this as a model-loading and experiment-specification handoff. You may write new code however you want, but the data split rules and leakage rules below must be followed exactly.

## Core Question

We have trained JetClass HLT/reconstruction/tagger models on the research compute. Some previous fusion analyses reported very large gains from stacked logistic regression. The purpose of this fresh implementation is to independently test whether those gains are real.

The fresh implementation should report:

1. Raw HLT baseline performance.
2. Raw stage2 model performance for each reconstruction model.
3. Optional offline teacher performance as an upper reference only.
4. Fusion performance from the stage2 model logits/probabilities.
5. Fusion performance from HLT + stage2 logits/probabilities.
6. Controls that should collapse if the fusion is not leaking labels or row identity.

## Research Compute Paths

Run on the research compute where the data and checkpoint files exist.

Main data path:

```text
/home/ryreu/atlas/PracticeTagging/data/jetclass_part0
```

Original PracticeTagging checkpoint root:

```text
/home/ryreu/atlas/PracticeTagging/checkpoints
```

Fresh-check repo may be located elsewhere, for example:

```text
/home/ryreu/atlas/Fresh_check
```

Do not hard-code a local Windows path. Use RC paths for data/checkpoints.

## Fixed-HLT Filename 1M Models To Load

These are the models from the fixed-HLT filename setup. They all use the same intended HLT corruption profile and filename-based class assignment.

Use this checkpoint root:

```text
/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview
```

Use this model source list:

```python
SOURCE_SPECS = [
    # HLT-only baseline and offline teacher are both stored in core01_base.
    ("hlt_baseline", "baseline_hlt", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core01_base"),
    ("offline_teacher", "offline_teacher", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core01_base"),

    # Twelve stage2 dual-view models.
    ("m2_base", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core01_base"),
    ("m2_consstrong", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core02_consstrong"),
    ("m2_budgetlite", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core03_budgetlite"),
    ("m2_genlow", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core04_genlow"),
    ("m2_genhigh", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core05_genhigh"),
    ("m2_splitstrong", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core06_splitstrong"),
    ("m2_splitlight", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core07_splitlight"),
    ("m2_physstrong", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core08_physstrong"),
    ("m2_offdropmid", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core09_offdropmid"),
    ("m2_offdrophigh", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core10_offdrophigh"),
    ("m2_topk60ish", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core11_topk60ish"),
    ("m2_antioverlap", "stage2", "/home/ryreu/atlas/PracticeTagging/checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename_core12_antioverlap"),
]
```

Recommended fusion groups:

```python
FUSION_GROUPS = {
    "m2_only": [
        "m2_base", "m2_consstrong", "m2_budgetlite", "m2_genlow", "m2_genhigh",
        "m2_splitstrong", "m2_splitlight", "m2_physstrong", "m2_offdropmid",
        "m2_offdrophigh", "m2_topk60ish", "m2_antioverlap",
    ],
    "hlt_plus_m2": [
        "hlt_baseline", "m2_base", "m2_consstrong", "m2_budgetlite", "m2_genlow",
        "m2_genhigh", "m2_splitstrong", "m2_splitlight", "m2_physstrong",
        "m2_offdropmid", "m2_offdrophigh", "m2_topk60ish", "m2_antioverlap",
    ],
}
```

The offline teacher must not be included in fusion groups. It is only an upper reference.

## Checkpoint Files To Load

Each model directory has an `args.json`. Always load it. It defines architecture settings, class assignment, HLT generation parameters, preprocessing, seed, and train/val/test file counts.

Expected checkpoint file names:

```text
baseline_hlt        -> baseline_hlt_best.pt, fallback baseline_best.pt or baseline.pt
offline_teacher     -> teacher_offline_best.pt, fallback teacher.pt
stage2 reconstructor -> offline_reconstructor_stage2.pt, fallback offline_reconstructor.pt
stage2 dual tagger   -> dual_joint_stage2.pt, fallback dual_joint.pt
```

Do not load teacher logits, cached score files, or old fusion `.npz` files. Regenerate logits from model checkpoints on the fresh fusion splits.

## Minimal Checkpoint Loading Helpers

Use this pattern for checkpoint discovery. Some checkpoints are raw `state_dict`; some may be dictionaries containing a nested state dict.

```python
from pathlib import Path
import json
import torch
from types import SimpleNamespace


def load_args(run_dir: str | Path) -> SimpleNamespace:
    run_dir = Path(run_dir)
    with (run_dir / "args.json").open("r") as f:
        return SimpleNamespace(**json.load(f))


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "net", "model", "reco_state_dict", "dual_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def load_first_state_dict(run_dir: str | Path, candidate_names: tuple[str, ...], map_location="cpu"):
    run_dir = Path(run_dir)
    for name in candidate_names:
        path = run_dir / name
        if path.exists():
            obj = torch.load(path, map_location=map_location)
            return unwrap_state_dict(obj), path
    raise FileNotFoundError(f"No checkpoint found in {run_dir}; tried {candidate_names}")
```

## Model Construction Contract

The fresh code must construct models with the same architecture implied by `args.json`.

The model kinds need these inputs and outputs:

```python
# baseline_hlt:
# input:  feat_hlt, mask_hlt
# output: logits [batch, n_classes]
logits = baseline_model(feat_hlt, mask_hlt)

# offline_teacher:
# input:  feat_offline, mask_offline
# output: logits [batch, n_classes]
# This is for reference only, not fusion.
logits = teacher_model(feat_offline, mask_offline)

# stage2:
# input:  feat_hlt, mask_hlt, const_hlt4
# steps:
# 1. reconstructor predicts corrected candidate constituents.
# 2. convert reconstructor output into a soft corrected view.
# 3. dual-view classifier consumes original HLT view plus corrected view.
# output: logits [batch, n_classes]
reco_out = reconstructor(feat_hlt, mask_hlt, const_hlt4, stage_scale=1.0)
feat_corr, mask_corr = build_soft_corrected_view(reco_out, weight_floor=1e-4)
logits = dual_view_model(feat_hlt, mask_hlt, feat_corr, mask_corr)
```

For stage2 models, load both:

```python
reco_sd, reco_ckpt = load_first_state_dict(run_dir, ("offline_reconstructor_stage2.pt", "offline_reconstructor.pt"))
dual_sd, dual_ckpt = load_first_state_dict(run_dir, ("dual_joint_stage2.pt", "dual_joint.pt"))
```

Important: the stage2 checkpoints here are `hybrid_ops` reconstructors. If your fresh code has not implemented the hybrid operation-aware reconstructor, loading will fail or produce meaningless output.

## Hybrid Ops Detection

The old implementation detected the reconstructor family from the checkpoint keys. This is useful if your code supports multiple families.

```python
def detect_reco_family(state_dict: dict) -> str:
    keys = tuple(state_dict.keys())
    if any(k.startswith("base.split_exist_head.") for k in keys) or any(k.startswith("base.gen_attn.") for k in keys):
        return "hybrid_ops"
    if "base.gate_temperature" in state_dict or any(k.startswith("base.op_gate_head.") for k in keys):
        return "confgen_ops"
    return "unknown"
```

For this specific fixed-HLT filename set, expect `hybrid_ops`.

## Data Split Protocol For Fusion

Use the original model file split from `args.json`, then subdivide a fresh source split for fusion. The previous fresh audit used the model's original `test` files as the source pool because these jets were not used to fit the base models.

Use this protocol:

1. Read `args.json` from `core01_base` as the reference.
2. Use the same class assignment and class order as the reference.
3. Collect JetClass files from `/home/ryreu/atlas/PracticeTagging/data/jetclass_part0`.
4. Reconstruct the original train/val/test file split using:
   - `train_files_per_class`
   - `val_files_per_class`
   - `test_files_per_class`
   - `shuffle_files`
   - `seed`
5. Load events from the original `test` files only for the fusion audit.
6. Build a fixed HLT view from the offline constituents using the same HLT parameters from `args.json`.
7. Slice the loaded events in order into stack/fusion splits.

Run both sizes:

```text
SMALL:
stack_train = 50,000
stack_val   = 20,000
final_test  = 100,000

LARGE:
stack_train = 250,000
stack_val   = 50,000
final_test  = 500,000
```

The source pool must contain at least the sum of those counts.

## Leakage Rules

These are mandatory.

1. Do not train the stacker on `final_test` labels.
2. Do not tune hyperparameters on `final_test` labels.
3. Do not select the best method by `final_test` accuracy.
4. Fit all stacker weights only on `stack_train`.
5. Choose among methods and hyperparameters only with `stack_val`.
6. Report `final_test` exactly once per method after the stacker is frozen.
7. Do not include `offline_teacher` in any deployable fusion group.
8. Do not use offline features as stacker inputs, except for the `offline_teacher` reference evaluation.
9. Do not use any cached old fusion scores or old `.npz` outputs.
10. Do not reuse labels to reorder rows, filter hard cases, or rebalance final test.
11. Save sample hashes for `stack_train`, `stack_val`, and `final_test` and verify zero overlap.
12. Verify every source emits logits for exactly the same event order.
13. Include negative controls.

## Required Negative Controls

At minimum, run these:

### Control 1: Permuted Labels

Fit the same stacker features on `stack_train`, but randomly shuffle `y_stack_train`. Evaluate on final test. Accuracy should collapse to about chance, roughly `0.10` for 10 classes.

### Control 2: Row-Shuffled Features

For each feature column, independently shuffle the `stack_train` feature column across rows while keeping labels fixed. Fit stacker and evaluate final test. Accuracy should collapse near chance or clearly below raw HLT.

### Control 3: Singleton Stacker Audit

For each source separately, train the fusion/logistic model on only that source's logits/probs. This is important because previous results showed a suspiciously large singleton stacker gain. Report raw vs singleton-stacked for every source.

### Control 4: Diagonal/Temperature-Only Calibration

Compare full multiclass logistic regression against simple calibration options:

- raw softmax
- temperature scaling only
- per-class bias only, if implemented
- diagonal scaling plus bias, if implemented
- full linear multinomial logistic regression

If only the full linear model creates a huge jump, inspect the learned weight matrix and class confusion changes.

## Fusion Feature Choices

Evaluate these feature modes separately:

```text
logits only
probabilities only
logits + probabilities
```

For a group of N models and 10 classes:

```text
logits only          -> 10*N features
probabilities only   -> 10*N features
logits + probabilities -> 20*N features
```

Do not include true labels, class names, sample indices, file IDs, jet counts, or offline kinematics as stacker features.

## Simple Fresh Fusion Implementation Sketch

This pseudocode is intentionally explicit. You can implement it differently, but the data flow must match.

```python
# 1. Load fresh source events from original test files.
off_tokens, off_mask, y = load_jetclass_events_from_original_test_files(...)

# 2. Split rows by fixed slices.
idx_stack_train = slice(0, 50_000)
idx_stack_val = slice(50_000, 70_000)
idx_final_test = slice(70_000, 170_000)

# 3. Generate HLT view once using fixed HLT parameters.
hlt_tokens, hlt_mask = build_fixed_hlt_view(off_tokens, off_mask, params_from_args_json, seed=...)

# 4. Compute features needed by baseline/reco models.
feat_hlt = compute_features(hlt_tokens, hlt_mask, feature_mode, feature_preprocessing)
feat_off = compute_features(off_tokens, off_mask, feature_mode, feature_preprocessing)
const_hlt4 = hlt_tokens[:, :, :4]

# 5. Load all sources and collect logits on the exact same row order.
logits_by_source = {}
for name, kind, run_dir in SOURCE_SPECS:
    source = load_source(name, kind, run_dir)
    logits_by_source[name] = collect_logits(source, feat_hlt, hlt_mask, const_hlt4, feat_off, off_mask)

# 6. Compute raw metrics for each source.
for name, logits in logits_by_source.items():
    probs = softmax(logits)
    report_raw_metrics(name, probs[idx_stack_val], y[idx_stack_val], probs[idx_final_test], y[idx_final_test])

# 7. Fit stacker on stack_train only.
for group_name, source_names in FUSION_GROUPS.items():
    X_train = make_stacker_features(logits_by_source, source_names, idx_stack_train, mode="logits_probs")
    X_val = make_stacker_features(logits_by_source, source_names, idx_stack_val, mode="logits_probs")
    X_test = make_stacker_features(logits_by_source, source_names, idx_final_test, mode="logits_probs")

    y_train = y[idx_stack_train]
    y_val = y[idx_stack_val]
    y_test = y[idx_final_test]

    stacker = fit_multinomial_logistic_regression(X_train, y_train)
    choose_or_report_using_val_only(stacker, X_val, y_val)
    report_final_test(stacker, X_test, y_test)
```

## Logistic Regression Requirements

If using scikit-learn:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

clf = make_pipeline(
    StandardScaler(),
    LogisticRegressionCV(
        Cs=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        cv=5,
        solver="lbfgs",
        scoring="accuracy",
        max_iter=2000,
        n_jobs=1,
    ),
)
clf.fit(X_stack_train, y_stack_train)
probs_stack_val = clf.predict_proba(X_stack_val)
probs_final_test = clf.predict_proba(X_final_test)
```

But also test a non-CV version where C is selected using `stack_val`, not internal CV. This helps avoid ambiguity about what split is being used for model selection.

## Reports To Save

Save at least:

```text
fusion_report.json
raw_source_metrics.csv
singleton_stacker_metrics.csv
group_fusion_metrics.csv
controls.json
stack_split_hash_audit.json
```

Report fields should include:

```text
accuracy
macro OVR AUC
signal-vs-background AUC if implemented
FPR at 50% signal efficiency if implemented
confusion matrix for HLT raw and HLT singleton-stacked
```

## What Would Make The Result Suspicious

Flag the result as suspicious if any of these occur:

1. Singleton stacker improves HLT by a huge amount, especially if confusion matrix shows systematic class remapping.
2. Permuted-label or row-shuffled controls do not collapse near chance.
3. Final-test accuracy is used in any selection path.
4. Stacker features include anything except logits/probs.
5. Event hashes overlap across `stack_train`, `stack_val`, and `final_test`.
6. Different sources are evaluated on different HLT views or different event row order.
7. Offline teacher information is used in a deployable stack group.
8. `class_assignment` or class ordering differs between source models.

## Specific Thing To Investigate

Previous fresh audit output showed approximately:

```text
raw HLT final-test accuracy: about 0.718
HLT singleton stacked-logreg final-test accuracy: about 0.789
m2-only stacked-logreg final-test accuracy: about 0.802
offline teacher raw final-test accuracy: about 0.817
```

That singleton HLT jump is the main concern. The new independent code should focus on verifying or falsifying that exact behavior.

If the new implementation reproduces the singleton jump, inspect:

1. HLT raw confusion matrix.
2. HLT singleton-stacked confusion matrix.
3. Learned logistic regression weight matrix.
4. Whether the raw model's class-index ordering matches the labels used for evaluation.
5. Whether filename label mapping and canonical label mapping are accidentally mixed.

## Class Assignment Warning

The fixed-HLT filename runs should use filename-based labels. Do not mix canonical-label order with filename-label order unless you explicitly prove the class index mapping is identical.

Always print and save:

```text
class_names list
class_to_idx mapping
args_json class_assignment
checkpoint output dimension
```

If raw HLT is low but singleton stacker is high, one possible explanation is class-index/mapping mismatch or a systematic remapping that logistic regression learns. That must be ruled out directly.


## Fresh-Check Loader Demo Script

A concrete non-fusion loader/evaluation script has been added here:

```text
/home/ryan/ComputerScience/ATLAS/fresh_start_check/scripts/demo_load_and_score_models_no_fusion.py
```

This script is intentionally not a fusion script. It does everything needed before fusion:

1. Builds model specs for the fresh-check HLT baseline and selected fresh-check dual-view reconstructor variants.
2. Loads frozen checkpoints through the `jetclass_fresh` library.
3. Runs inference on cached fixed-HLT splits.
4. Saves one prediction block per model per split.
5. Reloads those blocks to verify label and jet-identity alignment.
6. Builds candidate fusion feature matrices from logits/probs.
7. Saves raw model metrics and feature-matrix shape reports.
8. Does not train logistic regression, does not fit weighted averages, and does not combine model outputs.

### Script Commands

Small smoke/demo run, no final test:

```bash
cd /home/ryreu/atlas/Fresh_check
python -u scripts/demo_load_and_score_models_no_fusion.py \
  --hlt-cache-dir checkpoints/jetclass_fresh_hlt_cache \
  --hlt-checkpoint checkpoints/jetclass_fresh_hlt_baselines/single_hlt_seed101/best_model_val.pt \
  --reco-root checkpoints/jetclass_fresh_reco7 \
  --output-dir checkpoints/jetclass_fresh_model_loading_demo \
  --splits stack_train stack_val \
  --max-jets-per-split 2000 \
  --batch-size 128 \
  --device cuda
```

If you deliberately want to generate final-test prediction blocks too, you must explicitly confirm it:

```bash
cd /home/ryreu/atlas/Fresh_check
python -u scripts/demo_load_and_score_models_no_fusion.py \
  --hlt-cache-dir checkpoints/jetclass_fresh_hlt_cache \
  --hlt-checkpoint checkpoints/jetclass_fresh_hlt_baselines/single_hlt_seed101/best_model_val.pt \
  --reco-root checkpoints/jetclass_fresh_reco7 \
  --output-dir checkpoints/jetclass_fresh_model_loading_demo_with_final_test \
  --splits stack_train stack_val final_test \
  --confirm-final-test \
  --max-jets-per-split 2000 \
  --batch-size 128 \
  --device cuda
```

To only run a subset of variants:

```bash
python -u scripts/demo_load_and_score_models_no_fusion.py \
  --variants m2_base m2_budgetlite m2_topk60ish m2_antioverlap \
  --splits stack_train stack_val \
  --max-jets-per-split 2000
```

To validate that the external PracticeTagging fixed-HLT paths exist without trying to load those old architectures:

```bash
python -u scripts/demo_load_and_score_models_no_fusion.py \
  --splits stack_train stack_val \
  --max-jets-per-split 100 \
  --validate-practicetagging-paths
```

### What The Script Saves

Main report:

```text
<output-dir>/model_loading_demo_report.json
```

Prediction blocks:

```text
<output-dir>/predictions/<model_name>/<split>_predictions.npz
<output-dir>/predictions/<model_name>/<split>_predictions_metadata.json
```

Each `.npz` prediction block contains:

```text
logits                float32 [n_jets, 10]
probs                 float32 [n_jets, 10]
labels                int64   [n_jets]
jet_file_indices      int32   [n_jets]
jet_entries           int64   [n_jets]
```

The metadata contains:

```text
model_name
split
jet_identity_hash
prediction_content_hash
n_jets
num_classes
raw accuracy / cross entropy
checkpoint path
allowed input description
```

### How It Loads Fresh-Check Models

The script uses these functions from `jetclass_fresh.fusion`:

```python
from jetclass_fresh.fusion import (
    FusionModelSpec,
    collect_frozen_predictions,
    load_blocks_for_split,
    validate_prediction_alignment,
    stack_feature_matrix,
)
```

The HLT baseline spec is:

```python
FusionModelSpec(
    name="hlt_baseline",
    kind="hlt",
    checkpoint="checkpoints/jetclass_fresh_hlt_baselines/single_hlt_seed101/best_model_val.pt",
)
```

Each dual-view reconstructor/tagger spec is:

```python
FusionModelSpec(
    name="m2_base",
    kind="dual_view",
    checkpoint="checkpoints/jetclass_fresh_reco7/m2_base/stage2_dual_view/best_model_val.pt",
)
```

The helper that builds the default HLT + reco7 list is:

```python
from jetclass_fresh.fusion import default_reco7_plus_hlt_specs

specs = default_reco7_plus_hlt_specs(
    hlt_checkpoint="checkpoints/jetclass_fresh_hlt_baselines/single_hlt_seed101/best_model_val.pt",
    reco_root="checkpoints/jetclass_fresh_reco7",
    variants=["m2_base", "m2_budgetlite", "m2_topk60ish"],
)
```

Internally, the fresh-check library loads HLT checkpoints with:

```python
load_hlt_model_from_checkpoint(path, device=device)
```

and dual-view checkpoints with:

```python
load_dual_view_model_from_checkpoint(path, device=device)
```

The dual-view checkpoint records its reconstructor checkpoint, so loading the dual-view model also loads the matching Stage-A reconstructor:

```python
tagger, reconstructor, dual_payload, reco_payload = load_dual_view_model_from_checkpoint(path, device=device)
```

### How It Runs Inference

For HLT-only models, the fresh library calls:

```python
evaluate_hlt_model(
    model,
    fixed_hlt_view,
    model_name="hlt_baseline",
    batch_size=batch_size,
    num_workers=num_workers,
    device=device,
)
```

For dual-view models, the fresh library calls:

```python
evaluate_dual_view_model(
    model_name,
    tagger,
    reconstructor,
    fixed_hlt_view,
    batch_size=batch_size,
    num_workers=num_workers,
    device=device,
)
```

The dual-view inference path is:

```python
reco = reconstructor(batch["hlt_tokens"], batch["hlt_mask"])
hlt_inputs = build_part_inputs_torch(batch["hlt_tokens"], batch["hlt_mask"])
corrected_inputs = build_soft_corrected_view_torch(batch["hlt_tokens"], batch["hlt_mask"], reco)
logits = tagger(hlt_inputs, corrected_inputs)
```

This is the exact place where a future independent fusion script should obtain raw logits. Do not modify labels, row order, or split membership after this point.

### How It Prepares Candidate Fusion Features Without Fusing

After prediction blocks are saved, the script reloads them and verifies alignment:

```python
blocks = load_blocks_for_split(prediction_dir, model_names, split)
validate_prediction_alignment(blocks)
```

Then it prepares the matrix that a fusion method would consume:

```python
X = stack_feature_matrix(blocks, feature_mode="logits_probs")
y = blocks[0].labels
```

At this point the demo stops. A separate independent fusion script can take `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, and `y_test`, but it should not be written by copying old fusion behavior blindly.

### Why This Script Exists

Use this script to decouple two questions:

1. Can we load the models and reproduce raw logits correctly?
2. Does a fusion/meta-classifier actually improve performance in a fair way?

This script answers only question 1. It intentionally provides all model loading and raw prediction machinery while leaving question 2 to a fresh independent fusion implementation.
