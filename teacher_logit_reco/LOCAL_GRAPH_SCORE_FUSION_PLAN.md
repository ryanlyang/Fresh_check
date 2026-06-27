# Local Graph Score Fusion Plan

## Goal

Test whether the completed QCD-vs-Hgg HLT 0.6 models contain complementary decision information, even though the local graph models did not individually beat the HLT ParT baseline.

The primary question is:

```text
Can frozen-score fusion of HLT ParT + local graph adapters beat HLT ParT alone
on final_test FPR at 50% Hgg signal efficiency?
```

This is a diagnostic first, not yet the final deployable architecture. If fusion wins, it tells us the local graph models learned a useful alternate ranking or uncertainty signal that the baseline HLT ParT did not fully absorb.

## Fixed Experiment Realm

- Task: `QCD` vs `Hgg`
- Inference view: HLT only
- HLT degradation strength: `0.6`
- Existing trained models:
  - `hlt_part_baseline`
  - `local_edgeconv_adapter`
  - `local_point_attention_adapter`
  - `local_point_attention_adapter_warmstart` when available
- Checkpoint selection for base models: already done on `model_val`
- Fusion training split: `stack_val`
- Fusion final evaluation split: `final_test`
- Primary metric: `fpr_at_signal_eff_0p50`
- Secondary metrics: AUC, accuracy, FPR@30, background rejection@30/50

No fusion method may train on `final_test`.

## Inputs

The fusion script should accept:

- `--experiment-root`
- `--tagger-root`
- `--hlt-cache-dir`
- `--variants`
- `--baseline-variant hlt_part_baseline`
- `--stack-split stack_val`
- `--final-test-split final_test`
- `--output-dir`
- `--primary-metric fpr_at_signal_eff_0p50`
- split caps, usually:
  - `--max-stack-jets 150000`
  - `--max-final-test-jets 500000`

For each variant, load:

```text
taggers/<variant>/best_model_val.pt
```

Then predict logits on `stack_val` and `final_test`.

## Frozen Prediction Cache

First implement a prediction cache so fusion methods do not repeatedly rerun the neural models.

For each variant and split, write:

```text
predictions/<variant>/<split>_predictions.npz
```

Contents:

- `logits`, shape `[n_jets, 2]`
- `labels`, shape `[n_jets]`
- `indices`, shape `[n_jets]`
- `score_margin = logit_Hgg - logit_QCD`
- `prob_signal`
- `prob_background`

Also write:

```text
predictions/prediction_manifest.json
```

The manifest should record checkpoint paths, split names, split caps, model report paths, source commit, and HLT cache metadata/audit.

## Fusion Families

### 1. Single-Model Calibration Controls

Train a one-feature stacker on `stack_val` for each individual model:

```text
score = model_margin
```

This answers whether gains come from real multi-model complementarity or just recalibrating the baseline score.

Required controls:

- `calibrated_hlt_part_baseline`
- `calibrated_local_edgeconv_adapter`
- `calibrated_local_point_attention_adapter`
- `calibrated_local_point_attention_adapter_warmstart`, if present

### 2. Equal-Weight Ensembles

No learned parameters except optional score calibration.

Variants:

- average logits
- average probabilities
- average binary margins
- average log-odds

Model sets:

- HLT + EdgeConv
- HLT + PointAttention
- HLT + EdgeConv + PointAttention
- all available models

### 3. Grid-Searched Weighted Averages

Search weights on `stack_val`, select by FPR@50, evaluate once on `final_test`.

Use nonnegative weights that sum to 1:

```text
fused_score = sum_i w_i * score_i
```

Use a coarse grid first, for example step `0.05`, then optionally refine around the best weights with step `0.01`.

Run this for:

- margins
- logits/probability signal score
- ranks, if rank fusion is enabled

### 4. Logistic Regression Stacker

Train regularized logistic regression on `stack_val`.

Base feature set:

```text
[hlt_margin, edge_margin, point_margin, warm_margin?]
```

Try regularization strengths:

```text
C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
```

Select by stack-val FPR@50. Evaluate selected stacker on final test.

Implementation should have a NumPy fallback if scikit-learn is unavailable.

### 5. Product-of-Experts / Log-Odds Fusion

Combine model log-odds:

```text
score = sum_i w_i * logit(prob_signal_i)
```

Test:

- equal weights
- grid-searched nonnegative weights
- temperature-scaled log-odds if easy

This can help if models are independently confident on complementary cases, but it can also over-sharpen, so calibration controls are important.

### 6. Rank / Quantile Fusion

Convert each model score to its percentile rank using the `stack_val` distribution.

Train/evaluate:

- average ranks
- grid-weighted rank average
- logistic regression on ranks

This removes calibration differences and tests whether the models have complementary orderings.

### 7. Disagreement-Aware Fusion

Use small regularized logistic regression with score and disagreement features:

```text
hlt_margin
edge_margin
point_margin
abs(hlt_margin - edge_margin)
abs(hlt_margin - point_margin)
abs(edge_margin - point_margin)
hlt_confidence
edge_confidence
point_confidence
```

Optional additional features:

- entropy per model
- max probability per model
- pairwise rank disagreement

Keep this intentionally small. It is a diagnostic for whether local models help specifically in uncertain or disagreement regimes.

## Negative Controls

Negative controls are mandatory, because fusion can otherwise look better through calibration artifacts.

### Baseline-Only Stacker

Train logistic regression using only HLT ParT margin.

The real multi-model stacker must beat this, not just the raw baseline.

### Row-Shuffled Local Scores

Shuffle local model scores across jets on `stack_val` and `final_test` while keeping HLT baseline scores fixed.

Run:

- HLT + shuffled EdgeConv
- HLT + shuffled PointAttention
- HLT + shuffled EdgeConv + shuffled PointAttention

If shuffled local scores help, the method is suspect.

### Label-Shuffled Stacker

Train the stacker on `stack_val` with shuffled labels.

This should not beat baseline.

## Outputs

Write:

```text
fusion_report.json
fusion_report.md
fusion_metric_table.csv
fusion_weights.csv
fusion_controls.csv
fusion_prediction_manifest.json
```

Each fusion row should include:

- method name
- model set
- selected stack-val metric
- final-test accuracy
- final-test AUC
- final-test FPR@30
- final-test FPR@50
- final-test background rejection@30/50
- delta vs raw HLT baseline
- delta vs calibrated HLT-only stacker
- selected weights or coefficients
- whether this is a negative control

The report should clearly identify:

```text
best_valid_fusion_by_fpr50
best_control_by_fpr50
raw_hlt_baseline
calibrated_hlt_baseline
```

## Interpretation Rules

A fusion result is interesting only if it beats both:

```text
raw HLT ParT baseline
calibrated HLT-only baseline
```

on `final_test` FPR@50.

A fusion result is suspicious if:

- row-shuffled controls also improve,
- label-shuffled stacker improves,
- final-test improvement is large but stack-val improvement is absent,
- the selected fusion assigns nearly all weight to a weaker model while hurting AUC.

## Implementation Steps

1. Add a `teacher_logit_reco/local_graph_part/fusion.py` module with prediction loading, metric helpers, simple fusion feature builders, and stacker implementations.

2. Add a script `scripts/run_local_graph_score_fusion.py` that loads frozen checkpoints, writes prediction caches, runs all fusion families, and writes the final reports.

3. Add a Slurm wrapper `sbatch/run_local_graph_score_fusion.sh` for one existing experiment root.

4. Add tests with toy prediction arrays for:
   - FPR@50 selection direction,
   - baseline-only calibration,
   - weighted average selection,
   - logistic stacker shape and no-final-test-training rule,
   - row-shuffled controls.

5. Run first on the existing QCD-vs-Hgg HLT 0.6 Step 10 root:

```text
local_graph_part_step10_qcd_hgg_binary_hlt0p6_20260627_075757
```

6. If fusion beats HLT ParT, add a follow-up plan to distill or fold the complementary signal back into a single deployable HLT-only architecture.

