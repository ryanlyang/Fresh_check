# Local Graph Residual Expert Plan

This is an implementation plan for a diversity-aware fusion path aimed at beating the HLT Particle Transformer in the current working regime:

- Task: binary QCD vs Hgg
- Inference view: HLT only
- HLT degradation strength: 0.6
- Primary metric: false-positive rate at 50% Hgg efficiency, lower is better
- Baseline: strong HLT ParT trained normally on the same HLT cache and splits
- First residual expert: local point-attention graph adapter with the real/reference ParT backbone

The goal is not to train another standalone classifier and hope its scores are complementary. The goal is to train a model whose explicit job is to improve a frozen HLT ParT score near the operating region that determines FPR@50.

## Core Idea

Use additive residual logits:

```text
z_base(x) = frozen HLT ParT logit
r_phi(x, z_base) = residual expert logit correction
s(x) = z_base(x) + alpha * r_phi(x, z_base)
```

The residual expert sees the HLT jet and a small set of frozen baseline-score features. It predicts a correction, not a full replacement for HLT ParT.

This is Option B from the design discussion:

```text
local graph sees the HLT jet plus z_base(x)
outputs residual r(x, z_base)
final score = z_base(x) + alpha * r(x, z_base)
```

This is closer to boosting than ordinary ensembling. The second model is asked:

> Given what HLT ParT already believes, what correction should be made?

## Why This Is Different From Score Fusion

The existing score-fusion run trains several normal classifiers independently, then combines their scores. That can help, but the models are not trained to cover different error modes. If their predictions are highly overlapping, fusion has little to use beyond calibration.

The residual expert changes the training target:

```text
standalone training:
  learn y from x

residual expert training:
  learn how z_base should change, especially near the FPR@50 boundary
```

If successful, the local graph branch should produce a more meaningful fusion signal because it was trained to attack the baseline's operating-point mistakes.

## Split Discipline

The split protocol must be strict:

```text
model_train:
  train the HLT ParT baseline
  train residual/complement experts

model_val:
  select baseline checkpoint
  select residual expert checkpoint
  tune residual-loss hyperparameters

stack_train:
  train final score fusion or logistic stacker only

stack_val:
  select fusion method and fusion hyperparameters only

final_test:
  untouched until final reporting
```

The residual loss is built from baseline predictions on `model_train`, with checkpoint selection on `model_val`. `stack_train` and `stack_val` are reserved for the final fusion layer and should not be used to train residual experts.

## Baseline Logit Cache

After the HLT ParT baseline is trained and selected, freeze it and cache logits for:

```text
model_train
model_val
stack_train
stack_val
final_test
```

For residual expert training, only `model_train` and `model_val` are needed. The stack and final splits are cached for later fusion/reporting.

Each cached split should include:

```text
logits: [n_jets, 2] or binary positive logit/margin
positive_logit_or_margin: [n_jets]
probability_Hgg: [n_jets]
labels: [n_jets]
indices: [n_jets]
baseline_thresholds:
  threshold_at_signal_eff_0p50
  threshold_at_signal_eff_0p30
baseline_metrics:
  FPR@50
  FPR@30
  AUC
  accuracy
```

The binary residual logit should use a consistent baseline score:

```text
z_base = logit_Hgg - logit_QCD
```

This is the natural binary log-odds margin. It also makes additive corrections well-defined.

## Baseline Score Conditioning

The residual expert should not receive the full baseline network state. It should receive a small jet-level conditioning vector derived from the frozen score:

```text
c_base = [
  z_base,
  sigmoid(z_base),
  z_base - tau50_base,
  abs(z_base - tau50_base),
  z_base - tau30_base,
  baseline_low_margin_indicator_or_soft_weight
]
```

The initial implementation should inject `c_base` only into the final residual head:

```text
HLT particles -> local graph + ParT backbone -> jet embedding h
c_base -> small MLP -> condition embedding c
r_raw = MLP([h, c])
gate = sigmoid(MLP([h, c]))
r = gate * r_raw
s = z_base + alpha * r
```

This gives the residual expert context about the baseline score without letting baseline-score shortcuts dominate every particle token.

Possible later ablations:

```text
no score conditioning
head-only score conditioning
particle-token score conditioning
local-adapter score conditioning
```

The expected first serious setting is head-only conditioning.

## Architecture

Start from the best existing local-graph individual variant:

```text
local_point_attention_adapter_warmstart
```

The model should use:

- real/reference HLT ParT backbone, not the old raw-token prototype
- local point-attention adapter
- warm start from the selected HLT ParT baseline when possible
- local graph kNN over eta-phi with HLT particles
- residual output head instead of ordinary classifier head

The model still consumes only HLT particles plus frozen HLT ParT score features at inference. This is still HLT-only because `z_base` is produced by an HLT-only model.

## Alpha Handling

The fused score is:

```text
s = z_base + alpha * r
```

Use one of three alpha modes:

1. Fixed alpha, default `alpha=1.0`
2. Learnable scalar alpha during residual training
3. Post-training validation shrinkage grid

For the first implementation, use:

```text
learnable alpha initialized small, e.g. 0.1
alpha constrained positive by softplus or clipped range
residual L2 penalty on alpha * r
```

Then always evaluate validation shrinkage:

```text
alpha_grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25]
```

The selected checkpoint should report both:

```text
learned-alpha metrics
best validation-shrinkage-alpha metrics
```

Final-test evaluation should use the alpha selected on `model_val` or later by the fusion protocol, not by `final_test`.

## Operating-Point Regions

The main physics/metric target is the HLT ParT FPR@50 boundary.

Define baseline regions on `model_train` from frozen baseline scores:

```text
tau50_base:
  score threshold giving 50% Hgg efficiency

signal_boundary:
  Hgg jets with baseline score quantile roughly [0.45, 0.55]

hard_background:
  QCD jets near or above tau50_base

near_threshold:
  jets with |z_base - tau50_base| below a quantile or score-window threshold
```

Prefer quantile-defined regions over raw score windows because logits can shift by model or calibration:

```text
early training:
  Hgg boundary quantile [0.40, 0.60]
  hard QCD top 10-20% by baseline score

later training:
  Hgg boundary quantile [0.45, 0.55]
  hard QCD top 5-10% or QCD above tau50_base
```

For stability, the first version should use baseline-defined regions. A later version can let regions follow the current fused score.

## Loss Ladder

The implementation should support a staged ladder so we can learn which term actually matters.

### A. Additive Residual Plus Weighted BCE

This is the plumbing baseline.

```text
L_A = weighted_BCE(y, z_base + alpha * r)
      + lambda_res * mean((alpha * r)^2)
```

Weights emphasize:

- baseline false positives near threshold
- baseline false negatives near threshold
- low-margin baseline predictions

Easy baseline-correct examples get lower but nonzero weight.

This loss should answer:

> Does residual-logit training work at all?

### B. Additive Residual Plus Boundary Pairwise Loss

Main metric-aligned candidate.

For a batch, select:

```text
S = Hgg boundary positives
B = hard QCD negatives
```

Then optimize:

```text
L_pair =
  mean_{j in hard QCD}
  mean_{i in Hgg boundary}
    softplus((s_j - s_i) / temperature)
```

This directly pushes dangerous QCD below the Hgg boundary region.

Use a CVaR/top-k option:

```text
compute per-QCD pairwise loss
take mean of top rho fraction
```

This focuses on the worst background tail.

### C. Boundary Pairwise Loss Plus Tiny BCE Anchor

This should be the first serious candidate:

```text
L_C = L_pair
      + lambda_bce * weighted_BCE(y, s)
      + lambda_res * mean((alpha * r)^2)
```

The BCE anchor is intentionally small. It prevents weird metric hacking on a narrow boundary slice while leaving the pairwise term in charge.

Expected default:

```text
lambda_bce = 0.05 to 0.20
lambda_res = 1e-4 to 1e-3
```

### D. Boundary Pairwise Loss Plus Soft-FPR@50 Plus Tiny BCE Anchor

This is the best-bet hybrid:

```text
L_D = L_pair
      + lambda_rate * L_soft_FPR50
      + lambda_bce * weighted_BCE(y, s)
      + lambda_res * mean((alpha * r)^2)
```

Soft FPR@50:

```text
tau50 = stop_gradient(current or baseline Hgg 50% threshold)

L_soft_FPR50 =
  mean_{QCD} sigmoid((s_qcd - tau50) / epsilon)
```

Start with `tau50_base`, then optionally add a current fused threshold variant later.

The rate term keeps the objective aligned with the exact final metric. The pairwise term supplies stronger positive/negative gradients around the boundary.

### E. D Plus Validation Alpha Shrinkage

After training D, tune alpha on `model_val`:

```text
s_alpha = z_base + alpha_grid_value * r
select alpha_grid_value minimizing model_val FPR@50
```

Report:

- raw learned alpha
- selected shrinkage alpha
- FPR@50 before/after shrinkage
- whether shrinkage collapses to alpha=0

If alpha selection collapses to zero, the residual expert did not provide a useful correction.

## Metrics

Every residual expert report should include:

### Standard Metrics

For baseline alone, residual alone, and fused score:

```text
accuracy
AUC
FPR@30
FPR@50
background rejection at 50% Hgg efficiency
confusion matrix
```

### Complement Metrics

Compute on `model_val`, `stack_val`, and final report:

```text
baseline_FP_near_tau50_corrected_fraction
baseline_FN_near_tau50_corrected_fraction
hard_QCD_tail_score_shift_mean
Hgg_boundary_score_shift_mean
residual_mean
residual_std
residual_abs_mean
alpha_value
fused_delta_FPR50_vs_baseline
```

Also report overlap:

```text
baseline false positives at FPR50
fused false positives at FPR50
intersection size
new false positives introduced
old false positives removed
```

These metrics determine whether the residual expert is genuinely correcting baseline mistakes.

## Controls

Required controls:

1. Alpha zero control:

```text
s = z_base
```

2. Residual random/shuffled score conditioning:

```text
shuffle z_base conditioning across jets
```

3. Label-shuffled residual expert:

```text
train residual objective with shuffled labels or shuffled residual target region
```

4. Same-architecture second seed:

```text
train another HLT ParT seed and compare ordinary HLT ensemble vs residual expert
```

5. Calibration-only control:

```text
fit monotonic/logistic calibration on z_base only
```

The residual result is only interesting if it beats calibration-only and same-architecture ensemble controls.

## Fusion After Residual Training

After residual experts are trained and selected, train a final fusion model on `stack_train` and select on `stack_val`.

Candidate fusion inputs:

```text
z_base
r_local
z_base + alpha_local * r_local
r_multiscale
z_base + alpha_local*r_local + alpha_multiscale*r_multiscale
model margins
near-threshold indicators
```

Fusion methods:

- logistic regression
- ridge/logistic with strong regularization
- isotonic/calibration-only baseline
- small MLP only if controls are strong

Fusion should report whether residual training made the scores more useful than ordinary independently trained local graph scores.

## Sequential Expert Extension

After the first local graph residual expert:

```text
z_1 = z_base + alpha_1 * r_local
```

Train a second expert, for example multiscale-subjet:

```text
z_2 = z_1 + alpha_2 * r_multiscale(x, z_1)
```

Then a third expert could be local compression/subtoken:

```text
z_3 = z_2 + alpha_3 * r_subtoken(x, z_2)
```

This is stagewise boosting over HLT-only views. Each stage should be selected on `model_val`, while final fusion remains reserved for `stack_train/stack_val`.

## Initial Implementation Target

Start with the local graph model:

```text
local_point_attention_adapter_warmstart
```

But implement the residual expert path so any existing local graph variant can be used:

```text
hlt_part_baseline
local_edgeconv_adapter
local_point_attention_adapter
local_point_attention_adapter_warmstart
```

The expected first run should include:

```text
residual_weighted_bce
residual_boundary_pairwise
residual_boundary_pairwise_bce_anchor
residual_boundary_pairwise_soft_fpr_bce_anchor
residual_boundary_pairwise_soft_fpr_bce_anchor_alpha_shrink
```

## Implementation Steps

1. Add baseline-logit prediction/cache script.
   - Inputs: HLT cache dir, baseline checkpoint, split names.
   - Outputs: per-split baseline logits, labels, indices, threshold metadata.

2. Add residual expert model wrapper.
   - Reuse local graph ParT model.
   - Replace classifier output with residual logit head.
   - Add baseline-score conditioning MLP.
   - Add gated residual head.

3. Add residual loss module.
   - Weighted BCE.
   - Boundary pairwise loss.
   - CVaR/top-k hard-negative aggregation.
   - Soft-FPR@50 term.
   - Residual L2 penalty.
   - Alpha shrinkage utility.

4. Add training loop for residual experts.
   - Load HLT batches and matching baseline logits.
   - Compute residual/fused logits.
   - Select by fused `model_val` FPR@50.
   - Save baseline/fused/residual metrics.

5. Add diagnostics.
   - Boundary correction metrics.
   - FP/FN overlap changes.
   - Residual distributions.
   - Alpha reports.

6. Add CLI script.
   - `scripts/train_local_graph_residual_expert.py`
   - Support loss modes A-E.
   - Require final-test confirmation for final evaluation.

7. Add Slurm runner for one residual expert.
   - `sbatch/run_train_local_graph_residual_expert.sh`
   - Uses existing HLT cache and baseline checkpoint.

8. Add experiment submitter.
   - Queue baseline if needed, baseline-logit cache, residual experts A-E, report, optional score fusion.
   - Reuse existing QCD/Hgg HLT0.6 binary caches when provided.

9. Add final report builder.
   - Compare baseline, standalone local graph, residual-fused local graph, and fusion controls.
   - Primary metric remains final-test FPR@50.

10. Add tests.
    - Shape and cache alignment.
    - Loss finite/gradient checks.
    - Boundary selection correctness.
    - Alpha shrinkage selection.
    - No stack/final leakage in residual training.

## Success Criteria

The first run is successful if:

```text
fused residual expert FPR@50 < frozen HLT ParT FPR@50 on model_val
and
the improvement survives stack_val/final_test reporting
and
controls do not match the gain
```

The result is compelling if:

```text
residual expert removes a measurable fraction of baseline FPR@50 false positives
without introducing an equal or larger new false-positive set
and
final fusion improves beyond calibration-only and same-architecture HLT ensemble controls
```

The result is not compelling if:

```text
alpha shrinkage selects alpha=0
or
the label/shuffled controls get the same gain
or
only accuracy/AUC improves while FPR@50 does not
```
