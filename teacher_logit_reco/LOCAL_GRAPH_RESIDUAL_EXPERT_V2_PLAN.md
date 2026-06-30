# Local Graph Residual Expert V2 Plan

## Purpose

This plan describes the V2 implementation of the local-graph residual expert.
The V1 residual expert is a useful pilot, but it still uses a widened ParT-like
classifier output as the residual embedding. V2 should be the serious version:
it anchors directly on the exact trained HLT ParT baseline and uses that
baseline's real internal representation.

The fixed working regime is:

- Task: binary QCD vs Hgg
- Inference view: HLT only
- HLT degradation strength: 0.6
- Primary metric: `FPR@50`, false-positive rate at 50% Hgg efficiency
- Baseline to improve: exact HLT ParT checkpoint trained on the same HLT cache
- Residual expert: local graph model that predicts an additive correction to
  the frozen HLT ParT logit margin

The scientific question is:

> Given the exact HLT ParT score and representation, can a local-graph expert
> find correctable boundary mistakes that the baseline ParT leaves behind?

## Why V2 Exists

V1 did the right thing structurally:

```text
z_base = frozen HLT ParT margin
delta = local graph residual correction
score = z_base + gamma * delta
```

But the residual branch used a ParT wrapper configured with
`num_classes = embedding_dim`, then consumed those widened "logits" as a jet
embedding. That is not a correctness bug, but it is not ideal. It leaves an
architectural ambiguity:

```text
Is the residual expert using the real HLT ParT representation,
or just a newly trained classifier-head proxy?
```

V2 removes that ambiguity. It should consume:

```text
z_base(x)        = exact frozen 2-class HLT ParT margin
e_base(x)        = true frozen HLT ParT penultimate embedding
h_local(x)       = local graph representation from the HLT jet
c_base(x)        = safe score-conditioning features from model_train thresholds
delta(x)         = residual correction
score_v2(x)      = z_base(x) + gamma * delta(x)
```

This makes the residual expert a real "improve on HLT ParT" model rather than a
second classifier that happens to be fused with HLT ParT afterward.

## Non-Negotiable Constraints

1. HLT-only inference.

   The model may use HLT particles, the frozen HLT ParT logits, and the frozen
   HLT ParT embedding. It may not use offline particles or offline teacher
   predictions at inference.

2. Exact baseline anchor.

   The cached baseline score must come from the selected 2-class HLT ParT
   checkpoint, not from a reinitialized model and not from a widened-head proxy.

3. True embedding or fail loudly.

   V2 must not silently fall back to `num_classes=128` logits as the embedding.
   If a true penultimate embedding cannot be extracted from Weaver ParT, the V2
   run should fail with a clear error and remain unsubmitted.

4. Shared conditioning reference.

   Thresholds such as `tau50` and `tau30` must be computed once from
   `model_train` baseline scores and reused for every split. Held-out labels
   must not define held-out conditioning features.

5. Split discipline.

   Residual training uses only `model_train` and `model_val`.
   `stack_train` and `stack_val` are reserved for later fusion.
   `final_test` is used only for final reporting.

6. Primary selection metric.

   Checkpoint selection should be based on fused `model_val` `FPR@50`, lower is
   better. Accuracy is not the win condition for this project.

## Core Architecture

The V2 model has three information streams:

```text
1. Frozen baseline score:
   z_base = logit_Hgg - logit_QCD

2. Frozen baseline representation:
   e_base = true HLT ParT penultimate embedding

3. Trainable local HLT branch:
   h_local = LocalGraphAdapter(HLT particles)
```

The residual head combines them:

```text
c_base = ScoreConditionMLP([
    z_base,
    sigmoid(z_base),
    z_base - tau50_train,
    abs(z_base - tau50_train),
    z_base - tau30_train,
    near_boundary_weight_train
])

h = ResidualBody([
    LayerNorm(e_base),
    LayerNorm(h_local),
    c_base
])

gate = sigmoid(GateHead(h))
raw_delta = DeltaHead(h)
delta = residual_scale * gate * raw_delta
score = z_base + gamma * delta
```

The frozen baseline representation gives the residual expert access to the
exact features ParT used to make its decision. The local graph branch gives the
expert a different inductive bias: explicit eta-phi neighborhood reasoning.

## Embedding Extraction

This is the most important engineering piece.

### Preferred Path: Explicit ParT Embedding Wrapper

Add an HLT ParT embedding anchor that wraps the Weaver `ParticleTransformer`
model and returns both:

```text
logits:    [batch, 2]
embedding: [batch, embed_dim]
```

The intended embedding is the jet-level representation immediately before the
final classifier head. In a CLS-token transformer this is usually the final CLS
token after the class-attention blocks and before the final fully connected
classification layer.

The wrapper should expose a method like:

```python
forward_with_embedding(points, features, lorentz_vectors, mask)
```

and should guarantee:

```text
classifier_head(embedding) == logits
```

within numerical tolerance for the unmodified baseline checkpoint.

### Fallback Path: Forward Hook Into Final Head

If Weaver does not expose a clean embedding API, use a targeted forward hook on
the final classifier head. The hook should capture the final head input during
the normal forward pass.

The fallback must be strict:

- Locate exactly one final classifier module.
- Capture a 2D `[batch, embedding_dim]` tensor.
- Verify the captured tensor can reproduce the model logits through the final
  head, if the head is directly callable.
- Record `embedding_source = "final_head_forward_hook"` in metadata.
- Fail if the final head cannot be identified.

### Disallowed Fallbacks

These are not V2:

```text
num_classes=128 widened classifier logits
raw HLT summary MLP only
ordinary score-only fusion
local graph logits used as embedding
```

They can be useful controls, but the V2 serious path must use an exact frozen
HLT ParT embedding.

## Baseline Embedding Cache

After loading the selected HLT ParT baseline, write a V2 cache for:

```text
model_train
model_val
stack_train
stack_val
final_test
```

For training, only `model_train` and `model_val` are used. The other splits are
cached for final report/fusion convenience, but final-test metrics should not be
computed during cache creation.

Each split cache should store:

```text
logits:              [N, 2]
margin:              [N]
prob_hgg:            [N]
embedding:           [N, D]
labels:              [N]
indices:             [N]
condition_features:  [N, C]
```

Each metadata file should store:

```text
contract: local_graph_residual_expert_v2_baseline_embedding_cache_v1
split
num_jets
label_names
positive_class_name
checkpoint_path
checkpoint_identity_hash
model_config
embedding_dim
embedding_source
embedding_reproduces_logits_check
hlt_degradation_strength
hlt_cache_dir
hlt_content_hash
jet_identity_hash
split_manifest_hash
condition_reference:
  source_split: model_train
  tau50
  tau30
  near_boundary_scale
metric_splits:
  model_train
  model_val
```

Final-test cache metadata may contain labels for alignment because the dataset
loader materializes them, but it should not contain final-test performance
metrics unless the final-report job requested them.

## Cache Alignment

The training and report loaders must check:

```text
same checkpoint identity
same hlt_content_hash
same jet_identity_hash
same split_manifest_hash when available
same label_names
same positive class
same condition reference
same split indices/order
same embedding_dim across splits
```

If a residual training dataset and a baseline embedding cache disagree, fail.
Do not rely only on labels or row counts.

## V2 Residual Model

The V2 residual model should be separate from V1:

```text
teacher_logit_reco/local_graph_part/residual_v2_model.py
```

Suggested classes:

```text
HLTPartEmbeddingAnchorConfig
HLTPartEmbeddingAnchorOutput
HLTPartEmbeddingAnchor
LocalGraphResidualExpertV2Config
LocalGraphResidualExpertV2Output
LocalGraphResidualExpertV2
```

The trainable model should take:

```python
tokens
mask
baseline_logit
baseline_embedding
baseline_condition_features
```

and return:

```text
fused_logits
residual_logits
correction_logits
delta
gamma
gate
diagnostics
```

### Local Branch

The first serious local branch should reuse the best current local graph
machinery:

```text
local_point_attention_adapter
k = 16
eta-phi kNN
geometry-aware local attention
```

It should produce a compact jet-level representation:

```text
h_local = pool(LocalGraphTokens(HLT particles))
```

Reasonable pooling choices:

- attention pooling over particle tokens
- masked mean plus max pooling
- CLS token if already available from the local graph branch

The first implementation should keep this simple and robust. The key novelty is
the exact ParT embedding anchor, not another local-graph redesign.

### Residual Head

Use a conservative residual head:

```text
input = [
  LayerNorm(e_base),
  LayerNorm(h_local),
  ScoreConditionMLP(c_base)
]

hidden = MLP(input)
gate = sigmoid(gate_head(hidden))
delta_raw = delta_head(hidden)
delta = gate * delta_raw
score = z_base + gamma * delta
```

Initialize so the model starts close to the baseline:

```text
delta_head std: 1e-3
gate bias: negative, e.g. -1.0
gamma initial: 0.1
gamma selection: validation shrinkage over learned delta
```

The learned delta should be small at the start. If the residual expert helps,
training should grow it.

## Loss Ladder

Keep the same staged ladder, but do not submit duplicate E training jobs.

### A: Additive Residual Plus Weighted BCE

```text
loss = weighted_BCE(z_base + gamma * delta, y)
     + lambda_delta * mean((gamma * delta)^2)
```

Weights should emphasize:

- QCD near or above the baseline FPR@50 boundary
- Hgg near or below the baseline 50% signal boundary
- normal jets with small baseline weight so the model remains calibrated

This is the plumbing baseline.

### B: Boundary Pairwise Loss

This mode is useful for ablation but probably too unconstrained alone.

It compares hard QCD jets against boundary Hgg jets:

```text
score(signal_boundary) > score(hard_background) + margin
```

It teaches the residual expert the ordering we care about near FPR@50.

### C: Boundary Pairwise Loss Plus Tiny BCE Anchor

This is the first serious mode:

```text
loss = boundary_pairwise_loss
     + small_bce_anchor
     + residual_size_penalty
```

The tiny BCE anchor prevents the residual expert from learning pathological
rankings that help only the pairwise term.

### D: Boundary Pairwise Plus Soft-FPR@50 Plus Tiny BCE Anchor

This is the most serious V2 training mode:

```text
loss = boundary_pairwise_loss
     + soft_fpr50_loss
     + small_bce_anchor
     + residual_size_penalty
```

The soft-FPR term should target background scores around the baseline's
train-derived `tau50`.

### E: Validation Shrinkage Policy, Not Training Mode

E should not be a separate cluster training job.

For every trained mode A-D, report:

```text
learned_gamma row:
  score = z_base + learned_gamma * delta

validation_shrunk row:
  score = z_base + gamma_val * (learned_gamma * delta)
```

where `gamma_val` is selected on `model_val`, not final_test.

## Validation Shrinkage

Gamma shrinkage should apply to the learned correction:

```text
delta_learned = learned_gamma * delta
score(gamma_val) = z_base + gamma_val * delta_learned
```

This avoids scale non-identifiability between `delta` and learned gamma.

Recommended grid:

```text
gamma_grid = [
  0.0,
  0.01,
  0.02,
  0.05,
  0.1,
  0.2,
  0.35,
  0.5,
  0.75,
  1.0,
  1.25,
  1.5,
]
```

If validation selects `0.0`, the residual expert did not help.

## Diagnostics

Report diagnostics for learned and validation-shrunk scores separately:

```text
baseline_fpr50
fused_fpr50
delta_fpr50
baseline_auc
fused_auc
baseline_accuracy
fused_accuracy
selected_gamma
learned_gamma
residual_mean
residual_std
residual_abs_mean
gate_mean
gate_min
gate_max
embedding_norm_mean
embedding_norm_std
local_embedding_norm_mean
condition_feature_stats
```

Boundary diagnostics:

```text
baseline_false_positives_at_fpr50
fused_false_positives_at_fpr50
false_positive_intersection
old_false_positives_removed
new_false_positives_introduced
baseline_false_negatives_near_tau50
fused_false_negatives_near_tau50
hard_qcd_score_shift_mean
boundary_hgg_score_shift_mean
```

The important question is not just whether FPR@50 moves. It is whether the
residual expert removes baseline mistakes instead of merely recalibrating the
score.

## Controls

V2 should include or enable these controls:

1. Alpha/gamma zero:

```text
score = z_base
```

2. Calibration-only:

```text
fit logistic/isotonic calibration on z_base only
```

3. Embedding-only residual:

```text
delta = f(e_base, c_base)
```

This tests whether local graph information is adding anything beyond ParT's own
embedding.

4. Local-only residual:

```text
delta = f(h_local, c_base)
```

This tests whether the ParT embedding is necessary.

5. Shuffled conditioning:

```text
shuffle z_base/c_base across jets within split
```

This should hurt if conditioning is meaningful.

6. Label-shuffled residual:

```text
train residual with shuffled labels
```

This should not improve FPR@50.

7. Same-architecture HLT seed ensemble:

Compare V2 against ordinary extra HLT ParT seeds if available. If an HLT ParT
seed ensemble gives the same gain, the result is less distinctive.

## Reporting

The V2 report should compare:

```text
baseline HLT ParT
V1 residual expert, if present
V2 residual expert learned-gamma rows
V2 residual expert validation-shrunk rows
V2 residual-only control
calibration-only control
score-fusion control, if available
standalone local graph variants, if available
```

Primary ranking for binary QCD vs Hgg:

```text
final_test_fpr_at_signal_eff_0p50
lower is better
```

Secondary metrics:

```text
FPR@30
background rejection at 50%
AUC
accuracy
boundary correction diagnostics
runtime
parameter count
```

The report should never declare a winner by accuracy for the binary QCD/Hgg
paper table.

## Suggested Files

Add new V2 files rather than overloading V1:

```text
teacher_logit_reco/local_graph_part/residual_v2_anchor.py
teacher_logit_reco/local_graph_part/residual_v2_cache.py
teacher_logit_reco/local_graph_part/residual_v2_model.py
teacher_logit_reco/local_graph_part/residual_v2_train.py
teacher_logit_reco/local_graph_part/residual_v2_report.py
scripts/cache_local_graph_residual_v2_anchor.py
scripts/train_local_graph_residual_v2_expert.py
scripts/write_local_graph_residual_v2_report.py
sbatch/run_cache_local_graph_residual_v2_anchor.sh
sbatch/run_train_local_graph_residual_v2_expert.sh
sbatch/run_write_local_graph_residual_v2_report.sh
sbatch/submit_local_graph_residual_v2_experiment.sh
tests/test_local_graph_residual_v2_anchor.py
tests/test_local_graph_residual_v2_cache.py
tests/test_local_graph_residual_v2_model.py
tests/test_local_graph_residual_v2_train.py
tests/test_local_graph_residual_v2_report.py
```

If code reuse is clean, shared helpers can live beside V1. But V2 contracts
should remain explicit so old V1 reports are not misread as V2.

## Implementation Steps

1. Add the V2 plan and constants.

   - Define V2 contract names.
   - Define canonical task metadata: QCD vs Hgg, HLT0.6, binary labels,
     `FPR@50` primary metric.

2. Implement `HLTPartEmbeddingAnchor`.

   - Load exact selected HLT ParT baseline checkpoint.
   - Freeze the model.
   - Return logits plus true penultimate embedding.
   - Fail if only widened-head proxy embeddings are available.
   - Save embedding source diagnostics.

3. Add embedding extraction tests.

   - Logits from normal forward and embedding forward match.
   - Embedding is 2D `[batch, dim]`.
   - Frozen anchor has no trainable baseline parameters.
   - Failure mode is explicit when no embedding source can be found.

4. Implement V2 baseline embedding cache.

   - Cache logits, margin, probabilities, embeddings, labels, indices, and
     condition features.
   - Compute condition reference from `model_train` only.
   - Avoid final-test metrics during cache creation by default.

5. Add strict cache alignment.

   - Verify checkpoint identity, HLT cache hashes, jet identity hashes, split
     indices, labels, embedding dim, and condition reference.
   - Add tests that stale or mismatched caches fail.

6. Implement `LocalGraphResidualExpertV2`.

   - Inputs: HLT tokens/mask, baseline margin, baseline embedding, condition
     features.
   - Local point-attention branch over HLT particles.
   - Residual head consumes `[e_base, h_local, c_base]`.
   - Outputs fused logits, correction logits, delta, gamma, gate, diagnostics.

7. Implement V2 loss modes A-D.

   - Reuse safe V1 losses where possible.
   - Keep E as report-time shrinkage policy only.
   - Ensure all losses are finite and have gradients into the residual branch.

8. Implement V2 training loop.

   - Train only on `model_train`.
   - Select checkpoint on `model_val` fused `FPR@50`.
   - Save learned-gamma predictions and correction logits for model_val.
   - Record selected loss mode, cache identity, and embedding contract.

9. Implement validation shrinkage over learned correction.

   - Select `gamma_val` on `model_val`.
   - Store `shrinkage_applies_to = learned_correction_delta`.
   - Report learned and shrunk metrics separately.

10. Add CLI scripts.

    - Cache anchor embeddings.
    - Train one V2 residual expert mode.
    - Write final V2 report.
    - Require explicit final-test confirmation for final reporting.

11. Add Slurm scripts.

    - Use existing QCD/Hgg HLT0.6 split/cache roots.
    - Use existing selected HLT ParT baseline checkpoint.
    - Queue cache, A/C/D training modes, and report.
    - Do not submit E as training.

12. Add report builder.

    - Compare baseline, V1 if present, V2 A/C/D learned, V2 A/C/D shrunk, and
      controls.
    - Rank binary rows by final-test `FPR@50`.
    - Include boundary mistake-overlap diagnostics.

13. Add controls.

    - Calibration-only baseline.
    - Embedding-only residual.
    - Local-only residual.
    - Shuffled conditioning smoke run.
    - Optional label-shuffled residual smoke run.

14. Add focused tests.

    - Anchor embedding extraction.
    - Cache alignment and condition-reference safety.
    - Model forward/gradient checks.
    - Loss mode finite checks.
    - Gamma shrinkage semantics.
    - Report ranking by FPR@50.
    - Slurm script environment checks.

15. Run a small pilot.

    - Use a small max-jet cap.
    - Confirm V2 cache aligns with the existing HLT cache.
    - Confirm residual starts near baseline.
    - Confirm gamma shrinkage can choose zero if the expert does not help.

16. Run the serious QCD/Hgg HLT0.6 experiment.

    - Use the existing 3M/1M/1M split/cache if available.
    - Train A, C, and D.
    - Report learned and validation-shrunk rows.
    - Compare against baseline, local graph, score fusion, and V1 residual if
      available.

## Success Criteria

V2 is useful if:

```text
V2 validation-shrunk FPR@50 < exact HLT ParT FPR@50 on model_val
and
the gain survives final_test
and
calibration-only and shuffled controls do not reproduce the gain
```

V2 is especially interesting if:

```text
it removes a measurable subset of HLT ParT false positives near FPR@50
and
the local+embedding residual beats embedding-only and local-only controls
```

V2 has failed cleanly if:

```text
gamma_val selects 0
or
the residual only improves AUC/accuracy but not FPR@50
or
embedding-only control matches the full local+embedding residual
or
the true ParT embedding cannot be extracted
```

The final point is important: a failed embedding extraction is not a reason to
silently fall back to V1. It is a sign that V2 needs a better Weaver integration
before it deserves cluster time.
