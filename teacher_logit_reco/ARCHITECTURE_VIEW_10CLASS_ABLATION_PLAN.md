# Architecture-View 10-Class Ablation and Transfer Plan

## Short Name

AV10 Ablation, or Architecture-View 10-Class Mechanism Ablation.

## Goal

The high-data AV10 run produced the clearest win over HLT ParT so far:

```text
+----------------------------------------------------+---------------------+
| Model / fusion                                     | Final-test accuracy |
+----------------------------------------------------+---------------------+
| HLT ParT baseline                                  | 0.745424            |
| standalone HLT4 ensemble                           | 0.748611            |
| best individual AV10 adapter                       | 0.752787            |
| best AV10 scalar fusion                            | 0.752929            |
+----------------------------------------------------+---------------------+
```

The result is strong enough that the next question is no longer only:

```text
Can architecture-view context beat HLT ParT?
```

The next question is:

```text
Why did it beat HLT ParT?
```

This plan is about separating the mechanism from the possible explanations:

```text
extra parameters
extra post-embedding compute
better optimization / fine-tuning dynamics
feature-conditioned residual context
architecture-specific latent views
general ParT improvement beyond HLT
```

The purpose is not to run every possible variant forever. The purpose is to run
the smallest set of clean experiments that can make the AV10 result harder to
dismiss and easier to improve.

## Current Result To Explain

The high-data 10-class HLT0.6 run used fixed HLT-degraded JetClass inputs with
classes:

```text
QCD
Hbb
Hcc
Hgg
H4q
Hqql
Zqq
Wqq
Tbqq
Tbl
```

The best single model was:

```text
av10_context_mlp_control
final-test accuracy: 0.752787
```

The best fusion was:

```text
group: av10_with_controls
mode: scalar_weighted_logit_mean
final-test accuracy: 0.752929
```

The selected scalar weights were approximately:

```text
av10_pcnn_context_to_part   0.436
av10_context_mlp_control    0.354
av10_pn_context_to_part     0.126
av10_pfn_context_to_part    0.049
av10_baseline_recheck       0.034
```

The architecture-view core without the MLP control also performed well:

```text
av10_architecture_view_core scalar fusion: 0.752869
```

That matters. The MLP control winning individually suggests the injection site
and feature-conditioned adapter are powerful. But the architecture-view core
nearly matching the best fusion suggests PN/PFN/PCNN-style views are also
carrying useful complementary signal.

## Main Scientific Questions

### Q1. Is this just more parameters?

If a larger vanilla ParT matches AV10, the story weakens:

```text
AV10 may not be a new representation idea.
It may just be an under-sized ParT baseline.
```

If AV10 beats a larger ParT, the story strengthens:

```text
The placement and conditioning of the extra computation matters.
```

### Q2. Is this just extra Transformer compute after embedding?

AV10 injects a residual into the ParT embedding stream. A skeptic can say:

```text
Maybe any extra post-embedding block helps.
```

The right control is a parameter-matched ParT refinement block:

```text
raw HLT particles
  -> canonical ParT embedding h_base
  -> extra lightweight self-attention / ParT-style block
  -> delta_h or h_refined
  -> normal ParT backbone
```

No PN, PFN, PCNN, or feature-view branch.

### Q3. Is the useful signal coming from canonical particle features?

The current `av10_context_mlp_control` uses canonical particle features to
produce a per-particle embedding residual:

```text
canonical PF / ParT features
  -> MLP context adapter
  -> delta_h
  -> h_base + gated delta_h
  -> ParT backbone
```

This may mean the key is not external architecture views, but a learned
feature-conditioned correction before global attention.

There are two different ways to test that idea:

```text
embedding-space repair:
  canonical features -> MLP -> delta_h -> add after ParT embedding

input-feature repair:
  canonical features -> MLP -> delta_F -> add before ParT embedding
```

The existing AV10 result tests the first path. The new LC MLP Delta ablation
tests the second path. This matters because the two variants answer different
scientific questions. An embedding-space residual lets the adapter speak in
ParT's hidden language. A feature-space residual forces the adapter to express
its correction as small, bounded edits to the canonical PF feature rows that
ParT already knows how to consume.

If both work, the strongest interpretation is that feature-conditioned
per-particle repair is genuinely useful. If only the embedding-space variant
works, the gain likely comes from latent representation shaping rather than a
literal correction of ParT input features. If only the input-feature variant
works, the simplest story is that HLT0.6 particles enter ParT in a slightly
miscalibrated coordinate system and a small learned feature repair makes the
existing backbone stronger.

### Q4. Is the gain from architecture-specific latent views?

The PCNN/PFN branches did not beat the MLP control individually, but they were
very close, and the core architecture-view fusion remained strong. That means
we should not throw away the architecture-view idea. Instead, we should test
whether architecture-view branches become more useful with:

```text
repeat seeds
clean scalar fusion
maybe PCNN/PFN only
possibly better gating
```

### Q5. Does the adapter help if ParT is frozen?

If a frozen-ParT adapter improves over the frozen baseline, then the adapter is
adding useful representation under a strict constraint:

```text
ParT weights cannot move.
Only the injected adapter path can help.
```

If it does not improve, the current gain may rely on joint fine-tuning. That is
not bad, but it changes the interpretation:

```text
the adapter may be a better fine-tuning pathway rather than an independent
source of extra representation.
```

### Q6. Does this transfer to offline ParT?

This is the biggest new test.

So far the AV10 result is in the HLT0.6 setting, where HLT degradation creates
missing, smeared, merged, and noisy particle information. A natural explanation
is:

```text
The adapter helps repair HLT-specific degradation.
```

But if the best single adapter also beats an offline ParT baseline, the result
becomes much broader:

```text
The adapter may be a general way to improve ParT taggers, not just HLT taggers.
```

If it does not transfer, that is still a clean result:

```text
The method is likely HLT/degradation-specialized.
```

Both outcomes are useful.

## HLT Ablation Matrix

All HLT ablations should use the same 10-class HLT0.6 split discipline as the
main AV10 high-data run.

Default splits:

```text
model_train: train neural taggers
model_val: select checkpoints
stack_train: fit fusion/stacking models only
stack_val: select fusion hyperparameters
final_test: evaluate once at the end
```

Primary metric:

```text
10-class accuracy on final_test
```

Secondary metrics:

```text
macro per-class accuracy
cross entropy
macro one-vs-rest AUC, where available
per-class accuracy
confusion matrix
binary projections such as QCD vs Hgg, QCD vs Hbb, QCD vs Tbqq
```

### A0. HLT ParT Baseline Recheck

Purpose:

```text
Anchor every comparison to the same data, split, preprocessing, and metric.
```

Architecture:

```text
raw fixed-HLT particles
  -> canonical HLT ParT
  -> 10-class classifier
```

Interpretation:

```text
All ablations should report absolute delta against this baseline.
```

### A1. Larger Vanilla HLT ParT

Purpose:

```text
Test whether AV10 is simply winning because it has more parameters.
```

Architecture:

```text
raw fixed-HLT particles
  -> larger canonical HLT ParT
  -> 10-class classifier
```

Possible scaling knobs:

```text
embed_dim
number of attention blocks
number of heads
FFN expansion
classification head width
dropout adjusted only if needed
```

Important constraint:

```text
No architecture views.
No context MLP adapter.
No post-embedding residual injection beyond normal ParT layers.
```

Report:

```text
parameter count
trainable parameter count
accuracy delta vs baseline
accuracy delta vs best AV10 single model
```

Interpretation:

```text
If larger ParT closes the gap, the AV10 gain may mostly be capacity.
If larger ParT does not close the gap, AV10's structure matters.
```

### A2. Parameter-Matched Extra ParT Block

Purpose:

```text
Test whether any extra post-embedding Transformer/ParT compute is enough.
```

Architecture:

```text
raw fixed-HLT particles
  -> canonical ParT embedding h_base
  -> small ParT-style self-attention refinement block
  -> delta_h or h_refined
  -> normal ParT backbone
  -> classifier
```

This should roughly match the trainable parameter count of the successful AV10
context adapter. It should not use PN/PFN/PCNN views or canonical-feature MLP
conditioning.

Two implementation options:

```text
residual form:
  h_final = h_base + gamma * block(h_base)

replacement/refinement form:
  h_final = block(h_base)
```

Prefer residual form first because it preserves the baseline path and matches
the AV10 residual-injection hypothesis.

Interpretation:

```text
If A2 matches AV10, the key may be latent refinement after embedding.
If AV10 beats A2, external/canonical conditioning probably matters.
```

### A3. ParT-Only Adapter

Purpose:

```text
Test whether h_base itself contains everything needed for the residual adapter.
```

Architecture:

```text
h_base
  -> tiny MLP
  -> delta_h
  -> h_base + gated delta_h
  -> ParT backbone
```

No raw canonical features are fed to the adapter. No PN/PFN/PCNN views.

Interpretation:

```text
If this works, the adapter is mostly refining ParT's own embedding.
If it fails relative to feature MLP, raw/canonical feature conditioning matters.
```

### A4. Feature MLP Adapter

Purpose:

```text
Re-run and standardize the current best individual mechanism.
```

This is the current `av10_context_mlp_control` idea:

```text
canonical particle features
  -> per-particle MLP
  -> delta_h
  -> h_base + gated delta_h
  -> ParT backbone
```

This variant is the main single-model benchmark for the ablation suite.

Interpretation:

```text
If A4 remains best, the strongest story is feature-conditioned embedding
residual adaptation.
```

### A5. LC MLP Delta Feature-Input Adapter

Purpose:

```text
Test the input-feature version of the current best MLP mechanism.
```

This is the 10-class HLT0.6 counterpart of the local-compression
`lc_mlp_delta` idea described in:

```text
teacher_logit_reco/LC_MLP_DELTA_EXPLAINED.md
```

The crucial difference from A4 is the insertion point:

```text
A4 feature MLP adapter:
  canonical particle features
    -> MLP
    -> delta_h
    -> h_base + gated delta_h
    -> ParT backbone

A5 LC MLP Delta:
  canonical particle features
    -> MLP
    -> bounded delta_F
    -> adapted canonical features = features + delta_F
    -> ParT embedding
    -> ParT backbone
```

This ablation should not edit the raw cached HLT particles. It should edit only
the canonical PF feature tensor passed into ParT. The following should remain
unchanged:

```text
raw fixed-HLT tokens
particle ordering
particle mask
ParT points
ParT Lorentz vectors
exact ParT backbone class
10-class label mapping
split discipline
checkpoint selection metric
```

The adapter should be deliberately conservative:

```text
raw_delta = MLP(canonical_feature_row)
bounded_delta = tanh(raw_delta)
delta_F = bounded_delta * feature_delta_scale * feature_active_mask
adapted_features = original_features + delta_F
```

Use the local-compression feature-delta philosophy:

```text
zero-init the final delta projection
start exactly or nearly exactly at baseline ParT
use feature-wise delta scales
allow PID and geometry deltas to be frozen or tightly limited
add a small delta_F L2 regularizer
log per-feature delta diagnostics
```

This should use the same 10-class HLT0.6 data, split manifest, HLT cache,
baseline checkpoint constraints, and reporting format as the rest of AV10. It
is not a new dataset, not a reconstruction task, and not a replacement for
ParT. It is a near-identity learned input-feature repair layer in front of the
same ParT.

Interpretation:

```text
If A5 matches or beats A4, input-feature repair may be the cleaner mechanism.
If A4 beats A5, latent embedding repair is more expressive than feature repair.
If A5 beats A0 but not A4, both mechanisms help, but hidden-space adaptation is
currently stronger.
If A5 fails entirely, the successful MLP control should not be interpreted as
literal PF feature calibration.
```

The most interesting future extension, if both A4 and A5 work, is a combined
model:

```text
features -> delta_F -> ParT embedding -> delta_h -> ParT backbone
```

But the first ablation should keep A5 isolated so the mechanism is readable.

### A6. Deeper/Wider Feature MLP Adapter

Purpose:

```text
Test whether the winning feature-MLP adapter is under-scaled.
```

Architecture:

```text
canonical particle features
  -> deeper/wider MLP
  -> delta_h
  -> h_base + gated delta_h
  -> ParT backbone
```

Suggested ladder:

```text
small: current A4 adapter
medium: 2x hidden width or one extra hidden layer
large: parameter-matched to PCNN/PFN view adapters or extra ParT block
```

Do not overrun the experiment with too many sizes. One larger version is enough
for the first pass.

Interpretation:

```text
If bigger MLP improves, scale the adapter.
If bigger MLP does not improve, the current adapter may already be capacity-saturated.
```

### A7. Frozen-ParT Adapter-Only

Purpose:

```text
Separate representation gain from joint fine-tuning gain.
```

Architecture:

```text
frozen canonical HLT ParT
train only:
  adapter
  adapter gate
  optionally classifier head, depending on checkpoint compatibility
```

Preferred strict version:

```text
freeze all ParT weights and classifier head
train only adapter/gate
```

If that is too brittle, use a slightly relaxed version:

```text
freeze ParT body
train adapter/gate/classifier head
```

The report must say which version was used.

Interpretation:

```text
If frozen adapter improves, adapter adds useful representation directly.
If frozen adapter does not improve, current gains likely depend on co-adaptation
with ParT fine-tuning.
```

### A8. Shuffled-Feature Adapter Control

Purpose:

```text
Test whether the feature adapter is exploiting meaningful particle information.
```

Architecture:

```text
canonical particle features
  -> feature shuffle / cross-jet shuffle / random projection control
  -> same adapter capacity
  -> delta_h
  -> ParT backbone
```

Controls to consider:

```text
within-feature random permutation across jets
within-particle feature group shuffle
fixed random Gaussian features with same dimension
random view embeddings with same norm statistics
```

The control should preserve broad scale and shape enough that it is a fair
capacity test, but destroy useful semantics.

Interpretation:

```text
If shuffled control improves similarly, beware capacity/regularization artifact.
If shuffled control fails, semantic feature conditioning matters.
```

### A9. PCNN-Context Repeat

Purpose:

```text
Check whether the strongest architecture-view branch is stable.
```

The high-data scalar fusion put the largest weight on PCNN-context:

```text
av10_pcnn_context_to_part weight: about 0.436
```

Architecture:

```text
PCNN-style per-particle/context view
  -> delta_h
  -> h_base + gated delta_h
  -> ParT backbone
```

Run at least one repeat seed if compute allows.

Interpretation:

```text
If PCNN-context repeats near 0.7526-0.7528, it is a real branch.
If it swings widely, fusion may have selected a lucky seed.
```

### A10. PFN-Context Repeat

Purpose:

```text
Check the other strong architecture-view branch.
```

PFN-context was nearly tied with PCNN-context individually:

```text
av10_pfn_context_to_part final-test accuracy: 0.752681
```

PFN has a different inductive bias: global set aggregation, less explicit local
particle interaction. If it remains strong, that suggests the view mechanism is
not only local-geometry driven.

### A11. Core Architecture-View Scalar Fusion

Purpose:

```text
Measure architecture-view complementarity without the generic MLP control.
```

Fusion group:

```text
av10_pn_context_to_part
av10_pfn_context_to_part
av10_pcnn_context_to_part
av10_all_views_to_part
```

Use stack-only selection:

```text
fit weights on stack_train
select on stack_val
evaluate once on final_test
```

Interpretation:

```text
If core fusion stays near the best result, architecture-specific views are real.
If it falls to baseline, the MLP/context adapter is the real discovery.
```

## Offline Transfer Matrix

This is not a control. It is a transfer test.

### Why Offline Matters

The HLT result could be HLT-specific:

```text
HLT degradation creates missing/smeared/merged particle information.
The adapter helps ParT repair that degraded input.
```

But if the same adapter helps offline ParT, then the implication is bigger:

```text
This may be a general ParT improvement strategy.
```

That would be very good news. It would mean the method is not merely a clever
HLT patch; it may be a broadly useful way to add learned per-particle latent
context before ParT's global attention.

### O0. Offline ParT Baseline

Purpose:

```text
Establish the normal offline 10-class ParT baseline on the same JetClass split.
```

Architecture:

```text
offline particles
  -> canonical offline ParT
  -> 10-class classifier
```

Important requirements:

```text
same label mapping
same train/val/test discipline
same final-test holdout policy
same checkpoint selection metric
same reporting outputs as HLT runs
```

The offline baseline should not use HLT-degraded inputs.

### O1. Offline Best Single Adapter

Purpose:

```text
Test whether the best single AV10 mechanism improves offline ParT.
```

Start with the simplest winning mechanism:

```text
offline canonical particle features
  -> feature MLP adapter
  -> delta_h
  -> h_base + gated delta_h
  -> offline ParT backbone
  -> 10-class classifier
```

Do not start with fusion. The first offline transfer test should be a single
model so the result is interpretable.

If compute allows, add one architecture-view offline branch:

```text
offline PCNN-context -> offline ParT
```

But the minimum transfer test is:

```text
offline baseline ParT
offline feature MLP adapter ParT
```

### Offline Interpretation

If O1 beats O0 by a similar margin:

```text
This may be a general ParT enhancement.
Next: run broader offline ablations and compare to larger offline ParT.
```

If O1 beats O0 by a smaller but real margin:

```text
The method transfers, but HLT degradation amplifies its value.
```

If O1 does not beat O0:

```text
The method is likely specialized to degraded HLT inputs.
That is still scientifically useful and still valuable for trigger/tagging.
```

If O1 overfits badly:

```text
The adapter may need stronger regularization or offline-specific scaling.
```

## Parameter Accounting

Every variant must report:

```text
total parameters
trainable parameters
ParT parameters
adapter parameters
classifier/head parameters
ratio vs HLT ParT baseline
```

This is required because the central challenge is attribution. Without parameter
counts, a larger-capacity explanation cannot be ruled out.

Suggested reporting table:

```text
variant
final_test_accuracy
delta_vs_baseline
total_params
trainable_params
adapter_params
best_epoch
cross_entropy
macro_per_class_accuracy
```

## Required Diagnostics

For adapter variants, log:

```text
delta_h_l2_mean
delta_h_l2_p90
delta_h_l2_max
gate_mean
gate_p10
gate_p90
adapter_output_norm
embedding_norm
delta_to_embedding_norm_ratio
```

For LC MLP Delta / input-feature repair variants, log:

```text
delta_F_l2_mean
delta_F_l2_p90
delta_F_l2_max
delta_F_abs_mean
delta_F_abs_p90
delta_F_abs_max
delta_F_to_feature_norm_ratio
per-feature delta_F_abs_mean
per-feature delta_F_abs_p90
per-feature delta_F_sq_mean
per-feature delta_F_rms
final_delta_projection_weight_norm
final_delta_projection_bias_norm
feature_delta_scale summary
PID/geometry delta freeze flags
```

The input-feature repair diagnostics are not optional. A5 is only interpretable
if the report shows whether it learned small stable repairs or large hidden
feature rewrites.

For feature-conditioned adapters, also log:

```text
input feature group norms
feature dropout/shuffle status
adapter parameter count
```

For frozen-ParT variants, report:

```text
which modules were frozen
which modules were trainable
gradient norm by trainable module group
```

## Report Layout

The final report should have four sections.

### 1. Main HLT Ablation Table

Rows:

```text
A0 HLT ParT baseline
A1 larger vanilla HLT ParT
A2 parameter-matched extra ParT block
A3 ParT-only adapter
A4 feature MLP adapter
A5 LC MLP Delta feature-input adapter
A6 wider/deeper feature MLP adapter
A7 frozen-ParT adapter
A8 shuffled-feature control
A9 PCNN-context repeat
A10 PFN-context repeat
```

Columns:

```text
final accuracy
delta vs baseline
cross entropy
macro per-class accuracy
best epoch
total params
trainable params
adapter params
```

### 2. Fusion and Complementarity Table

Rows:

```text
best individual
core architecture-view scalar fusion
with-controls scalar fusion
standalone HLT4 ensemble
```

Columns:

```text
stack_val selected accuracy
final_test accuracy
selected weights
included models
```

### 3. Offline Transfer Table

Rows:

```text
O0 offline ParT baseline
O1 offline feature MLP adapter
optional offline PCNN-context adapter
```

Columns:

```text
offline final accuracy
delta vs offline baseline
delta compared to HLT improvement
parameter count
```

### 4. Interpretation Summary

This should be generated as plain text in the report:

```text
Did larger ParT close the gap?
Did extra ParT block close the gap?
Did frozen adapter improve?
Did shuffled control fail?
Did offline transfer work?
Which hypothesis is most consistent with the results?
```

## Decision Rules

### Strong Evidence For AV10 As A Real Mechanism

The result is strongest if:

```text
A4/A5/A9/A10 beat A0
A1 does not match A4
A2 does not match A4
A8 does not improve
A7 improves at least partially
O1 improves over O0, or clearly shows HLT-specialized behavior
```

### Evidence For Capacity Explanation

The result is likely mostly capacity if:

```text
A1 larger ParT matches or beats AV10
A2 extra ParT block matches or beats AV10
A8 shuffled control improves similarly
```

### Evidence For Feature-Conditioned Adapter Explanation

The result points to feature-conditioned adaptation if:

```text
A4 beats A3
A4 beats A2
A8 fails
A5 improves or matches A4
A5 beats A0
```

If A4 wins and A5 also improves, the mechanism is likely feature-conditioned
adaptation with embedding-space repair currently stronger than input-feature
repair. If A5 wins, the cleaner story is learned canonical-feature repair
before ParT embedding.

### Evidence For Architecture-Specific Views

The result points to architecture-view complementarity if:

```text
A9/A10 repeat strongly
A11 core fusion beats A4
PCNN/PFN weights remain nontrivial under stack-selected scalar fusion
```

### Evidence For HLT-Specific Repair

The result is HLT-specific if:

```text
HLT adapter improves clearly
offline adapter does not improve
```

That would still be a good result. It would mean AV10 found a meaningful
degraded-input improvement strategy.

## Implementation Steps

### Step 1. Add Ablation Variant Registry

Create a focused registry for AV10 ablations:

```text
av10_hlt_baseline_recheck
av10_larger_part
av10_extra_part_block
av10_part_only_adapter
av10_feature_mlp_adapter
av10_lc_mlp_delta_features
av10_feature_mlp_adapter_wide
av10_frozen_part_feature_adapter
av10_shuffled_feature_adapter
av10_pcnn_context_repeat
av10_pfn_context_repeat
```

Each variant should declare:

```text
input source
adapter type
ParT size
freeze policy
shuffle/control policy
expected parameter-count target
whether it is a control or candidate
```

### Step 2. Implement Capacity and ParT-Only Controls

Add:

```text
larger vanilla ParT
parameter-matched extra ParT block
ParT-only h_base -> delta_h adapter
```

Make sure all three use the same data loader, split policy, checkpoint
selection, metrics, and report format.

### Step 3. Implement Feature-Adapter Ablations

Add:

```text
current feature MLP adapter as a named canonical variant
wider/deeper feature MLP adapter
frozen-ParT feature adapter
shuffled-feature adapter control
```

Add adapter diagnostics and parameter accounting here.

### Step 4. Implement Offline Transfer Runs

Add offline-compatible runners for:

```text
offline ParT baseline
offline feature MLP adapter
optional offline PCNN-context adapter
```

The offline runner should reuse the same reporting format but should make it
impossible to confuse offline and HLT inputs in the output metadata.

### Step 5. Implement Ablation Report

Write a final report that merges:

```text
HLT ablation rows
fusion rows
offline transfer rows
parameter accounting
diagnostics
interpretation summary
```

The report should explicitly answer the decision-rule questions rather than
only dumping tables.

### Step 6. Implement LC MLP Delta Input-Feature Ablation

Add the input-feature repair counterpart to the current feature MLP adapter:

```text
av10_lc_mlp_delta_features
```

This variant should follow the `lc_mlp_delta` design from the local-compression
work, but run inside the 10-class AV10 HLT0.6 ablation protocol:

```text
fixed HLT particles
  -> canonical ParT feature builder
  -> MLP predicts bounded delta_F for each particle feature row
  -> adapted features = features + delta_F
  -> exact 10-class HLT ParT backbone
  -> logits
```

Implementation requirements:

```text
reuse the same 10-class AV10 dataloaders and split checks
reuse the exact HLT0.6 cache and manifest discipline
warm start from the same 10-class HLT ParT baseline family
do not alter raw cached HLT tokens
do not alter ParT points, Lorentz vectors, or masks
zero-init the final delta projection
bound deltas with tanh and feature-wise scales
support freezing or strongly limiting PID and geometry deltas
add delta_F L2 regularization
log full delta_F diagnostics by epoch and in final reports
```

The priority comparison is:

```text
A4 av10_feature_mlp_adapter:
  feature-conditioned delta_h after ParT embedding

A5 av10_lc_mlp_delta_features:
  feature-conditioned delta_F before ParT embedding
```

This is the cleanest way to separate:

```text
learned latent context helps ParT
```

from:

```text
learned canonical feature repair helps ParT
```

Do not combine `delta_F` and `delta_h` in the first implementation. If both A4
and A5 are positive, add a later combined variant.

### Step 7. Queue Pilot Then High-Data Runs

Run a smaller pilot first if any new architecture path is risky:

```text
smoke/pilot:
  A1, A2, A3, A4, A5, A8, O0, O1

full:
  all A variants
  core fusion
  offline transfer
```

If the code paths are straightforward and compile cleanly, the high-data run can
be queued immediately after the pilot dependency.

## Preferred First Full Run Set

If compute is limited, prioritize:

```text
A0 HLT baseline recheck
A1 larger vanilla ParT
A2 parameter-matched extra ParT block
A3 ParT-only adapter
A4 feature MLP adapter
A5 LC MLP Delta feature-input adapter
A7 frozen-ParT feature adapter
A8 shuffled-feature control
O0 offline ParT baseline
O1 offline feature MLP adapter
```

Then add:

```text
A6 wide feature MLP
A9 PCNN repeat
A10 PFN repeat
A11 core scalar fusion
```

The first set answers the most important scientific question:

```text
Is the win real mechanism, or just capacity/adaptation?
```

The second set asks how to push the mechanism further.
