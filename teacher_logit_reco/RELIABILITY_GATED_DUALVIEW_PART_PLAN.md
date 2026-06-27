# Reliability-Gated Dual-View ParT Plan

This document defines the next focused architecture branch for the privileged
HLT/offline project:

```text
Reliability-Gated Dual-View Particle Transformer
```

The target setup is fixed for this branch:

```text
task:        QCD vs Hgg
HLT scale:   0.6 degradation
train size:  500k total jets
val size:    150k total jets
test size:   500k total jets
views:       original HLT + PN reconstructed view
metric:      FPR at 50% Hgg signal efficiency, with FPR at 30% also reported
```

The central lesson from the DETR/free-slot and set-matching experiments is:

```text
PN reconstruction appears useful.
The current five-view tagger is too weak as a tagger backbone.
```

The next model should therefore not replace the strong HLT ParT baseline with a
new multiview transformer. It should preserve the strong HLT ParT baseline and
add the PN reconstructed view only through gated residual paths.

## Core Principle

The deployed model still sees only HLT-derived information:

```text
inference:
  HLT jet -> PN reconstructor -> PN reconstructed view
  HLT jet + PN reconstructed view -> dual-view tagger -> prediction
```

No offline particles are used at inference.

Offline information is used only during training:

```text
training:
  paired offline jet
  offline labels
  optional offline teacher logits
  optional offline-vs-HLT reliability targets
```

The architectural rule is:

```text
The model must initialize as a strong HLT ParT, not as a weaker new tagger.
```

The PN view is allowed to help, but it should not be able to immediately damage
the baseline. The simplest way to enforce that is a zero-initialized residual
gate:

```text
final_logits = hlt_logits + gate(hlt, pn) * delta_logits(hlt, pn)
```

At initialization:

```text
gate ~= 0
delta_logits can be arbitrary
final_logits ~= hlt_logits
```

If the PN reconstructed view has useful information, training can open the gate.
If it does not, the model should fall back to the HLT ParT.

## Why Not the Current Five-View Tagger?

The current five-view tagger is useful as an experimental probe, but it failed a
basic baseline-preservation test:

```text
standard HLT ParT      final FPR@50 ~= 0.0204
DETR five-view hlt_only final FPR@50 ~= 0.0592
```

That means its HLT-only pathway is much weaker than the real HLT ParT. A
multiview gain inside that weaker architecture does not prove that the method
can beat the best HLT-only model.

The new branch should instead compare against:

```text
standard HLT ParT on HLT
offline ParT on offline
HLT ParT + PN residual adapter
HLT ParT + PN cross-attention adapter
```

The win condition is not:

```text
beats the five-view hlt_only ablation
```

The win condition is:

```text
beats the standard HLT ParT baseline on QCD/Hgg HLT0.6
```

## High-Level Architecture

The recommended first serious model has three parts:

```text
HLT anchor branch
PN reconstructed-view branch
Reliability-gated fusion head
```

### 1. HLT Anchor Branch

The HLT branch should be as close as possible to the standard HLT ParT baseline:

```text
HLT raw particles -> ParT embedding -> ParT blocks with pairwise bias -> HLT logits
```

This branch should be initialized from a trained QCD/Hgg HLT0.6 ParT checkpoint.
There are two acceptable training modes:

```text
frozen-anchor:
  keep HLT ParT fixed
  train only PN encoder + adapter + residual head

warm-start-anchor:
  initialize from HLT ParT
  train the whole model with a small LR on HLT ParT and larger LR on new modules
```

The first run should use frozen-anchor because it gives the cleanest
interpretation:

```text
Any gain must come from the PN view or residual adapter.
```

After frozen-anchor is understood, warm-start-anchor can be tried.

### 2. PN Reconstructed-View Branch

The PN branch consumes the PN reconstructed view:

```text
PN reconstructed particles
PN existence/confidence scores
PN masks
optional reconstruction diagnostics
```

The branch should be intentionally smaller than the HLT anchor. Its role is not
to replace ParT; it is to extract useful residual evidence from the PN
reconstruction.

Recommended PN encoder:

```text
PN reco tokens -> particle feature embedding
              -> confidence embedding
              -> 2-4 lightweight transformer/ParT-style blocks
              -> PN memory tokens + PN pooled context
```

The PN branch should preserve confidence information:

```text
token_feature = raw_reco_feature_embed
              + view_type_embed("pn_reco")
              + confidence_embed(existence_prob)
```

The PN branch should not require matching HLT particles to PN particles. The
model should learn cross-view relationships through attention and pooled
summary features.

### 3. Reliability-Gated Fusion

The first production-worthy fusion should combine two mechanisms:

```text
global residual logit correction
optional cross-attention residual into HLT context
```

#### Global Residual Logit Correction

This is the safest first version:

```text
hlt_context = pooled representation from HLT ParT
pn_context  = pooled representation from PN encoder

delta_logits = MLP([hlt_context, pn_context, disagreement_features])
gate         = sigmoid(MLP([hlt_context, pn_context, disagreement_features]) + gate_bias)

final_logits = hlt_logits + gate * delta_logits
```

Initialize:

```text
last delta layer weights near 0
gate_bias = -4 to -6
```

This makes the initial model nearly identical to HLT ParT.

#### Cross-Attention Adapter

The second version should add an HLT-to-PN cross-attention adapter:

```text
HLT particle tokens query PN memory tokens.
The adapter output is projected back into the HLT token dimension.
A learned scalar/vector gate controls the residual strength.
```

Shape:

```text
hlt_tokens' = hlt_tokens + adapter_gate * CrossAttention(Q=hlt_tokens, K/V=pn_tokens)
```

Again:

```text
adapter_gate starts near 0
```

This follows the local/global lesson from the research report: preserve the
strong structured particle-level ParT reasoning, then inject the extra view as a
controlled residual source.

## Reliability Features

The gate should see more than just pooled embeddings. It should receive
handful-size diagnostic features that tell it when PN reconstruction might be
trustworthy:

```text
HLT confidence / margin
HLT entropy
PN predicted particle count
PN mean/max existence confidence
PN count vs HLT count
PN jet pt / eta / phi / energy summary
HLT-vs-PN jet summary disagreement
nearest-neighbor HLT/PN geometry disagreement
```

These features are not intended to dominate the model. They are useful because
they make reliability gating easier to learn.

## Training Objectives

The primary objective is supervised binary classification:

```text
L_cls = cross_entropy(final_logits, y)
```

The anchor-preserving objective should be explicit:

```text
L_anchor = KL(stopgrad(hlt_logits) || final_logits)
```

or equivalently a small penalty on the residual:

```text
L_residual = mean(||gate * delta_logits||^2)
```

Recommended first loss:

```text
L = L_cls
  + lambda_anchor * KL(softmax(hlt_logits / T), softmax(final_logits / T))
  + lambda_gate * mean(gate)
  + lambda_delta * mean(||delta_logits||^2)
```

Use small regularization weights:

```text
lambda_anchor = 0.05 to 0.2
lambda_gate   = 0.001 to 0.01
lambda_delta  = 0.0001 to 0.001
T             = 2
```

The anchor KL can be annealed down after a few epochs:

```text
early training: preserve HLT strongly
late training: allow PN branch to correct HLT where useful
```

Optional privileged training signal:

```text
offline_teacher_logits -> soft target
```

This should be tested only after the hard-label version is stable. The first
goal is to prove that PN reconstruction improves a strong HLT baseline.

## Evaluation

The main comparison table should always include:

```text
standard HLT ParT
offline ParT on offline
frozen HLT ParT anchor inside dual-view model
HLT ParT + PN residual head
HLT ParT + PN cross-attention adapter
HLT ParT + PN residual + cross-attention
```

Primary metric:

```text
FPR at 50% Hgg signal efficiency
```

Secondary metrics:

```text
FPR at 30% Hgg signal efficiency
AUC
accuracy
per-class accuracy
calibration / confidence
gate mean by class
gate mean by HLT correctness bucket
gate mean by HLT confidence bucket
```

The diagnostic question is:

```text
Does the gate open on jets where HLT ParT is uncertain or wrong?
```

If the model improves but gate behavior is random, the result is harder to
interpret. If the model improves and the gate opens on meaningful regimes, the
story is much stronger.

## Ablations

Run only a small number of ablations at first:

```text
HLT ParT baseline
HLT anchor reimplemented in wrapper, no PN
HLT + shuffled PN view
HLT + PN residual correction
HLT + PN cross-attention
HLT + PN residual correction + cross-attention
```

Important negative controls:

```text
shuffled PN view across jets
random PN confidence scores
gate forced to zero
gate forced to one
```

The shuffled PN view is crucial. If shuffled PN performs similarly to real PN,
the model is exploiting distributional artifacts rather than event-specific
reconstruction information.

## Storage Strategy

This branch should not create four full reconstructed-view caches. It should use
only PN:

```text
HLT cache
PN reconstructed-view cache
tagger checkpoints
diagnostics/final reports
```

For large 500k/150k/500k runs, avoid storing unnecessary intermediate arrays.
If possible:

```text
store PN reconstructed views as float16
store confidence/mask as compact dtypes
delete PN cache after final report if reproducible from reconstructor checkpoint
```

The first implementation can reuse existing PN cache machinery, but the long-run
design should support on-the-fly PN reconstruction to avoid 20-30 GB experiment
roots.

## Implementation Steps

### Step 1: Define The Experiment Contract

Create a small config module or dataclass for the dual-view branch:

```text
task labels: QCD, Hgg
HLT degradation: 0.6
train/val/test sizes: 500k/150k/500k
primary metric: fpr_at_signal_eff_0p50
positive class: Hgg
views: hlt, pn_reco
```

The config should make it hard to accidentally run a different label pair or
degradation setting.

### Step 2: Add A Baseline HLT ParT Loader

Implement a loader that can:

```text
load a trained QCD/Hgg HLT0.6 ParT checkpoint
return logits
optionally return pooled/context features
optionally freeze all parameters
```

If the current ParT wrapper cannot expose internal embeddings cleanly, start
with the residual-logit correction version using only `hlt_logits` and a pooled
feature from a small parallel HLT summary encoder.

### Step 3: Add PN Reconstructed-View Dataset Support

Build a dual-view dataset that returns:

```text
hlt_inputs
pn_reco_tokens
pn_reco_mask
pn_reco_confidence
label
jet_id / split metadata
```

This should reuse the existing set-matching/DETR reconstructed-view cache format
where possible.

### Step 4: Implement PN Memory Encoder

Implement a lightweight PN view encoder:

```text
raw PN features -> token embedding
confidence -> confidence embedding
mask-aware transformer/ParT-lite blocks
pooling -> pn_context
tokens -> pn_memory
```

The encoder must be mask-safe and finite-safe.

### Step 5: Implement Reliability Feature Builder

Add a small function that computes per-jet reliability features:

```text
HLT confidence features
PN count/confidence features
HLT-vs-PN summary disagreement
```

These should be normalized and passed to the gate MLP.

### Step 6: Implement Residual Logit Correction Model

First model:

```text
DualViewResidualParT
```

Forward pass:

```text
hlt_logits = hlt_part(hlt_inputs)
pn_context = pn_encoder(pn_reco)
features   = reliability_features(...)
gate       = sigmoid(gate_mlp([...]) + gate_bias)
delta      = delta_mlp([...])
logits     = hlt_logits + gate * delta
```

Initialize the gate closed.

### Step 7: Add Training Script

Add a training runner that supports:

```text
frozen anchor
warm-start anchor
different LR for anchor vs new modules
primary metric selection by FPR@50
final-test confirmation
diagnostics mirroring
```

The first production run should be frozen-anchor.

### Step 8: Add Diagnostics

Write diagnostics:

```text
gate mean/std
gate by class
gate by HLT confidence bucket
gate by HLT correct/wrong if labels available
delta logit norm
final-vs-HLT prediction changes
cases where final fixes HLT
cases where final breaks HLT
```

This is essential for deciding whether the PN branch is doing real work.

### Step 9: Add Shuffled-PN Negative Control

Add a flag:

```text
--shuffle-pn-view
```

This should shuffle PN reconstructed views across jets within a split while
keeping labels and HLT jets unchanged.

The real PN model must beat shuffled PN.

### Step 10: Run Smoke Test

Small run:

```text
train: 10k
val:   5k
test:  10k
epochs: 2
```

Verify:

```text
model starts near HLT baseline
gate starts near zero
loss is finite
metrics write correctly
diagnostics write correctly
shuffled control runs
```

### Step 11: Run Frozen-Anchor 500k QCD/Hgg HLT0.6

Full first real run:

```text
train: 500k
val: 150k
test: 500k
anchor: frozen
PN encoder: trainable
residual head: trainable
gate: trainable, closed init
```

Compare against standard HLT ParT and offline ParT.

### Step 12: Add Cross-Attention Adapter

After residual correction is stable, add:

```text
HLT token queries -> PN memory keys/values
gated residual adapter
```

Keep gate initialized closed.

Test:

```text
residual-only
cross-attention-only
residual + cross-attention
```

### Step 13: Warm-Start The HLT Anchor

Try unfreezing the HLT ParT with:

```text
anchor LR = 0.05x to 0.1x adapter LR
new module LR = normal LR
```

This tests whether the strong baseline can adapt to the PN view without
forgetting.

### Step 14: Optional Offline Teacher Distillation

Once hard-label training is stable, add:

```text
offline teacher KL on final logits
possibly only on HLT-uncertain jets
```

This should be treated as a second-stage improvement, not part of the first
proof.

### Step 15: Report

Final report should include:

```text
standard HLT ParT
offline ParT
residual-only frozen-anchor
cross-attention frozen-anchor
residual+cross-attention frozen-anchor
warm-start versions if run
shuffled-PN controls
```

Primary decision:

```text
Does any real-PN model beat standard HLT ParT on FPR@50?
```

Secondary decision:

```text
Does PN help most in the regimes where HLT is uncertain or wrong?
```

## Expected Outcomes

Best-case outcome:

```text
HLT + PN gated residual beats standard HLT ParT.
Gate opens mostly on hard/uncertain HLT jets.
Shuffled PN control does not improve.
```

Useful middle outcome:

```text
HLT + PN improves over frozen HLT anchor but not over the separately trained
standard HLT ParT.
```

That would mean PN contains useful evidence, but the adapter/backbone still needs
work.

Negative outcome:

```text
PN branch does not beat shuffled PN or gate stays closed.
```

That would suggest the PN reconstructed view does not add event-specific
information beyond what the HLT ParT already extracts in this QCD/Hgg HLT0.6
setting.

