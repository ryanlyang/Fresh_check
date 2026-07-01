# Architecture-View Residual Particle Transformer Plan

## Short Name

Architecture-View Residual ParT, or AV-Residual-ParT.

## Setup We Are Working In

The target task is binary QCD vs Hgg tagging on fixed HLT-degraded JetClass-like particle inputs, with:

- `QCD = 0`
- `Hgg = 1`
- HLT degradation strength `0.6`
- model selection by `fpr_at_signal_eff_0p50`, lower is better
- final comparison against the strong HLT ParT baseline on the same split/cache

The current HLT ParT baseline is strong, but the local-compression feature-adapter pilot showed something important:

```text
exact HLT ParT baseline
  + zero-init residual adapter
  + careful FPR@50 selection
  -> small but real final-test improvements
```

That result suggests a useful pattern:

```text
Do not replace ParT.
Keep ParT as the strongest final reasoner.
Give ParT additional baseline-safe residual information.
Let training learn whether that information is useful.
```

This plan applies that pattern to a new idea: use ParticleNet-like, PFN-like, and PCNN-like branches as architecture-specific particle-view generators.

## Motivation

Particle Transformer is the strongest baseline because it performs global particle-particle attention with physics pairwise bias. But weaker architectures still reason differently:

- ParticleNet/PN emphasizes local graph neighborhoods and EdgeConv-like local geometry.
- PFN emphasizes per-particle nonlinear summaries followed by permutation-invariant jet aggregation.
- PCNN emphasizes local ordered/sequence-style pattern extraction.

Even if these models underperform ParT as final classifiers, they may encode complementary information. The mistake would be to ask them to beat ParT directly. The better question is:

```text
Can PN/PFN/PCNN-style reasoning produce useful per-particle hints that ParT can use?
```

The final model should not be an ensemble of scores. It should be ParT with richer particle tokens.

## Core Hypothesis

ParT's first step is effectively:

```text
canonical particle features F_i -> ParT particle embedding h_i
```

where `h_i` is the learned internal particle representation used by the transformer blocks.

AV-Residual-ParT adds architecture-view information in that same embedding space:

```text
h_i_base = ParTEmbed(F_i)

view_i = combine(
  PN_view_i,
  PFN_view_i,
  PCNN_view_i
)

delta_h_i = ViewToParT(view_i)

h_i_final = h_i_base + gate_i * delta_h_i

h_i_final -> normal ParT attention blocks -> classifier
```

At initialization:

```text
gate_i = 0
delta projection = 0
h_i_final = h_i_base
model = exact HLT ParT baseline
```

This is the same baseline-preserving principle that made the local-compression experiment trustworthy.

## Why Embedding-Space Injection Is Preferred

There are two possible ways to feed architecture views into ParT.

### Feature-space residual

```text
architecture views -> delta_F
ParT sees F + delta_F
```

This is close to the local-compression setup and is easier to implement. But it can create physical inconsistency:

- changed `part_pt_log` without changed Lorentz vectors
- changed PID-like features without categorical consistency
- changed geometry features while pairwise geometry remains original

It can still work, but it is less clean conceptually.

### Embedding-space residual

```text
architecture views -> delta_h
ParT sees h_base + delta_h
```

This does not claim that the physical particle changed. It says:

```text
PN/PFN/PCNN branches provide semantic hints.
A learned adapter translates those hints into ParT's embedding language.
ParT uses attention to decide whether the hints matter.
```

This is the best-performance target.

## Main Architecture

### 1. Canonical HLT ParT Backbone

Use the existing `ParticleTransformerHLTClassifier` baseline configuration:

- canonical PF features from raw HLT particles
- original Lorentz vectors and masks
- original pairwise physics bias
- 2-class CE output
- warm-start from the FPR@50-selected HLT ParT checkpoint

The new wrapper must pass an exact baseline-recovery test:

```text
with all view gates/projections zero:
  logits(new wrapper) == logits(original HLT ParT)
```

This test is non-negotiable.

### 2. Architecture View Branches

Each branch consumes the same raw HLT particle tokens and mask.

Each branch outputs a per-particle embedding:

```text
PN_view:   [B, P, d_view]
PFN_view:  [B, P, d_view]
PCNN_view: [B, P, d_view]
```

Recommended default:

```text
d_view = 32 per branch
combined view dim = 96
```

The branches should be lightweight. They are not full separate taggers. They are view generators.

### PN View Branch

Purpose:

```text
local eta-phi neighborhood reasoning
```

Design:

- build kNN in eta-phi among valid particles
- use relative edge features:
  - `deta`
  - wrapped `dphi`
  - `deltaR`
  - `log(deltaR + eps)`
  - `delta_log_pt`
  - optionally pair mass / relative kT if cheap
- EdgeConv-style message MLP
- max + mean aggregation over neighbors
- residual update into per-particle state

Output:

```text
PN_view_i = local graph summary for particle i
```

This branch captures local structures ParT may learn but does not explicitly stage before global attention.

### PFN View Branch

Purpose:

```text
per-particle nonlinear particle identity/importance features
```

Design:

- shared per-particle MLP `phi(x_i)`
- optionally condition each particle on a global PFN jet summary:

```text
g = sum_i phi_global(x_i)
PFN_view_i = MLP([phi_local(x_i), g])
```

Output:

```text
PFN_view_i = particle identity plus global jet-context hint
```

This branch captures the PFN inductive bias: simple, stable, permutation-invariant global summarization.

### PCNN View Branch

Purpose:

```text
local ordered particle pattern extraction
```

Design:

- sort/keep particles in canonical HLT order, usually descending HLT pT
- apply 1D convolutions over the particle sequence
- mask invalid particles
- use residual Conv1d blocks with small kernels:
  - kernel sizes `3, 5`
  - channels `64 -> 64 -> d_view`

Output:

```text
PCNN_view_i = sequence-local pattern hint for particle i
```

This is not as geometrically clean as PN, but it gives a distinct inductive bias.

### 3. View Fusion And Gating

The three views are concatenated:

```text
view_cat_i = [PN_view_i, PFN_view_i, PCNN_view_i]
```

Then compressed:

```text
view_hidden_i = MLP(LayerNorm(view_cat_i))
```

Then projected into ParT embedding space:

```text
delta_h_i_raw = Linear(view_hidden_i)
```

The final projection should be zero-initialized.

Use a learned gate:

```text
gate_i = sigmoid(gate_mlp([h_base_i, view_hidden_i, quality_i]))
delta_h_i = gate_i * delta_h_i_raw
```

Default gate initialization:

```text
gate bias = -5
```

This makes the initial gate near zero and preserves the baseline.

Optional per-view gates:

```text
gate_PN_i
gate_PFN_i
gate_PCNN_i
```

Default for the first implementation:

```text
one combined gate is enough
```

### 4. ParT Embedding Injection

The best version injects into the particle embedding after ParT's input embedding and before the global transformer blocks:

```text
h_base = ParT particle feature embedding
h_final = h_base + delta_h
global ParT blocks consume h_final
```

Implementation detail:

The existing wrapper calls Weaver's `ParticleTransformer` as a black box:

```python
self.mod(features, v=lorentz_vectors, mask=mask)
```

For embedding injection, we need a wrapper around the internal forward path. The implementation should first inspect the installed Weaver `ParticleTransformer` module and identify:

- feature embedding module
- pairwise embedding path
- transformer blocks
- class token / class blocks
- classifier head

Then implement a repo-local adapter that reuses those exact modules but inserts:

```python
x = x + delta_h
```

after particle embedding.

If the Weaver internals are too brittle, the fallback is:

```text
architecture views -> delta_F -> existing local-compression ParT wrapper
```

But the target implementation is embedding injection.

## Training Objective

Primary loss:

```text
CrossEntropyLoss(logits, labels)
```

Do not use BCE unless deliberately converting to a single logit difference. The baseline is a 2-logit classifier.

Regularizers:

```text
loss = CE
     + lambda_delta * mean(||delta_h_i||^2 over valid particles)
     + lambda_gate * mean(gate_i)
```

Suggested defaults:

```text
lambda_delta = 1e-4
lambda_gate = 1e-4
```

These should be light. The point is to discourage wild embedding shifts, not prevent useful corrections.

Optional boundary emphasis after the base version works:

```text
increase weight near the HLT ParT FPR@50 boundary
```

Do not put this into the first serious version unless we need a V2. The first question is whether view injection helps at all.

## Training Schedule

Start from a strong HLT ParT checkpoint.

Recommended pilot schedule:

```text
epoch 0:
  exact baseline recovery check

epochs 1-2:
  freeze ParT
  train view branches + view-to-delta projection + gates

epochs 3+:
  unfreeze ParT
  adapter_lr = 3e-4
  part_lr = 1e-5 to 3e-5

selection:
  model_val fpr_at_signal_eff_0p50

final:
  evaluate final_test once after selection
```

The point of the freeze phase is attribution. If the model improves while ParT is frozen, the views are doing useful work. If it only improves after unfreezing, it may still be useful, but the story is less clean.

## Minimal Serious Variant Set

The first run should not explode into too many jobs. Use a compact set:

```text
av_baseline_recheck
av_all_views
av_pn_only
av_pfn_only
av_pcnn_only
av_random_view_control
av_context_mlp_control
```

### `av_baseline_recheck`

Exact HLT ParT baseline recovery.

Expected:

```text
same logits and metrics as baseline
```

### `av_all_views`

Main model:

```text
PN + PFN + PCNN per-particle views -> delta_h -> ParT
```

### `av_pn_only`

Tests local graph view.

### `av_pfn_only`

Tests PFN-style per-particle/global-summary view.

### `av_pcnn_only`

Tests sequence-local conv view.

### `av_random_view_control`

Same view dimensions and gates, but the per-particle view embeddings are detached/shuffled across jets or generated from random learned features with no real architecture information.

Purpose:

```text
Does extra capacity alone explain the gain?
```

### `av_context_mlp_control`

No PN/PFN/PCNN architecture branches. Use a generic per-particle MLP plus shallow global context to produce `delta_h`.

Purpose:

```text
Does any extra residual adapter help, or do architecture-specific views help?
```

## Additional Ablations After Pilot

Only run these if the main pilot is promising.

### Frozen-Views Control

Train standalone PN/PFN/PCNN taggers, freeze them, extract particle views, then train only view projection/gates.

Purpose:

```text
Are independently learned architecture representations complementary?
```

This is cleaner scientifically but may underperform joint training.

### Frozen-ParT Control

Keep ParT frozen for all epochs and train only view branches/projection/gates.

Purpose:

```text
Can views improve ParT without changing ParT?
```

If this works, the result is very strong.

### Feature-Residual Version

Use architecture views to predict `delta_F`, then reuse the local-compression feature adapter path.

Purpose:

```text
Is feature-space correction better than embedding-space correction?
```

### Score-Residual Version

Use architecture views to predict:

```text
z_final = z_part + alpha * r_view
```

Purpose:

```text
Is this just score fusion, or does particle-level injection matter?
```

### View-Dropout

Randomly drop one architecture view during training.

Purpose:

```text
Force robustness and prevent one branch from dominating.
```

### Larger ParT Control

Train a larger HLT ParT with similar parameter increase.

Purpose:

```text
Make sure gains are not just from more parameters.
```

This is expensive and should not block the first pilot.

## Diagnostics To Require

Every run should report:

- model-val, stack-val, final-test FPR@50
- model-val, stack-val, final-test AUC
- validation-threshold final-test FPR and signal efficiency
- best epoch
- exact baseline-recovery max logit diff at initialization
- `delta_h_l2_mean`
- `delta_h_l2_p90`
- `delta_h_abs_max`
- mean gate value
- p90 gate value
- per-view gate value if per-view gates are enabled
- per-view embedding norm
- false-positive overlap with HLT baseline at FPR@50
- false-negative recovery near the HLT threshold

The most important sanity checks:

```text
baseline_recheck final_test == HLT baseline final_test
init max logit diff <= 1e-6 or tightly justified
delta_h and gates are nonzero after training
improvement holds on stack_val and final_test
```

## Interpreting Outcomes

### Strong Win

```text
av_all_views beats HLT baseline
av_all_views beats av_context_mlp_control
av_all_views beats av_random_view_control
improvement holds on final_test
```

Interpretation:

```text
Architecture-specific particle views add useful information for ParT.
```

### Moderate Win

```text
all residual adapters beat HLT baseline
but architecture views do not beat context MLP control
```

Interpretation:

```text
Embedding residual adapters help, but architecture specificity is unproven.
```

### Weak Or No Win

```text
only model_val improves
stack/final do not
```

Interpretation:

```text
Overfit or selection noise.
```

### Interesting Failure

```text
single-view PN/PFN/PCNN wins but all-views does not
```

Interpretation:

```text
View fusion/gating needs work.
```

## Implementation Steps

### Step 1: Implement Architecture View Branches And Outputs

Create a new package:

```text
teacher_logit_reco/architecture_view_part/
```

Implement:

- config and variant registry
- canonical raw HLT input handling
- PN-style per-particle view branch
- PFN-style per-particle view branch
- PCNN-style per-particle view branch
- view fusion and gate modules

Tests:

- shapes `[B, P, d_view]`
- masks respected
- invalid particles zeroed
- gradients finite
- each branch can run independently

### Step 2: Implement Embedding-Injected ParT Wrapper

Implement:

```text
ArchitectureViewResidualParT
```

This wrapper should:

- load the exact HLT ParT checkpoint
- reuse the original ParT modules
- inject `delta_h` after ParT particle embedding
- preserve original pairwise bias, class token, transformer blocks, and classifier head
- support exact baseline recheck with zero gates/projections

Tests:

- loaded baseline has zero missing keys
- zero-injection logits match original HLT ParT
- nonzero injected delta changes logits
- strict split/HGG/HLT0.6 metadata checks match local-compression behavior

### Step 3: Implement Training, Reporting, And Ablation Runner

Implement scripts:

```text
scripts/train_architecture_view_part_tagger.py
scripts/train_architecture_view_part_variants.py
scripts/write_architecture_view_part_report.py
```

Implement sbatch:

```text
sbatch/run_train_architecture_view_part.sh
sbatch/run_write_architecture_view_part_report.sh
sbatch/submit_architecture_view_part_qcd_hgg_hlt0p6_experiment.sh
```

Use the same QCD/Hgg HLT0.6 reused caches and same baseline checkpoints as local compression.

Default pilot variants:

```text
av_baseline_recheck
av_all_views
av_pn_only
av_pfn_only
av_pcnn_only
av_random_view_control
av_context_mlp_control
```

### Step 4: Run A Small Pilot, Then Promote Only If It Beats Controls

First run:

```text
500k train
150k val
500k final_test
45 epochs
freeze_part_epochs = 2
adapter_lr = 3e-4
part_lr = 1e-5
```

Promote to high-data only if:

```text
av_all_views improves final_test FPR@50
and beats av_context_mlp_control or at least clearly beats av_random_view_control
```

If the best single-view branch wins, run a focused V2 around that branch instead of forcing all views.

