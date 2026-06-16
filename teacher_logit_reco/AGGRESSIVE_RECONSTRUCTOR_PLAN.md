# Aggressive Teacher-Logit Reconstructor Plan

This document defines the next reconstructor branch after the current
conservative teacher-logit reconstructors.  It is intentionally separate from
the completed Global Transformer, ParticleNet, Particle Flow, Particle CNN, and
cross-architecture 16x4 plans.

The goal is to test a sharper hypothesis:

```text
The current reconstructors may be too conservative to recover useful
HLT-only signal.  If every reconstructor family gets the same larger but
budgeted editing/generation interface, can offline teacher supervision produce
synthetic HLT-derived views that help more than direct HLT taggers?
```

This is not meant to be arbitrary generation.  The aggressive branch should be
more expressive, but still auditable and bounded.

## Short Summary

The aggressive reconstructor keeps the same outer experiment:

```text
fixed HLT view -> reconstructor -> reconstructed soft view -> frozen offline teacher -> reco logits
offline view                                  -> frozen offline teacher -> offline logits
```

but changes the reconstructor output mechanics:

```text
conservative:
  parent edits + 32 extra candidates

aggressive:
  looser parent edits
  parent pruning/reweighting
  64 extra candidates
  larger but budgeted extra-particle capacity
  global jet-level scale/calibration correction
```

The key design constraint is consistency:

```text
Same output contract.
Same losses.
Same budgets.
Same train/predict/fusion logic.
Different encoder families only.
```

So the four aggressive reconstructor families remain comparable:

```text
aggressive_gt
aggressive_pn
aggressive_pfn
aggressive_pcnn
```

Each architecture supplies particle/context embeddings.  A shared aggressive
soft-view head turns those embeddings into the final reconstructed view.

## What We Are Trying To Fix

The current reconstructors are clean and conservative.  That is useful for a
first paper-quality mechanism, but the results suggest they may not have enough
freedom:

- Existing HLT particles can only move/change within fairly tight caps.
- Generated extra particles are limited.
- Parent particles are mostly preserved rather than actively pruned.
- Jet-level calibration effects are only indirectly modeled through local
  particle edits.

If the HLT/offline gap includes:

- missing soft radiation,
- unresolved local splitting,
- misleading HLT constituents,
- broad energy-scale differences,
- class-relevant substructure that needs extra support,

then small per-particle correction may not be enough.  The aggressive branch
gives the reconstructor more tools, while keeping explicit budgets so it has to
"spend" that freedom.

## Non-Goals

The first aggressive version should not use class-aware conditioning.

Do not feed the true label into the reconstructor at inference or training.
Do not feed offline particles into the reconstructor at inference.  Do not
train fusion on `model_train` or `model_val`.

Class-aware variants can be a later branch, for example using HLT prior logits
as a conditional input.  This plan leaves that out so the first aggressive
story stays clean.

## Core Output Contract

Every aggressive reconstructor must return the existing soft-view object:

```text
SoftReconstructedView(
  tokens:  [B, N_parent + N_extra, 19],
  mask:    [B, N_parent + N_extra],
  weights: [B, N_parent + N_extra],
  labels,
  jet_ids,
  split,
  metadata,
  aux
)
```

The downstream teacher/tagger path must not care whether the view came from a
Global Transformer, ParticleNet, PFN, or P-CNN reconstructor.

The shared aggressive head should produce:

```text
parent corrected tokens
parent weights
extra candidate tokens
extra candidate weights
global jet calibration parameters
budget diagnostics
```

The concatenated view is:

```text
tokens  = concat(corrected_parent_tokens, extra_tokens)
mask    = concat(parent_mask, extra_mask)
weights = concat(parent_weights, extra_weights)
```

## Aggressive Mechanism

### 1. Looser Parent Edits

For existing HLT constituents:

```text
pt'     = pt * exp(delta_logpt)
eta'    = eta + delta_eta
phi'    = phi + delta_phi
energy' = energy * exp(delta_loge)
aux'    = aux + bounded_delta_aux
```

The caps should be larger than the conservative branch, but still explicit.
Suggested first aggressive defaults:

```text
max_delta_logpt = 1.0
max_delta_loge  = 1.0
max_delta_eta   = 0.20
max_delta_phi   = 0.20
```

These are intentionally bigger than tiny unsmearing, but still bounded enough
that the reconstructed view remains interpretable.

### 2. Learned Parent Pruning/Reweighting

Every existing HLT constituent gets a learned soft weight:

```text
parent_weight = sigmoid(parent_weight_logit + parent_weight_bias)
```

The current conservative behavior effectively assumes most parent particles
should survive.  The aggressive branch should allow the reco to downweight
misleading HLT constituents.

Important interpretation:

```text
parent edits answer: "How should this HLT particle move?"
parent weights answer: "How much should the teacher trust this HLT particle?"
```

The parent weights should be regularized, but not forced to one.  If this
mechanism works, some classes or kinematic regimes may show systematic pruning
patterns.

### 3. More Extra Candidate Slots

Increase extra candidates from 32 to 64:

```text
num_extra_candidates = 64
```

Each extra slot is produced from:

```text
global jet context + learned slot embedding -> shared extra decoder
```

The extra slots should be able to represent:

- missing low-pt structure,
- unresolved local splitting,
- teacher-useful support particles,
- soft class-specific shape information.

But they must have a cost.  The reconstructor should not be allowed to add
large arbitrary energy everywhere.

### 4. Extra Candidate Budget

The extra weights and extra pt should be budgeted.

Useful budget terms:

```text
extra_weight_sum = sum(extra_weights)
extra_pt_sum     = sum(extra_weights * extra_pt)
parent_pt_sum    = sum(parent_mask * parent_weight * parent_pt)
extra_pt_fraction = extra_pt_sum / clamp(parent_pt_sum, eps)
```

Suggested first aggressive default:

```text
max_total_extra_pt_fraction = 0.50
```

The loss should penalize excess rather than hard clipping everything to zero:

```text
extra_budget_loss =
  relu(extra_pt_fraction - max_total_extra_pt_fraction)^2
+ small_lambda * mean(extra_weight_sum)
```

This gives the model freedom to add structure when useful, but makes that
freedom visible and measurable.

### 5. Global Jet-Level Calibration

Add a global correction predicted from the full-jet context:

```text
global_logpt_scale
global_loge_scale
global_eta_shift
global_phi_shift
```

Apply it coherently to parent and extra particles:

```text
pt_global     = pt_local * exp(global_logpt_scale)
energy_global = energy_local * exp(global_loge_scale)
eta_global    = eta_local + global_eta_shift
phi_global    = phi_local + global_phi_shift
```

Suggested caps:

```text
max_global_logpt_scale = 0.35
max_global_loge_scale  = 0.35
max_global_eta_shift   = 0.05
max_global_phi_shift   = 0.05
```

Rationale:

Some HLT/offline mismatch may be coherent jet-level calibration, not just
particle-local noise.  A global correction makes this easy for the model to
express and easy for us to audit.

### 6. Shared Aggressive Soft-View Head

The implementation should avoid four independent aggressive mechanisms.
Instead, build one shared head:

```text
AggressiveSoftViewHead
```

Inputs:

```text
tokens
mask
particle_embeddings
global_context
```

Outputs:

```text
SoftReconstructedView
aux diagnostics
```

The four reconstructor families only differ in how they produce:

```text
particle_embeddings: [B, N, D]
global_context:      [B, D]
```

Then the shared head handles:

- parent edit deltas,
- parent weights,
- extra candidate decoding,
- extra candidate weights,
- global calibration,
- budget diagnostics,
- metadata.

This keeps the story straight:

```text
architecture diversity lives in the encoder;
aggressive reconstruction freedom is held constant.
```

## Architecture-Specific Encoders

### Aggressive Global Transformer

Use the existing ParT-ish/global transformer encoder style:

```text
tokens -> token embedding -> transformer blocks -> particle embeddings
```

Global context:

```text
masked mean/max pooling or class token
```

This is the most expressive all-to-all version.

### Aggressive ParticleNet

Use EdgeConv/dynamic graph blocks:

```text
tokens -> graph features -> EdgeConv blocks -> particle embeddings
```

Global context:

```text
masked pooling over final graph embeddings
```

This version emphasizes local neighborhoods and should be less redundant with
the transformer.

### Aggressive Particle Flow

Use DeepSets/PFN-style per-particle encoding:

```text
tokens -> phi MLP -> particle embeddings
global_context = pooled particle embeddings -> context MLP
```

Parent edits can use both local phi embeddings and broadcast global context.
Extra candidates should primarily decode from global context plus slot
embeddings.

This version has weak local geometry but strong set-level simplicity.

### Aggressive Particle CNN

Use sorted sequence/1D convolution features:

```text
tokens sorted by pt/rank -> masked 1D conv blocks -> particle embeddings
```

Global context:

```text
masked pooling over sequence embeddings
```

This version tests whether rank/sequence-local patterns produce different
errors from graph and transformer encoders.

## Loss Function

The first aggressive loss should stay close to the current teacher-logit
objective:

```text
loss =
  lambda_kl      * teacher_KL
+ lambda_ce      * label_CE
+ lambda_budget  * correction_budget
+ lambda_summary * weak_jet_summary_loss
+ lambda_extra   * extra_budget_loss
+ lambda_global  * global_calibration_budget
```

### Teacher KL

Compare frozen teacher probabilities on offline view vs reconstructed view:

```text
teacher_KL =
  KL(
    softmax(offline_teacher_logits / T),
    softmax(reco_teacher_logits / T)
  ) * T^2
```

This is the main teacher-logit reconstruction objective.

### Label CE

Use the true label only as a target for the teacher's prediction on the
reconstructed view:

```text
label_CE = CE(reco_teacher_logits, true_label)
```

This does not feed the label into the reconstructor.  It only tells the reco
that the teacher's reconstructed-view prediction should remain discriminative.

### Correction Budget

Penalize excessive parent edits:

```text
parent_edit_budget =
  mean(abs(delta_logpt))
+ mean(abs(delta_loge))
+ mean(abs(delta_eta))
+ mean(abs(delta_phi))
```

This budget should be weaker than the conservative branch so the model can
actually use the new freedom.

### Parent Weight Budget

Avoid pathological deletion of all parent particles:

```text
parent_keep_loss = relu(min_parent_weight_sum - sum(parent_weights))^2
parent_prune_loss = small_lambda * mean(abs(parent_weights - 1))
```

Do not make this too strong.  If parent pruning helps, we need to let it happen.

### Extra Budget Loss

Penalize excessive generated extra pt:

```text
extra_pt_fraction = extra_pt_sum / clamp(parent_pt_sum, eps)
extra_budget_loss = relu(extra_pt_fraction - max_total_extra_pt_fraction)^2
```

Optionally add:

```text
extra_count_loss = mean(sum(extra_weights))
```

with a small coefficient.

### Global Calibration Budget

Penalize excessive global scale and shift:

```text
global_budget =
  abs(global_logpt_scale)
+ abs(global_loge_scale)
+ abs(global_eta_shift)
+ abs(global_phi_shift)
```

This should be mild.  The hard caps already prevent runaway behavior.

### Weak Jet Summary Loss

Keep the current weak jet summary idea, but do not let it dominate.

Useful summary quantities:

- jet pt,
- jet energy,
- jet eta,
- multiplicity,
- charged/neutral/photon fractions if stable.

This is only a weak anchor.  The main objective remains teacher-logit matching.

## Diagnostics To Save

Every aggressive reconstructor should save additional diagnostics:

```text
mean_parent_weight
mean_parent_weight_sum
mean_extra_weight_sum
mean_extra_pt_fraction
mean_global_logpt_scale
mean_global_loge_scale
mean_global_eta_shift
mean_global_phi_shift
parent_edit_norms
extra_slot_usage_histogram
```

For prediction blocks, metadata should include:

```text
reconstructor_aggression_level = "aggressive_v1"
num_extra_candidates = 64
max_total_extra_pt_fraction
global_calibration_enabled = true
parent_reweighting_enabled = true
```

These diagnostics are important.  If aggressive models improve, we need to
know what mechanism they actually used.  If they fail, these numbers may tell
us whether they were too constrained, too unconstrained, or ignored the extra
capacity.

## Experiment Comparisons

The aggressive branch should not overwrite the conservative branch.

Use a fresh namespace:

```text
checkpoints/teacher_logit_reco_crossarch_aggressive_v1_500k
```

or, if reusing the crossarch root:

```text
checkpoints/teacher_logit_reco_crossarch_500k/aggressive_v1
```

The cleaner option is a top-level fresh root so old fusion/prediction jobs
cannot collide.

Recommended comparisons:

```text
HLT4 direct taggers
conservative 16 frozen-teacher reco predictions
conservative 16 adapted reco-domain taggers
aggressive 16 frozen-teacher reco predictions
aggressive 16 adapted reco-domain taggers
HLT4 + aggressive adapted all16
HLT4 + conservative adapted all16
HLT4 + conservative adapted all16 + aggressive adapted all16
```

The most important headline comparison is:

```text
HLT4 fusion
vs
HLT4 + aggressive adapted all16
```

because it tests whether aggressive offline-supervised reconstruction adds
information beyond a strong architecture-diverse HLT ensemble.

## Leakage Rules

Keep the same five-way split discipline:

```text
model_train:
  train reconstructors
  train adapted reco-domain taggers

model_val:
  select reconstructor checkpoints
  select adapted tagger checkpoints

stack_train:
  fit fusion models

stack_val:
  select/validate fusion models

final_test:
  final locked evaluation only
```

No final-test information may affect:

- reconstructor checkpoint selection,
- adapted tagger checkpoint selection,
- fuser hyperparameter selection,
- fuser group selection after looking at final-test results.

## Implementation Plan

### Step 1: Add Aggressive Config And Shared Head Skeleton

Create a new shared module, likely:

```text
teacher_logit_reco/aggressive_soft_view.py
```

Define:

```text
AggressiveSoftViewConfig
AggressiveSoftViewHead
AggressiveSoftViewDiagnostics
```

The first implementation should focus on shape correctness, stable bounds, and
the same `SoftReconstructedView` contract as existing reconstructors.

Do not change old reconstructor behavior in this step.

### Step 2: Implement Global Calibration And Parent Reweighting

Inside the shared head, implement:

```text
global_logpt_scale
global_loge_scale
global_eta_shift
global_phi_shift
parent_weight_logits
parent edit deltas
```

Add unit tests for:

- output token shape,
- output mask shape,
- weight range,
- finite outputs,
- global caps,
- parent edit caps.

### Step 3: Implement Extra Candidate Generation With 64 Slots

Add the extra slot decoder:

```text
global_context + learned slot embedding -> extra token + extra weight
```

Add budget diagnostics:

```text
extra_weight_sum
extra_pt_fraction
extra_slot_usage
```

Unit tests should check:

- `N_out = N_parent + num_extra_candidates`,
- generated tokens are finite,
- generated weights are in range,
- masks are valid,
- budget diagnostics are present.

### Step 4: Implement Aggressive Loss Additions

Extend the teacher-logit loss module without breaking conservative training.

Add:

```text
extra_budget_loss
parent_weight_budget_loss
global_calibration_budget_loss
```

The existing losses remain:

```text
teacher_KL
label_CE
correction_budget
weak_jet_summary_loss
```

Add tests that confirm each loss is finite and participates in the total loss
when its coefficient is nonzero.

### Step 5: Build Aggressive Global Transformer Reconstructor

Create the first full aggressive reconstructor:

```text
AggressiveGlobalTransformerReconstructor
```

It should reuse the transformer-style encoder idea, but call the shared
`AggressiveSoftViewHead` for all output mechanics.

This is the reference implementation for the other three families.

### Step 6: Add Aggressive PN, PFN, And PCNN Reconstructors

Implement:

```text
AggressiveParticleNetReconstructor
AggressiveParticleFlowReconstructor
AggressiveParticleCnnReconstructor
```

Each model should produce:

```text
particle_embeddings
global_context
```

and then call the same shared aggressive head.

Avoid architecture-specific output tricks unless absolutely necessary.

### Step 7: Add Checkpoint Builders And Loader Support

Extend reconstructor builders so checkpoints can identify:

```text
aggressive_global_transformer
aggressive_particle_net
aggressive_particle_flow
aggressive_particle_cnn
```

The loader should not confuse conservative `pn` with aggressive `pn`.
Use explicit names in metadata:

```text
reconstructor_architecture = "aggressive_particle_net"
aggression_level = "aggressive_v1"
```

### Step 8: Add Training CLIs

Add training scripts for the aggressive branch.

Either use one shared CLI:

```text
scripts/train_teacher_logit_aggressive_reco.py
```

with:

```text
--reco-architecture aggressive_gt|aggressive_pn|aggressive_pfn|aggressive_pcnn
```

or four architecture-specific scripts if that better matches the current repo
pattern.

The shared CLI is preferable if the implementation is clean.

### Step 9: Add Prediction CLIs

Add prediction collection for:

```text
fixed HLT -> aggressive reconstructor -> frozen offline teacher
```

Prediction blocks should use distinct model names, for example:

```text
aggt_reco_to_part_teacher
agpn_reco_to_pn_teacher
agpfn_reco_to_pcnn_teacher
agpcnn_reco_to_pfn_teacher
```

The exact names can be adjusted, but they must not collide with conservative
sources.

### Step 10: Add Adapted Reco-Domain Tagger Compatibility

The reco-domain adapted tagger strategy should work with aggressive
reconstructors too:

```text
fixed HLT -> frozen aggressive reco -> trainable tagger
```

Add a parallel adapted-tagger path or make the current adapted-tagger scripts
accept aggressive reconstructor architecture names.

The adapted tagger names must also be distinct, for example:

```text
agpn_reco_to_pfn_adapted_tagger
```

### Step 11: Add Fusion Groups

Create fresh aggressive fusion groups:

```text
aggressive_all16
aggressive_all16_plus_hlt4
aggressive_cross12_plus_hlt4
aggressive_part_teacher4_plus_hlt4
aggressive_pn_teacher4_plus_hlt4
aggressive_mixed4_plus_hlt4
```

Also add comparison groups:

```text
conservative_adapted_all16_plus_hlt4
aggressive_adapted_all16_plus_hlt4
conservative_plus_aggressive_adapted_all32_plus_hlt4
```

The last group is big, but it may reveal whether aggressive and conservative
views make complementary errors.

### Step 12: Add Slurm Runners

Write separate Slurm runners under a new namespace.

Recommended jobs:

```text
16 aggressive reconstructor train jobs
16 aggressive frozen-teacher prediction jobs
16 aggressive adapted tagger train jobs
16 aggressive adapted tagger prediction jobs
1 or more fusion jobs
```

Recommended first resources:

```text
aggressive reconstructor train:
  2 days, gpu:1, 160G, 8 CPUs

frozen-teacher prediction:
  8 hours, gpu:1, 160G, 8 CPUs

adapted tagger train:
  2 days, gpu:1, 160G, 8 CPUs

adapted tagger prediction:
  8 hours, gpu:1, 160G, 8 CPUs

fusion:
  1 day, 160G, 8 CPUs
```

The first submitter should support `DRY_RUN=1` and should refuse existing
output directories unless `OVERWRITE=1`.

### Step 13: Add Aggressive Audits

Add an audit script that summarizes:

- exact split sizes,
- checkpoint metadata,
- whether any final-test data entered training,
- parent weight distributions,
- extra slot usage,
- extra pt fraction,
- global calibration magnitudes,
- fusion group membership.

This audit is critical because aggressive reconstruction is easier to misuse.
The model gets more freedom, so the report must show how that freedom was used.

### Step 14: Run A Small Smoke Test

Before launching 16x or 64x jobs, run tiny sizes:

```text
model_train = 5k or 10k
model_val   = 2k
stack_train = 5k
stack_val   = 2k
final_test  = 10k
```

Smoke-test goals:

- no shape errors,
- no NaNs,
- budgets finite,
- prediction blocks align,
- fusion can load all groups.

Do not interpret smoke-test performance.

### Step 15: Launch Full Aggressive 16x4

Once the smoke test passes, launch the full 500k/150k/500k aggressive branch.

The first readout should compare:

```text
HLT4
conservative adapted all16 + HLT4
aggressive adapted all16 + HLT4
```

If aggressive helps, inspect diagnostics before making the mechanism more
complicated.  If it does not help, the diagnostics should tell us whether the
model ignored the new capacity or used it in unstable ways.

## Recommended First Defaults

```text
num_extra_candidates = 64
max_delta_logpt = 1.0
max_delta_loge = 1.0
max_delta_eta = 0.20
max_delta_phi = 0.20
max_total_extra_pt_fraction = 0.50
max_global_logpt_scale = 0.35
max_global_loge_scale = 0.35
max_global_eta_shift = 0.05
max_global_phi_shift = 0.05
parent_weight_bias = 2.0
extra_weight_bias = -2.0
```

Loss weights should start conservative enough to prevent collapse:

```text
lambda_kl = 1.0
lambda_ce = 0.5
lambda_budget = 0.02
lambda_summary = 0.05
lambda_extra = 0.05
lambda_global = 0.02
```

These should be treated as first-pass defaults, not final theory.

## Expected Failure Modes

### The Model Ignores Extra Slots

If `extra_weight_sum` stays near zero, the model may not need generated
particles, or the extra penalty may be too strong.

Try:

```text
lower lambda_extra
less negative extra_weight_bias
higher max_total_extra_pt_fraction
```

### The Model Deletes Parent Particles

If parent weights collapse, the pruning freedom is too strong.

Try:

```text
higher parent_weight_bias
stronger parent_keep_loss
lower extra budget
```

### The Model Uses Only Global Scale

If global scale dominates and particle edits/extras are unused, the model may
be solving only calibration.  That is still interesting, but it may not be the
substructure recovery we want.

Try:

```text
lower global caps
higher global budget penalty
inspect per-class global shifts
```

### Fusion Does Not Improve

If aggressive adapted taggers do not improve over HLT4, inspect diversity:

```text
pairwise disagreement
per-class confusion
row-shuffle controls
confidence/margin bins
extra-slot usage by class
```

It may still be useful if aggressive models are complementary in specific
classes or uncertainty regions.

## Why This Is A Good Paper Story

The mechanism is simple enough to explain:

```text
We give the reconstructor a controlled edit-and-generate interface.
It can edit, suppress, add, and globally recalibrate HLT constituents.
All freedom is budgeted.
The same head is used for every encoder architecture.
```

The research question is also clean:

```text
Does controlled reconstruction freedom plus architecture diversity recover
offline-supervised signal that direct HLT models miss?
```

That is a stronger story than trying many unrelated reconstructor tricks.  If
it works, the diagnostics can show which freedoms mattered.  If it does not,
the result is still interpretable.
