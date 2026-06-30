# Local Compression Residual Feature Adapter ParT Plan

## Purpose

This document replaces the earlier standalone LC-ParT plan with the stronger,
baseline-preserving version of the local-compression idea.

The target setting is fixed:

```text
task:              QCD vs Hgg
input view:        fixed HLT particles only
HLT degradation:   0.6
baseline:          exact HLT Particle Transformer
primary metric:    FPR@50, false-positive rate at 50% Hgg signal efficiency
goal:              beat HLT ParT at high training counts
```

The new mainline is not:

```text
local compression -> custom transformer -> classifier
```

The new mainline is:

```text
raw HLT tokens
  -> canonical ParT inputs
  -> local feature/modality compression
  -> small residual correction delta_F
  -> F' = F + delta_F
  -> exact warm-started HLT ParT
  -> classifier
```

This is the cleanest and highest-odds implementation of the idea because the
model starts as the exact HLT ParT baseline. Local compression only earns its
place if it learns useful feature corrections.

## The Key Shift

The first local-compression plan asked:

```text
Can we build a better particle token than ParT's input embedding?
```

That is interesting, but dangerous. If the model loses, we may not know whether
local compression failed or whether our custom global transformer was weaker
than the real HLT ParT.

This revised plan asks a cleaner question:

```text
Can local compression make the inputs to the exact HLT ParT better?
```

Instead of replacing the ParT backbone, we preserve it:

```text
same raw HLT cache
same canonical PF feature builder
same points/vectors/mask
same ParticleTransformerHLTClassifier backbone
same FPR@50 selection discipline
only new component: a local-compression adapter that predicts delta_F
```

At initialization:

```text
delta_F = 0
F' = F
logits = exact baseline HLT ParT logits
```

After training:

```text
delta_F != 0 only where it helps the supervised tagging objective
```

This gives us both scientific cleanliness and the best chance to beat the
baseline.

## Data Contract In This Repo

The fixed HLT cache stores raw particle tokens with the repo's canonical raw
contract:

```text
RAW_TOKEN_DIM = 14
```

The columns are:

```text
0   pt
1   eta
2   phi
3   energy
4   charge
5   isChargedHadron
6   isNeutralHadron
7   isPhoton
8   isElectron
9   isMuon
10  d0
11  d0err
12  dz
13  dzerr
```

The real HLT ParT baseline does not consume those raw tokens directly. It uses
the canonical ParT input builder, which creates:

```text
features:        [batch, PF_FEATURE_NAMES, particles]
points:          [batch, PF_POINT_NAMES, particles]
lorentz_vectors: [batch, PF_VECTOR_NAMES, particles]
mask:            [batch, particles]
```

The canonical PF features are:

```text
part_pt_log
part_e_log
part_logptrel
part_logerel
part_deltaR
part_charge
part_isChargedHadron
part_isNeutralHadron
part_isPhoton
part_isElectron
part_isMuon
part_d0
part_d0err
part_dz
part_dzerr
part_deta
part_dphi
```

The local-compression adapter must respect this contract. It predicts
corrections in exactly this feature space:

```text
delta_F shape: [batch, particles, len(PF_FEATURE_NAMES)]
F_rows shape:  [batch, particles, len(PF_FEATURE_NAMES)]
F_prime_rows = F_rows + delta_F
```

Then `F_prime_rows` is transposed back into ParT convention:

```text
features_prime = F_prime_rows.transpose(1, 2)
```

and passed to the exact HLT ParT backbone.

## Motivation

ParT is strong because it treats particles as tokens and performs global
particle-particle attention with physics-aware pairwise structure. That is
exactly why we should not replace it lightly.

The possible blind spot is one level lower. A particle feature vector is not one
homogeneous object. It contains multiple evidence streams:

```text
geometry:
  eta, phi, deta, dphi, deltaR

energy/momentum:
  pt, energy, log pt, log energy, relative pt, relative energy,
  px, py, pz, E where available

identity:
  charge, PID flags

tracking/error:
  d0, dz, d0err, dzerr

quality/consistency:
  valid mask, finite/clipped indicators, charged-vs-neutral consistency,
  track-applicability signals, pT rank, error-size summaries
```

The standard ParT input path gives ParT a fixed feature vector per particle.
The model can learn interactions among these fields through its input embedding,
but those interactions are compressed before global particle attention.

Local compression gives each particle an internal discussion first:

```text
within this particle:
  what does geometry say?
  what does energy/momentum say?
  what does identity say?
  what does tracking say?
  what looks reliable or inconsistent?

then:
  propose a small correction to the canonical ParT features
```

The model is not asked to reconstruct offline truth. It is asked to make the HLT
features more tag-useful while preserving the real ParT backbone.

## Research Inspiration

This design is inspired by a repeated lesson across transformer domains:
tokenization and local compression matter.

### Transformer-In-Transformer

TNT treats image patches as coarse tokens and sub-patches as inner tokens:

```text
sub-patch words -> local transformer -> patch token -> global transformer
```

The analogy here is:

```text
feature/modality subtokens -> local particle compressor -> particle token/features -> ParT
```

### Tokens-To-Token ViT

T2T-ViT showed that naive patch tokenization can be improved by progressive
local aggregation before global attention. For jets, particle tokenization is
much more physically meaningful than image patches, but each particle still has
internal structure.

### CANINE, MEGABYTE, BLT

Byte/character models often need local compression before expensive global
reasoning. The lesson is not that every scalar should attend globally. The
lesson is:

```text
fine tokens are useful if they are compressed locally first
```

For particles:

```text
feature/modality subtokens are useful if they are bound to a particle
and compressed before global particle attention
```

### FT-Transformer And Tabular Feature Tokens

Tabular transformers treat features within one row as tokens. A particle is
similar to a tiny physics row:

```text
pt, eta, phi, energy, charge, PID, d0, dz, errors
```

But in this project the row is not the whole event. The row is one particle, and
the jet is a set of many particles. That suggests a hierarchy:

```text
feature tokens inside each particle
particle tokens inside each jet
```

### Particle Transformer

ParT is the backbone we are trying to improve, not casually replace. The
strongest plan is to keep its real input contract and backbone, then learn a
local-compression correction before ParT sees the features.

## Main Architecture

The model name for the main implementation should be:

```text
LocalCompressionResidualFeatureAdapterParT
```

or shorter:

```text
LCFeatureAdapterParT
```

The forward path is:

```text
raw HLT tokens [B, N, 14], mask [B, N]
  -> build canonical ParT inputs:
       F      [B, C, N]
       points [B, P, N]
       vecs   [B, V, N]
       mask   [B, N]
  -> transpose F_rows = F.transpose(1, 2) [B, N, C]
  -> build modality subtokens per particle [B, N, M, D]
  -> local within-particle compressor [B, N, M, D]
  -> shallow particle context [B, N, D]
  -> context-aware gates [B, N, M]
  -> predict delta_F_rows [B, N, C]
  -> F_prime_rows = F_rows + gamma * delta_F_rows
  -> F_prime = F_prime_rows.transpose(1, 2)
  -> exact ParticleTransformerHLTClassifier(points, F_prime, vecs, mask)
  -> logits
```

The adapter is HLT-only. It uses no offline particles, no teacher logits, and no
final-test labels outside final reporting.

## Baseline Preservation

The implementation must prove baseline preservation.

The adapter has a final projection:

```text
hidden -> delta_F
```

That projection must be zero-initialized:

```text
weight = 0
bias = 0
```

The effective correction is:

```text
F_prime = F + gamma_F * delta_F
```

Recommended initialization:

```text
gamma_F = 1.0
delta_F projection = 0
```

This is better than setting `gamma_F = 0`, because a zero final projection gives
baseline recovery while still allowing gradients into the final layer. After the
first update, gradients can flow into the upstream local-compression path.

Required invariant:

```text
fresh model + copied HLT ParT weights + zero delta_F projection
  -> logits numerically match baseline HLT ParT
```

This should be checked in tests and written into run reports:

```text
baseline_recoverable_at_zero_delta: true
max_abs_logit_diff_at_init: small
```

## Modality Design

The default modality grouping should be practical, not ornamental. Avoid
splitting heavily redundant information into separate default branches unless an
ablation asks for it.

Recommended default groups:

### Geometry Token

```text
eta
phi
part_deta
part_dphi
part_deltaR
sin(phi)
cos(phi)
pT rank embedding or scalar
```

Purpose:

```text
where is this particle in the jet?
```

### Energy/Momentum Token

```text
pt
energy
part_pt_log
part_e_log
part_logptrel
part_logerel
px
py
pz
E
```

Purpose:

```text
how much momentum/energy does this particle carry?
```

The exact list can use raw and canonical values, but it should avoid redundant
columns if they cause instability. Log-scaled canonical features should be the
core.

### Identity Token

```text
charge
PID flags:
  charged hadron
  neutral hadron
  photon
  electron
  muon
```

Purpose:

```text
what kind of particle does HLT think this is?
```

### Tracking/Error Token

```text
d0
d0err
dz
dzerr
part_d0
part_d0err
part_dz
part_dzerr
```

Purpose:

```text
what does tracking/displacement say, and how reliable is it?
```

For neutral particles, track-like quantities may be zero, missing, or
non-applicable. The model should be told that with quality signals rather than
forced to infer it from zeros alone.

### Quality/Consistency Token

```text
valid particle mask
finite indicators
raw value clipping indicators if available
charged PID vs charge consistency
neutral PID vs track-feature applicability
track error magnitude summaries
pT rank / log pT percentile
local density summary if cheap
```

Purpose:

```text
how much should the model trust the other modalities?
```

This token is important. It should be more than a mask. HLT degradation creates
local corruption patterns, so reliability is part of the signal.

## Subtoken Embedding

Each modality gets its own small embedder:

```text
geometry_embed
energy_embed
identity_embed
tracking_embed
quality_embed
```

Each embedder maps its modality fields to a shared dimension:

```text
D_local = 64 or 96
```

Then add:

```text
modality type embedding
particle anchor embedding
optional pT rank embedding
```

The particle anchor binds all subtokens to the same particle:

```text
anchor_i = MLP(F_i)
subtoken_{i,m} = modality_embed_{i,m} + modality_type_m + anchor_i
```

This avoids the main failure mode of scalar tokenization: features losing their
attachment to the particle they came from.

## Local Within-Particle Compression

The local compressor operates over modalities inside each particle:

```text
input:  [B, N, M, D]
reshape [B * N, M, D]
run small transformer over M modality tokens
reshape [B, N, M, D]
```

Recommended default:

```text
layers:        2
heads:         4
embed dim:     96
dropout:       0.05
attention drop:0.05
MLP ratio:     2.0
```

The local compressor answers:

```text
inside this particle, how should geometry, energy, ID, track, and quality
reinterpret one another?
```

It should be small. The heavy global reasoning still belongs to ParT.

## Particle Context Before Gates

Reliability is not always particle-local. A particle can look trustworthy or
suspicious depending on its neighbors and the rest of the jet.

So the serious default should not compute gates from local subtokens alone.

Recommended context path:

```text
compressed modality subtokens
  -> provisional particle summary token
  -> 1-2 shallow particle-context blocks
  -> context-aware gates and delta_F
```

The context block can be one of:

```text
global lightweight self-attention over particles
local kNN eta-phi attention
cheap pairwise-biased particle block
```

For the first implementation, prefer the simplest reliable option:

```text
1 shallow global particle-context transformer block
with mask support
```

Then optionally test local kNN context later. This is not meant to replace ParT;
it is just enough context for gates to know whether a modality is reliable.

## Reliability Gates

The model should produce modality gates:

```text
gates [B, N, M]
```

Use sigmoid gates by default:

```text
gate_m = sigmoid(gate_head([local_modality, particle_context]))
```

Why sigmoid instead of softmax?

```text
softmax forces a fixed unit of trust across modalities
sigmoid lets all modalities be trusted or all modalities be distrusted
```

The gates can be used in two ways:

```text
1. diagnostics: which modalities matter where?
2. computation: weighted pooling / delta_F prediction
```

Do not make the gates too dominant at initialization. Initialize gate logits near
zero so sigmoid gates start near 0.5, not saturated.

Recommended regularization:

```text
small entropy/variance diagnostic only at first
no strong gate loss in the first serious run
```

We should not force a reliability story before seeing whether the task wants
one.

## Delta Feature Prediction

The adapter predicts a residual in canonical PF feature space:

```text
delta_F_rows [B, N, 17]
```

The prediction head should see:

```text
local pooled particle token
context particle token
original canonical F_i
optional gate summary
```

Recommended head:

```text
LayerNorm
Linear -> GELU -> Dropout
Linear -> GELU
zero-initialized Linear to 17 dims
```

Feature correction should be bounded enough to avoid destroying baseline ParT.
Use one or both of:

```text
delta_F = delta_scale * tanh(raw_delta)
feature-wise delta scales
L2 penalty on delta_F
```

Recommended initial setting:

```text
delta_scale = 0.25
delta_l2_weight = small, e.g. 1e-4 to 1e-3
```

For normalized/log PF features, a correction of 0.25 can already be meaningful.
The model should not be allowed to wildly rewrite the input on day one.

Use feature-wise scales rather than one global scale. The canonical feature
matrix and Lorentz vectors are consumed together by ParT, so unrestricted
changes to kinematic features can create inconsistent inputs:

```text
features say one thing
lorentz_vectors still say the original HLT four-vector
```

That inconsistency is acceptable only if corrections are small and measured.
The first serious run should strongly limit or freeze the most physically
coupled features:

```text
part_pt_log
part_e_log
part_logptrel
part_logerel
part_deltaR
part_deta
part_dphi
```

and should be especially conservative with PID one-hot-like fields:

```text
part_isChargedHadron
part_isNeutralHadron
part_isPhoton
part_isElectron
part_isMuon
```

Recommended first-run policy:

```text
small scales for kinematic/geometry deltas
very small scales or frozen deltas for PID flags
moderate scales for track/error and quality-derived corrections
```

The run report must include feature-consistency diagnostics:

```text
delta by PF feature
delta on kinematic/log-energy features
delta on PID flag features
estimated feature-vector vs Lorentz-vector inconsistency
fraction of particles with large absolute delta by feature
```

## Exact HLT ParT Backbone

The serious implementation must use:

```text
jetclass_fresh.hlt_baseline.ParticleTransformerHLTClassifier
```

The forward call should remain:

```text
part_model(points, features_prime, lorentz_vectors, mask)
```

This preserves:

```text
real Weaver ParticleTransformer module
canonical ParT inputs
existing model size/config choices
existing checkpoint loading path
```

The local-compression adapter should be implemented as a wrapper around this
backbone, similar in spirit to the multiscale-subjet branch's feature-adapter
pattern.

## Warm Start Policy

The best pilot should warm-start from the already-trained HLT ParT baseline:

```text
load hlt_part_baseline/best_model_val.pt
initialize adapter with zero delta_F
train adapter + optionally fine-tune ParT
```

The baseline checkpoint must be the actual strong target baseline:

```text
same QCD/Hgg task
same HLT0.6 cache
same split manifest
same label mapping
selected by fpr_at_signal_eff_0p50
not an older accuracy-selected checkpoint
```

The loader should reject or loudly warn on mismatched metadata. A win over a
weaker or differently-selected checkpoint is not meaningful for this project.

Recommended training schedule:

```text
phase 1:
  freeze ParT for 1-2 epochs
  train adapter only
  very small LR

phase 2:
  unfreeze ParT
  train adapter + ParT with smaller LR for ParT

phase 3:
  optional last epochs with delta penalty annealed lower
```

But if freezing causes optimization issues, default to:

```text
freeze_part_epochs = 0
```

because baseline preservation already protects the run.

Zero final projection is the right initialization, but it creates a short
optimization lag: upstream local-compression layers receive little or no useful
gradient on the very first step until the final projection moves. That is fine,
but the early run should avoid excessive delta regularization.

Required training diagnostics:

```text
delta_F_l2 from epoch 1 onward
delta_F_l2 by feature from epoch 1 onward
adapter final projection norm
adapter upstream gradient norm if easy to log
```

Recommended learning rates:

```text
adapter_lr: 3e-4
part_lr:    3e-5 to 1e-4
weight_decay: 1e-4
```

## Loss

The primary training loss is ordinary supervised binary classification:

```text
CrossEntropyLoss(logits_2class, label)
```

Use the two-logit ParT output directly. This keeps checkpoint compatibility with
the exact HLT ParT baseline. Only use `BCEWithLogitsLoss` if the implementation
explicitly converts the two-class output into:

```text
single_logit = logit_Hgg - logit_QCD
```

Add small feature-residual regularization:

```text
loss = CE + lambda_delta * mean(mask * ||delta_F||^2)
```

Optional auxiliary diagnostics, not main losses:

```text
gate entropy
delta norm by feature
delta norm by modality
baseline score shift
```

Do not add too many clever losses in the first implementation. We already have
the residual-expert project for specialized complementarity losses. This plan is
about the pure architecture question:

```text
does local compression improve the exact ParT input representation?
```

## Model Selection

For binary QCD vs Hgg:

```text
selection metric: fpr_at_signal_eff_0p50
direction:        lower is better
selection split:  model_val
```

Do not select by accuracy. Accuracy can hide the thing we care about most:
background rejection at fixed Hgg signal efficiency.

Final test requires explicit confirmation:

```text
--confirm-final-test
```

Report two versions of final-test FPR@50:

```text
oracle-style final_test FPR@50:
  compute the threshold directly on final_test for comparison with old reports

validation-threshold final_test FPR:
  choose the threshold on model_val at 50% Hgg efficiency
  apply that fixed threshold once to final_test
  report final_test signal efficiency and FPR
```

The validation-threshold number is the more deployable check. The oracle-style
number is still useful for continuity with prior experiments.

## Evaluation Splits

Use the already-established split discipline:

```text
model_train:
  train LC adapter and ParT fine-tune

model_val:
  checkpoint selection
  threshold selection for FPR@50 diagnostics

stack_train / stack_val:
  optional later fusion or residual-expert analysis
  not needed for the main standalone LC adapter result

final_test:
  final held-out result only after model selection
```

For the current high-data realm:

```text
model_train: 3M
model_val:   1M
final_test:  1M
```

For pilot:

```text
model_train: 500k
model_val:   150k
final_test:  500k
```

## Main Variants

The first implementation should not explode into too many modes. The key is to
isolate which piece helps.

### LC-A0: Zero Adapter Control

```text
delta_F projection zero
adapter present
no training or frozen adapter
same HLT ParT checkpoint
```

Purpose:

```text
prove exact baseline recovery
```

### LC-A1: MLP Front-End Delta

```text
F_i -> MLP -> delta_F_i
no modality subtokens
same delta bounds
same ParT backbone
```

Purpose:

```text
control for extra parameters and feature correction
```

If A1 matches A2, modality tokenization may not matter.

### LC-A2: Local Modality Compression

```text
modality subtokens
local within-particle transformer
pooled token
delta_F
```

Purpose:

```text
test local compression without context-aware gates
```

### LC-A3: Context-Aware Reliability Gates

```text
modality subtokens
local compressor
shallow particle context
sigmoid gates
delta_F
```

Purpose:

```text
best first serious model
```

### LC-A4: Context Delta Without Modalities

```text
F_i -> shallow particle context -> delta_F
no modality subtokens
no local compressor
same delta bounds
same ParT backbone
```

Purpose:

```text
separate "local compression helped" from "a small pre-ParT context adapter helped"
```

This control is required if LC-A3 wins. Otherwise a gain could be caused by the
extra particle-context block rather than by within-particle modality structure.

### LC-A5: Random Grouping Control

```text
same number of modality tokens
random/permuted feature groupings
same architecture
```

Purpose:

```text
test whether meaningful modality grouping matters
```

The random grouping should preserve group sizes. If LC-A3 beats one random
grouping, rerun this control with at least a couple random seeds before making a
strong claim.

### LC-A6: Larger ParT Control

```text
baseline ParT with similar parameter count increase
no local compression
```

Purpose:

```text
prove wins are not just extra parameters
```

This one may be expensive, but it is scientifically important if A3 wins.

## Diagnostics

Every run should report:

```text
final_test accuracy
final_test AUC
final_test oracle FPR@50
final_test validation-threshold FPR
final_test validation-threshold signal efficiency
final_test FPR@30 if useful
background rejection at 50%
best epoch
selection metric
```

Adapter-specific diagnostics:

```text
delta_F_l2_mean
delta_F_l2_p90
delta_F_by_feature
delta_F_by_particle_pt_rank
max_abs_delta_F
fraction_delta_clipped_if_tanh_scaled
```

Gate diagnostics:

```text
mean gate by modality
gate entropy
gate variance
gate by true class
gate by baseline score margin
gate by particle pT rank
gate by photon PID flag
gate by neutral fraction
```

Boundary diagnostics:

```text
baseline FPR@50 false positives
LC FPR@50 false positives
overlap fraction
newly fixed baseline false positives
newly introduced false positives
baseline false negatives near threshold fixed by LC
score shift vs baseline margin
```

Slice diagnostics for QCD vs Hgg:

```text
photon multiplicity
leading photon pT fraction
neutral energy fraction
charged multiplicity
track-feature availability
high d0/dz uncertainty
baseline score margin bins
```

## What Would Count As A Meaningful Win

A small random-looking improvement is not enough. We want evidence that local
compression found something real.

Strong evidence:

```text
A3 beats exact HLT ParT on model_val and final_test FPR@50
A3 beats A1 MLP front-end
A3 beats A4 context-delta without modalities
A3 beats A5 random grouping
A3 has stable gain across seeds
A3 fixes a nontrivial set of baseline false positives/false negatives
diagnostics show plausible modality/gate behavior
```

Weak evidence:

```text
A3 beats baseline once but not across seeds
A3 only beats A1 by noise
A3 improves accuracy but not FPR@50
A3 needs final-test threshold tuning
```

Failure evidence:

```text
A3 underperforms exact HLT ParT
A1 equals or beats A3
delta_F stays near zero and metrics unchanged
delta_F grows large but FPR worsens
```

## Relationship To Existing Branches

### Difference From `subtoken_part`

The existing subtoken branch tests a related idea with a custom global
pairwise-biased transformer. That is useful, but it is not the cleanest way to
beat exact HLT ParT.

This plan keeps the exact HLT ParT backbone and changes only canonical PF
features before that backbone.

### Difference From Local Graph

Local graph adds neighborhood structure before ParT-like reasoning. This plan
adds feature/modality structure inside each particle before ParT.

They are complementary:

```text
local graph:        particle-neighborhood prior
local compression:  feature/modality prior inside particle
```

### Difference From Residual Expert

The residual expert learns:

```text
final score = z_base + alpha * residual_score
```

This plan learns:

```text
features_prime = features + delta_F
final score = ParT(features_prime)
```

So this is not score-level fusion. It is input-level adaptation of the exact
baseline.

### Possible Future Hybrid With Multiscale Subjet

For QCD vs Hgg, a high-upside future branch is:

```text
local compression
  -> better particle features/reliability
  -> multiscale subjet adapter
  -> residual delta_F
  -> exact HLT ParT
```

That is probably more ambitious than the first implementation should be, but it
may be the best long-run architecture if both branches show partial gains.

## Implementation Steps

### Step 1: Rewrite The Plan Contract In Code

Add a new package:

```text
teacher_logit_reco/local_compression_part/
```

Initial files:

```text
__init__.py
config.py
features.py
subtokens.py
compressor.py
gates.py
adapter.py
model.py
train.py
reports.py
```

The config should make the mainline explicit:

```text
backbone_type = "exact_hlt_part"
adapter_mode = "residual_delta_features"
baseline_recoverable_at_zero_delta = True
selection_metric = "fpr_at_signal_eff_0p50"
label_names = ("QCD", "Hgg")
hlt_degradation_strength = 0.6
```

### Step 2: Build Canonical Inputs Wrapper

Implement a helper that takes raw HLT tokens and returns canonical ParT inputs:

```text
tokens [B, N, 14]
mask   [B, N]
->
CanonicalPartInputs:
  features        [B, C, N]
  points          [B, P, N]
  lorentz_vectors [B, V, N]
  mask            [B, N]
  feature_rows    [B, N, C]
```

Prefer reusing the existing repo functions/patterns from ParT and multiscale
subjet code. Do not duplicate preprocessing logic with ad hoc formulas unless
there is no existing helper.

Tests:

```text
raw token dim is 14
feature dim matches len(PF_FEATURE_NAMES)
feature order matches PF_FEATURE_NAMES
points/vectors/mask match baseline builder
```

### Step 3: Implement Modality Feature Builder

Create modality feature tensors from raw tokens and canonical features:

```text
geometry_fields
energy_momentum_fields
identity_fields
tracking_error_fields
quality_consistency_fields
```

The builder should return:

```text
LocalCompressionModalities:
  values_by_modality: dict[str, Tensor [B, N, C_m]]
  modality_mask: Tensor [B, N, M]
  particle_mask: Tensor [B, N]
  metadata: feature names and source columns
```

Tests:

```text
all outputs finite
invalid particles are zeroed
charged/neutral consistency features behave as expected
phi wrapping features use sin/cos or wrapped delta phi
```

### Step 4: Implement Subtoken Embedders

For each modality:

```text
MLP(C_m -> D_local)
```

Then add:

```text
modality type embedding
particle anchor embedding from F_i
pT rank embedding or scalar projection
```

Output:

```text
subtokens [B, N, M, D]
```

Tests:

```text
shape correct
mask correct
different modality type ids change subtokens
particle anchor binds modality tokens
gradients flow to modality embedders
```

### Step 5: Implement Local Compressor

Implement:

```text
LocalModalityCompressor
```

Forward:

```text
[B, N, M, D] -> [B, N, M, D]
```

Internally:

```text
reshape to [B*N, M, D]
TransformerEncoder over modalities
reshape back
```

Use `src_key_padding_mask` for dropped or invalid modalities.

Tests:

```text
mask-safe
finite with empty/invalid particles
permutation behavior understood and documented
gradients flow
```

### Step 6: Implement Provisional Pooling

Pool local modality outputs to a provisional particle token:

```text
local_particle_token [B, N, D]
pool_weights [B, N, M]
```

Use learned-query masked pooling by default. The pool weights are diagnostics,
not necessarily reliability gates.

Tests:

```text
inactive modalities get zero weight
active weights sum to one
invalid particles produce zero token
```

### Step 7: Implement Shallow Particle Context

Add a small context module:

```text
ParticleContextBlock
```

Default:

```text
1 global self-attention layer
mask-aware
small hidden dimension
```

Input:

```text
local_particle_token [B, N, D]
mask [B, N]
```

Output:

```text
context_token [B, N, D]
```

This is only for reliability/delta prediction. It is not the main classifier.

Tests:

```text
mask-safe
context changes when neighboring particles change
invalid particles remain zero
```

### Step 8: Implement Context-Aware Gates

Implement sigmoid modality gates:

```text
gate_input = concat(local_modality_token, context_token, F_i summary)
gate = sigmoid(MLP(gate_input))
```

Output:

```text
gates [B, N, M]
```

Tests:

```text
gates in [0, 1]
inactive modalities have zero diagnostic weight
changing context can change gates
no saturation at initialization
```

### Step 9: Implement Delta-F Adapter

Implement:

```text
LocalCompressionDeltaFAdapter
```

Input:

```text
F_rows
local pooled token
context token
gates
```

Output:

```text
delta_F_rows [B, N, len(PF_FEATURE_NAMES)]
adapter_diagnostics
```

The final projection must be zero-initialized.

Recommended:

```text
delta_F = delta_scale * tanh(raw_delta)
```

Tests:

```text
delta_F is exactly zero at initialization
nonzero after perturbing final layer
invalid particles get zero delta
bounded delta if tanh scaling enabled
```

### Step 10: Implement Exact ParT Wrapper

Implement:

```text
LocalCompressionFeatureAdapterParT
```

It should contain:

```text
part_model: ParticleTransformerHLTClassifier
adapter: LocalCompressionDeltaFAdapter
```

Forward:

```text
canonical = build canonical inputs(tokens, mask)
delta_F = adapter(tokens, canonical)
features_prime = canonical.features + delta_F.transpose(1, 2)
logits = part_model(canonical.points, features_prime, canonical.lorentz_vectors, canonical.mask)
```

Tests:

```text
with zero delta and same checkpoint, logits match baseline
changing adapter changes logits
model reports baseline_recoverable_at_zero_delta
uses exact ParticleTransformerHLTClassifier
```

### Step 11: Add Checkpoint Loading And Warm Start

Add checkpoint utilities:

```text
load_hlt_part_baseline_checkpoint(path)
load into part_model
verify config compatibility
record source checkpoint hash/path
```

The report should include:

```text
baseline_checkpoint_path
baseline_checkpoint_hash
baseline_checkpoint_selection_metric
baseline_checkpoint_hlt_degradation_strength
baseline_checkpoint_split_manifest_hash
part_config
adapter_config
init_logit_diff_vs_baseline
```

### Step 12: Add Training Runner

Add script:

```text
scripts/train_local_compression_part_tagger.py
```

Required args:

```text
--output-dir
--manifest-path
--hlt-cache-dir
--baseline-checkpoint
--label-names QCD Hgg
--label-filter-names QCD Hgg
--train-split model_train
--val-split model_val
--final-test-split final_test
--selection-metric fpr_at_signal_eff_0p50
--confirm-final-test
```

Training report:

```text
run_report.json
training_curves.json
best_model_val.pt
last.pt
diagnostics/
```

### Step 13: Add Variants

Implement variant names:

```text
hlt_part_baseline_recheck
lc_mlp_delta
lc_local_compression_no_context
lc_context_gated
lc_context_delta_no_modalities
lc_random_grouping
```

Do not submit the larger-ParT control by default unless cluster time allows.

Each variant should share:

```text
same split
same HLT cache
same FPR@50-selected HLT0.6 baseline checkpoint when applicable
same final-test confirmation
same FPR@50 selection
```

### Step 14: Add Final Report

Add:

```text
scripts/write_local_compression_part_report.py
teacher_logit_reco/local_compression_part/reports.py
```

The report should compare:

```text
exact HLT ParT baseline
LC MLP delta
LC local compression
LC context-gated
LC context-delta without modalities
random grouping control
optional larger ParT control
```

Primary sorting:

```text
final_test_fpr_at_signal_eff_0p50 lower is better
```

Also include:

```text
model_val selected metric
final_test AUC
final_test accuracy
final_test oracle FPR@50
final_test validation-threshold FPR
final_test validation-threshold signal efficiency
background rejection at 50%
bootstrap uncertainty if implemented
```

### Step 15: Add Slurm Submitter

Add:

```text
sbatch/run_train_local_compression_part.sh
sbatch/submit_local_compression_part_qcd_hgg_hlt0p6_experiment.sh
sbatch/run_write_local_compression_part_report.sh
```

Defaults:

```text
DATA_ROOT uses existing QCD/Hgg HLT0.6 cache
model_train 500k pilot first
model_val 150k pilot first
final_test 500k pilot first
epochs 30-45
selection metric fpr_at_signal_eff_0p50
```

Then a separate high-data command can target:

```text
3M / 1M / 1M
```

### Step 16: Smoke Tests

Add tests:

```text
tests/test_local_compression_part_step1_features.py
tests/test_local_compression_part_step2_subtokens.py
tests/test_local_compression_part_step3_adapter.py
tests/test_local_compression_part_step4_model.py
tests/test_local_compression_part_step5_training_smoke.py
tests/test_sbatch_scripts.py
```

Critical tests:

```text
exact baseline recovery at zero delta
delta projection initialized to zero
correct PF feature order
mask-safe subtokens/gates/pooling
selection metric defaults to FPR@50 for binary
training loss uses two-logit CrossEntropyLoss by default
baseline checkpoint metadata rejects wrong split/cache/selection metric
validation-threshold final-test reporting is present
```

### Step 17: 500k Pilot

Run:

```text
exact baseline recheck
LC MLP delta
LC local compression no context
LC context-gated
LC context-delta no modalities
random grouping control
```

Decision:

```text
if LC context-gated > baseline and > MLP/context-only/random controls:
  proceed to high-data run

if LC context-gated ~= baseline but fixes different boundary events:
  consider residual-expert or fusion version

if LC loses clearly:
  do not spend 3M run unless diagnostics suggest a fix
```

### Step 18: High-Data Run

Only after the 500k pilot is stable:

```text
model_train: 3M
model_val:   1M
final_test:  1M
```

The high-data result answers the real question:

```text
Can local feature/modality compression improve exact HLT ParT
when both have lots of data?
```

### Step 19: If It Works, Build The Hybrid

If LC wins or nearly wins, the next high-upside branch is:

```text
LC feature adapter
  -> multiscale subjet feature adapter
  -> exact HLT ParT
```

This combines:

```text
within-particle feature reliability
with
explicit photon/prong/subjet organization
```

That is likely the strongest long-term architecture for QCD vs Hgg if the
individual pieces show real signal.
