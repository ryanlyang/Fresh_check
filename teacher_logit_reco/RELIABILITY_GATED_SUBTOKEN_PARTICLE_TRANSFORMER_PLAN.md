# Reliability-Gated Subtoken Particle Transformer Plan

This document defines a new architecture branch for HLT-only jet tagging:

```text
Reliability-Gated Hierarchical Subtoken Particle Transformer
```

The short idea is:

```text
ParT reasons across particles.
This branch first reasons inside each particle, across physical feature
modalities, then lets enriched particles reason globally.
```

The deployed model is still HLT-only:

```text
inference:
  fixed HLT jet -> reliability-gated subtoken ParT -> prediction
```

Offline information can be used during training for an optional privileged
version:

```text
training only:
  paired offline jet
  offline teacher logits
  offline-vs-HLT residual/reliability targets
```

but offline constituents are never used at inference.

## Why This Branch Exists

The Particle Transformer is a very strong jet tagger because it picks an
excellent tokenization level:

```text
one particle / constituent = one transformer token
```

That gives the model direct particle-to-particle relational reasoning.  It also
lets ParT inject physics-aware pairwise information into attention, such as
relative angular and kinematic relations.

The possible blind spot is earlier in the model.  A standard ParT-style front
end usually does:

```text
full per-particle feature vector -> embedding MLP -> particle token
```

That means heterogeneous evidence streams are fused before the transformer does
global reasoning:

```text
kinematics
PID / charge / identity
track / displacement / quality
```

The architecture question is:

```text
Does fusing heterogeneous particle features too early hide useful structure?
```

This branch tests the alternative:

```text
Let kinematics, identity, and track/displacement evidence become local subtokens.
Let those subtokens negotiate inside each particle.
Use jet context to estimate which modalities are reliable.
Then build the particle token used by the global ParT stage.
```

The desired gain is not merely sample efficiency.  The goal is to change the
architecture floor: to make useful HLT evidence easier to extract than it is for
a standard HLT ParT, even when both models receive large training samples.

## Research Inspiration

This plan is inspired by a broad transformer pattern:

```text
better tokenization + hierarchy + controlled fusion
```

The relevant external precedents are:

- [Transformer in Transformer](https://arxiv.org/abs/2103.00112):
  fine-grained "visual word" tokens are processed inside coarser patch tokens,
  then patch tokens are processed globally.  This is the closest vision analogy
  to "feature subtokens inside particle tokens."
- [Tokens-to-Token ViT](https://arxiv.org/abs/2101.11986):
  naive patch tokenization is improved by local aggregation before global
  transformer processing.
- [CrossViT](https://arxiv.org/abs/2103.14899):
  different tokenizations of the same image can be fused effectively, especially
  when the fusion is cross-attention rather than a naive concatenation.
- [FT-Transformer](https://arxiv.org/abs/2106.11959),
  [TabTransformer](https://arxiv.org/abs/2012.06678), and
  [SAINT](https://arxiv.org/abs/2106.01342):
  feature values inside one object can be treated as tokens, and feature-level
  interactions can matter.
- [Set Transformer](https://arxiv.org/abs/1810.00825) and
  [Perceiver](https://arxiv.org/abs/2103.03206):
  learned latent bottlenecks and inducing points are useful when raw token
  counts are high.
- [Particle Transformer](https://arxiv.org/abs/2202.03772):
  particle-level attention with pairwise physics bias is the baseline to
  preserve and beat, not something to throw away.
- [PHAT-JeT](https://arxiv.org/abs/2605.21789), SAL-T, OmniLearn, Masked
  Particle Modeling, and OmniJet-alpha:
  HEP is already moving toward explicit tokenization, local/global hierarchy,
  and reusable tokenized jet representations, but not this exact
  within-particle modality hierarchy.

The important lesson is:

```text
Tokenization alone rarely wins.
Tokenization plus the right architecture often wins.
```

## Core Hypothesis

Standard ParT is strong at:

```text
particle_i <-> particle_j
```

This branch makes the following relations first-class:

```text
kinematics_i <-> identity_i
kinematics_i <-> tracking_i
identity_i   <-> tracking_i

kinematics_i reliability depends on jet context
identity_i reliability depends on jet context
tracking_i reliability depends on jet context
```

The model should learn context-aware statements like:

```text
this particle's track evidence is suspicious in this dense region
this particle's kinematics look merged relative to the surrounding topology
PID evidence is more useful for this particle than track displacement
track/IP information is the key discriminant for this local HLT pattern
```

That is the proposed missing inductive bias.

## Data Contracts In This Repo

There are two feature spaces that must not be confused.

### Raw JetClass Token Space

The repo's raw `JetView.tokens` contract is currently:

```text
RAW_TOKEN_DIM = 14
columns:
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

This is the safest source for subtokenization because the modality grouping is
direct.

### ParT Feature Space

`jetclass_fresh/part_inputs.py` converts raw tokens into ParT-style features:

```text
PF_FEATURE_NAMES:
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

This transformed feature space has 17 particle features.  User discussions often
refer to a "19-vector" or "full feature vector"; implementation should always
check the actual local contract rather than assuming the number.

## Version A And Version B

This branch should be implemented as two closely related tracks.

### Version A: HLT-Only Architecture Test

This version asks:

```text
Can the architecture alone beat basic HLT ParT?
```

Training and inference both use only HLT and labels:

```text
fixed HLT jet -> reliability-gated subtoken ParT -> label
```

No offline teacher, no offline target, no reconstructor.

This is the cleanest architecture baseline.  If Version A beats HLT ParT, then
the subtoken/reliability inductive bias itself is useful.

### Version B: Privileged Offline Training Test

This version asks:

```text
Can paired offline information shape an even better HLT-only representation?
```

Inference is still HLT-only:

```text
fixed HLT jet -> reliability-gated subtoken ParT -> label
```

Training can add offline-derived losses:

```text
classification loss
+ offline teacher distillation
+ HLT/offline modality residual prediction
+ masked-subtoken reconstruction
+ optional reliability pseudo-targets from HLT/offline disagreement
```

Version B is the scientifically richer branch, but Version A must exist so that
we can separate:

```text
architecture gain
```

from:

```text
privileged-training gain
```

## Architecture Overview

The base architecture is:

```text
raw HLT tokens
  -> modality subtokenizer
  -> within-particle subtoken mixer
  -> provisional particle token
  -> particle-context stage
  -> context-aware modality gates
  -> reliability-aware particle token
  -> ParT-style global particle transformer
  -> class head
```

Expanded:

```text
for each particle i:
  kin_i   = encode_kin(pt, eta, phi, energy, derived optional)
  id_i    = encode_id(charge, PID flags)
  trk_i   = encode_track(d0, d0err, dz, dzerr)
  anchor_i = encode_anchor(full raw or transformed particle vector)

  kin_token_i = kin_i + modality_embed_kin + anchor_i
  id_token_i  = id_i  + modality_embed_id  + anchor_i
  trk_token_i = trk_i + modality_embed_trk + anchor_i

  local_tokens_i = transformer_inside_particle([kin_token_i, id_token_i, trk_token_i])
  provisional_particle_i = pool(local_tokens_i)

particle_context = context_transformer(provisional_particles)

for each particle i:
  gates_i = softmax(gate_head([local_tokens_i, particle_context_i, anchor_i]))
  particle_i = anchor_i + sum_m gates_i[m] * local_tokens_i[m]

global_outputs = part_style_particle_transformer(particle_i, pairwise_features)
prediction = classifier(global_outputs)
```

## Subtoken Groups

The first implementation should use three modality tokens per particle.

### Kinematics Token

Raw-space inputs:

```text
pt, eta, phi, energy
```

Optional derived inputs:

```text
log_pt
log_energy
log_pt_rel
log_energy_rel
deltaR_to_jet_axis
deta_to_jet_axis
dphi_to_jet_axis
```

The derived features should match `part_inputs.py` where possible so the
baseline comparison remains fair.

### Identity Token

Raw-space inputs:

```text
charge
isChargedHadron
isNeutralHadron
isPhoton
isElectron
isMuon
```

This token captures particle identity evidence.

### Track / Displacement Token

Raw-space inputs:

```text
d0
d0err
dz
dzerr
```

This token captures impact-parameter and track-quality-like evidence.

### Why Not Scalar Tokens First?

Flat scalar tokenization would be expensive:

```text
128 particles * 14 raw features = 1792 tokens
128 particles * 17 ParT features = 2176 tokens
```

Full global attention would then scale as:

```text
O((N_particles * N_features)^2)
```

That is likely too expensive and also creates a binding problem:

```text
which scalar tokens belong to the same particle?
```

Modality tokens are the practical middle ground:

```text
128 particles * 3 modality tokens = 384 local tokens
```

but local attention is only performed within each particle:

```text
128 * 3^2 local attention entries
```

then the global stage returns to:

```text
128^2 particle attention entries
```

## Binding Mechanism

Every subtoken from the same particle must know that it belongs to the same
particle.  The preferred binding mechanism is a learned particle anchor:

```text
anchor_i = MLP(full_particle_features_i)
```

Then:

```text
subtoken_i_m = modality_encoder_m(values_i_m)
             + modality_type_embedding_m
             + anchor_i
```

This is better than a pure learned "particle index 17" embedding because jets
are sets.  We can optionally include sorted-rank or pT-rank information, but it
should be a controlled ablation rather than the only binding signal.

Recommended binding ablations:

```text
anchor only
anchor + pT-rank embedding
anchor + geometry/jet-axis embedding
learned particle index only
no anchor, modality embeddings only
```

The expected best default is:

```text
anchor + modality type embedding
```

with optional pT-rank only if needed.

## Within-Particle Subtoken Mixer

The local mixer should be small:

```text
num_modalities = 3
embed_dim = 128 or 192
local_layers = 1 to 2
local_heads = 4
dropout = 0.05
```

Because each particle has only three subtokens, this stage should not be deep.
The goal is not to build a huge transformer inside every particle; it is to let
modalities negotiate before they are compressed.

Candidate local mixer implementations:

1. Tiny self-attention over `[kin, id, track]`.
2. Attention pooling with a learned particle summary query.
3. Gated MLP over concatenated modality embeddings.

Start with option 1 because it is closest to TNT and easiest to reason about.
Keep option 3 as a cheap control.

## Context-Aware Reliability Gates

This is the central novelty.

After local subtoken mixing, create a provisional particle token:

```text
provisional_i = attention_pool(local_tokens_i)
```

Then run a lightweight particle-context stage:

```text
context_i = particle_context_transformer(provisional_particles)_i
```

The reliability gate is:

```text
gate_logits_i = MLP([context_i, provisional_i, anchor_i])
gates_i = softmax(gate_logits_i over modalities)
```

The reliability-aware particle token is:

```text
particle_i =
    anchor_i
  + gate_kin_i * local_kin_i
  + gate_id_i  * local_id_i
  + gate_trk_i * local_trk_i
```

A `sigmoid` gate can also be tested:

```text
particle_i = anchor_i + sum_m sigmoid(g_m) * local_m
```

but the first version should use `softmax` because it forces a clearer
allocation of trust among evidence streams and makes diagnostics easier.

### Gate Diagnostics

Every training/evaluation run should report:

```text
mean gate per modality
gate entropy
gate distribution by class
gate distribution by particle pT rank
gate distribution by HLT/offline agreement bucket when available
gate distribution by predicted class correctness
```

The gates should not be treated as guaranteed explanations, but they are useful
failure diagnostics.  For example, if the model always sets:

```text
kin = 0.98, id = 0.01, track = 0.01
```

then the architecture is collapsing back toward a kinematics-only tagger.

## Global ParT Stage

The global stage should preserve the strongest part of ParT:

```text
particle-level global self-attention
pairwise physics-aware attention bias
class token / class attention
mask handling for variable constituent counts
```

Implementation options:

1. Reuse the existing Weaver `ParticleTransformer` implementation by feeding
   reliability-aware particle embeddings into a compatible wrapper.
2. Implement a local ParT-like transformer body in this repo with pairwise bias.
3. Initially use a standard transformer encoder without pairwise bias, then add
   pairwise bias after the subtoken path is validated.

Recommended path:

```text
Step 1: local repo implementation with standard transformer body for smoke.
Step 2: add ParT-style pairwise bias / interaction embeddings.
Step 3: compare to the existing Weaver ParT baseline.
```

The final serious model must include pairwise physics bias or it will not be a
fair ParT competitor.

## Pairwise Physics Features

Pairwise particle features should be computed from the particle-level physical
inputs, not from subtokens:

```text
deltaR_ij
log_deltaR_ij
pair_mass_ij
relative_kT_ij
z_ij / momentum sharing if already available locally
```

The subtoken stage should not replace pairwise physics bias.  It should produce
better particle tokens for that bias to operate on.

The first implementation can reuse the pairwise feature builder already used by
the existing ParT path if accessible.  If not, implement a minimal pairwise
embedding with:

```text
deta
dphi
deltaR
log_deltaR
log_pair_mass
```

## Training Objectives

### Version A Loss

Version A is purely supervised HLT-only:

```text
loss = cross_entropy(labels, logits)
```

Optional regularizers:

```text
gate_entropy_regularization, weak
modality_dropout
feature_dropout
```

Do not over-regularize the first run.  The initial question is whether the
architecture can learn.

### Version B Loss

Version B adds privileged offline training signals:

```text
loss =
    classification_loss
  + lambda_teacher * offline_teacher_distillation
  + lambda_residual * modality_residual_loss
  + lambda_masked * masked_subtoken_loss
  + lambda_gate * optional_reliability_regularization
```

#### Offline Teacher Distillation

Use a frozen offline teacher trained on offline jets:

```text
teacher_logits = offline_teacher(offline_jet)
student_logits = subtoken_model(HLT_jet)
distill_loss = KL(student_temperature || teacher_temperature)
```

This is not new by itself, but it can be useful when paired with the new
architecture.

#### Modality Residual Prediction

Because HLT and offline jets are paired, we can estimate how much the HLT view
deviates from offline at a coarse modality level:

```text
kin_residual_target   = summary_distance(HLT kin, offline kin)
id_residual_target    = summary_distance(HLT PID/identity, offline PID/identity)
track_residual_target = summary_distance(HLT track, offline track)
```

The model predicts:

```text
residual_pred_i_m = MLP([context_i, local_token_i_m])
```

This gives the reliability gates a supervised reason to learn:

```text
which modalities are likely corrupted?
```

The first residual targets should be simple and stable:

```text
per-particle nearest-neighbor residual to offline, using deltaR matching
or
jet-level modality residual summary broadcast to particles
```

Avoid a complex Hungarian residual target in the first version; keep it as an
optional later improvement.

#### Masked Subtoken Reconstruction

Mask one modality subtoken for some particles:

```text
mask kin token
mask id token
mask track token
```

Ask the local+global model to reconstruct it.

This mirrors the lessons from masked particle modeling and tokenized
pretraining, but at the subtoken level:

```text
can the rest of the particle plus the jet context predict the missing modality?
```

This objective can be used:

```text
on HLT only
or
with offline target subtokens for privileged training
```

The first supervised Version B should use it lightly or leave it off until the
base distillation/residual path is stable.

## Modality Dropout

This architecture should support modality dropout:

```text
randomly drop kinematics token
randomly drop identity token
randomly drop track token
```

This helps test robustness and makes the reliability gates meaningful.  It is
also a bridge to detector-shift studies where some modalities may become less
available or less reliable.

Initial defaults:

```text
modality_dropout_prob = 0.0 for first smoke
modality_dropout_prob = 0.05 to 0.10 for serious Version B
```

## Model Variants

We should implement several named variants so results are interpretable.

### Baseline Controls

```text
part_baseline
```

Existing HLT ParT baseline.

```text
subtoken_no_gate
```

Within-particle subtokens are mixed, pooled, and sent to global ParT, but no
context-aware reliability gates are used.

```text
subtoken_gate_local_only
```

Reliability gates use only within-particle information, not global jet context.

```text
subtoken_gate_context
```

Full proposed model.

### Dual-View Variant

```text
dual_part_subtoken_cross_attention
```

Two branches:

```text
standard HLT ParT branch
subtoken/reliability branch
```

Fusion:

```text
class token cross-attention
or
small latent fusion transformer
```

This is inspired by CrossViT.  It is less elegant than the single hierarchical
model, but it is safer experimentally because the standard ParT branch remains
available as a strong view.

### Scalar-Token Research Control

```text
scalar_token_local_only
```

Every raw scalar becomes a token, but attention is only within particle before
pooling.  This tests whether the three-modality grouping is too coarse.

Do not run a flat global scalar-token model except on tiny smoke tests; it is
too expensive and likely not the right inductive bias.

## Evaluation Tracks

### Track 1: Binary HLT JetClass

Use the current binary workflow:

```text
QCD vs Hgg
QCD vs Tbqq
possibly QCD vs Hbb
```

Metrics:

```text
accuracy
AUC
FPR at 30% signal efficiency
FPR at 50% signal efficiency
background rejection at fixed signal efficiency
```

This is the fastest way to answer:

```text
Does the architecture beat HLT ParT on realistic trigger-style binary tasks?
```

### Track 2: Full 10-Class JetClass

Metrics:

```text
accuracy
macro accuracy
per-class accuracy
confusion matrix
class-conditional FPR where meaningful
```

This answers whether the gain generalizes beyond one binary signal/background
definition.

### Track 3: Data Scaling

Run:

```text
100k
500k
2M if available
```

The key question is:

```text
Does the subtoken architecture remain ahead as both models get more data?
```

If gains disappear with more data, the method may be mostly sample-efficiency.
If gains persist, the architecture story is much stronger.

### Track 4: HLT Degradation Strength

Test:

```text
HLT strength 1.0
HLT strength 0.6
possibly 0.3
```

The question:

```text
Does modality-aware reliability help more when HLT preserves enough signal to
infer missing structure?
```

## Must-Have Baselines

Every serious run should include:

```text
HLT ParT baseline
HLT ParticleNet baseline if practical
HLT PFN baseline if practical
HLT PCNN baseline if practical
offline ParT reference
subtoken_no_gate
subtoken_gate_local_only
subtoken_gate_context
```

For Version B:

```text
HLT-only subtoken_gate_context
privileged subtoken_gate_context
HLT-only distillation baseline if already available
```

The most important comparison is:

```text
HLT ParT
vs
HLT-only subtoken_gate_context
```

The second most important comparison is:

```text
HLT-only subtoken_gate_context
vs
privileged subtoken_gate_context
```

## Diagnostics

Save diagnostics outside checkpoint directories, mirroring the existing
diagnostics-root pattern.

Minimum diagnostics:

```text
run_config.json
model_config.json
best_epoch
metric history
final_test metrics
gate summaries
class-conditional gate summaries
gate entropy summaries
modality dropout settings
parameter count
rough FLOPs or attention-entry estimates
```

For gate diagnostics:

```text
mean_gate_by_modality
mean_gate_by_class
mean_gate_by_correctness
mean_gate_by_particle_rank
gate_entropy_by_class
```

For Version B:

```text
distillation loss history
residual prediction metrics
masked subtoken reconstruction metrics
teacher/student agreement
```

## Failure Modes To Watch

### Gate Collapse

The model may always choose one modality.

Mitigation:

```text
weak gate entropy regularization
modality dropout
temperature on gate softmax
residual auxiliary loss
```

### No Gain Over ParT

Possible reasons:

```text
ParT MLP already learns the needed feature interactions
subtoken model is underparameterized
global stage lacks pairwise physics bias
gates are too constrained
training schedule is not tuned
```

Required response:

```text
compare no_gate vs gate
compare local_only gate vs context gate
compare with/without pairwise bias
compare parameter-matched ParT
```

### Overfitting Small Binary Tasks

Binary tasks can be easy, especially offline.  Use FPR at fixed signal
efficiency and compare against large HLT ParT baselines.

### Interpretability Overclaiming

Reliability gates are not guaranteed causal explanations.  Treat them as
diagnostics and architectural routing weights, not proof of detector-level
truth.

## Implementation Location

Recommended new package:

```text
teacher_logit_reco/subtoken_part/
```

Suggested files:

```text
teacher_logit_reco/subtoken_part/__init__.py
teacher_logit_reco/subtoken_part/config.py
teacher_logit_reco/subtoken_part/features.py
teacher_logit_reco/subtoken_part/model.py
teacher_logit_reco/subtoken_part/pairwise.py
teacher_logit_reco/subtoken_part/losses.py
teacher_logit_reco/subtoken_part/train.py
teacher_logit_reco/subtoken_part/evaluate.py
teacher_logit_reco/subtoken_part/reports.py
```

Scripts:

```text
scripts/train_subtoken_part_tagger.py
scripts/evaluate_subtoken_part_tagger.py
scripts/write_subtoken_part_report.py
```

Slurm:

```text
sbatch/run_train_subtoken_part_tagger.sh
sbatch/run_evaluate_subtoken_part_tagger.sh
sbatch/run_write_subtoken_part_report.sh
sbatch/submit_subtoken_part_qcd_hgg_binary_experiment.sh
```

The implementation should reuse:

```text
jetclass_fresh.hlt_cache
jetclass_fresh.jetclass_data
jetclass_fresh.part_inputs
teacher_logit_reco diagnostics helpers where practical
binary label-filtered split/cache builders
```

Do not make a new data pipeline unless absolutely necessary.

## Proposed Implementation Steps

### Step 1: Write Architecture Contract And Configs

Add:

```text
teacher_logit_reco/subtoken_part/config.py
```

Define:

```text
SubtokenPartConfig
SubtokenFeatureConfig
SubtokenTrainingConfig
SubtokenVariantConfig
```

Include:

```text
num_classes
raw_token_dim = 14
embed_dim
num_modalities = 3
local_layers
local_heads
context_layers
context_heads
global_layers
global_heads
gate_mode
use_pairwise_bias
version = A or B
```

Add validation for invalid dimensions, heads, dropout, and feature group
indices.

### Step 2: Implement Feature Grouping

Add:

```text
teacher_logit_reco/subtoken_part/features.py
```

Implement:

```text
split_raw_tokens_into_modalities(tokens, mask)
build_derived_kinematics(tokens, mask)
build_subtoken_inputs(tokens, mask, config)
```

Return:

```text
kin_values:   [batch, particles, kin_dim]
id_values:    [batch, particles, id_dim]
track_values: [batch, particles, track_dim]
mask:         [batch, particles]
```

Tests:

```text
shape checks
mask zeroing
finite values
raw 14-dim contract
derived features match part_inputs.py where expected
```

### Step 3: Implement Modality Encoders And Particle Anchor

Add:

```text
KinematicsEncoder
IdentityEncoder
TrackEncoder
ParticleAnchorEncoder
```

Each maps its input to `embed_dim`.

Output:

```text
subtokens: [batch, particles, modalities, embed_dim]
anchor:    [batch, particles, embed_dim]
```

Tests:

```text
subtokens from same particle share anchor contribution
different modality embeddings change outputs
gradients flow to all modality encoders
```

### Step 4: Implement Within-Particle Subtoken Mixer

Implement:

```text
WithinParticleSubtokenTransformer
```

Input:

```text
[batch, particles, modalities, embed_dim]
```

Internally reshape:

```text
[batch * particles, modalities, embed_dim]
```

Run tiny transformer.

Output:

```text
local_tokens: [batch, particles, modalities, embed_dim]
```

Tests:

```text
mask handling
finite outputs
gradients
permutation only across particles should not affect within-particle local shape
```

### Step 5: Implement Local Pooling To Provisional Particle Tokens

Implement:

```text
SubtokenAttentionPool
```

Options:

```text
mean pooling
learned query attention pooling
CLS-like modality summary token
```

Default:

```text
learned query attention pooling
```

Return:

```text
provisional_particles: [batch, particles, embed_dim]
pool_weights:          [batch, particles, modalities]
```

### Step 6: Implement Particle Context Stage

Implement a lightweight particle transformer over provisional particles:

```text
ParticleContextTransformer
```

This stage provides context for reliability gates.  It does not replace the
final ParT stage.

Tests:

```text
particle mask respected
changing neighboring particles can change context for a particle
```

### Step 7: Implement Context-Aware Reliability Gates

Implement:

```text
ReliabilityGateHead
```

Input:

```text
local_tokens
provisional_particles
particle_context
anchor
```

Output:

```text
gate_logits: [batch, particles, modalities]
gates:       [batch, particles, modalities]
```

Default:

```text
softmax over modalities
```

Add gate diagnostics.

Tests:

```text
gates sum to one for valid particles
masked particles have zeroed diagnostics
context changes can change gates
gate entropy finite
```

### Step 8: Implement Reliability-Aware Particle Token Builder

Combine:

```text
particle_token = anchor + weighted_modalities
```

Support variants:

```text
no_gate
local_gate
context_gate
```

Tests:

```text
no_gate equals configured pooling behavior
context_gate uses global context
all modality paths receive gradients
```

### Step 9: Implement Initial Global Transformer Classifier

Implement a simple masked transformer classifier first:

```text
SubtokenParticleTransformerClassifier
```

Components:

```text
subtoken encoder
context gates
global transformer
class token / masked pooling
classification head
```

This smoke model can omit ParT pairwise bias initially, but the config must make
that explicit:

```text
use_pairwise_bias = false
```

Tests:

```text
forward pass
loss backward
variable particle counts
binary and multiclass output dims
```

### Step 10: Add ParT-Style Pairwise Bias

Implement:

```text
pairwise.py
PairwiseFeatureBuilder
PairwiseBiasEncoder
PairwiseBiasedAttentionBlock
```

Use raw kinematics to build pairwise features.

Tests:

```text
delta phi wraparound
pairwise mask
symmetry where expected
attention changes when pairwise features change
```

### Step 11: Replace Or Augment Global Stage With ParT-Like Blocks

Upgrade the classifier so the serious default uses:

```text
reliability-aware particle tokens
+ pairwise-biased particle attention
```

This is the first model that should be compared seriously to HLT ParT.

### Step 12: Implement Version A Training

Add:

```text
scripts/train_subtoken_part_tagger.py
teacher_logit_reco/subtoken_part/train.py
```

Support:

```text
binary/multiclass
label filters
HLT cache input
model_train/model_val/stack_val/final_test
selection metric
diagnostics root mirroring
```

Metrics:

```text
accuracy
AUC for binary
FPR at 30/50 signal efficiency for binary
per-class accuracy for multiclass
```

### Step 13: Add HLT ParT Baseline Runner Compatibility

Make sure the same split/cache can run:

```text
standard HLT ParT
subtoken_no_gate
subtoken_gate_local_only
subtoken_gate_context
```

The report must compare all of them on identical splits.

### Step 14: Add Version A Slurm Submitter

Add:

```text
sbatch/submit_subtoken_part_qcd_hgg_binary_experiment.sh
```

First real run defaults:

```text
labels: QCD Hgg
hlt_degradation_strength: 0.6 and/or 1.0
model_train: 500k
model_val: 150k
final_test: 500k
epochs: 45
variants:
  hlt_part_baseline
  subtoken_no_gate
  subtoken_gate_local_only
  subtoken_gate_context
```

### Step 15: Implement Report Builder

Add:

```text
scripts/write_subtoken_part_report.py
teacher_logit_reco/subtoken_part/reports.py
```

Report:

```text
best metric by selection target
full metric table
gate diagnostics
parameter counts
runtime/walltime
baseline comparison
```

For binary, prefer:

```text
FPR at 50% signal efficiency, lower is better
```

unless explicitly configured otherwise.

### Step 16: Implement Version B Offline Teacher Distillation

Extend training to accept:

```text
offline teacher checkpoint
offline view loader
distillation temperature
distillation weight
```

Training still feeds HLT to the student.

Tests:

```text
offline teacher only used in training
inference artifacts do not require offline
distillation loss finite
```

### Step 17: Implement Modality Residual Auxiliary Targets

Add coarse HLT/offline residual targets:

```text
kin residual
identity residual
track residual
```

Start with jet-level and nearest-neighbor residuals; do not overbuild.

Use them to supervise:

```text
residual_pred_by_modality
```

and optionally regularize reliability gates.

### Step 18: Implement Masked Subtoken Objective

Add masked modality modeling:

```text
mask one modality for some particles
predict raw/transformed modality values
```

Support:

```text
HLT self-reconstruction
offline target reconstruction for privileged training
```

Keep it optional and off by default until stable.

### Step 19: Implement Dual-View Cross-Attention Variant

Add:

```text
standard ParT branch
subtoken branch
cross-attention fusion
```

This is the CrossViT-inspired safety net.

Ablate:

```text
late logits average
concat pooled embeddings
cross-attention class-token fusion
```

### Step 20: Run Smoke Tests

Run tiny smoke:

```text
10k train
2k val
10k final test
2 epochs
QCD vs Hgg
HLT strength 0.6
```

Require:

```text
all variants complete
diagnostics written outside checkpoints
gate diagnostics finite
final report written
```

### Step 21: Run First Serious Version A Binary Test

Run:

```text
QCD vs Hgg
HLT strength 0.6
500k train
150k val
500k final test
45 epochs
```

Compare:

```text
HLT ParT
subtoken_no_gate
subtoken_gate_local_only
subtoken_gate_context
```

Primary metric:

```text
FPR at 50% signal efficiency
```

### Step 22: Run Version B Binary Test

Use same splits and HLT cache.

Compare:

```text
subtoken_gate_context HLT-only
subtoken_gate_context + offline distillation
subtoken_gate_context + offline residual
subtoken_gate_context + distillation + residual
```

### Step 23: Scale And Stress

Repeat best Version A/B on:

```text
HLT strength 1.0
QCD vs Tbqq
10-class JetClass subset
larger train counts if available
```

### Step 24: Decide The Paper Story

The idea is promising only if at least one of these is true:

```text
Version A beats parameter-competitive HLT ParT.
Version B beats Version A and HLT ParT.
Gains persist as training data increases.
Gate diagnostics reveal stable, class/task-relevant modality routing.
```

If gains are only from extra parameters or ensembling, the story becomes weaker.
If gains are strongest under HLT degradation or detector shift, the paper story
can become:

```text
context-aware modality reliability for trigger-level particle transformers
```

