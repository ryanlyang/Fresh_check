# Multi-Scale Subjet Particle Transformer Plan

## Purpose

This document defines a serious architecture plan for trying to beat the HLT
Particle Transformer in the current target regime:

- Task: QCD vs Hgg
- Inference view: HLT only
- HLT degradation strength: 0.6
- Primary metric: false-positive rate at 50% signal efficiency, `FPR@50`
- Baseline to beat: HLT ParT trained on the same HLT cache and splits
- Goal: improve the HLT-side classifier itself, especially at high data counts

The core hypothesis:

> HLT ParT is very strong at particle-particle global attention, but it is flat:
> it does not explicitly form intermediate prong/subjet tokens. QCD vs Hgg is a
> substructure problem, so a model with learned multi-scale particle-and-subjet
> tokens may expose structure that a flat particle transformer has to rediscover
> implicitly.

This is not a reconstruction plan. The deployed model uses only HLT particles.
Offline information can be added later as a training-only auxiliary signal, but
the first serious version should be an HLT-only architecture comparison.

## Why This Might Beat HLT ParT

ParT tokenizes jets as:

```text
one HLT particle / constituent = one token
```

That is a powerful choice. It lets every particle interact with every other
particle, and ParT injects physics-aware pairwise information into attention.
The possible missing structure is not "attention." The missing structure is
hierarchy.

For QCD vs Hgg, the decision is often closer to:

```text
Do these particles organize into signal-like local cores or prongs?
How many coherent regions are present?
How compact or diffuse are those regions?
How do the candidate prongs relate to each other?
Are the high-pT particles consistent with local support around them?
```

A flat ParT can learn those questions, but it is not architecturally forced to
ask them. A multi-scale subjet model asks those questions directly by adding
intermediate tokens:

```text
particle tokens:
  detailed constituent evidence

subjet/proto-prong tokens:
  learned local/regional summaries

jet token:
  final class decision
```

The desired win is a lower fixed-model/optimization floor, not a Bayes-limit
violation. With enough data, a sufficiently large and perfectly optimized
HLT-only model can approximate `p(y | HLT)`. Our claim is narrower and more
defensible:

> Given realistic architectures and optimizers, explicit multi-scale subjet
> tokens may extract the remaining HLT information more effectively than a flat
> particle-only ParT.

## Research Inspiration

The idea is inspired by several patterns from models that improved over plain
transformer baselines in other domains:

- Tokens-to-Token ViT: naive patch tokens improved when local tokens were
  progressively aggregated before global attention.
- Transformer-in-Transformer: fine tokens inside coarse tokens improved image
  classification by modeling structure inside a patch before global patch
  reasoning.
- CrossViT: multiple token scales can be complementary when fused by
  cross-attention instead of simple concatenation.
- GraphGPS: local message passing plus global attention can outperform using
  either local or global structure alone.
- Set Transformer / Perceiver: learned latent tokens can summarize sets without
  requiring hard clusters.
- ParticleNet: local geometric neighborhoods are valuable for jet tagging.
- Particle Transformer: pairwise physics bias and global particle attention are
  strong enough that they should be preserved, not casually replaced.

The HEP-specific translation is:

```text
ViT patch tokens        -> jet particle tokens
TNT/T2T local hierarchy -> subjet/proto-prong hierarchy
CrossViT multi-scale    -> particle branch + subjet branch fusion
GraphGPS local/global   -> local subjet formation + global ParT reasoning
```

## Design Principles

### Preserve The Real Baseline

The strongest implementation should not replace HLT ParT with a custom
"transformer-like" approximation. That mistake makes a result hard to interpret.

The model must include a real HLT ParT branch or a branch initialized from the
same canonical ParT input contract:

```text
raw HLT tokens
  -> canonical ParT PF features / lorentz vectors / mask
  -> reference ParticleTransformer backbone
```

The new subjet machinery should be an augmentation, not a weaker replacement.

The primary model should also be able to reproduce the trained HLT ParT baseline
at initialization. The preferred path is a zero-initialized residual adapter:

```text
canonical ParT PF features F
  -> subjet hierarchy computes delta_F
  -> F' = F + gamma * delta_F, with gamma initialized to 0
  -> real warm-started HLT ParT backbone
```

Late fusion can still be useful as a control, but the main high-data bet should
change what the real ParT backbone sees, not merely add a second classifier next
to it.

### Make Subjets Learnable Tokens

The strongest version should not only compute hand-engineered subjet variables.
It should create tokens that can interact with particles and each other.

Bad weak version:

```text
anti-kT subjets -> a few scalar features -> concatenate to classifier
```

Better serious version:

```text
seed-conditioned learned subjet queries
  -> radius-aware soft particle assignment at multiple angular scales
  -> subjet tokens
  -> particle readback into canonical ParT PF features
  -> real HLT ParT backbone
```

Pure learned queries should be kept as an ablation/control. The mainline should
use physically grounded seeds plus learned refinement, so the tokens are more
likely to behave like subjets/proto-prongs rather than generic Perceiver
latents.

### Use Multi-Scale Structure

Hgg vs QCD is not tied to one angular radius. HLT degradation and shower
structure can show up at several scales. The model should form tokens at
multiple effective scales:

```text
small scale:
  tight local cores

medium scale:
  candidate prongs / proto-subjets

large scale:
  broad radiation regions
```

### Keep The First Serious Comparison HLT-Only

The first version should answer the clean question:

```text
Can better tokenization/hierarchy beat HLT ParT using the same HLT inputs?
```

Privileged offline losses can be a later extension. They should not be required
for the first architecture win.

### Optimize For FPR@50

The training and reporting path must select checkpoints using the real binary
metric:

```text
fpr_at_signal_eff_0p50
```

Accuracy can be reported, but it should not choose the best checkpoint.

## Best-Bet Architecture

The best first serious model is:

```text
HLT raw particles
  -> canonical ParT inputs
  -> canonical PF feature tensor F
  -> seeded multi-scale soft subjet token builder
  -> subjet-subjet physics reasoning
  -> particles read back from subjet tokens
  -> zero-initialized residual delta_F
  -> F' = F + delta_F
  -> real warm-started HLT ParT backbone
  -> classifier
```

The important distinction is that the subjet hierarchy is not merely a second
classifier. It changes the particle representation consumed by the real ParT
backbone. If the residual is initialized to zero, the model starts at the HLT
ParT baseline; if the learned subjet hierarchy helps, the real ParT receives a
better HLT-side feature representation and can lower FPR@50.

### Reference HLT ParT Anchor

The anchor is the baseline-preserving path:

```text
raw HLT tokens
  -> build_part_inputs_torch(...)
  -> canonical PF features F, lorentz vectors, mask
  -> ParticleTransformerHLTClassifier / build_hlt_classifier
```

The model should expose diagnostics proving:

```text
uses_reference_part_backbone = true
serious_comparison_ready = true
warm_start_checkpoint = trained HLT ParT checkpoint, when used
residual_adapter_zero_init = true
```

The exact `hlt_part_baseline` variant must route through the same reference
backbone and raw-token-to-ParT-input contract. No custom raw-token transformer
should be labeled as the baseline.

### Residual Subjet Adapter

The primary architecture is a residual adapter over canonical PF features:

```text
F = canonical ParT PF features, shape [B, N, C]
H = particle embedding of F and/or raw HLT geometry
S = multi-scale subjet tokens built from H
H_readback = ParticleSubjetCrossAttention(H, S)
delta_F = MLP([H, H_readback, H_readback - H])
F' = F + gamma_F * delta_F
logits = reference_HLT_ParT(F', lorentz_vectors, mask)
```

`gamma_F` should initialize to zero. This is the cleanest baseline-preserving
setup:

```text
at initialization:
  F' = F
  model == warm-started HLT ParT, up to numerical noise

after training:
  F' can include learned multi-scale subjet corrections
```

The adapter should never modify Lorentz vectors in the first implementation.
Only PF features get a residual correction. Lorentz-vector modification is a
later, riskier ablation because it changes the geometry contract consumed by
ParT.

### Seeded Multi-Scale Subjet Token Builder

The subjet builder forms learned soft subjet tokens from HLT particles.

Input:

```text
canonical PF features F_i
raw geometry: pt, eta, phi, energy
particle mask
```

For each scale `s`, define `M_s` seed-conditioned learned subjet queries:

```text
Q_s = learned scale query + seed embedding + optional local context
```

Example default:

```text
small scale:  8 queries
medium scale: 8 queries
large scale:  4 queries
total:       20 subjet tokens
```

Each query attends over particles:

```text
a_{s,m,i} = softmax_i(score(query_{s,m}, particle_i, geometry_bias_{s,m,i}))
subjet_{s,m} = sum_i a_{s,m,i} * value(particle_i)
```

The geometry bias should make the token builder physically meaningful without
being too rigid.

Possible bias features:

```text
log_pt
pt_fraction
eta
phi
deltaR to learned/seed center
local density estimate
distance to leading particles
pair mass proxy
relative kT proxy
```

The model should support three query modes:

1. Seed-conditioned learned queries:

   Queries are attached to deterministic/semi-deterministic seed centers such as
   leading-pT particles, local-density peaks, and farthest-point eta-phi seeds.
   This is the recommended mainline.

2. Pure learned queries:

   Queries are learned vectors and discover regions by attention.

   This is an important control because it tests whether the "subjet" structure
   matters or whether generic latent tokens are enough.

3. Hard-cluster initialized queries:

   Queries are initialized from anti-kT/exclusive-kT-like cluster centers or
   simple hard seeded regions, then refined softly.

   This is useful as an ablation or auxiliary target, not the first core model.

## Subjet Token Formation Options

### Option 1: Seed-Conditioned Soft Query Pooling

This is the main recommended path.

For each scale:

```text
seed centers + scale embeddings -> queries
particle embeddings             -> keys/values
radius/geometry bias            -> attention logits
softmax over particles          -> assignment weights
weighted sum                    -> subjet token
```

Advantages:

- differentiable
- stable
- easy to batch
- learned but physically anchored
- less likely to collapse into generic latent tokens
- no dependence on external clustering libraries

### Option 2: Pure Learned Query Pooling

Pure learned queries are convenient and differentiable:

```text
learned scale queries
  -> soft attention over particles
  -> latent/subjet tokens
```

They should be implemented, but not as the mainline. If pure learned queries
match the seeded-subjet model, the result is more of a Perceiver/Set Transformer
latent-token story than a physics-subjet story.

### Option 3: Hard Seed Plus Soft Refinement

This version chooses seed centers from HLT particles:

```text
leading-pT seeds
local-density seeds
farthest-point seeds in eta-phi
```

Then each subjet token softly attends to particles around that seed. This gives
more explicit regional meaning.

Advantages:

- more interpretable
- good for diagnostics
- likely easier to learn at large scale

Risk:

- seed choice may be brittle
- high-pT seeds may bias too strongly toward QCD-like hard fragments

### Option 4: Physics Clustering Teacher Token

This version computes anti-kT or exclusive-kT-like assignments outside the model
and uses them as auxiliary targets or initialization.

Recommended only as an optional auxiliary/control, not the first core model.
The learned token branch should remain the main path.

## Multi-Scale Design

The model should explicitly distinguish scale families.

Recommended default:

```text
scale small:
  num_tokens = 8
  radius_bias = 0.05 to 0.12
  purpose = dense local cores, HLT merge/drop artifacts

scale medium:
  num_tokens = 8
  radius_bias = 0.12 to 0.25
  purpose = candidate prongs / proto-subjets

scale large:
  num_tokens = 4
  radius_bias = 0.25 to 0.50
  purpose = broad radiation pattern and jet regions
```

Do not hard-mask by radius at first. Use radius-aware bias:

```text
bias_s(i, center) = -alpha_s * deltaR(i, center)^2
```

with learned or configured `alpha_s`. Hard masks can be an ablation.

Each scale should have:

- scale type embedding
- optional scale radius embedding
- shared or scale-specific query projection
- scale-level diagnostics

## First-Class Physics Summaries

The subjet tokens should carry more than pooled neural embeddings. The model
should compute differentiable soft physics summaries from assignment weights:

```text
subjet px, py, pz, E
subjet pt, eta, phi, mass
subjet pt fraction
effective constituent count
assignment entropy
axis uncertainty / spread
```

For subjet-subjet pairs, compute:

```text
deltaR
pair mass
relative kT
z / energy sharing
pt balance
assignment overlap
cross-scale containment
```

For Hgg specifically, the architecture should be able to represent:

```text
best learned two-prong candidate
opening angle
symmetric energy sharing
between-prong radiation
outside-core radiation
local density around each core
```

These features can enter as attention biases, token side-channel embeddings, or
diagnostics. They should be implemented in a differentiable way using HLT-only
particles and soft assignment weights.

## Particle-Subjet Fusion

The important part is not merely creating subjet tokens. The particles and
subjets need to exchange information.

### Stage 1: Particles To Subjets

Subjet tokens read from particles:

```text
subjet_tokens = CrossAttention(
  queries = learned scale queries,
  keys    = particle tokens,
  values  = particle tokens,
  bias    = geometry/scale bias
)
```

### Stage 2: Subjets To Particles

Particles then read back from subjets:

```text
particle_tokens_enriched = particle_tokens + gamma_ps * CrossAttention(
  queries = particle tokens,
  keys    = subjet tokens,
  values  = subjet tokens,
  bias    = particle-subjet geometry bias
)
```

Initialize `gamma_ps` to zero or a tiny value. This keeps the model close to the
baseline at initialization.

For the main model, this readback should become a residual correction to
canonical ParT PF features:

```text
delta_F = FeatureDeltaMLP([particle_tokens, particle_tokens_enriched])
F' = F + gamma_F * delta_F
```

Initialize `gamma_F = 0`. This makes the integrated model reproduce the
warm-started HLT ParT baseline before training while still allowing the subjet
hierarchy to reshape what ParT sees.

### Stage 3: Subjet-Subjet Reasoning

Subjet tokens should attend to each other:

```text
subjet_tokens = SubjetTransformer(subjet_tokens, subjet_pair_bias)
```

Subjet-pair features can include:

```text
deltaR between subjet centers
soft pair mass
relative kT
pt fraction ratio
overlap / assignment similarity
scale pair type
```

This is where the model can explicitly ask:

```text
Do two learned prongs form a signal-like structure?
```

## Integration With HLT ParT

There are four useful integration modes. Only the first should be the main
high-data bet.

### Mode A: Residual Feature Adapter Into Reference ParT

```text
F = canonical ParT PF features
S = multi-scale subjet tokens
delta_F = particle-subjet readback MLP(F, S)
F' = F + gamma_F * delta_F
logits = real_HLT_ParT(F', lorentz_vectors, mask)
```

Pros:

- strongest scientific story
- preserves the real HLT ParT backbone
- baseline can be exactly recovered when `gamma_F = 0`
- gives ParT access to explicit multi-scale prong/subjet structure

Cons:

- requires careful implementation of the ParT input contract
- harder than a standalone branch

This should be the primary implementation.

### Mode B: Mid-Layer Particle-Subjet Integration

```text
insert particle-subjet cross-attention between ParT blocks
```

Pros:

- potentially very powerful
- lets subjets interact with deeper ParT representations

Cons:

- requires modifying or wrapping Weaver ParT internals
- higher implementation risk

This is a later upgrade if the residual feature adapter works or nearly works.

### Mode C: CLS / Embedding Fusion

```text
z_part   = ParT pooled embedding
z_subjet = subjet branch pooled embedding
z_fused  = MLP([z_part, z_subjet, z_part * z_subjet, z_part - z_subjet])
logits   = classifier(z_fused)
```

Pros:

- useful control
- preserves an exact ParT branch
- easier to debug than mid-layer integration

Cons:

- can behave like a larger ensemble
- weaker claim if it wins

This should be implemented as a comparison, not the main paper-worthy model.

### Mode D: Late Logit Fusion

```text
logits = logits_part + beta * logits_subjet
```

Pros:

- safest control
- easy to compare
- baseline can be exactly recovered when `beta = 0`

Cons:

- weak interaction between branches
- most ensemble-like

This should be included as a control.

## Recommended First Serious Variant

The strongest first full run should be:

```text
variant: multiscale_subjet_residual_part_adapter

Reference ParT backbone:
  real HLT ParT, same model size as baseline
  warm-start from trained HLT ParT checkpoint
  no custom transformer substitute

Subjet adapter:
  20 learned subjet tokens total
  scales: 8 small, 8 medium, 4 large
  seeded learned queries using leading-pT/local-density seeds
  2 particle-to-subjet cross-attention layers
  2 subjet-subjet transformer layers
  particle readback into canonical PF features
  zero-initialized gamma_F residual

ParT input:
  F' = F + gamma_F * delta_F
  lorentz vectors unchanged in v1
  mask unchanged

Controls:
  subjet-only
  pure learned latent tokens
  CLS/embedding fusion
  late logit fusion
  random/permuted subjet assignment

Training:
  HLT-only labels
  checkpoint selection by FPR@50
  final-test confirmation required
```

## Diagnostics

This architecture can look impressive while learning nonsense unless diagnostics
are built in from the start.

Required diagnostics:

### Assignment Diagnostics

For each scale:

```text
mean assignment entropy
max assignment weight
effective particles per subjet
fraction of dead subjet tokens
fraction of duplicate/overlapping subjet tokens
mean subjet pt fraction
subjet center eta/phi spread
```

### Scale Usage Diagnostics

```text
small/medium/large token norm
small/medium/large attention mass
scale contribution to residual delta_F or control-variant logits
scale dropout sensitivity
```

### Particle-Subjet Interaction Diagnostics

```text
particle-to-subjet attention entropy
subjet-to-particle attention entropy
top particle contributors per subjet
subjet overlap matrix
```

### Comparison Diagnostics

For each variant:

```text
FPR@30
FPR@50
AUC
accuracy
background rejection at 30/50
runtime
parameter count
peak memory if available
```

Also include HLT degradation slices if cache diagnostics are available:

```text
low/high drop_total
low/high drop_merge
low/high merge_count
low/high drop_eff
```

## Important Controls

A win is only meaningful if the controls rule out trivial explanations.

Required controls:

1. `hlt_part_baseline`

   Exact HLT ParT trained/evaluated on the same binary HLT cache.

2. `subjet_branch_only`

   Subjet branch without ParT branch. This tests whether subjets alone are
   strong or merely complementary.

3. `part_plus_random_subjet_tokens`

   Same parameter count, but subjet assignments are random/permuted. This tests
   whether learned multi-scale structure matters.

4. `part_plus_single_scale_subjets`

   Remove multi-scale hierarchy. This tests whether several scales matter.

5. `part_plus_hard_leading_pt_subjets`

   Use simple deterministic leading-pT seeded soft pools. This tests whether
   learned queries are necessary.

6. `late_logit_fusion`

   Safe fusion baseline.

7. `cls_embedding_fusion`

   Stronger branch-fusion control. Useful, but less central than residual
   readback into ParT.

8. `residual_adapter_no_physics_bias`

   Same residual adapter but remove pair mass, relative kT, z, density, and
   scale-radius bias. This tests whether physics bias matters.

9. `pure_perceiver_latent_control`

   Replace seeded subjet tokens with generic learned latent tokens. This tests
   whether the gain is really from subjet/prong structure or just extra latent
   capacity.

10. `larger_hlt_part_control`

   A parameter-matched or compute-matched larger HLT ParT, if feasible. This is
   the cleanest answer to "did we just add parameters?"

11. `two_part_ensemble_control`

   Two independently trained HLT ParT models fused at logits/probabilities. This
   is required if late or CLS fusion variants win.

12. `cross_attention_fusion`

   Strongest fusion candidate.

13. `subjet_token_shuffle_control`

   Shuffle subjet tokens across jets or labels where appropriate. This catches
   leakage-like or bookkeeping mistakes.

14. `token_count_sweep`

   Run 8, 12, 20, and 32 total subjet tokens. This checks whether the default
   token count is under- or over-parameterized.

15. `seed_strategy_sweep`

   Compare leading-pT, local-density, farthest-point, mixed-seed, and pure
   learned queries.

## Split Discipline

Use the same split discipline as the current QCD/Hgg HLT 0.6 runs:

```text
model_train: 500k
model_val:   150k
stack_train: optional, only for later fusion/stacking
stack_val:   150k
final_test:  500k
```

For direct end-to-end models:

```text
train on model_train
select checkpoint on model_val using FPR@50
evaluate stack_val and final_test only after checkpoint selection
```

For any late fusion/stacking:

```text
train base models on model_train
select base checkpoints on model_val
train fusion calibrator on stack_train
select fusion on stack_val
touch final_test once
```

## Training Strategy

### Stage 1: Baseline Reproduction

Train or reuse the exact HLT ParT baseline for the same cache.

The report must show:

```text
cache HLT degradation strength = 0.6
same split sizes
same label names: QCD Hgg
same final_test size
checkpoint selected by FPR@50
```

### Stage 2: Exact-Baseline Warm Start

For ParT-anchored variants:

```text
load HLT ParT checkpoint
initialize residual fusion gates to zero
fine-tune all parameters
```

Avoid freezing the ParT branch by default. Freezing can be useful later, but the
first serious run should not risk blocking gradients or trapping the new branch
behind an inactive residual.

### Stage 3: Subjet Branch Warm-Up Optional

If the subjet branch is unstable, optionally warm it up with:

```text
small learning rate on ParT branch
larger learning rate on subjet branch
residual gates initialized tiny, e.g. 0.01
```

Do not train a large number of epochs with ParT fully frozen unless diagnostics
show the branch is actually affecting logits.

### Stage 4: Full Fine-Tuning

Train end-to-end:

```text
optimizer: AdamW
metric: FPR@50
epochs: 45 initially
batch size: as large as stable
early stop patience: 6 to 8
```

Use parameter groups:

```text
ParT branch lr:      1x
subjet branch lr:    1x to 2x
fusion/gates lr:     2x
weight decay:        same as current HLT ParT unless evidence says otherwise
```

## Losses

The first serious model should use classification loss only:

```text
cross entropy on QCD vs Hgg labels
```

Then optional regularizers can be added carefully:

### Entropy Regularization

Avoid both collapse and total diffusion:

```text
assignment entropy should not be near 0 for every token
assignment entropy should not be uniform for every token
```

Use diagnostics first. Add a weak loss only if needed.

### Diversity Regularization

Encourage different subjet tokens to attend to different regions:

```text
penalize excessive assignment overlap within a scale
```

Keep this weak. Hgg may legitimately have overlapping evidence.

### Scale Dropout

Randomly drop one scale family during training:

```text
drop small/medium/large token family with small probability
```

This tests and encourages complementary scale usage.

Do not enable this in the first debugging run. Add after the main path is stable.

## Suggested Module Layout

Create a new package:

```text
teacher_logit_reco/multiscale_subjet_part/
```

Suggested files:

```text
protocol.py
  QCD/Hgg HLT0.6 protocol constants
  required variants
  metric policy

features.py
  canonical particle feature extraction helpers
  geometry helpers
  subjet summary helpers

seeds.py
  leading-pT seeds
  density seeds
  farthest-point eta-phi seeds

assignment.py
  soft particle-to-subjet assignment
  multi-scale query pooling
  assignment diagnostics

subjet_tokens.py
  MultiScaleSubjetTokenBuilder
  scale embeddings
  center/radius estimates

cross_attention.py
  particle-subjet cross-attention
  subjet-subjet transformer
  geometry bias modules

model.py
  HLT ParT baseline wrapper
  MultiScaleSubjetResidualPartAdapter
  control variants for subjet-only and fusion

train.py
  shared training loop
  checkpoint selection by FPR@50
  cache protocol verification

reports.py
  final report builder
  metric tables
  diagnostics tables
```

Scripts:

```text
scripts/train_multiscale_subjet_part_tagger.py
scripts/write_multiscale_subjet_part_report.py
```

Slurm:

```text
sbatch/run_train_multiscale_subjet_part_tagger.sh
sbatch/run_write_multiscale_subjet_part_report.sh
sbatch/submit_multiscale_subjet_qcd_hgg_step1.sh
```

## Candidate Variants

The first submission graph should include:

```text
hlt_part_baseline
subjet_branch_only
multiscale_subjet_residual_part_adapter
part_plus_single_scale_subjets
part_plus_multiscale_subjets_cls_fusion_control
part_plus_multiscale_subjets_late_fusion_control
pure_perceiver_latent_control
part_plus_random_subjet_control
```

If resources are tight, start with:

```text
hlt_part_baseline
multiscale_subjet_residual_part_adapter
pure_perceiver_latent_control
part_plus_random_subjet_control
```

If the residual adapter beats baseline, the next submission should add:

```text
larger_hlt_part_control
two_part_ensemble_control
residual_adapter_no_physics_bias
seed_strategy_sweep
```

## What Would Count As A Real Win

A meaningful win should satisfy all of:

```text
same HLT cache, degradation 0.6
same QCD/Hgg splits
same final_test
checkpoint selected by model_val FPR@50
final_test FPR@50 lower than HLT ParT
improvement larger than run-to-run noise
improvement survives repeated seeds or bootstrap uncertainty
improvement is not matched by a larger/deeper HLT ParT control
improvement is not matched by a two-ParT ensemble control when fusion is used
random/ablated subjet controls do not match the gain
diagnostics show non-collapsed, physically plausible subjet usage
```

Small wins matter in this regime. If HLT ParT FPR@50 is around `0.020`, then a
move to `0.0195` may be meaningful if it is stable and supported by controls.

The report should include two binary views of the final-test result:

```text
oracle-style final_test FPR@50:
  compute threshold from final_test scores for comparison with existing runs

validation-threshold final_test result:
  choose the threshold on model_val at 50% signal efficiency
  apply that threshold once to final_test
  report final_test FPR and signal efficiency
```

The validation-threshold number is less flattering but closer to a deployable
operating point.

## Main Risks

### Risk 1: The Residual Adapter Never Turns On

If `gamma_F` initializes to zero and never grows, the model will stay near ParT.

Mitigation:

- log `gamma_F` and feature-delta norms
- use slightly higher LR for residual/gate parameters
- optionally initialize `gamma_F` to a tiny nonzero value after debugging
- test whether the loss has a gradient path through the residual adapter

### Risk 2: Subjet Tokens Collapse

All learned tokens may attend to the same high-pT particles.

Mitigation:

- assignment overlap diagnostics
- optional diversity regularizer
- seed-based initialization
- scale-specific bias

### Risk 3: Model Is Stronger Only By Parameter Count

The new model may win simply because it is larger.

Mitigation:

- include random-subjet same-parameter control
- include pure Perceiver latent same-parameter control
- report parameter counts
- compare to a slightly larger or deeper HLT ParT if feasible
- compare to a two-HLT-ParT ensemble if any fusion branch wins

### Risk 4: HLT ParT Already Learns This

The model may tie ParT because ParT's pairwise attention is already enough.

Mitigation:

- inspect degradation slices
- inspect Hgg/QCD failure cases
- test whether subjet branch helps only on high-degradation jets

### Risk 5: Fusion Or Stacking Creates Leakage-Like Mistakes

Complex fusion and cached predictions can cause accidental split issues.

Mitigation:

- strict split discipline
- final-test confirmation flag
- run reports listing exactly which split trained each component
- prefer the residual adapter mainline before late learned fusion

## Implementation Steps

### Step 1: Protocol And Package Skeleton

Create `teacher_logit_reco/multiscale_subjet_part/`.

Define:

- QCD/Hgg label policy
- HLT degradation strength `0.6`
- primary metric `fpr_at_signal_eff_0p50`
- required split names
- variant names
- report contract

Add tests that the protocol is frozen to the intended regime.

### Step 2: Canonical Feature And Geometry Helpers

Implement helpers for:

- raw HLT token validation
- canonical ParT `PF_FEATURE_NAMES`/points/vectors/mask construction from the same raw HLT tokens
- eta/phi wrapping
- pt fraction
- local density
- pairwise deltaR
- scale radius metadata

Ensure all helpers respect masks, raw token dim `14`, and the exact ParT
preprocessing contract used by the HLT baseline.

### Step 3: Seed Builder

Implement seed selection:

- leading-pT seeds
- local-density seeds
- farthest-point eta-phi seeds

Return seed centers, seed masks, and diagnostics. Tests should cover empty jets,
padded particles, phi wrapping, and deterministic behavior.

### Step 4: Soft Subjet Assignment

Implement differentiable particle-to-subjet assignment:

```text
queries + particle keys + geometry bias -> assignment weights
```

Support:

- pure learned queries
- seed-conditioned queries
- scale-specific radius bias
- assignment masks

Diagnostics must include entropy, max weight, effective particle count, and
dead-token fraction.

The default path should be seed-conditioned. Pure learned queries must be
implemented as a control, not the default serious variant.

### Step 5: Multi-Scale Subjet Token Builder

Build `MultiScaleSubjetTokenBuilder`.

It should create small/medium/large subjet tokens with scale embeddings and
return:

- subjet tokens
- subjet mask
- assignment weights
- estimated centers
- estimated pt fractions
- soft four-vector summaries
- soft pair-observable summaries
- diagnostics

### Step 6: Subjet-Subjet Transformer

Implement a small transformer over subjet tokens with optional pairwise subjet
bias.

Bias features:

- deltaR between centers
- pair mass
- relative kT
- pt ratio / z
- pt balance
- assignment overlap
- cross-scale containment
- scale-pair type embedding

### Step 7: Particle-Subjet Cross-Attention

Implement particle-subjet cross-attention and residual readback:

- particles read from subjets
- subjets read from particles
- particle readback becomes `delta_F`
- `F' = F + gamma_F * delta_F`

Support zero/tiny residual gates and diagnostics for attention entropy, feature
delta norms, and gate values. The default serious path should initialize
`gamma_F = 0` and leave Lorentz vectors unchanged.

### Step 8: Baseline-Faithful ParT Wrapper

Wrap the real HLT ParT baseline exactly as in existing HLT training.

The wrapper must make clear:

```text
uses_reference_part_backbone = true
baseline_variant = hlt_part_baseline
```

Do not use a custom raw-token transformer as the baseline.

### Step 9: Multi-Scale Subjet Classifier

Implement the main model:

- `hlt_part_baseline`
- `multiscale_subjet_residual_part_adapter`
- `subjet_branch_only`
- `pure_perceiver_latent_control`
- `part_plus_random_subjet_control`

The default serious variant is `multiscale_subjet_residual_part_adapter`. CLS
embedding fusion, late logit fusion, and cross-attention branch fusion should be
implemented later as controls/ablations, not as the mainline.

### Step 10: Training Loop

Implement shared training for all variants:

- train on `model_train`
- select on `model_val`
- select by `fpr_at_signal_eff_0p50`
- evaluate `stack_val` and `final_test` only after selection
- verify HLT cache degradation params equal `0.6`
- save diagnostics separately from large checkpoints

### Step 11: Report Builder

Build a report that compares:

- baseline
- each subjet variant
- controls

Include:

- metric table
- parameter counts
- runtime
- assignment diagnostics
- scale diagnostics
- residual feature-delta diagnostics
- fusion diagnostics for control variants
- validation-threshold final-test FPR and signal efficiency
- HLT degradation slice metrics if available

### Step 12: Slurm Runner

Create a submitter for the QCD/Hgg HLT0.6 experiment.

Defaults:

```text
model_train: 500k
model_val:   150k
stack_val:   150k
final_test:  500k
epochs:      45
metric:      FPR@50
```

Queue:

- baseline
- main residual ParT adapter
- pure Perceiver/latent control
- random-subjet control
- final report

### Step 13: First Serious Run

Run the first serious comparison.

Expected first question:

```text
Does multi-scale subjet hierarchy beat HLT ParT at FPR@50?
```

If no:

- check whether assignment collapsed
- check whether `gamma_F` and feature deltas stayed near zero
- check whether controls match main variants
- inspect HLT degradation slices

If yes:

- repeat with seeds
- run larger ParT and two-ParT ensemble controls
- run no-physics-bias and seed-strategy controls
- test larger split or repeated seed

### Step 14: Ablations

Run ablations:

- no scale bias
- one scale only
- no seeded queries
- no subjet-subjet transformer
- no particle readback
- residual adapter vs late fusion vs CLS fusion vs cross-attention branch fusion
- fewer/more subjet tokens
- seeded queries vs pure learned queries
- physics bias removed
- larger HLT ParT control
- two-HLT-ParT ensemble control

### Step 15: Optional Privileged Training Extension

Only after the HLT-only architecture is stable, add optional offline training
signals:

- offline teacher logits
- HLT/offline residual reliability targets
- auxiliary prong/subjet consistency targets

The final deployed model must remain HLT-only.

### Step 16: Write The V2 Local-Graph Plus Subjet Plan

After the V1 multi-scale subjet residual adapter has been implemented and at
least one serious QCD/Hgg HLT0.6 run has produced diagnostics, write a separate
V2 plan for a hybrid architecture:

```text
raw HLT particles
  -> canonical ParT PF features F
  -> local graph context G
  -> seed-conditioned multi-scale subjet tokens using F + G
  -> particle readback
  -> residual delta_F
  -> real HLT ParT backbone
```

The purpose of V2 is to make the subjet tokens smarter by giving each particle
local eta-phi neighborhood context before subjet assignment. The local graph
stage should not be a late-fused extra classifier. It should improve:

- particle embeddings used for subjet assignment
- seed selection, especially local-density and farthest-point seeds
- local reliability / HLT-artifact bias terms
- assignment logits and scale-specific radius biases
- diagnostics for high-degradation HLT slices

Use these files as starting points:

- `teacher_logit_reco/LOCAL_GRAPH_PARTICLE_TRANSFORMER_PLAN.md`
- `teacher_logit_reco/local_graph_part/knn.py`
- `teacher_logit_reco/local_graph_part/local_blocks.py`
- `teacher_logit_reco/local_graph_part/model.py`
- `teacher_logit_reco/MULTISCALE_SUBJET_PARTICLE_TRANSFORMER_PLAN.md`

The V2 plan should answer:

1. Should the local graph module run before seed selection, before assignment,
   or both?
2. Should EdgeConv, local point attention, or both be used?
3. Should local graph outputs affect only subjet assignment, or also the final
   residual `delta_F`?
4. What controls prove that local graph plus subjets is more than just extra
   parameters?
5. Does the hybrid help specifically on high HLT degradation slices?

Do not implement V2 until V1 has enough results to show whether assignment
collapse, weak residual gates, or missing local context is the main failure
mode.
