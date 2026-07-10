# Canonical Multi-Scale Jet State Residual Predictor Plan

## Core Idea

The goal is to improve an HLT-only particle tagger without asking it to
hallucinate offline particles and without forcing arbitrary neural hidden
spaces to align.

The central object is a deterministic, shared, residual-friendly jet
representation:

```text
Phi(HLT particles)     = canonical HLT jet-state tokens
Phi(offline particles) = canonical offline jet-state tokens
```

Because the same `Phi` function is applied to HLT and offline particles, the
residual target is meaningful:

```text
delta_Phi_true = Phi(offline) - Phi(HLT)
```

The model learns:

```text
delta_Phi_hat = residual_predictor(HLT particles, Phi(HLT))
Phi_pred      = Phi(HLT) + delta_Phi_hat
```

The final tagger sees:

```text
original HLT particle tokens
+ canonical HLT jet-state tokens
+ predicted residual/corrected jet-state tokens
```

It should not replace the HLT particles. The particle sequence remains the
main source of tagging power. The canonical jet-state tokens provide a
structured, count-invariant map of likely offline-level corrections.

## Why This Is Different From Previous Reconstructor Attempts

Previous reconstruction-style ideas often tried to solve one of these hard
problems:

```text
HLT particles -> offline particles
HLT per-particle embeddings -> offline per-particle embeddings
HLT ParT hidden vector -> offline ParT hidden vector
```

Those are difficult because:

```text
HLT and offline may have different particle counts.
Particle correspondence is ambiguous.
Raw neural representations can be arbitrary coordinate systems.
Exact offline-particle reconstruction may be harder than the tagging problem.
```

This plan avoids those traps.

Instead of reconstructing individual particles, it predicts corrections in a
canonical jet-state basis:

```text
global moments
+ radial/angular profiles
+ soft anchor/subjet slots
+ PID/channel moments
```

This representation is fixed-size, physically interpretable, and shared
between HLT and offline by construction.

## What The Canonical State Should Capture

The canonical state should be rich enough to describe meaningful offline-HLT
differences but coarse enough that residual prediction is learnable.

It should encode things like:

```text
overall jet pT/mass/width shifts
charged/neutral/photon fraction shifts
multiplicity and softness changes
radial energy profile changes
angular/subjet energy redistribution
local subjet mass/width corrections
PID-channel-specific missing energy or over-smoothing
```

It should not try to encode:

```text
exact offline particle identities
exact one-to-one HLT/offline particle matching
ultra-fine sparse image bins as the primary target
arbitrary neural hidden-vector coordinates
```

## Canonical Multi-Scale Jet State Tokens

Define `Phi(jet)` as a set of typed state tokens:

```text
Phi(jet) = [
  global tokens,
  radial profile tokens,
  angular sector tokens,
  soft anchor/subjet slot tokens,
  optional PID-channel summary tokens
]
```

Each token has:

```text
token_type
scale_id
slot_id / ring_id / sector_id
fixed feature vector
valid mask
```

The token identity is fixed by construction, so the tagger can learn stable
semantics:

```text
"global jet width token"
"radial ring 2 photon-energy token"
"medium-scale anchor slot 7 neutral-fraction token"
```

These semantics are implemented numerically through token-type embeddings,
scale embeddings, slot/ring/sector embeddings, and fixed feature ordering.

## Recommended Phi Layout

Start with a compact but expressive state:

```text
K_global = 4 to 8 tokens
K_radial = 6 to 8 radial rings
K_angular = 8 angular sectors
K_anchor_coarse = 4 soft anchor slots
K_anchor_medium = 8 soft anchor slots
K_anchor_fine = 16 soft anchor slots
```

Total tokens:

```text
K_state roughly 32 to 48 tokens
```

This is enough to provide spatial and subjet resolution without becoming a
particle-reconstruction problem.

## State Token Feature Fields

Each state token should store a fixed set of normalized moments.

Recommended base fields:

```text
sum_pt_frac
sum_energy_frac
log1p_count
mean_pt_frac
max_pt_frac
pt_weighted_mean_deta
pt_weighted_mean_dphi
pt_weighted_var_deta
pt_weighted_var_dphi
mass_proxy
width_proxy
charged_pt_frac
neutral_pt_frac
photon_pt_frac
electron_pt_frac
muon_pt_frac
hadron_pt_frac
quality_or_missingness_proxy
```

For tokens where a field is not meaningful, fill with zero and expose a
feature-valid mask if needed.

Important: all fields should be normalized into stable ranges. Examples:

```text
fractions in [0, 1]
counts as log1p(count) / scale
angular means normalized by jet radius or clipping window
variance fields log-scaled or clipped
mass/width proxies scaled by jet pT or dataset robust statistics
```

## Global Tokens

Global tokens summarize whole-jet structure.

Possible global token fields:

```text
jet_pt_proxy
jet_mass_proxy
total_constituent_count
charged_fraction
neutral_fraction
photon_fraction
electron_fraction
muon_fraction
jet_width
ptd
soft_fraction
hard_core_fraction
leading_particle_pt_frac
subleading_particle_pt_frac
```

Use multiple global tokens if one token would become too dense. For example:

```text
global_energy_shape
global_pid_fractions
global_multiplicity_softness
global_leading_structure
```

## Radial Tokens

Radial tokens describe how energy and particle content are distributed away
from the jet axis.

Use fixed annuli in local coordinates:

```text
r = sqrt(deta^2 + dphi^2)
```

Example bins:

```text
[0.00, 0.03)
[0.03, 0.06)
[0.06, 0.10)
[0.10, 0.15)
[0.15, 0.22)
[0.22, 0.30)
[0.30, 0.40)
[0.40, inf)
```

Each radial token stores energy, count, width, and PID moments for that ring.

This directly supports residuals like:

```text
"offline has more neutral energy at moderate radius"
"HLT loses soft charged multiplicity in the outer ring"
"HLT core is too narrow"
```

## Angular Sector Tokens

Angular sector tokens describe azimuthal asymmetry and multi-prong structure.

Use local polar angle:

```text
theta = atan2(dphi, deta)
```

Example:

```text
8 sectors around the jet axis
```

Each sector token stores similar moments:

```text
sum_pt_frac
count
mean radius
radial variance
PID fractions
leading pt fraction
```

These tokens help represent lopsided missing energy and coarse prong
orientation without depending on individual particle identities.

## Soft Anchor/Subjet Slot Tokens

Anchor slots are the most important part of the plan.

They give the model a medium-resolution representation of local subjet
structure without requiring one-to-one particle reconstruction.

There are two options:

### Option A: Deterministic Anchor Slots

Use fixed anchor locations in local coordinates and soft-assign particles to
anchors.

Example anchors:

```text
coarse: 4 anchors around main quadrants
medium: 8 anchors around eta/phi circle plus center/core anchors
fine: 16 anchors covering a small local eta/phi grid or polar grid
```

Particle-to-anchor weight:

```text
w_ik = softmax_k(-distance(p_i, anchor_k)^2 / temperature_scale)
```

Then slot moments are weighted sums:

```text
slot_k.sum_pt = sum_i w_ik * pt_i
slot_k.count  = sum_i w_ik
...
```

### Option B: Deterministic Soft Subjet Seeds

Use a few deterministic seeds such as leading particles or simple soft-kmeans
centers. Then compute slot moments around those seeds.

This may better track actual prongs but introduces more moving parts.

### First Implementation Choice

Start with Option A, deterministic anchor slots.

It is simpler, stable, reproducible, and easy to compare across HLT/offline.
Later, add soft-subjet seeds if deterministic anchor slots underperform.

## Why Residuals In Phi Space Make Sense

Because each token and each feature has a stable meaning:

```text
Phi_off[k, d] - Phi_hlt[k, d]
```

has a real interpretation:

```text
slot 7 neutral_pt_frac increased
ring 3 count increased
global mass proxy increased
sector 5 width changed
```

That is exactly what makes the residual prediction easier than predicting
offline particles.

## Residual Predictor

The residual predictor is the heart of this experiment.

It should not be a plain MLP over `Phi_hlt`, because the corrections should be
informed by the actual HLT particle cloud. It also should not be a full second
ParT clone, because the output structure is fixed: a correction for each
canonical state token and field.

The best first implementation is:

```text
Geometry-Biased Canonical State Residual Decoder
```

The asymmetry is important:

```text
HLT particles are evidence.
Canonical state tokens are the things being corrected.
```

So the predictor should make state tokens query the particle cloud.

### Predictor Inputs

The predictor consumes:

```text
HLT particle features
Phi_hlt state tokens
optional HLT ParT early/pooled context
```

More explicitly:

```text
particle_features:
  shape [batch, N_particles, D_particle]
  usually the same canonical 19 JetClass particle fields used by ParT

particle_mask:
  shape [batch, N_particles]

phi_hlt:
  shape [batch, K_state, D_phi]

state_token_metadata:
  token_type_ids
  scale_ids
  slot/ring/sector ids
  anchor positions or radial/angular centers
  state token mask
  field names / field scales
```

Optional later inputs:

```text
baseline HLT ParT logits
baseline HLT ParT pooled representation
HLT degradation/profile metadata
```

Do not include those optional inputs in the first mainline unless needed. They
can improve performance but also create shortcut risk.

### Predictor Output

The predictor outputs:

```text
delta_Phi_hat:
  shape [batch, K_state, D_phi]

Phi_pred:
  Phi_hlt + delta_Phi_hat

diagnostics:
  delta norms
  token-family residual norms
  attention summaries
  optional uncertainty
```

The output is a residual, not a full state reconstruction:

```text
Phi_pred = Phi_hlt + delta_Phi_hat
```

This keeps the model anchored to what HLT actually observed.

### Particle Memory Encoder

First encode HLT particles into a lightweight memory bank:

```text
particle_features
  -> Linear/Dense projection to d_model
  -> particle type / rank / optional feature embeddings
  -> 1 to 2 lightweight encoder blocks
  -> particle_memory [batch, N_particles, d_model]
```

Recommended first settings:

```text
d_model = 128
particle encoder layers = 1 or 2
heads = 4
dropout = 0.05
```

This encoder should be smaller than ParT. Its role is not to replace ParT; it
is only providing evidence for canonical state corrections.

### State Query Embedding

Embed each canonical state token as a decoder query:

```text
state_query_k =
  Linear(phi_hlt_k)
  + token_type_embedding[type_k]
  + scale_embedding[scale_k]
  + slot_position_embedding[slot_k]
  + state_family_embedding["Phi_hlt"]
```

For anchor/ring/sector tokens, include explicit geometry features:

```text
anchor_deta
anchor_dphi
anchor_radius
ring_center
ring_width
sector_center
sector_width
```

These geometry features are not only metadata; they should feed the attention
bias and/or a small geometry MLP.

### Decoder Block

Use a repeated decoder block:

```text
state self-attention
  lets global/radial/anchor tokens share information

geometry-biased cross-attention
  state queries attend to HLT particle memory

feed-forward block
  updates each state token representation
```

Recommended first settings:

```text
decoder layers = 3
d_model = 128
heads = 4
mlp ratio = 2 or 4
dropout = 0.05
attention dropout = 0.05
norm_first = true
```

The order can be:

```text
state self-attention
cross-attention to particles
FFN
```

or:

```text
cross-attention to particles
state self-attention
FFN
```

The first implementation should use:

```text
self-attention -> cross-attention -> FFN
```

because it lets state tokens establish global context before querying
particles.

### Geometry-Biased Cross-Attention

This is the main inductive bias.

For state token `k` and particle `i`, cross-attention logits should include:

```text
attention_logit(k, i) =
  q_k dot k_i / sqrt(d_head)
  + geometry_bias(k, i)
  + optional pid_bias(k, i)
```

The geometry bias depends on token type.

For anchor tokens:

```text
geometry_bias(k, i) =
  - ||position_i - anchor_position_k||^2 / sigma_k^2
```

For radial tokens:

```text
geometry_bias(k, i) =
  - (radius_i - ring_center_k)^2 / sigma_r_k^2
```

For angular sector tokens:

```text
geometry_bias(k, i) =
  - angular_distance(theta_i, sector_center_k)^2 / sigma_theta_k^2
```

For global tokens:

```text
geometry_bias(k, i) = 0
```

or a weak learned broad bias.

This should be a soft bias, not a hard mask. The model should be encouraged to
look locally, but it must be able to override locality when useful.

Recommended implementation:

```text
bias_scale is learnable per token family or per head
geometry bias is clipped to a reasonable range, e.g. [-8, 0]
invalid particles receive -inf attention mask
```

Why this matters:

```text
radial tokens naturally attend to particles in their ring
anchor tokens naturally attend to nearby particles/subjets
global tokens can attend everywhere
```

This lets the predictor learn corrections like:

```text
"this medium anchor slot is underpopulated"
"this radial ring is missing neutral energy"
"this sector looks smeared relative to expected offline structure"
```

without forcing exact particle reconstruction.

### Residual Output Head

After decoder blocks:

```text
state_hidden
  -> LayerNorm
  -> Linear d_model -> hidden
  -> GELU/SwiGLU
  -> Linear hidden -> D_phi
  -> feature-wise bounded residual
```

The final projection must be zero-initialized:

```text
raw_delta starts at zero
Phi_pred starts equal to Phi_hlt
```

Use feature-wise scales:

```text
delta_Phi_hat[..., d] =
  residual_scale[d] * tanh(raw_delta[..., d])
```

Feature-wise scales should come from the state config. Examples:

```text
fractions: scale 0.25 to 0.5
log-count fields: scale 0.5 to 1.0
angular mean fields: scale based on jet radius
variance fields: smaller scale
```

This prevents impossible giant corrections early in training.

### Optional Uncertainty Head

A later stronger version can predict uncertainty:

```text
log_sigma_hat [batch, K_state, D_phi]
```

Then the state loss can downweight ambiguous residuals:

```text
L = Huber(error) * exp(-log_sigma) + log_sigma
```

Do not make uncertainty mandatory in the first implementation. It is useful
but adds interpretation risk:

```text
the model may learn to inflate uncertainty instead of correcting hard fields
```

If added, clamp `log_sigma` and report uncertainty by token family and class.

### Mainline Predictor Summary

The first serious residual predictor should be:

```text
particle memory encoder:
  1-2 light transformer/PointNet-style blocks

state query encoder:
  phi_hlt values + type/scale/slot/geometry embeddings

state decoder:
  3 blocks
  state self-attention
  geometry-biased cross-attention to particles
  FFN

output:
  zero-init bounded delta_Phi_hat
  Phi_pred = Phi_hlt + delta_Phi_hat
```

This is the best first architecture because it is:

```text
structured around the residual target
particle-count invariant on the output side
able to use full particle evidence
biased toward physically meaningful locality
small enough not to become just another ParT
```

## Residual Predictor Targets

Train the residual predictor on:

```text
delta_Phi_true = Phi_off - Phi_hlt
```

Use a masked robust regression loss:

```text
L_state =
  weighted Huber(delta_Phi_hat, delta_Phi_true)
  + optional relative-error terms for positive quantities
  + optional sign/ordering constraints for fractions
```

Recommended first target loss:

```text
Huber on normalized fields
+ L1 on high-value energy/PID fields
+ small smoothness penalty across adjacent radial/anchor tokens
```

Weight fields by importance:

```text
energy/momentum fields: high
PID fractions: high
count/softness fields: medium
angular mean/variance: medium
diagnostic-only fields: low
```

## Corrected State

Compute:

```text
Phi_pred = Phi_hlt + delta_Phi_hat
```

Then sanitize or constrain:

```text
fractions clipped or renormalized where appropriate
counts non-negative
energy fractions non-negative
optional total-energy consistency projection
```

The classifier may consume:

```text
Phi_hlt
delta_Phi_hat
Phi_pred
```

Do not only feed `Phi_pred`; the difference itself can be highly informative.

## Tagging-Side Architecture

The final tagger must keep original HLT particles as the main input.

Recommended model:

```text
HLT particles -> normal ParT particle embedding

Phi_hlt / delta_Phi_hat / Phi_pred
  -> state token encoder
  -> state context tokens

particle embeddings attend to state context
  -> state-conditioned particle embeddings
  -> ParT blocks / classifier
```

There are two implementation options.

### Option 1: Cross-Attention Adapter

Use a small cross-attention adapter before or between ParT blocks:

```text
queries: particle embeddings
keys/values: encoded state tokens
output: delta_h_i
h_i' = h_i + gate * delta_h_i
```

Advantages:

```text
keeps ParT particle pathway intact
lets each particle ask for relevant global/state corrections
easy to zero-init and ablate
```

This is the recommended first implementation.

### Option 2: Prepend State Tokens

Prepend encoded state tokens to the particle sequence and let attention mix:

```text
[state tokens, particle tokens] -> transformer
```

Advantages:

```text
simple conceptual implementation
full mixing between state and particles
```

Disadvantages:

```text
changes ParT sequence semantics more aggressively
may require pairwise/attention mask care
can disrupt warm-start compatibility
```

Use this as a later ablation, not the first mainline.

## How The Tagger Knows What The Residuals Mean

The state tokens must not be anonymous.

Each token embedding should include:

```text
state family embedding:
  Phi_hlt vs delta_Phi_hat vs Phi_pred

token type embedding:
  global vs radial vs angular vs anchor

scale embedding:
  global/coarse/medium/fine

slot position embedding:
  ring index, sector index, anchor index

field-value projection:
  normalized physical moments
```

Then the tagger receives structured context:

```text
observed HLT state
predicted missing/corrected residual
predicted offline-like state
```

The model can learn which state tokens are useful for each particle and class.

## Training Objectives

Use a multi-task objective:

```text
L =
  L_CE
  + lambda_logit * L_logit_KD
  + lambda_state * L_state_residual
  + lambda_delta * L_delta_norm
  + lambda_smooth * L_state_smoothness
```

Where:

```text
L_CE:
  normal 10-class cross entropy on final logits

L_logit_KD:
  optional offline teacher logit KD

L_state_residual:
  Phi_pred should match Phi_off, or delta_Phi_hat should match Phi_off - Phi_hlt

L_delta_norm:
  discourages huge corrections

L_state_smoothness:
  optional smoothness across adjacent radial/anchor tokens
```

Final-test primary evaluation must be HLT-only:

```text
allowed:
  HLT particles
  Phi_hlt computed from HLT particles
  residual predictor
  tagger checkpoint

not allowed:
  offline particles
  Phi_off
  offline teacher logits
  offline teacher representations
```

## Training Schedule

The safest schedule is staged.

### Stage 0: Baseline Warm Start

Train or load the HLT ParT baseline on the same split/cache.

This gives:

```text
baseline checkpoint
baseline metrics
warm-start weights
```

### Stage 1: State Predictor Pretraining

Freeze the HLT ParT baseline.

Train:

```text
Phi_hlt encoder
residual predictor
state-token encoder
```

Loss:

```text
L_state_residual + L_delta_norm + optional light logit auxiliary
```

Purpose:

```text
learn predictable HLT -> offline canonical-state corrections
without destabilizing the tagger
```

### Stage 2: Tagger Adapter Warmup

Freeze most or all of ParT.

Train:

```text
residual predictor
state-token encoder
cross-attention adapter
classifier head optionally
```

Loss:

```text
L_CE + L_logit_KD + L_state_residual + L_delta_norm
```

Purpose:

```text
teach the classifier to use the state tokens
```

### Stage 3: Gentle Unfreeze

Unfreeze upper ParT blocks.

Keep:

```text
residual predictor trainable
state adapter trainable
upper ParT trainable
lower ParT frozen or low LR
```

Purpose:

```text
let ParT integrate state context without forgetting baseline particle reasoning
```

### Stage 4: Full Fine-Tune Control

Optional final phase:

```text
full ParT gentle LR
state components trainable
early stopping on model_val
```

Use this only if Stage 3 is stable.

## Main Variants

### S0: HLT ParT Baseline

```text
HLT particles only
normal ParT
CE only
```

This is the anchor.

### S1: Phi_hlt Context Only

```text
HLT particles + Phi_hlt tokens
no residual prediction
```

Tests whether the canonical state itself gives useful global context.

### S2: State-Only Tagger

```text
Phi_hlt / Phi_pred tokens only
no particle tokens
```

Expected to underperform ParT, but useful to measure how much standalone
information the canonical state contains.

### S3: Residual State Prediction, No Tagging Use

```text
train residual predictor to match Phi_off
do not feed Phi_pred into tagger
```

This is a training-only reconstruction control. It should not be the main
model.

### S4: Residual State Context Mainline

```text
HLT particles
+ Phi_hlt tokens
+ delta_Phi_hat tokens
+ Phi_pred tokens
cross-attention adapter into ParT
CE + state residual loss
```

This is the first serious model.

### S5: Residual State Context + Logit KD

```text
S4 + offline teacher logit KD
```

This tests whether offline teacher decision boundaries help the state context
become more tagger-useful.

### S6: Residual State Context + Feature MLP Adapter

```text
feature MLP adapter from AV10
+ canonical state residual context
```

This tests whether the best earlier AV10 feature adapter stacks with the new
canonical state correction.

### S7: Full Best Bet

```text
feature MLP adapter
+ residual canonical state predictor
+ state-token cross-attention adapter
+ offline logit KD
+ staged freeze/unfreeze schedule
```

This is the highest-upside version.

## Controls

### C1: Shuffled Residual Tokens

Shuffle `delta_Phi_hat` across jets within the batch.

Expected:

```text
should hurt or collapse relative to real residual tokens
```

If it still helps, the gain may be regularization or capacity rather than
meaningful residual correction.

### C2: Random Noise State Tokens

Replace residual tokens with learned noise tokens independent of the jet.

Expected:

```text
should not match real residual-token performance
```

### C3: Phi_hlt With No Phi_pred

Feed only observed HLT state.

Expected:

```text
some gain possible, but less than predicted residual state
```

### C4: Oracle Phi_off Upper Bound

Feed `Phi_off` to the tagger during validation only as an explicitly marked
oracle diagnostic.

Never use this for final-test primary results.

Purpose:

```text
measure the maximum possible value of this canonical state representation
if predicted perfectly
```

### C5: Frozen Residual Predictor

Train residual predictor in Stage 1, then freeze it while training the tagger.

Tests whether joint end-to-end updates help.

### C6: No State Residual Loss

Feed a state adapter with the same capacity but train only on CE/KD.

Tests whether explicit HLT -> offline state supervision matters.

## Residual Predictor Ablations

The predictor ablations should be small enough to run, but strong enough to
tell us why the main predictor works or fails.

The mainline predictor is:

```text
geometry-biased state decoder
state queries attend to HLT particle memory
zero-init bounded delta_Phi output
```

The ablations below should keep the same `Phi` layout, same split, same
training budget, and same downstream tagger whenever possible.

### P0: Mainline Geometry-Biased State Decoder

This is the main predictor:

```text
Phi_hlt state tokens -> decoder queries
HLT particles -> memory
state self-attention + geometry-biased cross-attention
delta_Phi_hat output
```

Expected behavior:

```text
best or near-best state residual prediction
best tagging among clean predictor variants
attention maps aligned with token geometry
```

### P1: No Geometry Bias Decoder

Same architecture as P0, but remove the additive geometry bias from
cross-attention.

Purpose:

```text
tests whether the canonical token geometry is actually helping
```

Expected behavior:

```text
P0 > P1 if geometry/locality is useful
P1 close to P0 if the model learns geometry from features alone
```

This is a high-priority ablation.

### P2: DeepSets / Pooled Particle Summary Predictor

Replace cross-attention with a simpler pooled particle summary:

```text
particles -> DeepSets/PointNet pooling -> global summary g
concat(Phi_hlt token, g) -> MLP/transformer over state tokens -> delta_Phi
```

Purpose:

```text
tests whether full particle-to-state cross-attention is necessary
```

Expected behavior:

```text
good on global/radial summaries
worse on local anchor/sector corrections
```

This is a useful cheap baseline.

### P3: State-Only Predictor

Use only `Phi_hlt`:

```text
Phi_hlt -> state transformer/MLP -> delta_Phi
```

No HLT particles are provided beyond what is already summarized in `Phi_hlt`.

Purpose:

```text
tests how much correction is predictable from the canonical HLT state alone
```

Expected behavior:

```text
better than zero residual for global trends
worse than particle-aware predictors
```

If P3 is close to P0, the `Phi_hlt` state may already contain most useful
information.

### P4: Particle-Only Learned Query Predictor

Use learned state queries with token metadata, but do not feed `Phi_hlt`
values into the query embedding.

```text
token identity + geometry -> state query
HLT particles -> memory
cross-attention -> delta_Phi
```

Purpose:

```text
tests whether the predictor needs observed Phi_hlt values or mostly learns a
class/geometry prior from particles
```

Expected behavior:

```text
worse than P0
possibly decent for common average corrections
```

If P4 performs too well, inspect whether the state target is too dominated by
class priors rather than event-specific corrections.

### P5: Hard-Locality Decoder

Replace soft geometry bias with hard attention masks for local tokens:

```text
anchor tokens only attend to nearby particles
radial tokens only attend to particles in/near ring
sector tokens only attend to particles in/near sector
global tokens attend everywhere
```

Purpose:

```text
tests whether locality should be a soft prior or a hard constraint
```

Expected behavior:

```text
P0 >= P5
```

Hard locality may improve interpretability but could miss long-range evidence.
This should be a second-round ablation unless P0 attention looks chaotic.

### P6: Uncertainty-Weighted Residual Predictor

Add an uncertainty head:

```text
delta_Phi_hat
log_sigma_hat
```

Use uncertainty-weighted residual loss:

```text
Huber(error) * exp(-log_sigma_hat) + log_sigma_hat
```

Purpose:

```text
tests whether ambiguous HLT -> offline corrections benefit from learned
uncertainty
```

Expected behavior:

```text
better state prediction calibration
maybe better tagging if the tagger receives uncertainty tokens
```

This should not be first unless P0 has clear high-error ambiguous regions.

### P7: No State Self-Attention

Keep geometry-biased cross-attention, but remove state self-attention.

```text
each state token independently queries particles
no token-token communication
```

Purpose:

```text
tests whether global/local state tokens need to communicate
```

Expected behavior:

```text
worse than P0 if corrections require consistency across rings/anchors
```

### Recommended Predictor Ablation Order

Run first:

```text
P0 mainline geometry-biased state decoder
P1 no geometry bias
P2 DeepSets pooled predictor
P3 state-only predictor
```

Run second if P0 is promising:

```text
P4 particle-only learned query
P5 hard-locality decoder
P6 uncertainty-weighted predictor
P7 no state self-attention
```

The first-round comparison answers the key question:

```text
Do we need particle-aware, geometry-biased state-token decoding?
```

The hoped-for ordering is:

```text
P0 > P1 > P2 > P3
```

But any result is informative:

```text
P3 strong:
  Phi_hlt itself is a powerful sufficient summary.

P2 strong:
  global particle summaries are enough; full cross-attention may be overkill.

P1 close to P0:
  geometry bias may not matter, or the model learns it from coordinates.

P0 uniquely strong:
  canonical token geometry and particle evidence are both important.
```

## Evaluation Metrics

Tagging metrics:

```text
accuracy
macro accuracy
per-class accuracy
confusion matrix
cross entropy
top-2 accuracy
optional macro AUC if supported
```

State prediction metrics:

```text
Phi_hlt -> Phi_off baseline error
Phi_pred -> Phi_off error
relative improvement by token family
relative improvement by field
delta norm
correlation between predicted and true residuals
per-class residual error
```

Mechanism diagnostics:

```text
state-token attention mass
which token families are attended to
delta_Phi norm by class
Phi_pred physical validity rates
oracle Phi_off gap
shuffled/noise control gaps
adapter gate values
ParT fine-tuning magnitude
```

## Fusion And Ensembling Strategy

Fusion should be part of the experiment, not an afterthought.

The reason is practical:

```text
different residual/state predictors may recover different failure modes
```

One predictor may be better at global energy/fraction corrections, another at
subjet/anchor corrections, another at pure discriminative tagging. If their
errors are not identical, fusion can turn small individual gains into a more
stable result.

There are three useful fusion levels.

### Level 1: Logit Fusion

This is the cheapest and should always be run.

Train several deployable HLT-only models, then fit a small fusion model on
`stack_train` / `stack_val`:

```text
inputs:
  logits from HLT ParT baseline
  logits from S4 residual state context
  logits from S5 residual state + logit KD
  logits from S6/S7 feature-MLP + residual state variants
  optional logits from best AV10 feature/context models

fusion:
  logistic regression / ridge multinomial regression
  temperature-scaled average
  constrained convex weights
```

Use `stack_train` to fit fusion and `stack_val` to select:

```text
weighting
regularization
temperature
which models to include
```

Then apply once to `final_test`.

Advantages:

```text
cheap
hard to break
good diagnostic for complementary signal
easy to compare to old HLT4 fusion baselines
```

Disadvantages:

```text
does not let particle-level reasoning use multiple views internally
can overstate deployability if too many models are needed
```

### Level 2: State-Token Fusion

This is the most natural fusion level for this plan.

Instead of fusing only final logits, let the tagger consume multiple predicted
state views:

```text
Phi_hlt
Phi_pred_from_geometry_decoder
Phi_pred_from_deepsets_predictor
Phi_pred_from_feature_mlp_or_context_model
optional uncertainty tokens
```

Represent each view as a separate state-token family:

```text
state_family_embedding:
  observed_hlt_state
  geom_decoder_residual
  deepsets_residual
  feature_adapter_residual
  predicted_state_geom
  predicted_state_deepsets
```

Then the state-conditioned ParT can attend to all of them:

```text
HLT particles -> ParT particle embeddings
state views -> state encoder
particles cross-attend to state views
classifier -> logits
```

This is stronger than logit fusion because the model can use the state views
while still reasoning over individual HLT particles.

The first state-token fusion candidate should be:

```text
observed Phi_hlt
+ predicted Phi_pred from P0 geometry-biased decoder
+ predicted Phi_pred from P2 DeepSets predictor
+ optional delta tokens from both
```

This tests whether local geometry-aware and global pooled predictors recover
different useful corrections.

### Level 3: Particle-View Fusion

Particle-view fusion means giving ParT multiple particle-level or
particle-conditioned views, not only state tokens.

Possible views:

```text
original HLT particles
feature_mlp_adapter-adjusted particle embeddings
state-conditioned particle embeddings
local-compression adjusted features
architecture/context adapter particle deltas
```

The cleanest implementation is not to duplicate the full particle sequence
several times. Instead, use separate adapters that produce per-particle
corrections:

```text
delta_h_feature_mlp
delta_h_state_context
delta_h_local_compression
```

Then fuse them with a gated mixture:

```text
h_fused = h_base
  + gate_feature * delta_h_feature_mlp
  + gate_state   * delta_h_state_context
  + gate_local   * delta_h_local
```

Gates can be:

```text
global per-model learned scalars
per-particle gates
per-class/context gates
```

Start with global learned scalar gates initialized near zero. Per-particle
gates are powerful but can overfit.

This level is the highest-upside deep fusion, but it is also the easiest to
make scientifically muddy. It should come after the state-token and logit
fusion results are understood.

## Recommended Fusion Ladder

Run fusion in this order:

### F0: Baseline Logit Fusion

Fuse:

```text
HLT ParT baseline
best old AV10 feature_mlp_adapter
best old AV10 context/repeat model if available
```

Purpose:

```text
anchors against the previous fusion baseline
```

### F1: Canonical State Logit Fusion

Fuse:

```text
HLT ParT baseline
S4 residual state context
S5 residual state + logit KD
S6/S7 if trained
```

Purpose:

```text
tests whether canonical-state residual models add complementary score-level
signal
```

### F2: Predictor Diversity Logit Fusion

Fuse variants using different residual predictors:

```text
P0 geometry-biased decoder tagger
P1 no-geometry decoder tagger
P2 DeepSets predictor tagger
P3 state-only predictor tagger
```

Purpose:

```text
tests whether different predictor inductive biases capture different residual
information
```

### F3: State-Token View Fusion

Train one tagger that consumes multiple predicted state views:

```text
Phi_hlt
Phi_pred_P0
Phi_pred_P2
delta_Phi_P0
delta_Phi_P2
optional uncertainty/confidence
```

Purpose:

```text
lets ParT use complementary residual views before final classification
```

This is the most promising fusion strategy specific to the canonical-state
idea.

### F4: Particle-View Gated Adapter Fusion

Fuse per-particle deltas:

```text
feature_mlp_adapter delta_h
state_context_adapter delta_h
optional local_compression delta_h
```

with learned gates:

```text
h_fused = h_base + sum_j gate_j * delta_h_j
```

Purpose:

```text
tests whether multiple internal particle-view corrections stack better than
score fusion
```

This should be a second-stage experiment after F1/F3.

## Fusion Controls

Fusion needs controls because it is easy to win by capacity or leakage.

Required controls:

```text
shuffle one model's logits across jets before logit fusion
shuffle one state-view family across jets before state-token fusion
random/noise state-view fusion
equal-weight average vs learned fusion
calibration-only baseline
same-number-of-models seed ensemble baseline
```

The seed ensemble baseline is important:

```text
HLT ParT seed 1 + seed 2 + seed 3
```

If a fancy fusion only matches a simple seed ensemble, the new representation
may not be adding unique information.

## Best Ensemble Candidates

The first ensemble should include models that are likely to make different
mistakes:

```text
HLT ParT baseline:
  strongest canonical baseline

AV10 feature_mlp_adapter:
  known strong embedding-adapter behavior

S4/S5 canonical residual state model:
  structured HLT -> offline state correction

P2 DeepSets predictor model:
  global/pooled residual style

P0 geometry-biased predictor model:
  local/anchor/ring residual style
```

Do not fuse every model blindly in the main claim. Use broad fusion as a
diagnostic, but define a small deployable ensemble:

```text
2-model ensemble:
  HLT ParT baseline + best canonical-state model

3-model ensemble:
  HLT ParT baseline + AV10 feature_mlp_adapter + best canonical-state model

4-model diagnostic ensemble:
  add best predictor-diversity model
```

The final report should show both:

```text
best single model
best small deployable ensemble
best diagnostic broad ensemble
```

## Training Models For Complementary Fusion

Do not rely only on accidental diversity.

The best fusion candidates should be trained so they remain strong individually
but are nudged toward complementary information.

There are two competing goals:

```text
each model must be accurate on its own
models should make different useful corrections/errors
```

If the diversity pressure is too strong, models become weird and weak. If there
is no diversity pressure, fusion may collapse into a seed ensemble.

### Warm-Start Policy

Most serious variants should warm-start from the same strong HLT ParT baseline.

Use:

```text
HLT ParT baseline checkpoint selected on model_val
same split
same HLT cache
same label mapping
same optimizer family
```

Warm-starting matters because the question is:

```text
what additional structured correction/view improves an already strong ParT?
```

Not:

```text
can a randomly initialized model eventually learn a different solution?
```

Recommended default:

```text
warm-start ParT body and classifier head from HLT baseline
zero-init all adapters/residual heads
freeze most of ParT for a short warmup
then unfreeze upper blocks
then optionally gentle full unfreeze
```

This makes every variant start from the same strong baseline and lets the
residual/context modules prove their value.

### When Not To Warm-Start

Include a small number of from-scratch controls only if compute allows:

```text
from-scratch canonical-state model
from-scratch feature_mlp_adapter model
from-scratch seed ensemble
```

These are not the mainline. They answer whether the gain is from architecture
alone or from baseline-preserving adaptation.

### Complementarity Without Hurting Accuracy

The safest way to encourage complementarity is to diversify architecture and
training objective, not to directly punish agreement too aggressively.

Good diversity sources:

```text
different residual predictor inductive biases:
  geometry-biased decoder vs DeepSets vs state-only

different context insertion points:
  state-token cross-attention vs feature MLP delta_h

different auxiliary losses:
  CE only vs CE + state residual vs CE + logit KD + state residual

different freeze schedules:
  frozen warmup -> upper unfreeze vs joint fine-tune

different random seeds:
  only as a baseline/control
```

Riskier diversity sources:

```text
explicitly penalizing same predictions too strongly
forcing low correlation of logits
training models to specialize on hard examples without guardrails
```

Those can reduce individual quality.

### Recommended Complementary Training Ladder

Train these as the first complementary pool:

```text
M0: HLT ParT baseline
  warm-start anchor / no adapter

M1: AV10 feature_mlp_adapter repeat
  known strong per-particle embedding adapter

M2: S4 canonical residual state context
  geometry-biased decoder, CE + state residual

M3: S5 canonical residual state context + logit KD
  same as M2, plus offline teacher logit KD

M4: P2 DeepSets residual state model
  global/pooled residual predictor, different inductive bias

M5: S7 full best bet
  feature_mlp_adapter + canonical residual state + logit KD
```

Then evaluate:

```text
single-model performance
pairwise error overlap
pairwise logit correlation
per-class complementarity
fusion gain over best individual
fusion gain over same-size seed ensemble
```

### Residual-Diversity Loss

If ordinary architectural diversity is not enough, add a mild diversity loss.

Use it only for second-round experiments.

The safest version targets residuals, not logits:

```text
encourage delta_Phi_hat from predictor A and predictor B to explain different
components of delta_Phi_true
```

Example:

```text
residual_error_A = delta_true - delta_hat_A
residual_error_B = delta_true - delta_hat_B

diversity term encourages:
  delta_hat_B to improve examples/token-fields where A has high residual error
```

A practical version:

```text
train model B with state loss weights upweighted where model A has large
model_val/token-family residual error
```

This is better than saying:

```text
make logits different
```

because it asks the second model to fix different canonical-state failures,
not simply disagree.

### Boundary/Hard-Example Complementarity

Another second-round option is hard-example specialization.

Train a complementary model with slightly higher weight on:

```text
model_train examples where baseline ParT is wrong
examples near the decision boundary
classes with large HLT/offline gap
token families where Phi_hlt -> Phi_off residual is large
```

This must be done using model_train/model_val only.

Do not use final_test to choose hard examples.

Recommended form:

```text
sample_weight =
  1
  + a * baseline_error_indicator
  + b * boundary_weight
  + c * normalized_state_residual_magnitude
```

Keep weights mild:

```text
max weight 2x or 3x
```

### Fusion Selection Discipline

Fusion models must be selected on stack/model validation data, not final test.

Recommended:

```text
train base models on model_train
select base checkpoints on model_val
fit fusion on stack_train
select fusion hyperparameters on stack_val
evaluate final_test once
```

If stack splits are tiny or unavailable for a specific experiment, use
model_val for fusion diagnostics only and clearly label it as not final-clean.

### Best Expected Fusion Strategy

The highest-probability path is:

```text
1. warm-start all serious models from HLT ParT
2. train M1/M2/M3/M4/M5 with different architectures/objectives
3. run logit fusion to measure complementarity
4. if M2/M4 complement each other, train F3 state-token view fusion
5. if M1 and M2 complement each other, train F4 particle-view gated adapter fusion
```

The best deployable ensemble is likely:

```text
HLT ParT baseline
+ AV10 feature_mlp_adapter
+ canonical residual state context model
```

The best single model is likely:

```text
feature_mlp_adapter
+ canonical residual state context
+ staged warm-start/unfreeze
```

## Split Discipline

Use the same split discipline as the AV10 experiments when comparing against
AV10 results:

```text
model_train: train taggers and residual predictors
model_val: checkpoint selection and hyperparameter decisions
stack_train/stack_val: only if score fusion is run
final_test: held out until final evaluation
```

For offline supervision:

```text
Phi_off may be computed for model_train and model_val.
Phi_off for final_test must not be used in primary evaluation.
```

Optional final-test oracle diagnostics can exist, but they must be:

```text
explicitly labeled oracle/non-deployable
excluded from model selection
excluded from primary claims
```

## Cache Artifacts

The experiment should cache canonical states to avoid recomputation.

For each split:

```text
canonical_state_hlt/{split}_phi_hlt.npz
canonical_state_offline/{split}_phi_off.npz
canonical_state_metadata/{split}_metadata.json
```

Each cache should include:

```text
phi_tokens
token_mask
token_type_ids
scale_ids
slot_ids
field_names
normalization_metadata
source_manifest_hash
source_view
hlt_cache_hash if HLT
offline_cache_hash if offline
phi_builder_version
```

The trainer must fail if:

```text
manifest hash mismatches
HLT cache identity mismatches
Phi metadata version mismatches
field order mismatches
token layout mismatches
```

## Implementation Steps

### Step 1: Input Split And HLT v2 Cache Contract

Implement the input contract for this campaign.

Deliverables:

```text
split builder/reuse hooks for:
  model_train = 5M
  model_val   = 1M
  stack_train = 3M
  stack_val   = 1M
  final_test  = 1M

HLT cache builder locked to:
  hlt_profile = fixed_hlt_v2_realistic
  hlt_degradation_strength = 2.5

cache audit that records:
  source manifest hash
  HLT profile/version/strength
  per-split metadata hashes
  label mapping
  split sizes
```

The trainer must fail if the cache is not `fixed_hlt_v2_realistic` at strength
`2.5`.

Tests:

```text
wrong profile rejected
wrong strength rejected
manifest mismatch rejected
missing split rejected
expected high-data split sizes recorded
```

### Step 2: Canonical Jet-State Layout

Implement the canonical state specification.

Deliverables:

```text
CanonicalJetStateConfig
CanonicalJetStateLayout
token families:
  global
  radial
  angular
  anchor_coarse
  anchor_medium
  anchor_fine

field names and field order
feature scales
residual scales
token metadata:
  token_type_id
  scale_id
  slot/ring/sector id
  geometry center/width
```

The field contract should be frozen before large caches are built.

Tests:

```text
stable field order
stable token order
expected K_state and D_phi
metadata roundtrip
residual scale sanity
```

### Step 3: Deterministic Phi Builder

Implement deterministic `Phi(particles)`.

Builder components:

```text
global token builder
radial ring token builder
angular sector token builder
soft anchor-slot token builder
normalization/sanitization
diagnostics
```

Inputs:

```text
canonical particle features
particle mask
optional source-view metadata
```

Outputs:

```text
phi_tokens: [batch, K_state, D_phi]
state_mask: [batch, K_state]
token metadata
diagnostics
```

Tests:

```text
finite values
mask handling
empty/padded particles
simple hand-computed jets
HLT/offline same builder path
deterministic reproducibility
```

### Step 4: Phi Cache Writer And Audit

Implement cache generation for canonical states.

Required caches:

```text
Phi_hlt:
  model_train
  model_val
  stack_train
  stack_val
  final_test

Phi_off:
  model_train
  model_val
  stack_train
  stack_val

Phi_off final_test:
  optional oracle diagnostics only
```

Each cache must include:

```text
phi_tokens
state_mask
token metadata
field names
normalization metadata
source manifest hash
source HLT/offline cache identity
builder version
```

Tests:

```text
cache provenance checks
field/layout mismatch rejection
HLT/offline split alignment
oracle final-test cache cannot be used by primary trainer
```

### Step 5: Residual Predictor Models

Implement the residual predictor family.

Mainline:

```text
GeometryBiasedStateResidualDecoder
```

Architecture:

```text
HLT particles -> particle memory encoder
Phi_hlt tokens + metadata -> state queries
state self-attention
geometry-biased cross-attention to particle memory
FFN
zero-init bounded delta_Phi head
Phi_pred = Phi_hlt + delta_Phi_hat
```

Required predictor variants:

```text
P0 geometry-biased decoder
P1 no geometry bias
P2 DeepSets/global pooled predictor
P3 state-only predictor
P4 particle-only learned-query predictor
P5 hard-locality predictor
P6 uncertainty predictor
P7 no state self-attention
```

Tests:

```text
shape/mask checks
zero-init gives delta_Phi ~= 0
geometry bias affects attention logits
invalid particles masked
variant switches are active
diagnostics emitted
```

### Step 6: State-Conditioned ParT Tagger

Implement the tagger that keeps HLT particles as the main input and feeds
canonical state context into the classifier path.

Mainline:

```text
HLT particles -> warm-started ParT particle pathway
Phi_hlt / delta_Phi_hat / Phi_pred -> state token encoder
particle embeddings cross-attend to state tokens
state-conditioned particle embeddings -> ParT/classifier
```

Supported modes:

```text
particles only
Phi_hlt context only
delta_Phi context
Phi_pred context
Phi_hlt + delta_Phi + Phi_pred
state-only tagger
shuffled/noise state controls
oracle Phi_off diagnostic context
feature_mlp_adapter + state context
```

Tests:

```text
warm-start compatibility
zero-init adapter preserves baseline logits approximately
state family embeddings work
shuffled/noise controls actually break semantics
oracle context blocked from primary final-test path
```

### Step 7: Losses And Training Schedules

Implement the training objectives and schedules.

Losses:

```text
10-class CE
state residual Huber/L1 loss
optional offline logit KD
delta norm penalty
optional smoothness penalty
optional uncertainty-weighted state loss
```

Schedules:

```text
warm-start frozen warmup -> upper unfreeze
warm-start joint from start
full ParT frozen adapter/head training
from-scratch canonical-state model
from-scratch ParT baseline
predictor pretrain -> tagger train
joint end-to-end no pretraining
```

Report per phase:

```text
trainable module groups
LR per group
loss weights
freeze/unfreeze state
state cache identity
teacher/cache identity
```

Tests:

```text
loss terms active/inactive by variant
freeze schedule changes trainable params
from-scratch variants do not load warm start
teacher-free final-test evaluation
nonfinite batch guard
```

### Step 8: Variant Registry For A0-G3

Add a registry for every planned run:

```text
A0 A1 A2 A3
B0 B1 B2 B3
C0 C1 C2 C3 C4 C5 C6
D0 D1 D2 D3 D4 D5
E0 E1 E2 E3 E4 E5 E6
F0 F1 F2 F3 F4 Fseed Fshuffle
G0 G1 G2 G3
```

Each spec should define:

```text
model components
predictor variant
state context families
warm-start policy
loss weights
freeze schedule
whether offline teacher logits are used
whether oracle inputs are allowed
primary-vs-diagnostic status
```

Tests:

```text
all run IDs registered
oracle variants marked non-primary
final-test restrictions enforced
baseline/fusion dependencies declared
```

### Step 9: Metrics And Reports

Implement reports that answer the scientific questions.

Report tables:

```text
tagging metrics
state prediction metrics
per-token-family residual metrics
per-field residual metrics
per-class metrics
control gaps
oracle gaps
fusion comparison
seed ensemble comparison
```

Required comparisons:

```text
D2/D3 vs A0/A1/A2/A3
B0 vs A0
D2/D3 vs B0
D5 vs D2
C0 vs C1/C2/C3
E3 vs E4
F1/F3/F4 vs Fseed
G0/G1 vs D2/D3
```

Tests:

```text
missing required run fails report unless explicitly allowed
provenance consistency enforced
oracle diagnostics separated
fusion rows separated from single-model rows
```

### Step 10: Slurm Submitters

Add submitters for pilot and high-data campaigns.

Jobs:

```text
build/reuse splits
build HLT v2 strength-2.5 cache
cache Phi_hlt/Phi_off
train A-E single-model variants
cache predictions/logits
run F fusion variants
write G oracle diagnostics
write final report
```

Submitter requirements:

```text
pilot and high-data modes
strict cache/provenance checks
dependency graph between caches, models, fusion, report
optional skip-existing with completeness checks
clear job names
record source commit/status hash
```

High-data default:

```text
model_train = 5M
model_val   = 1M
stack_train = 3M
stack_val   = 1M
final_test  = 1M
```

Tests:

```text
sbatch dry-run includes all expected jobs
wrong cache profile/strength rejected before queue
skip-existing rejects incomplete outputs
all A0-G3 run IDs appear in expected jobs/report
```

## Recommended First Run

If compute is limited, run:

```text
S0 HLT ParT baseline
S1 Phi_hlt context only
S4 residual state context mainline
S5 residual state context + logit KD
C1 shuffled residual tokens
C2 random/noise state tokens
C4 oracle Phi_off upper bound on model_val only
```

Then, if S4/S5 beat the baseline and controls fail:

```text
S6 feature_mlp_adapter + residual state context
S7 full best bet
```

## Full Ablation Campaign

If compute is available, run the full campaign below.

The goal is not only to find the best number. The goal is to separate:

```text
canonical state information
residual prediction quality
tagger use of residual context
training schedule effects
warm-start vs from-scratch effects
fusion complementarity
oracle headroom
```

All runs should use the same split/cache and label mapping unless explicitly
marked as an oracle diagnostic. The preferred comparison setting is the same
AV10 10-class HLT0.6 high-data split/cache when comparing against the existing
AV10 results.

### Tier A: Anchors

These runs define the ground truth for interpretation.

```text
A0: HLT ParT baseline
  normal HLT particles only
  canonical baseline
  selected on model_val

A1: HLT ParT fine-tune-only control
  same warm-start/schedule budget as adapter models
  no Phi tokens
  no residual predictor
  tests whether gains come from fine-tuning schedule alone

A2: AV10 feature_mlp_adapter repeat
  repeat known strong AV10 feature MLP adapter
  same split/cache as canonical-state runs
  anchors against previous best adapter behavior

A3: HLT ParT seed ensemble baseline
  same architecture as A0
  multiple seeds/checkpoints
  same number of models as small fusion comparisons
  tests whether fusion gains beat ordinary seed diversity
```

Required outputs:

```text
single-model metrics
training curves
final-test metrics
per-class accuracy/confusion
seed ensemble fusion report for A3
```

### Tier B: Does Phi Help At All?

These runs test whether canonical state tokens contain useful tagging
information before residual prediction.

```text
B0: HLT particles + Phi_hlt context
  no residual prediction
  state-token encoder/cross-attention adapter active
  tests whether observed canonical HLT state helps ParT

B1: Phi_hlt-only tagger
  no original HLT particle tokens
  state tokens only
  expected to underperform ParT
  measures standalone information content of Phi

B2: HLT particles + shuffled Phi_hlt
  Phi_hlt state tokens shuffled across jets
  same capacity as B0
  tests whether semantic state conditioning matters

B3: HLT particles + random/noise state tokens
  learned/random state tokens independent of the jet
  same adapter capacity
  tests capacity/regularization-only explanation
```

Key interpretation:

```text
B0 > A0:
  Phi_hlt observed state is useful.

B2/B3 close to B0:
  gain may be capacity/regularization, not meaningful state semantics.

B1 close to A0:
  Phi is surprisingly powerful by itself.
```

### Tier C: Residual Predictor Quality

These runs train/evaluate the HLT -> offline canonical-state residual predictor
independent of the final tagging claim.

```text
C0: geometry-biased decoder predictor
  mainline P0 predictor
  state queries attend HLT particles with geometry bias

C1: no-geometry-bias decoder
  same as C0, but no additive geometry bias
  tests whether token geometry matters

C2: DeepSets/global pooled predictor
  particles -> pooled summary
  Phi_hlt + pooled summary -> delta_Phi
  tests whether full state-particle cross-attention is needed

C3: state-only predictor
  Phi_hlt -> delta_Phi
  no particle evidence beyond Phi_hlt
  tests how much residual is predictable from state alone

C4: particle-only learned-query predictor
  token identity/geometry queries + particles
  no Phi_hlt values in state query
  tests whether observed state values are needed

C5: hard-locality predictor
  hard local attention masks for ring/sector/anchor tokens
  tests soft bias vs hard constraint

C6: uncertainty predictor
  predicts delta_Phi and log_sigma
  uncertainty-weighted state residual loss
  tests ambiguous residual calibration
```

Required outputs:

```text
Phi_hlt -> Phi_off baseline error
Phi_pred -> Phi_off error
relative improvement by token family
relative improvement by field
per-class residual error
delta norm
predicted-vs-true residual correlation
attention diagnostics for C0/C1/C5
uncertainty calibration for C6
```

Expected ordering:

```text
C0 > C1 > C2 > C3
```

But any deviation is informative.

### Tier D: Does Predicted Residual Context Help Tagging?

These are the main tagging experiments.

```text
D0: HLT particles + Phi_hlt + delta_Phi_hat
  feed observed state and predicted residual
  do not feed Phi_pred separately

D1: HLT particles + Phi_hlt + Phi_pred
  feed observed state and corrected predicted state
  do not expose delta separately

D2: HLT particles + Phi_hlt + delta_Phi_hat + Phi_pred
  full state context
  CE + state residual loss
  main clean residual-state model

D3: D2 + offline logit KD
  adds teacher decision boundary guidance
  likely stronger than D2 if teacher logits are useful

D4: D2 with state residual loss only as auxiliary, no logit KD
  isolates state supervision without teacher logits

D5: D2 architecture, CE only, no state residual loss
  same capacity but no explicit HLT -> offline state supervision
  tests whether residual target matters
```

Key interpretation:

```text
D2 > B0 > A0:
  predicted residual state adds real tagging value.

D5 close to D2:
  architecture/capacity may matter more than residual supervision.

D3 > D2:
  offline teacher logits help make state context tagger-relevant.

D0 vs D1:
  tells whether the tagger prefers explicit residuals or corrected states.
```

### Tier E: Training Schedule And Warm-Start Ablations

These runs test whether the gain depends on warm-starting and staged training.

```text
E0: warm-start, frozen warmup -> upper-unfreeze
  default stable schedule

E1: warm-start, joint from start
  all trainable modules active immediately
  tests whether staged training is necessary

E2: warm-start, full ParT frozen
  train residual predictor + state adapter/head only
  tests how much can be gained without moving ParT body

E3: from-scratch canonical-state model
  same architecture as main residual-state model
  no HLT ParT warm start
  tests whether architecture itself is better

E4: from-scratch ParT baseline
  matched schedule/epochs to E3
  required comparison for E3

E5: residual predictor pretrain -> tagger train
  pretrain delta_Phi predictor on state loss
  then train tagger with predictor initialized
  tests whether stable predictor pretraining helps

E6: joint end-to-end, no predictor pretraining
  train predictor and tagger together from the beginning
  comparison for E5
```

Key interpretation:

```text
E0 > E1:
  staged training stabilizes residual use.

E2 improves:
  residual/context adapter can help without changing ParT body.

E3 > E4:
  canonical-state architecture helps even from scratch.

E5 > E6:
  predictor pretraining matters.
```

### Tier F: Fusion And Ensemble Runs

These runs test complementarity.

```text
F0: logit fusion, HLT ParT + best canonical-state model
  smallest deployable ensemble

F1: logit fusion, HLT ParT + AV10 feature MLP + best canonical-state model
  likely strongest small practical ensemble

F2: predictor-diversity logit fusion
  fuse C0/D-style geometry predictor model
  + C2/D-style DeepSets predictor model
  + optionally no-geometry/state-only variants
  tests whether different predictors recover different signal

F3: state-token view fusion
  one tagger consumes multiple predicted state views:
    Phi_hlt
    Phi_pred_C0
    Phi_pred_C2
    delta_C0
    delta_C2
  lets ParT use residual views before logits

F4: particle-view gated adapter fusion
  fuse per-particle deltas:
    feature_mlp_adapter delta_h
    state_context_adapter delta_h
    optional local_compression delta_h
  learned gates initialized near zero

Fseed: same-size HLT ParT seed ensemble
  required advisor-facing control

Fshuffle: fusion with one model/view shuffled across jets
  required leakage/capacity control
```

Fusion discipline:

```text
train base models on model_train
select base checkpoints on model_val
fit fusion on stack_train
select fusion hyperparameters on stack_val
evaluate final_test once
```

If stack splits are unavailable, label the fusion diagnostic as model-val-only
and do not use it as a primary final-test claim.

### Tier G: Offline/Oracle Diagnostics

These are not deployable. They measure headroom and failure mode.

```text
G0: oracle Phi_off context on model_val only
  feed true offline canonical state to tagger
  measures upper bound of canonical state usefulness

G1: oracle delta_Phi_true context on model_val only
  feed true Phi_off - Phi_hlt residual
  tests whether residual form itself is useful

G2: predicted Phi_pred vs oracle Phi_off gap
  compare D/S models against oracle state-context performance
  estimates predictor bottleneck

G3: per-class/token-family residual error analysis
  identify which classes and state fields are predictable/useful
  guide second-round representation changes
```

Oracle rules:

```text
never use oracle Phi_off for training final-test decisions
never report oracle as deployable
keep oracle diagnostics separate from primary final-test tables
```

### Full Campaign Summary

Run IDs:

```text
A0 A1 A2 A3
B0 B1 B2 B3
C0 C1 C2 C3 C4 C5 C6
D0 D1 D2 D3 D4 D5
E0 E1 E2 E3 E4 E5 E6
F0 F1 F2 F3 F4 Fseed Fshuffle
G0 G1 G2 G3
```

This is intentionally broad. The key primary comparisons are:

```text
D2/D3 vs A0/A1/A2/A3
B0 vs A0
D2/D3 vs B0
D5 vs D2
C0 vs C1/C2/C3
E3 vs E4
F1/F3/F4 vs Fseed
G0/G1 vs D2/D3
```

The result we want:

```text
D2/D3 improve over HLT ParT and Phi_hlt-only context.
Controls B2/B3/D5/Fshuffle do not explain the gain.
C0 predicts residuals better than simpler predictors.
Fusion beats same-size seed ensemble.
Oracle diagnostics show remaining headroom.
```

## Expected Outcomes

The ideal result:

```text
S4 or S5 > S1 > S0
C1/C2 do not match S4/S5
oracle Phi_off is above S4/S5
S6/S7 stack on top of feature_mlp_adapter
```

That would suggest:

```text
canonical HLT -> offline state residuals are learnable,
the predicted residuals are meaningful,
and ParT can use them as structured context.
```

The concerning result:

```text
C1/C2 match S4/S5
```

That would suggest:

```text
the gain is mostly extra capacity, regularization, or tuning dynamics.
```

The still-useful result:

```text
S1 helps, but S4/S5 do not improve much beyond S1
```

That would suggest:

```text
canonical HLT state tokens are useful, but offline residual prediction is not
yet accurate or not tagger-relevant.
```

## Bottom Line

This plan is the clean bridge between reconstruction and tagging.

It does not ask the model to create fake particles. It does not ask arbitrary
neural hidden dimensions to align. It builds a shared, fixed, multi-scale
state space where HLT/offline residuals have physical meaning, then gives a
ParT tagger both the original HLT particles and a predicted offline-like
state context.

The line in the sand:

```text
HLT particles remain the main input.
Predicted canonical state residuals must be fed into the classifier path.
Final-test primary evaluation must be HLT-only.
```

## Input HLT Cache Contract

This experiment should use a fresh HLT cache built with the HLT v2 realistic
degradation profile:

```text
hlt_profile = fixed_hlt_v2_realistic
hlt_degradation_strength = 2.5
```

This should be treated as a new canonical input setting for this campaign.
Do not silently reuse old AV10 `hlt0p6` / `fixed_hlt_v1` caches for the primary
results.

The HLT v2 cache should be built once per split family and then reused by every
run in the A0-G3 campaign.

Required cached splits:

```text
model_train
model_val
stack_train
stack_val
final_test
```

Primary comparison split sizes should match the high-data AV10-style regime
unless a pilot is explicitly labeled:

```text
high-data:
  model_train = 5M
  model_val   = 1M
  stack_train = 3M
  stack_val   = 1M
  final_test  = 1M

pilot:
  model_train = 500k
  model_val   = 150k
  stack_train = 500k
  stack_val   = 150k
  final_test  = 500k or 150k, clearly recorded
```

Every report must record:

```text
hlt_profile
hlt_degradation_strength
hlt_builder_version
hlt_params
source_manifest_hash
per-split HLT cache metadata hash
```

The trainer and report writer must fail if:

```text
an HLT cache was built with fixed_hlt_v1
the degradation strength is not 2.5
the source manifest hash does not match
the split sizes do not match the run contract
```

Old AV10 `hlt0p6` results can still be used as historical context, but the
primary canonical-state campaign should be internally consistent on:

```text
fixed_hlt_v2_realistic, strength 2.5
```
