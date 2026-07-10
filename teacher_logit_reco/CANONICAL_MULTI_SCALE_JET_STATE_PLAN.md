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

### Step 1: Canonical State Specification

Add a module defining:

```text
state token families
token counts
field names
normalization rules
feature scales
token metadata contract
```

Deliverables:

```text
CanonicalJetStateConfig
CanonicalJetStateLayout
field order tests
token identity tests
```

### Step 2: Phi Builder

Implement deterministic `Phi(particles)`:

```text
build_global_tokens
build_radial_tokens
build_angular_tokens
build_anchor_slot_tokens
```

Inputs:

```text
canonical particle features
particle mask
```

Outputs:

```text
phi_tokens: [batch, K_state, D_phi]
state_mask: [batch, K_state]
metadata
diagnostics
```

Tests:

```text
shape checks
mask checks
field order checks
identity/reproducibility
finite values
simple hand-computed jets
```

### Step 3: State Cache Writer

Write caches for:

```text
Phi_hlt for model_train/model_val/stack/final_test
Phi_off for model_train/model_val
optional Phi_off for final_test oracle diagnostics only
```

The cache writer must record manifest/cache provenance and reject mismatches.

### Step 4: Residual Predictor

Implement:

```text
CanonicalStateResidualPredictor
GeometryBiasedStateResidualDecoder
```

Inputs:

```text
Phi_hlt tokens
state token metadata
HLT particle features/mask
optional HLT particle summary
```

Outputs:

```text
delta_Phi_hat
Phi_pred
diagnostics
```

Use:

```text
particle memory encoder
state token query encoder
state self-attention
geometry-biased cross-attention from state tokens to particles
zero-init residual projection
feature-wise residual scales
token masks
```

The mainline implementation should be:

```text
d_model = 128
particle encoder layers = 1-2
state decoder layers = 3
heads = 4
dropout = 0.05
norm_first = true
soft geometry bias enabled
zero-init bounded residual head
```

The implementation must expose switches for first-round predictor ablations:

```text
no geometry bias
DeepSets pooled predictor
state-only predictor
```

Second-round switches:

```text
particle-only learned query predictor
hard-locality attention masks
uncertainty head
no state self-attention
```

### Step 5: State-Conditioned Tagger

Implement:

```text
HLT particles -> ParT embedding
state tokens -> state encoder
cross-attention adapter state -> particles
ParT/classifier -> logits
```

The tagger must support:

```text
particles only
Phi_hlt only
Phi_hlt + delta
Phi_hlt + Phi_pred
all state families
shuffled/noise controls
oracle Phi_off diagnostic mode
```

### Step 6: Losses And Training Schedule

Implement:

```text
state residual loss
tagging CE
optional offline logit KD
delta norm penalty
smoothness penalty
staged freeze/unfreeze schedule
```

Report phase metadata:

```text
which modules are trainable
LR per group
loss weights
state cache identity
teacher/cache identity
```

### Step 7: Reports

Report:

```text
tagging metrics
state prediction metrics
control gaps
oracle Phi_off upper bound
state attention diagnostics
per-class residual diagnostics
final-test primary metrics
```

The report should explicitly answer:

```text
Does Phi_hlt context help?
Does predicted residual context help beyond Phi_hlt?
Does Phi_pred beat shuffled/noise controls?
How close is Phi_pred to oracle Phi_off performance?
Does this stack with feature_mlp_adapter?
```

### Step 8: Slurm Submitter

Add submitters for:

```text
cache canonical states
train baseline/control/main variants
write final report
```

Support:

```text
reuse existing AV10 split/HLT cache
high-data run
pilot run
strict provenance checks
optional offline oracle diagnostics
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
