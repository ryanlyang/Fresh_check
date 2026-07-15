# Adaptive Binary Pseudo-Offline Particle Hierarchy Plan

## 1. Objective

The objective is to beat a strong HLT-only Particle Transformer on the same
10-class task by reconstructing useful offline-like structure from HLT
particles and exposing that structure to a baseline-preserving tagger.

The method must remain deployable:

```text
training:   HLT particles + paired offline particles + labels
inference:  HLT particles only
```

This is a conditional reconstructor or hallucinator. It cannot recover
information that is fundamentally absent from HLT. Its purpose is to model the
conditional distribution of offline structure that remains statistically
predictable from the observed HLT jet.

The design is explicitly performance-first. Training cost, model size, cache
size, and wall time are secondary to final tagging accuracy, calibration, and
robustness.

## 2. Primary Hypothesis

The previous constrained coarse-to-fine design used a dense spatial hierarchy:

```text
global jet -> 8 fixed cells -> 32 fixed cells -> 128 fixed cells -> particles
```

At fine resolution, most cells are empty. This spends capacity predicting the
absence of particles and gives the model a representation unlike the variable
sets on which ParT performs well.

This plan replaces fixed cells with a nested hierarchy of occupied particle
groups:

```text
HLT particles
-> predicted offline-like global state
-> up to 2 occupied groups
-> up to 4 occupied groups
-> up to 8 occupied groups
-> up to 16 occupied groups
-> up to 32 occupied microgroups
-> variable-length pseudo-offline particles
```

The powers of two are capacity limits, not required occupancies. A singleton
group is carried forward without inventing an empty sibling. For a particular
jet the active counts may be:

```text
1 -> 2 -> 4 -> 7 -> 12 -> 21 -> particles
```

Every active group represents an occupied subset of offline constituents.
Padding exists only for tensor batching and is always masked. A padded token
is not treated as a physical empty region.

## 3. Recommended Best Model

The primary performance candidate is named:

```text
ABPH-KT32-MH4-DUALCROSS
```

It has the following fixed configuration:

```text
trusted encoder:
  strong warm-started HLT ParT

offline target hierarchy:
  recursive exclusive-kT binary partition
  capacities 2, 4, 8, 16, 32

root predictor:
  typed semantic-query transformer
  residual physical targets relative to HLT
  calibrated probabilistic heads

reconstructor:
  parent-conditioned binary transformer decoder
  every stage can cross-attend to all HLT particles
  exact accounting projection at every split

hypotheses:
  one deterministic conditional mean
  four coherent stochastic pseudo-offline hypotheses
  one jet-level latent propagated through the complete tree per hypothesis

particle rendering:
  variable-length local set decoder
  exact parent four-momentum, count, type-count, and charge accounting
  local optimal-transport particle supervision

tagger:
  separate HLT and pseudo-particle ParT streams
  multiscale hierarchy-token memory
  ancestry and tree-distance attention biases
  geometry-aware bidirectional cross-attention
  learned uncertainty/trust gates
  zero-initialized residual correction around the HLT baseline

primary tagger objective:
  label CE
  maintained reconstruction/accounting supervision during joint fine-tuning
  no offline-teacher logit KD in the primary recipe
```

A second complementary target hierarchy based on Cambridge/Aachen is trained
for ablation and fusion. The expected absolute best campaign model fuses the
best exclusive-kT and Cambridge/Aachen pseudo views while retaining one trusted
HLT stream.

### 3.1 Locked Primary Dimensions

The maximum-performance configuration uses the existing `large` ParT family
for both particle branches:

```text
embed_dims: [192, 768, 192]
pair_embed_dims: [96, 96, 96]
num_heads: 8
num_layers: 12
num_cls_layers: 3
```

The clean A0 baseline uses this same large HLT ParT. A separately reported
canonical-base HLT result preserves comparison to older campaigns.

New hierarchy modules use:

```text
model dimension: 256
attention heads: 8
feed-forward dimension: 1024
dropout: 0.10 during reconstruction pretraining
root semantic-query blocks: 4
binary decoder blocks per hierarchy level: 4
fusion cross-attention blocks per insertion: 2
view-aggregator blocks: 4
hypothesis latent dimension: 64
```

Linear adapters map between ParT's 192-dimensional particle representation and
the 256-dimensional hierarchy/fusion representation. The pseudo ParT weights
are shared across stochastic hypotheses but not shared with the HLT ParT.

Fusion is inserted after ParT layers 4 and 8 and immediately before final class
pooling. These locations correspond to early-middle, late-middle, and
preclassification interaction in the 12-layer primary branches.

### 3.2 Locked Primary Optimization Defaults

Use AdamW, bfloat16 autocast on supported GPUs, float32 accounting/projection
math, and float32 loss accumulation.

```text
AdamW beta1: 0.9
AdamW beta2: 0.95
AdamW epsilon: 1e-8
weight decay: 0.01
global gradient-norm clip: 1.0
learning-rate warmup: first 5% of updates
post-warmup schedule: cosine decay to 5% of peak LR
EMA decay for reconstructor selection/prediction: 0.9999
```

Primary peak learning rates:

```text
new root/hierarchy/renderer modules: 3e-4
HLT encoder during reconstruction pretraining: 3e-5
new fusion and view-aggregation modules: 2e-4
pseudo ParT during frozen-reconstructor tagger phase: 5e-5
reconstructor during joint tagger fine-tuning: 1e-5
HLT ParT upper blocks during joint fine-tuning: 1e-5
HLT ParT lower blocks during final full unfreeze: 3e-6
```

Use gradient accumulation to reach these effective global batch sizes without
reducing model capacity:

```text
root and deterministic hierarchy: 1024 jets/update
particle renderer and stochastic hierarchy: 512 jets/update
single-hierarchy fused tagger: 512 jets/update
dual-hierarchy multi-hypothesis tagger: 256 jets/update
```

Microbatch size is hardware-dependent and is recorded. Effective batch size is
fixed by accumulation. All primary runs use three independent training seeds;
pilot debugging may begin with one seed, but a selected claim requires three.

## 4. Non-Negotiable Contracts

### 4.1 Deployment Contract

No deployable forward path may load or receive:

```text
offline particles
offline hierarchy targets
offline group masks
offline teacher representations
offline teacher logits
cached pseudo particles generated with offline inputs
```

At model validation and final test, pseudo views must be generated from HLT
particles only. Offline data may be loaded in a separately named diagnostic
pass that cannot affect checkpoint selection, fusion fitting, thresholds, or
reported deployable predictions.

### 4.2 Consistency Contract

For the primary model, parent-child accounting is required by construction.
It is not an optional regularizer and cannot be disabled by a forgotten loss
weight.

The implementation may include an explicitly named unconstrained ablation,
but that ablation must use a distinct registry entry, output directory, and
report label.

### 4.3 Baseline Contract

The primary result is always measured against a clean HLT ParT that:

```text
uses the same split manifest
uses the exact same HLT cache content
uses the same class mapping
uses the same model-selection split
does not load reconstruction targets
does not receive zero-padded reconstruction fields
```

The fused model is initialized to reproduce this baseline before fusion
training begins.

### 4.4 Coordinate Contract

Every offline target is expressed in a coordinate frame available from HLT
information alone.

Primary frame:

```text
origin: HLT jet axis
eta coordinate: eta_particle - eta_HLT_jet
phi coordinate: wrapped(phi_particle - phi_HLT_jet)
orientation: no offline-derived rotation
```

The root predictor estimates the offline-axis displacement. The offline axis
must never be used to define an input coordinate transformation at inference.

## 5. Data Regime

Use the same 10-class JetClass labels, split semantics, and HLT-v2 degradation
contract as the comparison campaigns.

Primary high-data setting:

```text
HLT profile: fixed_hlt_v2_realistic
HLT degradation strength: 2.5
model_train: 5,000,000
model_val: 1,000,000
stack_train: 2,000,000
stack_val: 1,000,000
final_test: 1,000,000
```

Pilot setting:

```text
model_train: 500,000
model_val: 150,000
stack_train: 300,000
stack_val: 150,000
final_test: 150,000
```

On Tigris, the current dataset root is:

```text
/home/ryreu/atlas/PracticeTagging/data/jetclass_part1
```

The plan should support rebuilding inputs from an empty checkpoint root. Reuse
is allowed only after content hashes bind every artifact to the active split,
HLT cache, offline cache, target hierarchy, and source checkpoint.

Final test remains locked until the variant set, checkpoint selection, and
fusion membership are frozen from model-validation and stack-validation
results.

## 6. Terminology and Tensor Conventions

For each jet:

```text
X_hlt       HLT particle features
M_hlt       HLT particle validity mask
H_hlt       final HLT particle embeddings
z_hlt       pooled HLT jet representation

S_root      predicted offline-like root accounting state
T_l         active group tokens at hierarchy level l
M_l         active group mask at hierarchy level l
P_l         parent index for each active group at level l

X_pseudo[h] pseudo-particle hypothesis h
M_pseudo[h] pseudo-particle mask for hypothesis h
U_pseudo[h] uncertainty and provenance features for hypothesis h
```

Hierarchy levels are named by maximum active capacity:

```text
L1: K <= 2
L2: K <= 4
L3: K <= 8
L4: K <= 16
L5: K <= 32
```

The particle renderer follows L5 and produces at most the configured ParT
constituent limit. The primary limit is 128.

## 7. Offline Target Hierarchy

### 7.1 Primary Grouping: Recursive Exclusive-kT

The primary target builder creates a deterministic nested binary partition of
the offline constituents.

For each jet:

1. Place all valid offline constituents in the root group.
2. If a group contains one constituent, mark it terminal.
3. Otherwise run exclusive-kT on only that group's constituents and request
   exactly two nonempty subgroups.
4. Store the two children as an unordered sibling set.
5. Repeat for every nonterminal active group until the 32-group capacity is
   reached or every group is terminal.
6. Retain full constituent membership so particle targets can be supervised
   locally within each final microgroup.

Use generalized-kT exponent `p=1`, radius `R=0.8`, and E-scheme recombination.
The distance calculation is performed in the HLT-centered eta-phi frame with
wrapped phi.

Tie breaking must be deterministic and use stable jet identity plus original
constituent index. It must not use labels.

### 7.2 Complementary Grouping: Cambridge/Aachen

Build a second hierarchy with the same recursion, `R=0.8`, E-scheme
recombination, and `p=0`.
This hierarchy emphasizes angular locality and is expected to make different
errors around soft radiation and prong boundaries.

The C/A hierarchy is not silently substituted for the primary hierarchy. Its
targets, checkpoints, pseudo predictions, and reports use separate hashes and
variant names.

### 7.3 Why Not Reverse a Naive Clustering Tree

The implementation should not simply reverse the final merges of one global
agglomerative tree. That can make the first target split a hard core versus one
very soft wide-angle constituent.

Each parent is reclustered into exactly two occupied children. This gives every
stage a local, supervised bipartition task and keeps sibling matching small.

### 7.4 Target Masks and Variable Topology

Target masks distinguish three states:

```text
active_and_split
active_and_terminal
padding
```

Terminal groups are carried forward conceptually but are not decoded into an
empty sibling. In fixed tensors, a terminal node occupies one valid slot and
all unused slots are masked.

The model predicts a split/terminal distribution, but predicted topology is
also constrained by the predicted count budget:

```text
predicted parent count <= 1  -> terminal
predicted parent count >= 2  -> split is permitted
```

During early curriculum phases, target topology and parent masks are teacher
forced. During rollout training, predicted topology is used with a
straight-through or differentiable relaxed gate. Deployment uses hard valid
masks derived only from HLT-conditioned predictions.

### 7.5 Local Matching

Sibling order has no physical meaning. For every parent, compute both possible
two-child assignments and use the lower matching cost. This is the exact
two-element equivalent of Hungarian matching and avoids unstable global
ordering.

Matching cost uses normalized target quantities:

```text
four-momentum distance
angular centroid distance
constituent-count distance
type-composition distance
shape/covariance distance
constituent-membership transport distance where available
```

Matching is performed within each parent. A child may never match a target
belonging to another parent.

### 7.6 Group Geometry and Support

Each target group records:

```text
pT-weighted centroid in the HLT frame
2x2 eta-phi covariance
radial quantiles r50, r80, r95
maximum member radius
principal-axis orientation
member count
member indices
```

The covariance is represented by a Cholesky factor or positive eigenvalues
plus a bounded correlation. No rectangular cell is assigned. This support is
used for local cross-attention bias, diagnostics, and target matching.

## 8. Root Offline-Like State

### 8.1 Root Predictor Architecture

Use a strong HLT ParT encoder followed by a semantic-query transformer
readout. Do not compress the complete task into one pooled vector and one
anonymous MLP.

Learned root queries:

```text
q_p4           four-vector and axis correction evidence
q_count        multiplicity evidence
q_composition  particle-type count and momentum evidence
q_shape        angular width and covariance evidence
q_charge       charge and tracking evidence
q_uncertainty  shared aleatoric uncertainty evidence
```

Each query has a distinct type embedding and passes through three or more
cross-attention decoder blocks over all valid HLT particle embeddings. The
queries may self-attend to capture correlations. Pairwise HLT geometry is
available as an attention bias.

A shared root context token combines the semantic queries with `z_hlt`. The
semantic heads receive both their typed query and the shared context.

### 8.2 Stable Residual Parameterization

Predict physical residuals relative to HLT rather than unrelated absolute
targets:

```text
pT_root   = pT_hlt * exp(delta_log_pT)
eta_root  = eta_hlt + delta_eta
phi_root  = wrap(phi_hlt + delta_phi)
mass_root = (mass_hlt + epsilon) * exp(delta_log_mass)
```

The root four-vector is deterministically compiled from these quantities.
Use robust clipping only outside a wide physically observed range and record
the clip fraction as a diagnostic.

Multiplicity is a distribution over the bounded integer `N_offline`, with an
auxiliary residual view `delta_N = N_offline - N_hlt`. Use categorical or
ordinal likelihood, not rounded scalar regression.

Composition heads predict:

```text
type-count simplex over:
  charged_hadron
  neutral_hadron
  photon
  electron
  muon
  other

type-scalar-pT simplex over the same categories
```

Count fractions are compiled into exact integer type counts that sum to the
sampled or selected root constituent count. Use differentiable soft counts in
training and deterministic largest-remainder or minimum-cost allocation for
hard inference.

Shape heads predict residual log widths, a bounded correlation, radial
quantiles, and pT-dispersion statistics. The authoritative covariance uses a
positive-semidefinite parameterization.

### 8.3 Canonical Root Ledger

The typed physical heads are compiled into one versioned accounting schema.
The primary hard ledger contains:

```text
four-momentum:
  E, px, py, pz

discrete accounting:
  total constituent count
  count by particle type
  total integer charge

positive scalar accounting:
  scalar sum_pT
  scalar energy by particle type
  scalar pT by particle type
  absolute-charge sum

geometric accounting:
  pT-weighted first moments in HLT-centered eta and wrapped phi
  pT-weighted eta/phi second moments and cross moment
  pT-weighted radial first and second moments
```

Derived quantities such as jet mass, axis, type fractions, covariance, and
width are computed from this ledger. A derived quantity cannot be separately
treated as another authoritative prediction.

The ledger schema, transform parameters, feature order, units, and training
normalization statistics are saved and hashed.

### 8.4 Root Losses

Use distribution-aware losses:

```text
continuous residuals: heteroscedastic Gaussian or Student-t NLL
wrapped delta_phi: circular likelihood
count: categorical/ordinal NLL
composition: logistic-normal or Dirichlet-style simplex loss
covariance: PSD-aware normalized loss
derived observables: robust Huber diagnostics and auxiliary losses
```

Targets are normalized with robust model-train statistics. Validation reports
must convert every metric back to physical units.

Learned loss balancing may operate within bounded ranges, but cannot drive a
required accounting channel's weight to zero. Every required head has a fixed
minimum weight and a reported gradient norm.

## 9. Exact Binary Accounting

### 9.1 Four-Momentum Split

Every active binary split conserves the parent four-vector exactly.

The primary implementation is a differentiable two-body phase-space layer:

1. Transform the parent four-vector to its rest frame.
2. Predict valid child mass fractions with `m1 + m2 <= M_parent`.
3. Predict a unit split direction in the parent rest frame.
4. Compute the child momentum magnitude from two-body kinematics.
5. Construct back-to-back child four-vectors.
6. Boost both children to the HLT frame.

Therefore, up to floating-point tolerance:

```text
P_child_1 + P_child_2 = P_parent
```

Near-massless numerical cases require an explicit collinear-safe branch and
tests. Silently replacing the split with unconstrained vectors is forbidden.

### 9.2 Count and Composition Split

For a parent with integer count `N`, predict a distribution over valid child
counts:

```text
N1 in [1, N-1] when N >= 2
N2 = N - N1
```

For terminal `N=1`, no split occurs.

Type counts are allocated conditionally on the total child counts. The final
integer allocations must satisfy:

```text
N1_type + N2_type = N_parent_type for every type
sum_type N1_type = N1
sum_type N2_type = N2
```

Use a differentiable relaxed allocation during training and an exact
minimum-cost integer allocator for evaluation and pseudo-particle rendering.

### 9.3 Positive Scalar Budgets

For each nonnegative additive scalar channel `A`, predict sibling logits and
compile:

```text
f1, f2 = softmax(logit1, logit2)
A1 = f1 * A_parent
A2 = f2 * A_parent
```

This applies to scalar-pT, type-specific scalar momentum, energy summaries,
and absolute-charge accounting.

### 9.4 Signed Moment Budgets

Signed additive channels use a parent-preserving center-plus-residual form:

```text
A1 = w * A_parent + r
A2 = (1 - w) * A_parent - r
```

The residual is bounded by a physically derived scale. The exact sum is
independent of the learned residual.

Second-moment and covariance outputs must remain physically valid. Use the law
of total covariance to divide parent spread into within-child and between-child
components. Parameterize all within-child covariance matrices as PSD.

### 9.5 Hard Accounting Audit

Every forward pass exposes maximum and mean residuals for:

```text
four-momentum conservation
total count conservation
per-type count conservation
positive scalar conservation
first-moment conservation
second-moment consistency
```

Training fails closed on nonfinite values or residuals above a configured
numerical tolerance. Reports must not call a model constrained if these audits
are absent.

## 10. Multi-Hypothesis Conditional Reconstruction

### 10.1 Why It Is Required in the Best Model

The inverse map from HLT to offline particles is one-to-many. A deterministic
MSE predictor tends to average incompatible outcomes and can generate diffuse,
low-information pseudo particles.

The best model predicts a conditional distribution, while the deterministic
mean remains a required control.

### 10.2 Coherent Jet-Level Latent

For hypothesis `h`, sample one latent `z_h` from a learned conditional prior:

```text
p(z | H_hlt, z_hlt, semantic_root_queries)
```

The same latent is supplied to the root heads, every hierarchy split, and the
particle renderer for that hypothesis. Independent noise at every node is not
the primary design because it can create globally incoherent trees.

Primary hypothesis set:

```text
h=mean: deterministic conditional mean
h=1..4: four stochastic samples
```

The stochastic model is a conditional variational hierarchy with:

```text
training-only posterior q(z | HLT evidence, offline target hierarchy)
deployable prior p(z | HLT evidence)
64-dimensional z
8 rational-quadratic-spline coupling layers in the conditional prior
```

The posterior encoder is used only to train the conditional latent model and
is absent from deployable prediction. At inference, the mean view uses the
deterministic root path and the four stochastic views sample only from the
HLT-conditioned prior.

### 10.3 Training the Distribution

Use a proper conditional likelihood or variational objective, plus bounded
anti-collapse terms. Do not rely only on best-of-K reconstruction loss.

Recommended components:

```text
root and split negative log-likelihood
conditional latent KL or flow likelihood
energy score across complete pseudo views
small diversity reward in uncertain fine structure
penalty for diversity in hard-conserved totals
calibration loss for predicted intervals
```

Hypotheses may differ in fine structure but cannot violate the same sampled
root ledger. Diversity must not be created by changing energy or count outside
the predicted root uncertainty distribution.

### 10.4 Hypothesis Selection Is Forbidden

At deployment, no offline target may select the best hypothesis. The tagger
receives all configured hypotheses and their uncertainty, or uses a fixed
HLT-only deterministic aggregation rule learned on stack-train.

Oracle best-hypothesis selection is permitted only as a clearly labeled
model-validation diagnostic ceiling.

## 11. Hierarchical Reconstructor Architecture

### 11.1 Shared HLT Evidence

Use the strong HLT ParT as the evidence encoder. Preserve:

```text
all valid particle embeddings from multiple ParT depths
final pooled jet representation
pairwise geometry tensors
HLT particle masks
```

The hierarchy decoder must never depend only on its predicted parent. Every
stage can revisit the original HLT particles.

### 11.2 Level-Specific Binary Decoders

Use separate decoder parameters for `1->2`, `2->4`, `4->8`, `8->16`, and
`16->32`. Weight sharing is an ablation, not the primary model. Different
levels solve different spatial and compositional tasks and receive enough
capacity to specialize.

For each active parent, build a parent query from:

```text
parent accounting state
parent geometric support
parent uncertainty
root semantic tokens
hypothesis latent
hierarchy depth embedding
ancestor summary
```

Each binary decoder contains:

```text
parent/peer self-attention
global cross-attention to all HLT particles
support-biased local cross-attention to nearby HLT particles
cross-attention to all coarser hierarchy tokens
gated residual MLP blocks
typed output heads for split variables
```

Local attention is a bias, not a hard crop. Missing offline structure can lie
where no HLT particle remains, so every parent retains a global evidence path.

### 11.3 Support-Aware Attention Bias

For an HLT particle at displacement `d` from a predicted group center, compute
a Mahalanobis distance from the predicted support covariance. Add a learned,
bounded attention bias based on:

```text
Mahalanobis distance
delta_eta
wrapped delta_phi
delta_R
relative log_pT
particle type
hierarchy depth
```

The uncertainty head controls the strength of this local bias. A highly
uncertain group should attend more globally.

### 11.4 Error-Propagation Safeguards

To keep an early mistake from controlling the complete rollout:

```text
every level reattends to HLT particles
every level receives the root semantic tokens
parent predictions enter through gated residual paths
teacher-forced and predicted-parent batches are mixed during training
later levels receive auxiliary direct target supervision
gradient flow through early levels is clipped and monitored
```

The model does not replace an erroneous parent with an unconstrained child.
It can use new HLT evidence to choose the best child allocation still allowed
by the parent budget.

### 11.5 Group Token Output

Each active group token retains:

```text
compiled accounting state
learned hidden representation
support geometry
aleatoric uncertainty
epistemic/hypothesis disagreement summary
parent index
depth
terminal probability
```

These tokens are saved for hierarchy-aware tagging. They are not discarded
after rendering particles.

## 12. Variable-Length Pseudo-Particle Renderer

### 12.1 Renderer Input

The renderer operates independently inside each active L5 microgroup and
receives:

```text
microgroup accounting state
microgroup hidden token
complete ancestor path
global root tokens
hypothesis latent
all HLT particle embeddings
predicted local constituent count and type counts
```

It produces a local unordered set. The complete pseudo jet is the masked union
of every local set.

### 12.2 Slot Count

The number of active local particle slots is fixed by the compiled integer
count budget, not by an unconstrained existence threshold. A maximum of 128
particles is supported.

If the predicted root count exceeds the model maximum, the compiler applies a
documented overflow policy that preserves the highest expected pT content and
records the truncated count and budget. Overflow must be rare and reported.

### 12.3 Particle Features

Render the canonical ParT feature set required by the offline-style branch,
including:

```text
log pT and energy features
eta/phi or jet-relative angular features
mass-related features
charge
particle type probabilities or hard type identity
track-quality and impact-parameter features when predictable
neutral and electromagnetic indicators
```

Every rendered particle also carries side-channel metadata that is not
pretended to be a measured detector feature:

```text
prediction uncertainty
hypothesis identity
microgroup confidence
ancestor-depth confidence
predicted-observed support score
source flag: pseudo
```

The pseudo branch has an explicit input adapter for soft type probabilities
and uncertainty. Do not force uncertain categorical predictions into
overconfident hard one-hot values during training.

### 12.4 Exact N-Body Four-Momentum Projection

The local renderer first emits raw rest-frame momenta and particle attributes.
A differentiable N-body phase-space projection then enforces the microgroup
four-vector.

For a local set of `N` particles:

1. Center the raw three-momenta so their vector sum is zero in the parent rest
   frame.
2. Assign valid nonnegative effective masses from the particle-type-conditioned
   mass head.
3. Solve one positive common scale so the sum of child energies equals the
   parent invariant mass.
4. Construct rest-frame four-vectors whose spatial sum is zero.
5. Boost every particle to the HLT frame.

This guarantees:

```text
sum_i P_particle_i = P_microgroup
```

The solver must be differentiable, bracketed, numerically tested, and fail
closed on infeasible mass assignments. Raw effective masses are compiled
through a simplex allocation of at most `(1 - epsilon) * M_microgroup`, so the
sum of particle masses cannot exceed the parent invariant mass. Target masses
remain directly supervised. A massless renderer is a named ablation, not an
automatic fallback.

### 12.5 Exact Discrete Accounting

Particle type identities are assigned with a differentiable relaxed transport
during training and an exact minimum-cost allocation during evaluation. The
hard output satisfies the microgroup type-count budgets exactly.

Charge assignments satisfy:

```text
allowed charge for selected particle type
sum particle charges = microgroup integer charge
```

If a predicted budget is infeasible under allowed particle types, the root or
group compiler must repair it before rendering and record the repair. The
renderer cannot silently violate the budget.

### 12.6 Local Particle Matching Loss

Because target membership is known, match predicted particles only to the
offline constituents inside their target microgroup.

Use entropically regularized optimal transport or Hungarian matching with a
cost containing:

```text
log-pT error
eta/phi and delta_R error
four-vector error
particle-type cross entropy
charge mismatch
track-feature error where supervised
uncertainty-normalized residual
```

Local matching is substantially easier and more stable than one global
128-by-128 assignment.

### 12.7 Particle-Level Auxiliary Observables

After rendering, recompute and supervise:

```text
jet four-vector
constituent count and type counts
energy correlation functions
N-subjettiness summaries
radial pT profile
leading and subleading pT fractions
charge and track summaries
particle-set energy distance
```

Hard ledger quantities should already match by construction. These losses
teach useful fine structure and expose bugs in the renderer.

## 13. Reconstruction Objectives

### 13.1 Required Objective Groups

The complete reconstruction objective is:

```text
L_reco =
    lambda_root * L_root
  + sum_l lambda_group[l] * L_group[l]
  + lambda_topology * L_topology
  + lambda_particle * L_particle_OT
  + lambda_feature * L_particle_features
  + lambda_distribution * L_conditional_distribution
  + lambda_calibration * L_uncertainty_calibration
  + lambda_aux * L_derived_observables
```

After per-family normalization, the locked deterministic primary weights are:

```text
lambda_root: 1.00
lambda_group[2]: 1.00
lambda_group[4]: 1.00
lambda_group[8]: 0.75
lambda_group[16]: 0.50
lambda_group[32]: 0.50
lambda_topology: 0.50
lambda_particle: 1.00
lambda_feature: 0.50
lambda_distribution: 0.25
lambda_calibration: 0.10
lambda_aux: 0.25
```

For probabilistic training, `lambda_distribution` begins at `0.05` for the
first 10% of updates and linearly rises to `0.25`. The bounded diversity term
inside `L_conditional_distribution` has weight `0.05`; hard-total diversity
penalties have weight `1.0` after normalization.

During joint tagger fine-tuning:

```text
lambda_reco_joint: 0.10
HLT anchor CE weight: 0.20
pseudo-only auxiliary CE weight in F3 only: 0.10
hierarchy-only auxiliary CE weight in F3 only: 0.05
offline logit KD weight in F4/F5 only: 0.25
offline logit KD temperature in F4/F5 only: 2.0
```

The label loss is ordinary 10-class CE with no class weighting because the
campaign subsets are balanced. Label smoothing is `0.02` for A0 and every
deployable tagger so it cannot become a fusion-only advantage.

Hard accounting residuals are audited but are not expected to require a
learned penalty in the primary model. A small numerical residual term may be
used to detect gradient or precision problems.

### 13.2 Deep Supervision

Every hierarchy level has direct offline target supervision. Later levels are
not trained only through the final particle loss. This ensures that the
`2/4/8/16/32` tokens retain meaningful, reportable semantics.

### 13.3 Loss Scaling

Normalize physical residuals using model-train statistics. Use bounded learned
task weighting or GradNorm only after fixed minimum weights are applied.

Report per-objective:

```text
raw loss
normalized loss
effective weight
gradient norm
nonfinite/skip count
```

### 13.4 No Label Leakage Into Reconstruction Targets

The reconstructor may be jointly optimized with tagging CE in later phases,
but hierarchy construction, matching, and accounting targets must not use the
class label. A class-conditional generator is outside this plan because labels
are unavailable at deployment.

## 14. Hierarchy-Aware Tagger

### 14.1 Three Information Streams

The strongest tagger consumes:

```text
1. trusted original HLT particle stream
2. one or more pseudo-offline particle streams
3. multiscale hierarchy-token memory
```

The hierarchy stream is essential. Coarse root and group predictions are
usually more reliable than exact particle hallucinations and should remain
available to classification.

### 14.2 HLT Stream

Initialize the HLT stream from the selected clean HLT ParT checkpoint. Preserve
its original input dimensionality and preprocessing exactly.

The HLT stream must be capable of producing the original baseline logits when
all fusion residuals are zero.

### 14.3 Pseudo Stream

Use a separate ParT encoder for pseudo particles. Do not share all weights with
the HLT stream because measured HLT particles and predicted particles have
different noise and uncertainty semantics.

When feature contracts permit, initialize the pseudo encoder from a strong
offline ParT. Add an input adapter for uncertainty, soft PID, source identity,
and hypothesis metadata. The offline warm start is an initialization only; the
deployable pseudo stream still receives HLT-generated particles.

### 14.4 Ancestor Injection

For every pseudo particle, gather its ancestor tokens at capacities
`2/4/8/16/32`. Project each depth separately and combine them with a learned
depth gate:

```text
h_particle_augmented = h_particle
                     + sum_l gate_l * project_l(h_ancestor_l)
```

The gate depends on ancestor and particle uncertainty. Parent identities are
used only as connectivity; arbitrary numeric group IDs are never embedded as
physical categories.

### 14.5 Tree-Relative Attention Bias

Pseudo-pseudo attention receives a learned bias based on the deepest shared
ancestor. Particles sharing a recent microgroup can interact differently from
particles that meet only at the root.

Recommended relation features:

```text
deepest shared depth
tree path distance
same/different hypothesis
ancestor support overlap
relative uncertainty
```

### 14.6 Bidirectional Cross-View Fusion

Insert cross-attention after multiple ParT stages rather than only at logits.
The primary configuration uses three fusion locations:

```text
after early particle embedding refinement
after a middle particle-attention block
before final jet pooling/classification
```

At each location:

```text
HLT queries attend to pseudo particles and hierarchy tokens
pseudo queries attend to HLT particles
```

Cross-view attention biases include:

```text
delta_eta, wrapped delta_phi, delta_R
relative log-pT
predicted correspondence/support score
pseudo uncertainty
hypothesis disagreement
shared hierarchy support
```

### 14.7 Trust Gates

Every pseudo-to-HLT update passes through a bounded trust gate conditioned on:

```text
pseudo uncertainty
group uncertainty
hypothesis disagreement
local HLT support
hierarchy depth
current HLT token
```

The gate is initialized near zero. The model begins as the trusted HLT
baseline and learns where pseudo information is useful.

### 14.8 Baseline-Preserving Residual Classifier

Let `z_base` and `logits_base` be the original HLT representation and logits.
The fused classifier computes:

```text
delta_z = FusionHead(HLT, pseudo, hierarchy)
z_fused = z_base + alpha * delta_z
logits_fused = HLTClassifier(z_fused) + beta * delta_logits
```

Initialize `alpha`, `beta`, and the final fusion projections to zero or a very
small value. Verify numerically before training that the fused model matches
the baseline logits within tolerance.

### 14.9 Multiple-Hypothesis Aggregation

Do not concatenate all hypotheses into one undifferentiated particle set.
Encode each pseudo view with shared pseudo-branch weights, preserve a hypothesis
token, then aggregate hypothesis representations and particles with a
permutation-invariant view transformer.

The aggregator receives uncertainty and disagreement. It can learn:

```text
consensus across hypotheses
useful minority structures
regions where all hypotheses are unreliable
```

The HLT stream remains available during this aggregation.

### 14.10 Dual-Hierarchy Fusion

The maximum-performance candidate uses one trusted HLT stream plus selected
exclusive-kT and C/A pseudo streams. The two reconstructors may share the HLT
evidence encoder but have separate target-specific hierarchy decoders.

Use a view-type embedding and view-level transformer to combine them. Train
with view dropout so the model remains stable when one pseudo family is weak.

## 15. Tagging Objectives and Ablations

### 15.1 Primary Objective

The primary tagger objective deliberately excludes offline-teacher KD:

```text
L_primary = L_fused_label_CE
          + 0.20 * L_hlt_anchor_CE
          + lambda_reco_joint * L_reco
```

During the frozen-reconstructor phase, `L_reco` is reported but has no
trainable reconstructor path. During gentle joint fine-tuning it prevents the
reconstructor from drifting into an unconstrained class shortcut.

### 15.2 Optional Branch Losses

Test, but do not assume the benefit of:

```text
HLT-only auxiliary CE
pseudo-only auxiliary CE
hierarchy-only auxiliary CE
cross-view agreement loss
view dropout
```

The `0.20` HLT-anchor CE term is part of the locked primary recipe. Pseudo-only
and hierarchy-only CE remain ablations because they can force uncertain
predictions to carry more classification burden than is useful.

### 15.3 KD Is an Explicit Ablation

Offline-teacher logit KD has not consistently helped prior campaigns. It is
therefore tested in paired variants:

```text
CE only
CE + maintained reconstruction
CE + offline logit KD
CE + maintained reconstruction + offline logit KD
```

Use identical initialization, schedule, and seed for the paired comparison.
Do not use representation KD in the primary campaign.

### 15.4 From-Scratch Comparisons

Warm starts are expected to perform best, but include:

```text
HLT branch from scratch
pseudo branch from scratch
both branches from scratch
reconstructor and tagger joint from start
```

These determine whether the gain comes from reconstructed structure or simply
from a favorable fine-tuning schedule.

## 16. Training Curriculum

### Phase 0: Inputs, Baselines, and Ceilings

1. Build and audit the split manifest.
2. Build HLT-v2 strength-2.5 caches.
3. Build paired offline caches for training supervision.
4. Train or validate a clean HLT ParT baseline.
5. Train or validate an offline ParT ceiling.
6. Cache offline hierarchy targets for both grouping definitions.
7. Confirm exact labels, jet identities, and content hashes across all inputs.

No final-test offline hierarchy cache is required for deployable training.

### Phase 1: Root Pretraining

Train the HLT evidence encoder and semantic root queries on root targets only.

Start from the HLT ParT checkpoint for the primary variant. Use a smaller
learning rate for the ParT body and a larger rate for new semantic queries and
heads.

Select on a composite model-validation score that includes calibrated root
NLL and normalized physical errors. Do not use tagging accuracy yet.

Train for at most 150,000 optimizer updates. Evaluate every 2,000 updates and
stop after 12 evaluations without improvement. Restore the best EMA checkpoint.

### Phase 2: Progressive Hierarchy Curriculum

Add levels in order:

```text
1 -> 2
2 -> 4
4 -> 8
8 -> 16
16 -> 32
```

For each new level:

1. Begin with true parent accounting and masks.
2. Train the new decoder and keep earlier levels trainable at a lower rate.
3. Mix true and predicted parent states.
4. Increase predicted-parent probability according to validation stability.
5. End with complete predicted rollout.

Use a schedule based on update counts, not only wall-clock epochs. Save the
best checkpoint at every depth.

Allocate at most 80,000 updates to each newly introduced depth. Use true
parents for the first 20%, linearly move from 100% to 25% teacher forcing over
the next 50%, and use 25% teacher-forced plus 75% rolled-out batches for the
last 30%. Complete zero-teacher-forcing validation is required throughout.

### Phase 3: Deterministic Particle Rendering

Train the local particle renderer first with true L5 parent groups, then with
predicted groups. Maintain all hierarchy losses.

Selection emphasizes both local transport quality and complete pseudo-jet
quality. A renderer that improves set matching while worsening the compiled
jet state is rejected.

Train for at most 200,000 updates with model-validation every 2,000 updates and
patience of 15 evaluations.

### Phase 4: Probabilistic Multi-Hypothesis Training

Initialize from the deterministic hierarchy. Add the conditional latent prior,
probabilistic heads, and four stochastic hypotheses.

Train complete coherent rollouts. Monitor mode collapse, calibration, and
hypothesis diversity in class-independent physical observables.

Train for at most 200,000 updates. Use the deterministic checkpoint for all
shared weights and introduce the posterior/prior modules with a 10,000-update
warmup before stochastic loss reaches full weight.

### Phase 5: Frozen-Reconstructor Tagger

Freeze the reconstructor. Initialize:

```text
HLT stream from the clean HLT baseline
pseudo stream from the offline ParT when compatible
fusion modules near zero
```

Train the pseudo input adapter, hierarchy memory, cross-attention, trust gates,
view aggregator, and residual classifier. Confirm epoch zero reproduces the
HLT baseline.

Train for at most 120,000 updates with model-validation every 1,000 updates and
patience of 15 evaluations.

### Phase 6: Staged Unfreezing

Unfreeze in this order:

```text
1. pseudo ParT upper blocks
2. hierarchy-to-tagger projections
3. hierarchy decoder upper levels and particle renderer
4. HLT ParT upper blocks
5. complete model with very small encoder learning rates
```

Maintain reconstruction supervision whenever reconstructor parameters are
trainable. Use separate optimizer groups and report every group learning rate.

Use 30,000 updates for each unfreeze stage and 100,000 updates for the final
full gentle-unfreeze stage. Restore the best model-validation checkpoint across
the complete staged run, not merely the final stage.

### Phase 7: Objective Ablations

From the same selected initialization, train CE-only, auxiliary-loss, KD, and
from-scratch variants. Model selection remains based on model-validation CE or
accuracy according to one frozen campaign rule.

### Phase 8: Multi-Hypothesis and Dual-Hierarchy Fusion

Train the four-view aggregator, then the combined exclusive-kT/C/A model. Use
stack-train only for any post-hoc scalar/logit fusion. Neural early fusion is
trained on model-train and selected on model-val.

### Phase 9: Final Claims

Freeze:

```text
selected checkpoint hashes
selected variant names
hypothesis count and sampling seed policy
fusion membership and weights
preprocessing and target schema hashes
```

Only then run one approved final-test deployable pass. Offline final-test
targets may be evaluated afterward in a separate non-selection diagnostic job.

## 17. Campaign Registry

The registry is versioned and machine-readable. Every run report stores the
complete resolved configuration, not only a short variant name.

### Tier A: Baselines and Ceilings

`A0_hlt_part`:

Clean large HLT ParT baseline using the locked dimensions in Section 3.1. No
reconstruction caches are required by its dataset or forward path.

`A0b_hlt_part_canonical_base`:

Canonical base HLT ParT using `[128, 512, 128]`, eight particle-attention
layers, and two class-attention layers. This preserves comparison to older HLT
campaigns but is not the trusted branch of the maximum-performance model.

`A1_hlt_schedule_control`:

Warm-start A0 and apply the same number of optimizer updates and HLT-branch
unfreeze schedule as the primary fused model, without pseudo or hierarchy
inputs.

`A2_hlt_capacity_control`:

HLT-only large ParT with additional particle-attention and feed-forward blocks
chosen so its trainable parameter count is within 5% of E5 excluding the
frozen reconstructor. It uses the same update budget and label objective as E5.

`A3_hlt_from_scratch`:

Same clean HLT model trained from scratch.

`A4_offline_part_ceiling`:

Offline ParT trained and evaluated on offline particles. It is a ceiling, not
a deployable baseline.

### Tier B: Root Prediction

`B0_pooled_mlp_root`:

Pooled HLT representation plus MLP root predictor.

`B1_semantic_query_root`:

Primary typed semantic-query deterministic root predictor.

`B2_semantic_query_probabilistic`:

Probabilistic semantic root with calibrated uncertainty.

`B3_root_multi_hypothesis`:

One mean plus four coherent stochastic root states.

`B4_oracle_root_diagnostic`:

True offline root state supplied on model-val only. Never a deployable claim.

### Tier C: Adaptive Hierarchy Depth and Target Definition

`C0_direct_8_group_set`:

Predict eight groups directly from the root. This tests the benefit of binary
curriculum and local sibling matching.

`C1_kt_2`:

Exclusive-kT hierarchy through capacity 2.

`C2_kt_4`:

Exclusive-kT hierarchy through capacity 4.

`C3_kt_8`:

Exclusive-kT hierarchy through capacity 8.

`C4_kt_16`:

Exclusive-kT hierarchy through capacity 16.

`C5_kt_32`:

Primary deterministic exclusive-kT hierarchy through capacity 32.

`C6_ca_32`:

Depth-matched Cambridge/Aachen hierarchy.

`C7_shared_level_weights`:

Exclusive-kT hierarchy with shared decoder weights across levels.

`C8_unconstrained_split_control`:

Explicit unconstrained child prediction control. It must never be reported as
the constrained primary method.

`C9_oracle_parent_rollout`:

True parent groups supplied at every level on model-val. This measures error
propagation and is not deployable.

### Tier D: Particle Reconstruction

`D0_kt32_mean_particles`:

Deterministic particle renderer from C5.

`D1_kt32_mh4_particles`:

Primary C5 hierarchy with one mean and four stochastic pseudo views.

`D2_ca32_mh4_particles`:

C/A counterpart to D1.

`D3_global_particle_set`:

One global particle set decoder without group-local matching.

`D4_no_nbody_projection`:

Renderer without exact N-body four-momentum projection. Diagnostic control.

`D5_oracle_groups_particles`:

Particle renderer receives true L5 groups on model-val. Measures renderer
ceiling.

`D6_true_offline_particles`:

True offline particles passed to the pseudo branch on model-val only. Measures
tagger architecture ceiling and is never a deployable result.

### Tier E: Tagger Fusion

`E0_pseudo_only`:

Pseudo-particle ParT without HLT particles.

`E1_hlt_pseudo_logit_mean`:

Independent HLT and pseudo taggers with untrained logit averaging.

`E2_late_representation_fusion`:

Fuse pooled HLT and pseudo representations before classification.

`E3_single_cross_attention`:

One mid-network cross-attention block, no hierarchy tokens.

`E4_hierarchy_memory_fusion`:

E3 plus multiscale hierarchy tokens and ancestor injection.

`E5_kt32_mh4_dualcross`:

Primary ABPH-KT32-MH4-DUALCROSS model with three cross-fusion locations,
hierarchy memory, uncertainty gates, and baseline-preserving residual output.

`E6_ca32_mh4_dualcross`:

C/A counterpart to E5.

`E7_dual_hierarchy_dualcross`:

One HLT stream fused with selected exclusive-kT and C/A pseudo streams. This
is the expected maximum-performance single model.

`E8_no_hierarchy_tokens`:

E5 without hierarchy-token memory or ancestor injection.

`E9_no_uncertainty_gates`:

E5 with fixed ungated pseudo updates.

`E10_no_baseline_residual_init`:

E5 without zero-initialized baseline-preserving fusion.

### Tier F: Training Objective and Initialization

All F variants start from the same selected E5 or E7 configuration unless the
name says otherwise.

`F0_ce_reco_primary`:

Label CE plus maintained reconstruction supervision. Primary recipe.

`F1_ce_only_frozen_reconstructor`:

Frozen reconstructor and label CE only.

`F2_ce_only_joint`:

Jointly train reconstructor and tagger with CE but no maintained reconstruction
loss. This is a drift diagnostic.

`F3_ce_reco_branch_aux`:

Primary recipe plus HLT/pseudo/hierarchy auxiliary CE.

`F4_ce_logit_kd`:

Label CE plus offline-teacher logit KD, no representation KD.

`F5_ce_reco_logit_kd`:

Primary recipe plus offline-teacher logit KD.

`F6_hlt_from_scratch`:

Primary recipe with HLT branch from scratch.

`F7_pseudo_from_scratch`:

Primary recipe with pseudo branch from scratch.

`F8_all_from_scratch`:

Both tagger branches from scratch after reconstructor pretraining.

`F9_joint_from_start`:

Reconstructor and tagger jointly trained from their initial states without the
frozen-reconstructor fusion phase.

### Tier G: Multi-View and Ensemble Results

`G0_kt_hypothesis_aggregator`:

Learned aggregation of E5's mean plus four stochastic hypotheses.

`G1_kt_ca_early_fusion`:

E7 dual-hierarchy particle and hierarchy-token fusion.

`G2_kt_ca_logit_fusion`:

Stack-train-fitted scalar or classwise fusion of E5 and E6 logits.

`G3_particle_and_logit_fusion`:

Early-fused E7 combined at logits with the strongest complementary independent
seed/model selected on stack-val.

`G4_seed_ensemble_primary`:

Three independently seeded F0 models with frozen stack-train fusion.

`G5_best_complementary_ensemble`:

Predeclared maximum-performance ensemble using diversity-aware membership from
E5, E6, E7, and selected objective/seed variants. Membership is chosen without
final test.

## 18. Expected Ranking and Main Scientific Comparisons

Expected strongest single model:

```text
E7_dual_hierarchy_dualcross trained with F0_ce_reco_primary
```

Expected strongest deployable result:

```text
G5_best_complementary_ensemble
```

Most important comparisons:

```text
A0 vs A1/A2:
  separate reconstruction gain from schedule and capacity gain

C0 vs C3/C5:
  test direct group prediction against progressive binary prediction

C1/C2/C3/C4/C5:
  determine useful hierarchy depth

C5 vs C6:
  compare exclusive-kT and C/A target semantics

D0 vs D1:
  measure value of probabilistic hypotheses

E3 vs E4/E5:
  measure hierarchy information and deeper fusion

E5 vs E7:
  measure complementary hierarchy fusion

F0 vs F4/F5:
  test KD rather than assuming it helps

F0 vs F6/F7/F8/F9:
  identify warm-start and curriculum effects
```

## 19. Diagnostics

### 19.1 Root Diagnostics

Report overall and per class:

```text
delta_log_pT bias, resolution, and coverage
axis-shift eta/phi bias and resolution
mass residual
count exact accuracy, MAE, and confusion matrix
type-count and type-pT fraction errors
charge error
shape/covariance errors
uncertainty coverage and calibration
```

### 19.2 Hierarchy Diagnostics

For every depth:

```text
active group count
split/terminal accuracy
sibling matching cost
four-momentum error before and after projection
count and type-count accuracy
centroid and support error
per-group pT response and resolution
parent-child hard accounting residuals
teacher-forced versus rolled-out degradation
```

Plot error growth from root through 32 groups. This directly tests whether the
binary curriculum controls compounding error.

### 19.3 Particle Diagnostics

Report:

```text
local and global OT cost
particle count and type-count accuracy
per-particle pT and angular residuals after matching
jet four-vector closure
radial profile and energy-correlation agreement
leading-particle response
soft-particle response
particle-type confusion
uncertainty-stratified errors
```

### 19.4 Hypothesis Diagnostics

Report:

```text
pairwise hypothesis distance
coverage of the true offline set
calibration of root and local intervals
best-of-K oracle diagnostic on model-val
mean aggregation versus learned aggregation
diversity in hard totals, which should remain bounded
diversity in fine structure, which should be nonzero
```

### 19.5 Tagger Diagnostics

Report at minimum:

```text
accuracy and loss
macro one-vs-rest AUC
per-class accuracy and AUC
confusion matrix
gap versus A0, A1, A2, and A4
HLT-only, pseudo-only, hierarchy-only, and fused logits
top-1 agreement between branches
uncertainty-binned accuracy
trust-gate distributions by class and hierarchy depth
performance versus HLT constituent count and degradation severity
```

### 19.6 Fusion Diagnostics

Measure whether the model uses reconstructed information:

```text
zero all pseudo particles at evaluation
shuffle pseudo views between jets
shuffle hypotheses within a jet
remove hierarchy tokens
remove one hierarchy depth at a time
replace pseudo particles with matched noise
disable cross-attention at each fusion location
force trust gates to zero or one
```

These are diagnostic evaluation passes. They do not alter selected checkpoints.

## 20. Provenance, Leakage, and Reuse Safety

Every artifact records and validates:

```text
split manifest hash
per-split jet identity hash
label hash and class mapping hash
HLT cache content hash
HLT profile, version, strength, and parameter hash
offline cache content hash where training targets are used
hierarchy target content and schema hash
grouping algorithm and parameter hash
root ledger schema and normalization hash
source checkpoint hash
resolved variant configuration hash
source git commit and dirty-status hash
```

Prediction and report reuse must recompute file hashes and bind them to the
currently active input hashes. A self-consistent stale artifact is not enough.

Final-test deployable reports additionally attest:

```text
offline inputs loaded: false
teacher logits loaded: false unless an explicitly approved training-only cache
hypothesis selection used offline target: false
fusion fitted on final_test: false
```

Any missing required provenance field makes the campaign report fail.

## 21. Reporting and Selection

Use model-val for neural checkpoint selection. Use stack-train and stack-val
only for declared post-hoc fusion fitting and selection. Do not use final test
to choose:

```text
hierarchy depth
grouping definition
hypothesis count
objective weights
fusion membership
ensemble weights
random seed
```

The final report must fail closed on:

```text
missing required primary runs
failed run_report status
unavailable metrics represented as an empty mapping
provenance conflicts or missing hashes
unexpected final-test access
fusion membership differing from its frozen declaration
```

Write machine-readable JSON plus CSV tables for all tiers, accounting audits,
reconstruction diagnostics, and tagging metrics.

## 22. Implementation Steps

### Step 1: Schemas, Registry, and Input Contracts

Implement:

```text
versioned root ledger schema
group and particle target schemas
variant registry A0-G5
resolved configuration hashing
HLT/offline/split contract validation
clean HLT-only dataset path
```

Acceptance tests:

```text
schema round trip
feature-order hash changes on any schema change
HLT-only baseline runs without target caches
wrong split/cache/profile fails before training
all registry names resolve to complete configurations
```

### Step 2: Adaptive Binary Target Builder

Implement recursive exclusive-kT and C/A hierarchy construction, target masks,
terminal handling, local membership, deterministic tie breaking, and cache
metadata.

Acceptance tests:

```text
every active child is nonempty
siblings exactly partition parent membership
all target particles occur once per frontier
terminal singleton never creates a physical empty sibling
phi wrapping is correct
same inputs produce byte-identical target identities
```

### Step 3: Semantic Root Predictor and State Compiler

Implement typed semantic queries, probabilistic physical heads, residual target
transforms, exact integer composition compiler, ledger conversion, and root
metrics.

Acceptance tests:

```text
physical transforms invert target preprocessing
compiled four-vector matches predicted physical root
type counts sum exactly to total count
all fractions are valid
no offline input is required for deployable root prediction
```

### Step 4: Exact Binary Accounting Layers

Implement two-body four-vector splitting, integer count/type allocation,
positive scalar allocation, signed moment partition, PSD covariance handling,
and hard accounting audits.

Acceptance tests:

```text
random valid parents conserve all hard channels
gradients are finite
near-massless parents use the documented safe branch
infeasible inputs fail or are explicitly repaired and counted
mixed precision remains within tolerance
```

### Step 5: Recursive Hierarchy Decoder

Implement level-specific parent transformers, global/local HLT cross-attention,
support bias, local sibling matching, topology masks, teacher forcing, and
predicted rollout.

Acceptance tests:

```text
shapes and masks are correct at every capacity
padding cannot receive attention or contribute loss
children only match within their parent
every level can access original HLT embeddings
teacher-forced and rollout modes are separately reported
```

### Step 6: Multi-Hypothesis Distribution

Implement conditional latent sampling, deterministic mean output, coherent
latent propagation, likelihood/calibration losses, fixed evaluation sampling,
and anti-collapse diagnostics.

Acceptance tests:

```text
same fixed seed reproduces hypotheses
different hypotheses preserve sampled hard root budgets
offline targets never select deployment hypotheses
mean and stochastic outputs have distinct report identities
```

### Step 7: Constrained Particle Renderer

Implement local set queries, exact slot count, soft/hard PID assignment,
N-body phase-space projection, local OT/Hungarian loss, complete pseudo-jet
assembly, and particle diagnostics.

Acceptance tests:

```text
local particle four-vectors sum to parent
complete pseudo jet sums to root
counts, types, and charges close exactly
local matching cannot cross group boundaries
final-test rendering uses HLT inputs only
```

### Step 8: Reconstructor Curriculum Trainer

Implement phases 1-4, optimizer groups, staged depth growth, teacher-forcing
schedule, nonfinite guards, checkpoint selection, restart safety, and complete
run reports.

Acceptance tests:

```text
short overfit test at each depth
resume reproduces schedule state
optimizer metadata records all learning rates/trainability
required losses cannot silently disappear
rollout validation is always produced
```

### Step 9: Deployable Pseudo-View Prediction and Caches

Implement HLT-only pseudo prediction for mean and stochastic views, hierarchy
token caches, uncertainty caches, content hashes, and memory-bounded shard
writing.

Acceptance tests:

```text
offline cache path is rejected in deployable mode
shards bind to checkpoint and HLT cache hashes
reassembled predictions preserve jet order and identity
partial/stale shards fail reuse validation
high-data generation is streaming and memory audited
```

### Step 10: Hierarchy-Aware Dual-Stream Tagger

Implement separate ParT streams, offline-compatible pseudo warm start,
ancestor injection, tree biases, hierarchy memory, multi-location bidirectional
cross-attention, uncertainty gates, view aggregation, and zero-init residual
classification.

Acceptance tests:

```text
zero-init logits reproduce A0
masked pseudo/hierarchy inputs cannot affect logits
tree relations are permutation-equivariant
hypothesis order does not affect aggregate output
HLT-only final-test path plus generated pseudo views remains deployable
```

### Step 11: Objective, Fusion, and Campaign Reporting

Implement F-tier paired objectives, dual-hierarchy fusion, stack-only post-hoc
fusion, diagnostic ablations, full provenance comparison, and fail-closed
reports.

Acceptance tests:

```text
KD variants fail if teacher logits are missing
non-KD primary variants never load teacher logits
all required A0-G variants are checked
final metrics require successful run reports
frozen fusion membership cannot be overridden
```

### Step 12: Slurm Campaign Orchestration

Implement pilot, approved high-data, prediction, fusion, diagnostic, report,
and final-claims submitters for Tigris and tier3-style environments.

The graph must support an empty checkpoint root and submit in this order:

```text
splits
-> HLT/offline caches
-> baselines and target hierarchies
-> root/hierarchy reconstructors
-> particle renderers
-> deployable pseudo predictions
-> taggers and objective ablations
-> fusion/ensemble jobs
-> report
```

Acceptance tests execute representative environment combinations, including
partial rebuilds, rather than only searching submitter source text.

High-data and final-test stages require approval inside the canonical
submitter, not only in convenience wrappers.

## 23. Queue Strategy

First queue the complete pilot through model-validation reporting without
final-test claims. Use pilot results to verify:

```text
root accuracy
error growth by depth
particle closure
hypothesis calibration
whether hierarchy tokens are used
whether E5 beats A0/A1/A2
```

The high-data campaign may run concurrently when explicitly approved, but its
final claims remain blocked until the pilot and high-data selection reports are
valid.

Reconstruction stages should use enough CPU memory for streaming target and
pseudo caches. Training jobs should use the largest practical GPU memory and
avoid reducing model capacity merely to fit a low-memory queue when a larger
partition is available.

## 24. Success Criteria

### Mechanical Readiness

```text
all unit and integration tests pass
hard accounting residuals meet tolerance
pilot graph completes from an empty root
deployment prediction loads no offline information
reports fail closed on missing or stale artifacts
```

### Scientific Readiness

Before claiming improvement:

```text
E5 or E7 beats A0, A1, and A2 on model_val and stack_val
gain persists across at least three seeds for the selected recipe
gain is not reproduced by the schedule/capacity controls
pseudo shuffle and zeroing remove the expected portion of the gain
trust gates correlate with reconstruction uncertainty
hard accounting remains exact after joint fine-tuning
```

### Performance Goal

The primary goal is a statistically stable gain over the clean HLT ParT. The
preferred result is not merely better reconstruction loss; it is better
10-class tagging while retaining the original HLT branch as a trusted anchor.

The expected route to the largest gain is:

```text
strong HLT warm start
+ semantic residual root prediction
+ exact recursive binary accounting
+ four coherent pseudo-offline hypotheses
+ exclusive-kT and C/A complementary hierarchies
+ constrained variable-length pseudo particles
+ hierarchy-aware early/mid fusion
+ uncertainty gates
+ CE and maintained reconstruction training
```

KD, representation matching, and post-hoc logit ensembles are supporting
experiments. The central model must be able to beat HLT ParT without relying on
them.
