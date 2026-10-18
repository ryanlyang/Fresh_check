# Five-run EdgeValue study: message content, routing, and aggregation

Status: implemented as a separate five-run study; see the
[tier3 queue guide](../part_ev_message_study/README.md). Models and launchers are
locally tested; GPU admission still requires cluster preflight. No jobs have
been submitted here. Richer TRACK/PT/REGION relations are deferred.

## Recommendation

Run exactly **five new seed-101 models**, all with the same shortened Ranger
recipe. Reuse the existing ParT and ordinary-EV controls; those are not two
additional training runs.

1. **State-conditioned EV:** make the added pair message depend nonlinearly on
   the current representations of both particles, not just their fixed pair
   features or one pooled jet context.
2. **Separate-encoder EV:** give the value pathway its own trainable pair encoder,
   initialized as a copy of the ordinary EV encoder, while retaining ParT's
   existing scalar-bias pathway.
3. **Their combination:** use the separate pair representation in both ordinary
   EV and its state-conditioned residual.
4. **EV-specific routing:** let EV messages use their own corrected attention
   weights, while native particle values retain ordinary ParT attention.
5. **Variance-augmented EV:** retain ordinary EV and add the attention-weighted
   spread of pair features, rather than only their weighted average.

My higher-upside architectural hypothesis is (1); (2) is a particularly clean,
small-parameter test of representation sharing. Their combination is worth a
prespecified run, not an assumed winner. The last two runs test different
mechanisms, rather than additional combinations or optimizer choices. None has
demonstrated gains yet.

Use the [reusable short-budget protocol](REUSABLE_SCREEN_PROTOCOL.md), existing
`PART_RANGER` and `EV_RANGER` controls, and fresh seed-101 initialization. Start
with Ranger only: changing the optimizer again would obscure these questions.

## 1. What limitation are we testing?

Let `e_ij` denote the four native kinematic pair features, `z_ij` the shared
64-channel pair embedding, and `x_i^l` the 128-channel particle state supplied to
particle attention in block `l`. There are eight particle blocks, eight heads,
and 16 value channels per head.

Ordinary EV currently uses, for each head `h`:

\[
z_{ij}=\phi(e_{ij}),\qquad
m_{ij}^{\ell h}=W_{\ell h}z_{ij},\qquad
\Delta y_i^{\ell h}=\sum_j a_{ij}^{\ell h}m_{ij}^{\ell h}.
\]

Here `a` denotes native per-head attention weights **after attention dropout**.
At evaluation it equals the softmax probabilities; during training its row sums
need not be one.

The native attention weights `a` already depend on evolving particle states and
the scalar pair bias. The limitation is specifically the **added message's
content**, not an absence of context anywhere in attention. The pair embedding
is nonlinear and learned jointly; only its final EV projection is linear.

The previous jet-context extension rescales pair channels using a pooled jet
representation, with the same channel scaling for all pairs in that layer. The
previous relation gate modulates sender values using pair-derived coefficients.
Neither directly constructs a nonlinear added message from both current
endpoint states. Small context gains therefore do not settle this new hypothesis.

ParticleNet provides precedent for particle-feature-dependent edge functions;
this proposal is not a claim to invent that general concept. The research
question is its utility as a controlled extension of ParT's value messages.
[ParticleNet, Sec. III](https://arxiv.org/pdf/1902.08570)

## 2. State-conditioned EV: a directed low-rank residual

### 2.1 Exact proposed module

Retain the existing EV term and add a new residual. Use rank `r=32`, all eight
particle-attention blocks, and separate parameters per block. Head-specific
output maps share a 32-channel nonlinear pair representation within a block.

For a block, define:

\[
u_i=\operatorname{LN}_{128}(x_i),\qquad
\bar z_{ij}=\operatorname{LN}_{64}(z_{ij}),
\]
\[
p_i=A u_i,\qquad q_j=B u_j,\qquad g_{ij}=E\bar z_{ij},
\]
\[
r_{ij}=\operatorname{GELU}\!\left(
  \operatorname{LN}_{32}\left(g_{ij}+p_i+q_j+p_i\odot q_j\right)
\right),
\]
\[
m_{ij}^{\ell h}=W_{\ell h}z_{ij}+D_{\ell h}r_{ij}.
\]

Dimensions: `A,B: 128 -> 32`, `E: 64 -> 32`, and each `D_h: 32 -> 16`.
All four projection types are bias-free. LayerNorms have no affine parameters,
epsilon `1e-5`, and normalize feature channels only. Compute normalization
statistics in FP32; use the established AMP policy for linear operations.
The symbols `A,B` are new residual projections, not replacements for native Q/K.

Use the exact states entering native MHA, after its existing block normalization;
the additional non-affine normalization above applies only to this new branch.
Do not recompute a separate backbone state or detach the particle states.

Why this version:

- `p_i` and `q_j` distinguish receiver and sender roles. The new message can be
  directed even though the native kinematic pair features are symmetric.
- The product `p_i * q_j` explicitly represents endpoint compatibility; the
  nonlinear sum also couples those states to pair geometry.
- Later-layer states already include information gathered from other particles.
  This gives pair-specific contextualization without another global pooling
  module or new measured features.
- Rank 32 bounds pair activation width, while eight distinct output maps retain
  head-specific messages. This is a chosen compromise, not a proven optimal rank.

All standard scalar biases, native attention weights, FFNs, class-attention
blocks, and classifier remain architecturally unchanged and trainable.

### 2.2 Initialization and training behavior

First construct the exact seed-101 ordinary EV model, including its original
nonzero EV projections. Initialize `A,B,E` with Xavier uniform draws from
isolated, layer-specific RNG streams. Initialize **only `D` to zero**.

Consequently, the initial model implements ordinary EV, not plain ParT. The new
output projection receives a gradient immediately; upstream new projections
normally begin receiving gradients after `D` moves away from zero. Test both
stages. Do not zero every projection, add a second zero multiplier, or insert a
long activation warmup: those can prevent or delay useful learning.

No branch-specific LR, weight decay, clipping, dropout, or extra training steps
are introduced. No donor trained weights or probe states are loaded. A fresh
initialization copied within a new model is not checkpoint warm-starting.

### 2.3 Memory-safe exact evaluation

Use native per-head, post-dropout attention weights for both ordinary EV and the
new residual. Do not sample a second dropout mask or renormalize those weights.

Compute `p_i,q_j` once per layer. For query chunks of 16 particles, construct
`r_ij` and aggregate first:

\[
t_i^{\ell h}=\sum_j a_{ij}^{\ell h}r_{ij},\qquad
\Delta y_i^{\ell h}=
W_{\ell h}\sum_j a_{ij}^{\ell h}z_{ij}+D_{\ell h}t_i^{\ell h}.
\]

This is algebraically identical to projecting every pair through `D`, but avoids
a `[batch, heads, N, N, head_dim]` tensor. The nonlinear computation of `r_ij`
must stay **before** neighbor aggregation; moving GELU after the sum changes the
model. Do not retain full pair-state activations from all chunks or layers.

Use non-reentrant activation checkpointing around the stateless chunk function
during training. It contains no BatchNorm, dropout, RNG draws, or module-state
mutation. At batch 512, 128 particles, rank 32 and chunk size 16, one chunk's
`r_ij` alone occupies 64 MiB in FP16 (128 MiB in FP32). This is not a peak-memory
estimate: attention, normalized inputs, autograd, and other activations add cost.

Keep all key particles, self-pairs, and the native trimming behavior; query
chunking is an exact memory technique, not sparsification. Mask padded pair
residuals after the nonlinear maps, then mask the added query output. Retain
native bias-path padding semantics rather than modifying its BatchNorm samples.
Never mirror the directed state residual from the lower triangle.

Concatenate head corrections and apply the existing native attention output
projection without its bias, exactly as ordinary EV does; add that correction
once to the native attention result. Do not add the output-projection bias twice.

## 3. Separate-encoder EV: untie the representation, not the data

### 3.1 Exact proposed module

Keep two encoders of the same native four-channel pair features:

\[
z^{\mathrm{bias}}_{ij}=\phi_b(e_{ij}),\qquad
b_{ij}=P_b(z^{\mathrm{bias}}_{ij}),\qquad
z^{\mathrm{value}}_{ij}=\phi_v(e_{ij}),
\]
\[
m_{ij}^{\ell h}=W_{\ell h}z^{\mathrm{value}}_{ij}.
\]

`P_b` is the unchanged native final bias projection, including its BatchNorm and
activation. Both stems use the exact native prefix up to that projection:
input BatchNorm on four channels, then three pointwise maps `4 -> 64 -> 64 -> 64`,
each with its original bias, BatchNorm, and GELU. Do not substitute an unrelated
encoder, widen channels, or change normalization in this first experiment.

The value stem is shared across all eight blocks, just as the current stem is.
Layer/head-specific EV projections remain unchanged. This separates two learning
roles without simultaneously introducing eight independent pair networks.

Hypothesis: a representation that serves attention routing may be a compromise
for constructing value messages. Independence lets both specialize. It could
also lose useful regularization and perform worse; shared representations are
not known to be the cause of the current small gains.

### 3.2 Copy initialization, independent optimization

Construct ordinary EV first. Deep-copy the freshly initialized native prefix,
including BatchNorm parameters and buffers, into `phi_v`. Assert that no tensors
share storage with `phi_b`; register every trainable tensor exactly once.

At initialization, both stems compute the same features, so this model starts
with ordinary EV's predictions. Then train their parameters independently with
the same optimizer recipe and no special warmup. All existing backbone, bias,
and EV-projection initial values and the training RNG stream must stay unchanged.

There is no direct value-content gradient into `phi_b` after the split, but EV
still depends on attention weights. Therefore `phi_b` can still receive gradients
through attention routing. **Do not describe this as complete gradient isolation**,
and do not detach attention weights or freeze either stem.

### 3.3 Pair construction and memory details

Compute native raw pair features once using exactly the same trimmed/permuted
four-vectors, lower-triangle packing, diagonal handling, epsilon, and transforms.
Feed that same packed input to both stems; scatter scalar bias and value features
with the existing symmetric packing convention. The duplicate stem is evaluated
once per forward, not once per block.

A replacement for `_BiasAndStem` can return `[native_bias_8, private_value_64]`
so that the current pair wrapper still scatters 72 channels. Do not also scatter
and retain a second unused full 64-channel bias stem. The packed internal
activations still increase memory and computation; measure both.

Raw kinematic feature construction may retain the native `no_grad` behavior.
Learned encoder forwards must remain under autograd. Do not checkpoint a
BatchNorm-containing stem naively: recomputation could update running statistics
twice. Query-chunk recomputation in Section 2 does not have that issue.

The inspected pinned prefix has **9,032 trainable parameters**. This variant
therefore has 2,218,054 parameters versus 2,209,022 for ordinary EV. Buffer memory
and FLOPs are not included in these parameter counts.

## 4. Combined model

Use `z_value` everywhere the state-conditioned EV branch consumes `z`:

\[
m_{ij}^{\ell h}=W_{\ell h}z^{\mathrm{value}}_{ij}
                +D_{\ell h}r_{ij}
                  (z^{\mathrm{value}}_{ij},x_i^{\ell},x_j^{\ell}).
\]

Scalar attention bias continues to use only `z_bias`. Copy-initialize the value
stem and zero-initialize `D`, so the combined model also starts as ordinary EV.
Use identical initialization streams for mechanisms shared with the individual
arms. Do not add the previous pooled-context or relation-gating modules.

The combination tests whether a specialized pair representation is especially
useful for state-dependent messages. A positive interaction is possible, not
guaranteed; report the individual arms even if the combined run wins.

## 5. Two additional architectural hypotheses

These arms start from ordinary EV, not from the combined model in Section 4.
Keep the shared native pair encoder and existing four kinematic relations in
both. This makes each result interpretable as one new mechanism.

### 5.1 EV-specific routing

Hypothesis: the neighbors worth borrowing particle features from need not have
the same importance as the pair relationships worth summarizing. Ordinary EV
forces both pathways to use one attention map.

Let `s_ij^lh` be the native logits, including scaled Q/K products, scalar pair
bias, and key masking. Define pre-dropout probabilities:

\[
\alpha_{ij}^{\ell h}=\operatorname{softmax}_j(s_{ij}^{\ell h}),\qquad
\delta_{ij}^{\ell h}
  =2\tanh\!\left(R_{\ell h}\operatorname{LN}_{64}(z_{ij})\right),
\]
\[
\beta_{ij}^{\ell h}
  =\operatorname{softmax}_j(s_{ij}^{\ell h}+\delta_{ij}^{\ell h}).
\]

Use `alpha` for native particle values and `beta` for existing EV messages
`W_lh z_ij`. The correction has a separate bias-free `64 -> 8` projection in
each block, non-affine LayerNorm with epsilon `1e-5`, and no new Q/K projections.
Initialize every `R` to zero. The bounded correction is in `[-2,2]` logit units;
this is a fixed initial design choice, not a tuned optimum.

This adds 4,096 parameters overall. It changes **where EV aggregates from**,
not the information supplied to its message encoder. It is different from the
previous relation gate, which modulates sender values without introducing a
separately normalized EV attention distribution. The two maps still share
native logits, so they are not fully independent attention networks.

Preserve the existing native MHA call and its realized dropout mask. With
`a = dropout(alpha)`, the EV post-dropout weights can be computed as:

\[
Z_i^{\ell h}=\sum_j\alpha_{ij}^{\ell h}\exp(\delta_{ij}^{\ell h}),\qquad
\widetilde\beta_{ij}^{\ell h}
  =a_{ij}^{\ell h}\frac{\exp(\delta_{ij}^{\ell h})}{Z_i^{\ell h}}.
\]

This is exactly `dropout(beta)` using the same mask and scaling. In particular,
do **not** treat post-dropout weights as softmax probabilities when computing
`Z`, renormalize dropped weights, or recover logits with a clipped logarithm.

Obtain `alpha` from the actual native Q/K projections, scale, bias and masks,
without a second stochastic attention call. Query-chunk the extra pre-dropout
score calculation; use FP32 for the correction exponentials and normalization.
Do not detach either distribution. Compare against an explicit tiny
`softmax(s + delta)` reference, including gradients and a fixed dropout mask.
Matching native softmax precision is an acceptance requirement, not an excuse
to relax parity tolerances or replace the baseline attention implementation.

At zero correction the EV path must match ordinary EV within the frozen
numerical tolerances, in training as well as evaluation. Mask padded query
corrections as usual. The original output projection and all particle-value
operations remain unchanged. Added Q/K-score recomputation costs time and memory
despite the small parameter count; benchmark it.

Implementation detail: native pre-dropout softmax probabilities are converted
to FP32 and row-normalized to remove FP16 row-sum rounding. The denominator is
evaluated as `1 + sum(alpha * expm1(delta))`, so a zero correction is exactly
identity even with quantized native weights. This is algebraically equivalent
in exact arithmetic; reference and mixed-precision parity checks guard it.

### 5.2 Variance-augmented EV

Hypothesis: two sets of relationships can have similar weighted means but
different spreads. For example, a mixture of very similar and very dissimilar
neighbors can average to the same feature value as uniformly intermediate
neighbors. Exposing dispersion may help distinguish them.

Ordinary EV already averages **learned nonlinear features**, and the backbone
has multiple heads and layers. It is not inherently incapable of representing
such distinctions. This arm tests an explicit aggregation bias, not a proof
that the baseline lacks the information.

Keep the ordinary EV correction unchanged. For the new branch alone, set
`zbar_ij = LN64(z_ij)`, using non-affine feature-channel normalization with
epsilon `1e-5`. For each head and valid query, normalize the existing
post-dropout weights over valid keys **only to compute these statistics**:

\[
S_i^h=\sum_j a_{ij}^h,\qquad
\pi_{ij}^h=
\begin{cases}
a_{ij}^h/S_i^h,&S_i^h>0,\\
0,&S_i^h=0.
\end{cases}
\]
\[
\mu_i^h=\sum_j\pi_{ij}^h\bar z_{ij},\qquad
\nu_i^h=\max\!\left(
  \sum_j\pi_{ij}^h(\bar z_{ij}\odot\bar z_{ij})
  -\mu_i^h\odot\mu_i^h,\;0\right),
\]
\[
\Delta y_i^{\ell h}
  =W_{\ell h}\sum_j a_{ij}^{\ell h}z_{ij}
   +U_{\ell h}\nu_i^{\ell h}.
\]

The max is elementwise. `U_lh: 64 -> 16` is bias-free and zero-initialized,
with independent parameters for every head and block. This adds 65,536
parameters and starts at the ordinary-EV function. Do not add endpoint
conditioning, a private encoder, or the routing correction in this arm.

Accumulate moments in FP32 and clamp roundoff-negative variances to zero.
Use an explicit safe denominator for zero-weight rows before division, then
force their statistics to zero; evaluating `0/0` and masking afterward is not
safe for autograd. Retain self-pairs and mask invalid queries. Use variance,
not a square root, to avoid introducing a near-zero square-root derivative.

During training this is the variance of the **dropout-surviving weighted
neighbors**, not an unbiased estimate of pre-dropout variance. At evaluation it
uses the full native attention distribution. The original EV term never gets
renormalized, and no additional dropout draw is introduced.

Compute first and second moments in query chunks of 16 and checkpoint the
stateless chunk calculation. Aggregate the shared 64-channel features into
per-head moments before applying `U`; do not materialize per-head pair-value
tensors or a full channel-by-channel covariance matrix. Use the same native
output projection once for the combined head correction.

The variance describes feature dispersion, not calibrated model uncertainty.
Multiple neighborhood aggregators have precedent in graph learning; this is a
targeted EV experiment, not a claim to invent moment aggregation.
[Principal Neighbourhood Aggregation](https://arxiv.org/abs/2004.05718)

## 6. Exactly five new runs

Reuse the verified donor `PART_RANGER` and `EV_RANGER` results. The recommended
first screen adds **five** single-seed runs. All five are architecture experiments;
the previously suggested static-MLP capacity control is deferred, not a sixth
run. These are proposed future tasks, not additions to the active seven-run array.

| Proposed run ID | New mechanism relative to ordinary EV | Expected parameters |
|---|---|---:|
| `EV_STATE_RANGER` | Nonlinear message from both current particle states | 2,323,710 |
| `EV_SEPARATE_RANGER` | Separate pair encoder for value messages | 2,218,054 |
| `EV_STATE_SEPARATE_RANGER` | Both of the above | 2,332,742 |
| `EV_ROUTING_RANGER` | EV-specific attention distribution | 2,213,118 |
| `EV_VARIANCE_RANGER` | Weighted pair-feature variance alongside ordinary EV | 2,274,558 |

The state residual adds `128*32 + 128*32 + 64*32 + 8*16*32 = 14,336`
parameters per block, or 114,688 overall. Count instantiated models in tests;
these totals assume the bias-free projections and non-affine LayerNorms above.

For routing, the addition is `8 blocks * 8 heads * 64 = 4,096`; for variance it
is `8 * 8 * 16 * 64 = 65,536`. Ordinary EV has 2,209,022 parameters and the
reused ParT control has 2,143,486. Different overheads are part of the comparison.

Primary contrasts on the same full-validation population:

- Each of the five models minus `EV_RANGER`; also report each minus `PART_RANGER`.
- Combined minus state-only, and combined minus separate-only.
- Factorial interaction: `combined - state - separate + ordinary_EV`.
- Routing versus ordinary EV isolates its routing modification; variance versus
  ordinary EV isolates its aggregation modification, including added capacity.

The interaction is a descriptive single-seed difference, not a significance test.
Do not compare a new model's best intermediate check with an old model's final
full-validation metric. Use the same reporting convention for every arm,
including unsuccessful ones.

### Deferred capacity controls, outside this five-run budget

This first stage prioritizes exploring mechanisms. It does not fully separate
their effects from extra trainable capacity. If a model wins, choose a targeted
control and additional matched seeds before making a causal mechanism claim.

Retain the following specification for a possible `EV_STATIC_MLP_RANGER`
follow-up: keep ordinary EV and add the same zero-output
residual structure, but without either particle state:

```text
LN64(z) -> Linear(64,107,bias=False) -> GELU
        -> Linear(107,32,bias=False) -> LN32 -> GELU
        -> head-specific Linear(32,16,bias=False), initialized to zero
```

Use all eight blocks, identical aggregation/masking/precision, and the same
chunking/recomputation scheme. Width 107 is the nearest integer width matching
the state branch's parameter count: the control adds only 256 more parameters
over the whole model. Its FLOPs are not matched; report timing separately.

This would test whether endpoint states improve upon simply making the geometry-only
message network more expressive. It does not fully resolve every capacity
confound for the separate encoder. If untying wins, a follow-up should compare
against a similarly sized widened **shared** stem before attributing all of the
gain specifically to separation. If the combined model is the winner, also add
the static-MLP-plus-separate-stem control before claiming special synergy.
Routing and variance winners likewise need targeted capacity/aggregation
controls; do not describe their first single-seed gains as proof of mechanism.

## 7. Training and execution contract

Inherit, without tuning, the donor's `ranger_short_v1` recipe: full 100M training
population, batch 512, seed 101, 200,000 attempted updates, validation every 20,000
updates on 1.28M examples, compressed 15-step decay from 140k to 196k, and
highest-accuracy checkpoint selection with first-tie retention. Use the same
continuous streams and two workers. Full selected-model evaluation uses the
same 5M validation jets; no 20M test data are read.

Run on the donor-compatible x86 SPORC A100 configuration and dedicated environment,
with FP16 training, FP32 parameters, normal fresh GradScaler, and FP32 evaluation.
Do not change batch/accumulation, clipping, optimizer, TF32, or precision to fit a
model silently: such a change needs matched controls and a new recipe.

Create a separate source snapshot, schema, campaign directory, and task registry.
Validate donor controls as specified in the reusable protocol. A missing donor
evaluation is pending; it is not permission to manufacture a baseline or rerun
all controls automatically. Leave active jobs and frozen source unchanged.

Before training, benchmark every new architecture with full-128-particle batches,
normal trimmed batches across multiple fetch windows, forward/backward precision,
finite losses/gradients, AMP skips, checkpoint writes, and validation cost.
Every pre-production allocation must be under six hours. Set training walltime
from measured sustained throughput, validation costs, and headroom. The old AMP
recovery approval does not authorize these new architectures.

## 8. Diagnostics that distinguish success from unused capacity

Use a fixed diagnostic validation sample and restore RNG/trimmer state afterward.
Collect summaries, not full population-sized pair tensors:

- Per-layer new-residual norm / ordinary-EV norm, and total-EV / native-output
  norm, with safe denominators. Record actual `D` norms and gradient norms.
- On valid particles, measure within-jet variation of `p_i,q_i` and the change
  in residual messages when endpoint inputs are replaced by their valid-jet
  means **inside the diagnostic branch only**. Keep attention weights and the
  rest of the forward fixed for this sensitivity check.
- For the separate stem, compare valid-pair feature similarity and differences
  from the bias stem, and confirm independent parameter/buffer storage.
- If affordable, record cosine similarity between bias-route and value-content
  gradients on the shared stem for a small ordinary-EV diagnostic batch.
  Gradient conflict is evidence to inspect, not an assumption or proof of harm.
- After training, temporarily disable the new state residual on the same small
  validation sample to measure reliance. This is a diagnostic ablation, not a
  substitute for training the proper control.
- For routing, compare pre-dropout alpha/beta entropies and their mean
  total-variation distance on valid queries; record correction magnitudes and
  saturation of tanh. Temporarily set delta to zero to measure reliance.
- For variance, record valid-query variance magnitudes, zero-weight-row counts,
  and the projected variance term's norm relative to ordinary EV. Temporarily
  disable only that term to measure reliance. Do not interpret it as uncertainty.
- Report parameters, peak memory, sustained jets/s, total training time, and
  cumulative skipped updates alongside accuracy, AUC, and all class rejections.

Similar representations do not by themselves prove redundancy; a large residual
does not prove usefulness; zero-initialization makes early small residuals expected.
The full-validation comparison remains the primary screen.

## 9. Implementation and acceptance tests

Implementation package: `part_ev_message_study/`, separate from the existing
`part_ev_screen/` implementation. Build reusable control validation first, then
the separate stem, then state residual, combined arm, routing arm, and variance
arm. The deferred static control is not required for this five-run launcher.
Use the existing [backbone/EV implementation](../part_ev_fullscale/model.py),
[precision policy](precision.py), and [screen trainer](../scripts/run_part_ev_screen.py)
as references; do not copy their fixed seven-task assumptions into a generic API.

Required tests before queue readiness:

1. **Identity at initialization:** all arms match ordinary EV logits on fixed
   same-platform inputs within frozen FP32/AMP tolerances. Original parameter
   tensors and RNG states match exactly. Test train mode with controlled dropout
   RNG as well as evaluation; independent BN buffers update exactly once.
2. **Gradient paths:** first-step `D` gradients are finite/nonzero on a
   nondegenerate fixture; new input projections learn after `D` is nonzero.
   Separate encoders have independent gradients and no optimizer duplication.
   Test first-step routing `R` and variance `U` gradients on nondegenerate
   fixtures, and moment/encoder gradients after `U` becomes nonzero.
3. **Math reference:** chunked and projected-after-aggregation results and
   gradients match an explicitly materialized tiny all-pairs reference. Use
   fixed nonzero `D`; an all-zero residual is not a useful equivalence test.
   Compare routing against explicit shifted-logit softmax with the same dropout
   mask. Compare variance against weighted squared deviations around the mean;
   check constant features, one valid key, and fully dropped rows.
4. **Symmetries/masks:** permutation equivariance, padded outputs, one-particle
   jets, self-pairs, and consistent trimming; include a nonsymmetric state-pair
   fixture to catch accidental triangular mirroring. Preserve native bias/BN
   behavior when testing padding rather than assuming it is padding-independent.
5. **Recomputation:** chunk size changes do not change results beyond tolerance;
   no duplicate BN updates, detached gradients, or extra dropout draws. Do not
   use dropout-bearing native MHA inside the stateless recomputed chunk.
6. **Training protocol:** exact schedule boundaries, data fingerprints,
   first-tie selection, complete optimizer/scaler/RNG resume, donor artifact
   rejection on mismatch, and all-five-million evaluation identity coverage.
7. **Resource checks:** assert counts above and profile real GPU peak memory;
   no full per-head pair-value tensor or per-layer retained pair-state history.
8. **Launcher:** mock the dependency graph; submit only new runs, never reused
   control training; protect against active duplicate submissions. Assert
   exactly the five IDs in Section 6, each once with seed 101.

An inference-only comparison across changed hardware/software does not establish
matched training. Use the same-runtime controls unless comparability is reviewed.

## 10. Decision rules and scope of conclusions

Keep overall full-5M accuracy as the primary metric; report macro OVO/OVR AUC and
the existing complete set of classwise QCD rejections as secondary metrics.
Assess these alongside runtime, not just the largest favorable number. Choose
any numerical practical-gain threshold before submission, not after seeing results.

- If the state arm wins, test it against the deferred static control next.
  A win over ordinary EV alone does not establish that particle-state
  conditioning, rather than added message capacity, caused the improvement.
- If separation wins, verify the gain across matched seeds and the shared-stem
  capacity follow-up before claiming the sharing constraint was causal.
- If the combination wins, quantify both individual contributions and its cost;
  do not omit unsuccessful individual arms.
- If routing or variance wins, inspect whether the relevant mechanism was used
  and replicate the result before extending it with the other mechanisms.
  A one-seed winner is a candidate, not a reason to launch every combination.
- If none improve meaningfully, do not automatically launch a rank/width/optimizer
  sweep. Inspect whether the branches learned and consider stopping this direction.

Single-seed validation is a screening result. Its evaluation overlaps checkpoint
validation and is not an independent test. Promising configurations should receive
matched additional seeds and, later, independent test evaluation with the design
fixed. More repeated validation experiments do not turn a tiny gain into evidence
of generalization.

## 11. Prior work and novelty boundaries

ParT already introduces pair interactions into attention; keep that baseline and
its recipe explicit. [ParT](https://arxiv.org/abs/2202.03772)

Learned endpoint-dependent edge functions are established in graph networks,
including ParticleNet. Our state branch uses ParT's dense attention weights and
adds low-rank value messages; it does not implement ParticleNet's dynamic k-nearest
neighbor graph. [ParticleNet](https://arxiv.org/pdf/1902.08570)

MIParT explores richer interaction embeddings and interaction-based attention.
It is relevant prior work, not evidence that these particular modifications
will work or that they are novel. [MIParT](https://arxiv.org/html/2407.08682v2)

PNA combines multiple neighborhood aggregators and degree scalers. It motivates
testing aggregation beyond a single weighted mean, but is not the same as our
attention-weighted variance branch and does not establish its effectiveness in
jet tagging. [PNA](https://arxiv.org/abs/2004.05718)

The immediate aim is a well-controlled empirical test. A novelty claim requires
a broader literature review once the mechanism and evidence are established.
