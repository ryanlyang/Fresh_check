# Support-aware aggregation on full Layerwise State-EV

Recorded: 2026-11-16. Detailed first-design and implementation plan.
**Status: implemented; independent SPORC benchmark queue-ready.**
See [implementation and queue instructions](../part_ev_support/README.md).
No jobs were submitted by this implementation. GPU admission, classification
gains and the later production launcher remain separate work.

This specifies idea C in the
[three-addition overview](STATE_EV_READOUT_TRIANGLE_SUPPORT_PLAN.md).
It is independent of the [pair-readout plan](DIRECT_PAIR_CLASSIFIER_READOUT_PLAN.md)
and [triangle-aware updates plan](TRIANGLE_AWARE_UPDATES_PLAN.md). Each first
experiment adds one mechanism to the same complete Layerwise State-EV backbone.
This document fixes the first support design, its correctness requirements, and
its benchmark path; it does not authorize changes to current jobs or experiments.

## 1. Objective: retain content and the amount of learned support

Ordinary attention mixes messages with normalized weights. A weighted mean can
describe what a collection of relationships says without preserving how many
relationships support that description. One copy and several identical copies
can have the same mean. The proposed branch supplies particle updates with both
learned relational content and its unnormalized amount of support.

For each receiver particle, four learned channels independently score its
relationships to the other valid constituents. Each channel reports:

- The gate-weighted mean content of those relationships.
- The logarithm of one plus the total gate weight, a soft support count.
- The total gate weight divided by the number of eligible neighbors, a support
  fraction that distinguishes selectivity from simply having more constituents.

A small particle-level MLP turns these summaries into an additional message.
There are no physics labels assigned to the four channels, no thresholded votes,
and no new supervision. All learning uses the existing classification loss.

The main hypothesis is that several moderately useful relationships can support
a different decision from one isolated relationship even when their average
content is similar. This is an information-preservation and optimization-bias
hypothesis, not a claim that existing ParT or State-EV cannot infer multiplicity.
Their particle states already contain contextual information, and their other
operations can encode aspects of the constituent population.

Combining aggregation statistics with degree information has precedent in
[Principal Neighbourhood Aggregation](https://arxiv.org/abs/2004.05718). The design
here is a separate learned-gate residual for jet tagging, not the full PNA
architecture or evidence of an expected jet-classification gain.

### Why this might help, and why it might not

The opportunity is a more direct way to preserve evidence strength while leaving
the successful normalized messages available. Four patterns can have different
content and support, and the nonlinear fusion can decide which combinations
matter. A bounded count representation avoids injecting a raw, potentially
large sum of signed features into every block.

However, counts and content may already be recoverable from the backbone. The
channels may become redundant, their gates may saturate, or the branch may learn
mostly constituent multiplicity rather than relation-specific evidence. Extra
parameters or different optimization may explain a gain. There is no justified
prediction of a particular accuracy increase or of which jet class benefits.
The checks and controls below are part of the scientific design, not optional
explanations to invent after seeing a favorable maximum.

## 2. Unchanged reference and locked first design

Build from `build_model('state_layerwise', ...)` in [the reference assembly](model.py):

- Standard physics relations, one shared 64-channel pair stem, and eight
  independent layerwise scalar-bias heads; not eight pair encoders.
- Eight particle blocks, eight attention heads, particle width 128.
- Ordinary EV plus noncompact width-32 State-EV in **all eight** particle blocks.
- Both original class-attention blocks, class normalization, and classifier.
- Original pair-normalization cache, query chunking, and activation checkpointing.
- **2,327,462 trainable parameters** before the addition.

Its historical seed-101 two-A100/SyncBatchNorm full-schedule run reached
**86.3151% training-time validation accuracy at check 49 of 50**. This is not an
independent test result, a multi-seed estimate, or a threshold that by itself
establishes a new architectural benefit. Audit runtime and recipe before reuse.

| Component | Locked first implementation |
| --- | --- |
| Placement | Additional message in all eight particle blocks |
| Input | Each block's existing directed width-32 State-EV pair features |
| New input normalization | Non-affine channelwise LayerNorm, epsilon `1e-5` |
| Support channels | Four, independently initialized, not attention heads |
| Gate projection | `32 -> 4`, with bias, followed by independent sigmoids |
| Content projection | Bias-free `32 -> 16`, shared across support channels within a block |
| Pooling set | All valid ordered off-diagonal pairs; one summary per receiver |
| Statistics per channel | 16 content coordinates, `log1p(count)`, support fraction |
| Summary width | `4 * (16 + 1 + 1) = 72` per receiver |
| Fusion | `72 -> 32 -> 128`, GELU between linears |
| Fusion biases | Hidden bias yes; output bias **no** |
| Identity initialization | Zero output matrix only; other weights nonzero, biases zero |
| Denominator epsilon | `1e-6`, applied once to the complete support count |
| Count/weighted-sum arithmetic | FP32, including under FP16 autocast |
| New dropout / auxiliary losses | None / none |
| Weight sharing across blocks | None for new modules; existing shared pair stem retained |
| Receiver chunk size | 64, matching the optimized base |
| First scientific task | Fresh seed 101, full matched Ranger recipe |
| Intended initial runtime | SPORC tier3, two same-node A100s, DDP/SyncBatchNorm |

These are reasoned starting choices, not validated optima. The output-bias choice
and exact mask/layout rules below remove ambiguities in the overview. Do not
silently change to fewer blocks, ordinary EV, a compact State-EV, selected
relations, attention-weighted counts, or a new optimizer to improve a benchmark.
Do not combine this arm with pair readout, triangles, state-conditioned bias,
relation-state multiplication, or evolving pair memory in its first experiment.

## 3. Pair features, time ordering, and masks

Index `i` is the receiver, `j` is the sender. Use the features computed from
the same pre-update particle states as the original State-EV message in layer `l`:

```text
zbar_ij = LN64(shared_physics_pair_stem_ij)
xbar_i  = LN128(native_attention_query_i)
g_ij    = E_l zbar_ij
p_i     = P_l xbar_i
q_j     = Q_l xbar_j
r_ij    = GELU(LN32(g_ij + p_i + q_j + p_i * q_j))
```

The native query already passed through the particle block's pre-attention
normalization. Preserve the state module's additional normalization and the
reference arithmetic order, precision, and valid-pair masking. Do not alter
`r_ij` in place or remove its diagonal values from the original State-EV branch.

All masks use the native post-trimming/reordering axes. Define a separate mask
for the new branch:

```text
support_valid_ij = valid_i AND valid_j AND (i != j)
M_i = sum_j support_valid_ij
has_neighbors_i = M_i > 0
```

`M_i` is the actual retained nonself degree, not the padded length, the number
of records before the 128-particle limit, or the dataset's original multiplicity.
For a valid receiver it is `n_valid - 1`; for a padded receiver it is zero.
Native stochastic trimming can change this population during training. Counts
must follow that same population; do not introduce a second trimmer or read
untrimmed constituents solely for this mechanism.

Preserve both directions: `r_ij` and `r_ji` can differ even when raw physics
relations are symmetric. There is no upper-triangle packing of these summaries,
no nearest-neighbor selection, and no pooling across jets or DDP ranks.

Create the support-only view by masking invalid and diagonal `r` entries to
zero **before** its new normalization/projections. Mask projected gates and
values again for reductions. This prevents excluded entries from contributing
through biases and makes the isolated support branch insensitive to them.
It does not change the base's diagonal convention or authorize hiding invalid
nonfinite features on eligible pairs. Reject empty jets under the existing
input contract; one-particle jets are valid with a zero support message.

## 4. Exact aggregation equations

For support channel `a = 0,...,3`, let `G_l` have four output rows and let `V_l`
produce a common 16-channel content vector. All parameters are specific to layer `l`:

```text
u_ij   = LayerNorm(support_only_r_ij, affine=False, eps=1e-5)
z_aij  = (G_l u_ij + b_gate_l)_a
s_aij  = sigmoid(z_aij)                    # independent scores, NOT softmax
v_ij   = V_l u_ij                          # bias-free, width 16
```

On excluded pairs set `s_aij=0` and `v_ij=0`. For each receiver and channel:

```text
C_ia   = sum_j s_aij
S_iad  = sum_j s_aij * v_ijd
mu_iad = S_iad / max(C_ia, 1e-6)
ell_ia = log1p(C_ia)
rho_ia = C_ia / max(M_i, 1)

f_i = concat_a(concat(mu_ia[0:16], ell_ia, rho_ia))   # 72 channels
h_i = GELU(H_l f_i + b_hidden_l)                    # 32 channels
e_i = O_l h_i                                     # 128 channels, no output bias
e_i = e_i * has_neighbors_i
```

The feature order is channel-major: 16 content entries, one log count, then one
fraction for channel 0, followed by channels 1, 2, and 3. Freeze this layout in
the model config and tests. If `M_i=0`, define **all** summary features and `e_i`
as zero, including after the hidden bias has learned a nonzero value.

For numerical underflow to `C_ia=0` despite eligible neighbors, masked gates
also make `S_iad=0`; the clamped denominator then gives zero content and zero
count features without `0/0`. At very small positive counts below epsilon,
content is deliberately attenuated. It is no longer an exact weighted mean
there; measure how often that safety region is used rather than hiding it.

Do not detach `C` or `S`, normalize gates across channels, divide counts by
sequence length before taking their logarithm, or apply another softmax.
No LayerNorm/BatchNorm is added to `f`, `h`, or `e`. The existing surrounding
backbone normalizations remain unchanged; this does not guarantee all support
information survives them, but avoids deliberately discarding it in the new
summary itself.

### What the support statistics mean

With `M_i <= 127`, `0 <= C_ia <= M_i`, `0 <= rho_ia <= 1`, and
`0 <= ell_ia <= log(128)`, approximately 4.852. Content is signed and not clipped
to these ranges. Four channels may all support the same pair: they are not a
partition of constituents and their counts should not sum to `M_i`.

The count is a learned soft score, not calibrated confidence, independent
evidence, a number of subjets, or a probability of a physical hypothesis.
Relationships sharing endpoints are correlated. The channel fraction is also
not a predicted class probability.

For fixed pair features, replicating every eligible neighbor `t` times changes
`C` and `S` by `t`, preserves `mu` above the epsilon floor, preserves `rho`, and
changes `ell` to `log1p(t*C)`. Thus the representation can distinguish equally
composed collections of different sizes. This is a mathematical unit test on
frozen pair features, not an instruction to duplicate real jet particles or a
claim that the whole backbone is invariant to such duplication.

Count and fraction are intentionally redundant with degree: where the count
is nonzero, `C/rho = M`. Keeping both makes absolute and relative support easy
to access, but can also supply multiplicity cheaply. Their presence alone
does not prove learned relational evidence is being used.

## 5. Message placement: add a parallel path, do not change attention

This is an additional **particle message**, not a class-token readout and not a
new attention-logit term. Its gates read `r_ij`, not the native attention weights.
Native softmax row sums cannot measure the desired support; after dropout those
sums fluctuate because of dropout, not because of a learned evidence count.

The four support channels are not the eight attention heads. Map the 128-channel
`e` into the native head slots only at the output boundary:

```text
support_heads = e.reshape(B, N, 8, 16).permute(0, 2, 1, 3)
state_plus_support = original_state_message + cast_like(support_heads, original_state_message)
joined_heads = ordinary_EV_message + state_plus_support
joined = joined_heads.permute(2, 0, 1, 3).reshape(N, B, 128)
extra = linear(joined, native_attention.out_proj.weight, bias=None)
extra = extra * valid_receiver_mask
attention_output = original_native_attention_output + extra
```

This specifies the addition order. The original native output already contains
its own output-projection bias: do not add it again to any extra message.
Retain native head scaling, particle-block normalization, residual scaling,
dropout, and feed-forward paths outside this adapter. The class-attention
readout and classifier are untouched.

Original EV and State-EV still use the single native post-dropout attention
weight tensor. The support message does not receive a second attention factor
or its own dropout; it passes through the existing downstream block operations.
Do not call MHA again, reconstruct its weights, or resample dropout. All base
state features and new support summaries describe states before this block's
particle update; do not let receiver chunk order feed updated states backward
into other receivers.

## 6. Chunked computation and bounded activation memory

The [optimized state module](../part_ev_state_speed/model.py) already computes
`r_ij` for a receiver chunk while the normalized pair cache is live. Extend that
computation jointly rather than running the state feature network twice.

Notation: `B` local batch, `N <= 128` trimmed particles, `Q <= 64` receiver chunk,
`A=4` support channels, `D=16` content width.

| Tensor | Shape |
| --- | --- |
| Existing pair features | `[B,Q,N,32]` |
| Support-only normalized view | `[B,Q,N,32]` |
| Gate logits / gates | `[B,Q,N,A]` |
| Projected content | `[B,Q,N,D]` |
| Count | `[B,Q,A]` |
| Weighted sum / mean | `[B,Q,A,D]` |
| Neighbor count | `[B,Q]` |
| Summary features | `[B,Q,72]` |
| Added message before head layout | `[B,Q,128]` |
| Old-plus-new state output | `[B,8,Q,16]` |

Recommended chunk order:

1. Compute the original `r` and original State-EV pooled/projected message in
   their existing operation order, without modifying their tensors in place.
2. Compute `u`, gate logits, and content while `r` is still live.
3. With autocast disabled, mask gates/content, sum gates across senders, and
   contract gates with values using an einsum or equivalent batched matmul.
4. Compute complete-row mean/log-count/fraction and the small fusion MLP.
5. Mask no-neighbor rows, cast/reshape the added output, and return old state
   output plus support output through the ordinary state-module interface.
6. Concatenate receiver chunks on their receiver axis, as the base already does.

Do **not** materialize `[B,Q,N,A,D]` to multiply every channel and content
coordinate. Use `einsum('bqja,bqjd->bqad', gates, values)` with the required
accumulation precision. Each receiver chunk includes all senders in the initial
implementation, so no global merge across receiver chunks is needed.

If later sender tiling is required, merge raw additive statistics only:

```text
C = sum_tiles C_tile
S = sum_tiles S_tile
M = sum_tiles M_tile
```

Then divide and take `log1p` **once**. Averaging per-tile means/fractions, adding
per-tile log counts, or applying epsilon separately to every tile is wrong.
Sender tiling is a same-equation optimization, but its floating-point order,
gradients, speed, and memory require parity tests and updated benchmark evidence.

### Memory and checkpointing contract

At `B=256,Q=64,N=128`, a separate FP32 support-normalized width-32 view contains
**256 MiB**; width-16 values contain **128 MiB**; four FP32 gates contain **32 MiB**.
Logits, temporary tensors, recomputation, and gradients add to these figures.
The forbidden expanded `[B,Q,N,4,16]` product would alone contain **512 MiB**.
These are tensor sizes, not a peak-memory prediction or sums of necessarily
simultaneous live allocations. The original model's pair cache remains present.

Use the existing non-reentrant chunk-checkpoint strategy. Pass pair slices,
endpoint projections, native attention weights, valid masks, and the correctly
offset off-diagonal mask explicitly. Do not consult a mutable cache during
backward after the outer forward clears it. Keep all computations differentiable
through the original `r` into endpoint and physics-stem projections.

The chunk has no random operations, so the inherited `preserve_rng_state=False`
is appropriate only while that remains true. Keep MHA/dropout, native trimming,
and all BatchNorm updates outside the recomputed chunk. Do not log, append to
external lists, or update diagnostic buffers from inside recomputation. PyTorch
documents this state-consistency requirement in its
[checkpointing guidance](https://docs.pytorch.org/docs/2.5/checkpoint.html).

New support summaries and gates are layer-local: do not reuse them as another
layer's features or carry them into another batch, jet, update, or restart.
Autograd may retain forward activations for backward; that is not a persistent
model state. Preserve the existing shared per-forward pair cache unchanged.
The new receiver summaries need no additional global pair cache or disk cache.

## 7. Initialization, precision, and gradient safety

### Identity start without a dead branch

Construct the fully seeded base first, on CPU under its normal initialization
protocol. Create each new module within an isolated per-layer RNG stream derived
from `(seed, layer, 'support-v1')`; do not advance the base's RNG streams.

- Xavier-uniform gain 1 for gate, content, and hidden projection weights.
- Zero gate and hidden biases; no content or output bias.
- Zero the entire `128 x 32` support output matrix `O_l`.
- Initialize channels independently, not as four identical copied gates.
- No additional zero scalar multiplier, special optimizer group, warm start,
  freezing, gate-entropy regularizer, diversity loss, or new dropout.

Zero bias does not imply gates are identically 0.5: their weights are nonzero.
This allows initial content-dependent differences between support channels.
Initially `O_l=0` recovers the base function while permitting gradients on `O_l`.
Upstream new parameters initially have zero gradients through `O_l`; after a
successful update they should receive finite nonzero gradients on informative
examples. Do not mark legitimate initial zeros as unused or numerical failures.

Keep the existing zero-initialized original State-EV output independent of this
new output projection. In particular, do not feed support features exclusively
through the original state output and then add another zero gate: that can delay
or prevent the intended new learning path.

Never bypass the branch when its output matrix is zero. Such a shortcut removes
its initial gradient and yields a misleading cheap benchmark. All masks should
retain graph-connected zero outputs, including a rank whose entire batch has
one-particle jets. Do not enable DDP unused-parameter discovery to hide missing
graph connections.

### Precision contract

Retain FP32 master parameters, the reference FP16-autocast training, FP32
validation, and pinned TF32/determinism settings. The new normalization uses the
existing FP32 helper (preserving FP64 in mathematical tests). Dense gate/content
and fusion linear layers follow the declared native autocast context.

Cast gate preactivations and content to FP32, then disable autocast for sigmoid,
masked gate sums, gate-content contractions, denominators, `log1p`, fractions,
and construction of the 72-channel summary. In particular, `.float()` alone
outside a disabled-autocast region does not establish the precision of einsum.
The small fusion MLP follows autocast; cast its final output to the original
state-message dtype at the addition boundary. Do not upcast or reorder the
old message just because the new reductions use FP32.

Keep a graph-connected denominator; no stop-gradient normalization. Use finite
masked values and clamped denominators rather than an unguarded `0/0` branch
hidden behind `where`. Test finite extreme logits, zero counts from underflow,
counts close to epsilon, and gates near one. No `nan_to_num`, arbitrary clipping,
or silent replacement of unhealthy valid-pair features with zeros.

Check eligible gate preactivations and projected content for finiteness before
sigmoid could conceal infinities. Aggregate health flags on device and coordinate
failure outside pure chunks across ranks; avoid per-chunk host synchronization.
Preserve the reference's controlled AMP scaler-recovery policy. Persistent
saturation, tiny denominators, nonfinite forward features, or sustained skipped
steps require investigation, not an automatic precision or science change.

## 8. Integration boundaries and analytic parameter budget

Use a new package, provisionally `part_ev_support`. Prefer an instance-local
subclass/adapter of `FastStateMessage` installed in all eight existing
`LayerwiseAttention.attention.state` slots. Retain the exact original receiver,
sender, edge, output, and pair-cache objects, and register new projections once
under a distinct support namespace. Return the combined state message with the
existing `[B,8,N,16]` interface.

This arm does not need a new top-level class-token forward or a modified native
MHA implementation. The existing `MessageAttention` can retain its one native
attention call, ordinary EV, output projection, and masks. Use a dense oracle
with the same specified addition order when checking the adapter.

Preserve base parameter names/buffers and audit their exact identity before
training. Do not register a second copy of base modules inside a retained
reference model, patch installed Weaver, change old model globals, or weaken
the base builder's count assertion. The count check should pass before adding
support modules, then a separate candidate count check should pass afterward.

Provide a support-only diagnostic disable mode if needed. The existing broad
`disable_new` switch can disable original State-EV as well; it is **not** a
valid support-only ablation. Do not expose a disabled training/benchmark mode as
the admitted production model. Unsupported export or external-pair modes should
fail explicitly instead of silently falling back to the base.

| Added component per block | Calculation | Parameters |
| --- | --- | ---: |
| Four gate logits | `4 * 32 + 4` | 132 |
| Content projection | `16 * 32` | 512 |
| Hidden fusion | `32 * 72 + 32` | 2,336 |
| Bias-free output fusion | `128 * 32` | 4,096 |
| **Per block** | | **7,076** |
| **Eight blocks** | | **56,608** |

Expected model total: **2,384,070 parameters**, approximately **2.432%** above
Layerwise State-EV. Baseline shared-bias ParT has **2,143,486** parameters.
These addition/total figures are analytic design counts, not measured candidate
counts. Count actual registered parameters and confirm every new trainable
tensor appears exactly once in the optimizer before admitting a benchmark.

### Compute and persistent storage

The added pair processing remains quadratic: gate/content projections and a
small gate-content contraction. Fusion is per particle. There is no triangle
dimension, sort, additional attention matrix, or second pair-state encoder.

At fixed `N=128`, counting two FLOPs per multiply-add, dense execution of the
new linears and content contractions alone adds approximately **0.198 GFLOPs
per jet per forward** across eight blocks:

```text
2 * 8 * [128^2 * (32*4 + 32*16 + 4*16)
         + 128 * (72*32 + 32*128)] = 197,656,576
```

This includes computed diagonal entries even though they are later masked. It
excludes normalization, nonlinearities, count reductions/divisions, masks, bias
additions, casts, backward, and checkpoint recomputation. It is not a measured
whole-model FLOP ratio or runtime prediction. Additional FP32 normalization and
kernel/memory traffic in eight blocks may matter more than the small weight count
suggests. Measure training and inference separately on the intended hardware.

Extra FP32 weights contain 226,432 bytes before serialization. This does not
bound campaign storage: full optimizer/Lookahead restart state, selected model
copies, atomic-write temporaries, snapshots, logs, and full-validation prediction
outputs also consume space. Budget all concurrently admitted campaigns. Never
persist pair activations or gate maps, delete data, or silently remove previous
artifacts to satisfy a storage guard.

## 9. Required correctness tests

### A. Dense aggregation oracle and distinguishing tests

Write a simple small-input oracle with explicit loops over eligible directed
pairs. Compare its counts, sums, content, features, fused messages, input
gradients, and parameter gradients against vectorized/chunked execution.

Required cases:

- Receiver chunks 1, a non-divisor, 64, and `>=N`; lengths crossing a boundary.
  If sender tiling is implemented, uneven tiles and all-masked sender tiles.
- Noncontiguous masks, different valid lengths in the same batch, one valid
  particle, two valid particles, and invalid receivers. Empty jets reject.
- Deliberately different `r_ij` and `r_ji`, and diagonal/padded feature changes.
  The isolated support branch must ignore excluded features and have zero
  gradients there; the whole model retains its original diagonal behavior.
- Constant gates and repeated content: verify `C=M*s`, `rho=s`, and unchanged
  content above epsilon. Replicate frozen eligible pair features and test the
  predicted count/log-count change; this catches accidental normalization away
  of support.
- Unequal tile counts and heterogeneous content for which averaging tile means
  gives the wrong answer. Verify merging `C/S/M` before nonlinear operations.
- Different learned gates with identical raw degree: demonstrate that counts
  are not hard-coded to degree, native attention sums, or one across channels.
- Finite logits near saturation, zero numerical gate weights, tiny counts on
  either side of epsilon, and signed content. Test gradcheck away from the
  denominator clamp's kink; test explicit finite behavior at the kink separately.
- Particle permutations jointly applied to both pair axes: summaries/messages
  must permute on the receiver axis. Eval-mode logits should be invariant with
  trimming disabled; stochastic training dropout is a separate equality question.

Use FP64 gradchecks for the pure equations and declared FP32 forward/gradient
tolerances, initially `rtol=1e-5, atol=1e-6` and `rtol=1e-4, atol=1e-5` respectively.
AMP uses a separate pinned, like-precision policy. Do not compare a different
precision path with an impossibly strict tolerance and then label it broken;
do not loosen tolerances without diagnosing the difference either.

### B. Full-model preservation and active learning

1. Verify the base subset, buffers, initialization RNG progression, parameter
   order, and total count. Identical seeds must produce identical candidates.
2. With `O=0`, compare logits and base gradients to the unchanged base, including
   nonzero original State-EV outputs and distinct layerwise bias heads. Merely
   testing a fresh base with its original state outputs zero is insufficient.
3. With support `O` deliberately nonzero, compare the dense oracle and chunked
   adapter through full-model forward/backward, checkpointing on/off, and AMP.
4. Check nonzero support-output gradients at initialization, then upstream gate,
   value, and hidden gradients after a successful update. Confirm original EV,
   all eight State-EV branches, pair stem, all bias heads, and classifier still
   receive their expected gradients.
5. Count trimmer, pair encoder, MHA, and BatchNorm calls/updates. Backward
   recomputation must not rerun native stochastic/stateful operations.
6. Test learned nonzero hidden bias on no-neighbor rows: added messages remain
   zero. Confirm safe graph-connected zero gradients rather than missing grads.
7. Test eval-mode batch independence, two outstanding forwards followed by
   backward, exception cleanup, and no stale pair/cache state on the next batch.
8. Verify support-only disabling leaves original State-EV active. A sensitivity
   diagnostic must not inadvertently compare with ordinary EV or ParT instead.

### C. DDP and complete-state restart

Test the intended two-rank SyncBatchNorm runtime, unequal per-rank trimmed
lengths, and a complete rank batch without eligible pairs under
`find_unused_parameters=False`. Match single/DDP comparisons under compatible
normalization and loss reduction, not native BatchNorm versus SyncBatchNorm.
Both replicas must agree on update success, scaler state, and parameter state.

Round-trip model, complete Ranger state including Lookahead slow weights and
counter, GradScaler, rank RNG, native trimming state, update/validation counters,
best-checkpoint bookkeeping, and exact data-stream replay information. Compare
uninterrupted repeats first, then uninterrupted versus resumed sequences across
Lookahead and validation boundaries. Do not relax the restart contract to hide
reduction-order nondeterminism or state omissions. Recompute support statistics
after restart; none belong in persistent checkpoint state.

## 10. Full recipe and scientific comparison

Preserve the authenticated `ranger_full_v1` recipe and effective reference runtime:

| Setting | First scientific run |
| --- | --- |
| Data | Official 100M train pool; unchanged preprocessing, maximum 128 particles |
| Training budget | 1,000,000 update attempts, global batch 512; nominal 512M exposures |
| Optimizer | Pinned Weaver RAdam plus complete-state Lookahead |
| RAdam | LR start `1e-3`, betas `(0.95,0.999)`, epsilon `1e-5`, weight decay 0 |
| Lookahead | Every 6 successful steps, mixing coefficient 0.5 |
| LR schedule | Unchanged reference schedule to `1e-5`, decays at reference checks 35 through 49 |
| Gradient clipping / early stopping | None / disabled |
| Validation | 50 checks, every 20,000 attempts, existing 1.28M-event streaming procedure |
| Selection | Highest validation accuracy, first tie retained |
| Post-training evaluation | Selected checkpoint on official full 5M validation |
| Seed | Fresh 101; no trained-model or benchmark-weight warm start |

Import the audited LR function rather than translating its boundary indexing
from prose. Record actual layerwise architecture, SyncBatchNorm, precision,
worker counts, and effective settings even if legacy recipe descriptive fields
use older shared/single-GPU labels. Do not silently rewrite reference hashes.

Validation checks are not full dataset epochs. Log attempted and successful
updates, skips, LR, per-check accuracy and AUC, and selected-check identity. Full
5M validation overlaps checkpoint selection; it is not independent testing.
Keep the 20M test split untouched during this design and screening phase.

Compare primarily with full Layerwise State-EV at matched update cutoffs and
accuracy-selected checkpoints. Include baseline ParT and ordinary EV as context,
but label their runtime differences. One-GPU native BatchNorm and two-GPU
SyncBatchNorm contrasts do not isolate architecture alone. Disclose historical
restart/segmentation differences, even when exact stream replay is implemented.
Any newly trained control or extra seed needs separate approval.

### Diagnostics that can distinguish support from counting particles

On a bounded, fixed validation diagnostic population in isolated eval mode:

- Summarize gate mean, within-jet/pair variation, saturation fractions, count,
  log-count, fraction, and denominator-floor usage for each layer/channel.
- Report cross-channel correlations and content-summary similarity. Collapse
  to similar channels is evidence to inspect, not an automatic training failure.
- Compare `C` with `M`, and gate/fraction variation within multiplicity bins.
  Nearly constant gates yield approximately `C=s*M`; that weakens a claim of
  relation-specific support even if accuracy improves.
- Measure added-message RMS relative to original State-EV and the full native
  message, plus new-parameter gradient norms from ordinary training updates.
- Report validation accuracy and classwise QCD rejection in fixed retained-
  multiplicity bins with adequate counts; also record pre-truncation multiplicity
  for diagnostics only if already available without changing model inputs.
- Check effects near the truncation cap. If an independent domain/generator
  validation sample is later available and approved, test robustness there;
  do not assume multiplicity-sensitive gains transfer to detector conditions.

Do not collect full pair maps every step, consume the live training RNG/stream,
update BN statistics in diagnostics, or add auxiliary optimizer steps. Use small
aggregated reports, not persistent pair caches. Gate visualizations are not
causal physics explanations; large support counts are not proof of strong gains.

Post-training sensitivity checks may disable only support or remove count
coordinates while retaining content. Label these as distribution-changing
interventions on a trained model, not substitutes for independently trained
ablations or estimates of guaranteed generalization benefit.

### Follow-up controls if this first arm is promising

Propose separately, do not launch automatically:

1. **Content only:** keep learned gates, content, and fusion layout, but replace
   both log-count and fraction input coordinates with zero during all training
   and evaluation. This tests support statistics beyond a parallel learned
   mean-content branch. Some fusion inputs become inactive; report effective
   capacity and compute rather than claiming a perfect parameter match.
2. **Mean content plus raw degree:** use uniform fixed gates, uniform mean of
   the same projected content, and `log1p(M)`/constant fraction features with
   comparable fusion. This tests whether adaptive pair support improves over
   a generic content-and-multiplicity path; report any capacity mismatch.
3. **Capacity and reproducibility:** a suitably documented larger base control
   and matched additional seeds, before claiming a repeatable architectural gain.

Use accuracy, macro AUC, and classwise rejection at declared signal efficiencies,
with QCD passing counts and uncertainty where estimable. Tail rejection changes
can be noisy with few passing backgrounds; avoid averaging rejection ratios as
a universal score. Match selection rules and populations, and distinguish
finite-evaluation uncertainty from variation across training seeds.

## 11. Independent benchmark and admission requirements

Implemented benchmark deliverables (including aggregation, oracle and diagnostic helpers):

```text
part_ev_support/{model,protocol,benchmark}.py
scripts/run_part_ev_support.py
scripts/submit_part_ev_support.py
sbatch/run_part_ev_support.sh
sbatch/submit_part_ev_support.sh
tests/test_part_ev_support_{model,aggregation,benchmark,launcher}.py
```

Reuse audited infrastructure, not unverified copies of whole trainers. Use a new
campaign directory with hashed source snapshot, full model config and parameter
inventory, authenticated reference, exact train/validation inventory, and frozen
recipe/environment/runtime. Freeze precision and chunking policies as well as
dimensions. Existing jobs must keep using their existing snapshots, not today's
mutable checkout. Never edit old manifests to make a comparison pass.

Environment inventories should use stable interpreter installation roots, not
lazy additions to `sys.path`. A real mismatch requires diagnosis or an explicit
compatibility recovery path, not blanket exclusion of environment checks. This
arm needs no new compiled backend, dependency, or package upgrade for its first
implementation.

Initial scheduling intent:

- CPU prepare: 2 hours. GPU benchmark: **5 hours 30 minutes**. CPU summary:
  15 minutes. All preliminary requests remain below six hours.
- Benchmark on two matching same-node A100s, DDP/SyncBatchNorm, global batch 512,
  local batch 256, one training worker per rank. Reference rank-zero validation
  uses batch 512 and two workers. Start with 8 CPUs and 64 GiB host RAM.
- Verify current partition/account/QoS limits and aggregate persistent-storage
  headroom. Do not silently reroute to another cluster or change normalization.

Measure the unchanged full Layerwise State-EV and the new arm under the same
allocation/settings where feasible, using separate disposable model instances
and controlled streams. Record measurement order and cold/cache effects. Aim
for the inherited 20 warm-up and 4,200 measured steps with repeated fetch windows,
not only a short cached-batch loop. Production always starts fresh.

Required GPU evidence:

1. Zero-output identity and active nonzero full-model forward/gradient checks,
   with an explicit record of tolerances and the mask/degenerate cases tested.
2. Sustained real-data FP16 training and confirmed successful optimizer steps;
   variable native lengths, full-128 backward, and graph-connected no-pair cases.
3. FP32 end-to-end validation timing on at least the inherited 128k-event sample,
   and full-length rank-zero validation memory at its actual batch size.
4. Fixed-128 inference throughput and batch-one latency in declared precision;
   report input preparation separately from device-only timing where measured.
5. Complete-state restart smoke test, finite state, and checkpoint I/O timing.
6. Per-device allocated and reserved peaks, host RAM, branch diagnostic health,
   and actual trainable parameter count.

Require peak allocation below 90% of each GPU's capacity (10% headroom), and
retain the sustained AMP skip guard: initially below 5% after the allowed
calibration phase. Report
reserved memory too; allocation-only success does not eliminate fragmentation
or other runtime-memory risks. Do not disable AMP, reduce batch size, omit the
new branch at initialization, or suppress a failed health check to get a pass.

Enforce a safe internal deadline. Missing required measurements mean incomplete
admission; Slurm `COMPLETED` alone is not approval. If a benchmark is too long or
memory-heavy, profile and document same-equation optimizations first. Changes
to widths, blocks, support definitions, precision, or training recipe require a
new declared variant and evidence, not a hidden fallback.

Project walltime from conservative measured training windows, all 50 validation
checks, checkpoint I/O, and an explicit margin such as 30%. Queue delay,
contention, and restart replay are additional uncertainties. A parameter/FLOP
estimate does not replace these measurements.

## 12. Later production path and completion criteria

Production implementation/submission follows a separate review of passed GPU
evidence. Intent remains one long allocation, not a scheduled chain of three-day
segments. Derive walltime from the benchmark and live limits rather than assume
twelve days will fit. Retain safe complete-state restart despite that intent.

Use an independent model-local `train -> full-validation evaluate -> report`
chain. It must not depend on unrelated readout, triangle, or existing extension
jobs. Do not cancel/requeue other campaigns, create a cross-cluster duplicate,
or initiate unapproved additional model/control runs.

Retain atomic latest/full-state checkpoints every 1,000 attempts and at
validation/stop boundaries, along with safe accuracy-selected model candidates.
Resume from latest complete training state, not best weights alone. Authenticate
source, recipe, inventory, environment, and runtime before resume. Require the
full 1,000,000-attempt budget and all 50 completed validation checks before
evaluation. Planned interruption is not completion.

Persistent storage admission must include other registered campaigns and
temporary checkpoint/evaluation writes. Reuse completed artifacts only after
identity checks; never overwrite previous results or delete them automatically.

## 13. Implementation milestones and definition of ready

1. **Reference audit:** pin source/environment/data, effective runtime, base
   parameters/buffers, and complete optimizer/schedule/restart conventions.
2. **Math oracle:** implement masked gates, additive statistics, safe means,
   exact feature layout, zero-neighbor handling, and dense forward/gradient tests.
3. **Minimal integration:** install independent support adapters in all eight
   state modules; preserve the native attention/classification path and RNG.
4. **Efficient execution:** jointly compute original state/support chunks, avoid
   expanded products, checkpoint pure chunks, and prove active-path parity.
5. **Runtime integrity:** verify mixed precision, DDP, degenerate rank batches,
   cache lifetime, parameter membership, complete-state restart, and dry runs.
6. **Benchmark queue readiness:** independent frozen prepare/benchmark/summary
   launcher, evidence schema, resource/storage guards, README commands, and
   preliminary walltimes below six hours. No production submission in this stage.
7. **Production readiness after review:** evidence-bound model-local chain,
   measured walltimes, guarded resume/evaluation/reporting, explicit queue steps.

Expected total is **2,384,070 parameters**. No new branch can bypass work at
zero initialization; excluded pairs and zero-neighbor rows must be safe;
counts must remain unnormalized soft support; the base's complete architecture
must remain available. CPU math tests are necessary, not proof of GPU runtime
safety or classification improvement.

A successful research outcome is an interpretable improvement in accuracy
and/or useful rejection at documented cost, confirmed with matched comparisons.
It is not merely a count-correlated feature, a large message norm, a single
favorable checkpoint, or evidence that more parameters were trainable.

## 14. Code entry points

- [Full Layerwise State-EV composition and counts](model.py).
- [Optimized pair-state computation, cache, and chunk checkpointing](../part_ev_state_speed/model.py).
- [Original state features, precision helper, and message integration](../part_ev_message_study/model.py).
- [Layerwise bias wrappers and native pair/mask binding](../part_layerwise_screen/model.py).
- [Pinned native model wrapper and configuration](../part_ev_fullscale/model.py).
- [Complete-state Ranger optimizer](../part_ev_fullscale/optimizer.py).
- [Full schedule source](../part_ev_state_fullscale/protocol.py).
- [Existing two-A100 benchmark contract](../part_ev_state_bias/protocol.py).
- [Existing production admission and restart reference](../part_ev_state_production/README.md).

**Implementation principle:** retain what useful relationships say and how much
learned support they have, through a small parallel message path that preserves
the full successful backbone, with explicit tests for whether the benefit is
relational evidence, multiplicity, or simply extra capacity.
