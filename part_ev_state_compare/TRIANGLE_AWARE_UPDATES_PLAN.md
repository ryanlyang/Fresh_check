# Sparse triangle-aware updates on full Layerwise State-EV

Recorded: 2026-11-16. Detailed first-design and implementation plan.
**Status: model and independent SPORC benchmark implemented; queue-ready.**
See [implementation and queue instructions](../part_ev_triangles/README.md).
No jobs submitted by the implementation; GPU admission and accuracy remain
unmeasured. Production is intentionally not implemented yet. The learning
probe uses eight successful Ranger updates to cross early rectification and
Lookahead before checking every new matrix has changed; the model and recipe
remain as specified below.

This specifies idea B in the
[three-addition overview](STATE_EV_READOUT_TRIANGLE_SUPPORT_PLAN.md).
It is independent of the [direct pair-readout plan](DIRECT_PAIR_CLASSIFIER_READOUT_PLAN.md)
and the support-aware aggregation proposal. Its neighbor-cutoff tie rule is an
explicit refinement of the overview, with a corresponding cost caveat below.

## 1. Objective: relationships reasoning about other relationships

Retain full, noncompact, standard-relations, all-eight Layerwise State-EV and
add a small late-block message that combines three related particle pairs.
For target pair `i-j`, use the current features of `i-k` and `k-j` to supply
additional context, with the target pair deciding which components are useful.

The existing index convention is receiver first, sender second:

```text
r_ij: relation used for a message from j to i
r_ik and r_kj: the two-edge path j -> k -> i
```

The pair being updated is therefore `i-j`; `k` is an actual third constituent,
not a learned token, truth label, or reconstructed subjet. The target-pair gate
closes the triangle by also reading `r_ij`.

### Why this could improve classification

Different configurations can have similar individual pair features but different
patterns among surrounding constituents. An explicit product of two related
edges can make such configurations easier to learn than requiring every useful
relation to be compressed into a particle state first. Late-layer pair features
already include learned context, so the new branch combines contextual evidence,
not just raw angular distances.

This is a hypothesis about useful information flow and inductive bias. Ordinary
attention and State-EV already transmit information about third particles through
particle states; the base is not fundamentally unable to learn three-particle
patterns. A gain could be substantial if that indirect representation is a
bottleneck, but could be negligible if it already suffices. No accuracy increase
or particular physics-class improvement can be predicted reliably yet.

Triangle interactions in
[AlphaFold's pair representation](https://www.nature.com/articles/s41586-021-03819-2)
are a precedent for edge-to-edge reasoning. This plan is a sparse, directed
path-style adaptation for jet classification, not the full Evoformer, a copy
of its exact contraction conventions, or evidence of jet-tagging performance.

### What this is not

- It is not pair memory: no persistent pair state is carried between layers.
- It is not the earlier relation-state multiplication arm: the multiplicative
  inputs here come from different pairs, not only the target pair's endpoints.
- It is not a new attention-bias model: attention logits and probabilities
  retain their original construction.
- It is not a replacement for State-EV or a last-two/last-four State-EV model.
- It does not add Selected Relations, pair-to-classifier readout, support counts,
  compaction, a clustering objective, or additional training labels.

## 2. Base model and locked first design

Build on `state_layerwise` from [the reference assembly](model.py), with
**2,327,462 parameters**. Keep all eight width-128 particle blocks, all eight
width-32 State-EV branches, ordinary EV, the width-64 shared physics-pair stem,
independent layerwise bias heads, two original class-attention blocks, and the
original classifier. Preserve the base's same-math caching and checkpointing.

The motivating seed-101 two-A100/SyncBatchNorm run achieved **86.3151%
training-time validation accuracy at check 49 of 50**. This is historical
validation evidence, not an independent test result or a reproducibility estimate.

| Component | First implementation |
| --- | --- |
| Extra triangle blocks | Blocks 6 and 7, zero-based: final two particle blocks |
| Original State-EV | Unchanged, width 32, all eight blocks |
| Triangle width | 16 |
| Intermediary neighborhood | Eight nearest-neighbor units per receiver; cutoff ties share weight |
| Geometry | Native rapidity/azimuth distance from the trimmed four-vectors |
| Neighborhood lifetime | Once per forward, shared between the two triangle blocks |
| Learned projections per triangle block | Bias-free `A:32->16`, bias-free `B:32->16`, `C:32->16` with bias |
| Factors / target gate | `tanh(A u_ik)`, `tanh(B u_kj)`, `sigmoid(C u_ij + c)` |
| Intermediary aggregation | Weighted mean over distinct eligible third particles |
| Sender aggregation | Original attention weights, including the same dropout realization |
| Triangle output | Separate head-specific `[8,16,16]` projection, initialized to zero |
| New normalization | Non-affine channelwise LayerNorm on the readout view of `r`, epsilon `1e-5` |
| New losses / dropout / temperature | None |
| First science task | Fresh seed 101; unchanged full Ranger recipe |
| Intended training runtime | SPORC tier3, two same-node A100s, DDP/SyncBatchNorm |

The extra branch is late because this exposes richer endpoint states while
limiting cost. It does not reduce the original State-EV coverage. Width 16 and
neighbor budget eight are fixed first-model choices, not validated optima or
automatic settings to shrink when a benchmark fails.

## 3. Geometry, neighbor weights, and order-independent cutoff ties

### Geometry contract

Compute geometry once **after the native trimmer** has permuted/truncated all
particle arrays. Run it under `no_grad`, outside autocast, using FP32 versions
of the same four-vectors supplied to the physics-pair encoder:

```text
(pt, rapidity, phi) = pinned to_ptrapphim(v, return_mass=False, eps=None)
distance2_ik = (rapidity_i - rapidity_k)^2 + delta_phi(phi_i, phi_k)^2
```

Use the pinned native wrapped `delta_phi`, not unwrapped subtraction, and use
rapidity rather than silently substituting pseudorapidity. No square root is
needed for ordering. Sanitize padding before geometry evaluation so invalid
four-vectors do not contaminate reductions; reject nonfinite geometry of valid
constituents instead of replacing it with invented coordinates.

Eligible neighbor candidates have valid `i`, valid `k`, and `k != i`.
No new geometry features are supplied to the learned model: geometry selects
the message graph only. Gradients do not pass through neighbor selection;
gradients through the existing learned pair features remain unchanged.

### Exact selection rule

Store nonnegative, forward-local weights `nu_ik` with `nu_ii=0` and zero padding.
For a valid receiver with at most eight eligible neighbors, assign weight one
to every eligible neighbor. Otherwise let `tau_i` be its eighth-smallest
eligible distance squared, and define:

```text
L_i = candidates with distance2_ik < tau_i
E_i = candidates with distance2_ik == tau_i

nu_ik = 1                            if k in L_i
nu_ik = (8 - len(L_i)) / len(E_i)     if k in E_i
nu_ik = 0                            otherwise
```

Equality is equality of the computed FP32 distance, not a tunable `isclose`
radius. Obtain the cutoff value, not an arbitrary subset of tied indices.
For invalid receivers, set the whole row to zero. Store the weights in FP32.
The row mass is `min(8, number_of_valid_nonself_neighbors)`, within rounding.

This fractional boundary rule is intentional. When several candidates tie
across the cutoff, selecting by particle position would make model behavior
depend on how those particles were ordered. Sharing the remaining weight treats
the tied candidates symmetrically without adding a learned ranking or extra data.

**Cost qualification:** ordinary rows have at most eight nonzero neighbors.
A cutoff tie can have more than eight actual contributing particles, despite
total neighbor weight eight. Therefore the worst case is not strictly
`O(N^2 * 8)`; pathological all-tied inputs can require cubic work. The dedicated
tie path, maximum `N=128`, and benchmark requirements below make this explicit.
Do not call ties rare without measuring their incidence on the real stream.

This is preferable to an undocumented index tie-break, dropping tied jets, or
claiming strict eight-particle cost while secretly evaluating more neighbors.
It refines the overview's hard-set formula only at cutoff ties.

### Excluding the target sender

For target `i-j`, exclude `k=j` as well as `k=i`, and require valid `i != j`.
Do not choose a new neighborhood per target or backfill the removed sender:

```text
eligible_weight_ijk = nu_ik * valid_j * (i != j) * (k != j)
denominator_ij = sum_k eligible_weight_ijk
```

For valid off-diagonal targets, the denominator is equivalently
`sum_k nu_ik - nu_ij`. It can be noninteger under tie sharing. A target with no
eligible third particle contributes exactly zero. The first version uses a
weighted mean, not an unnormalized count of supporting paths.

## 4. Pair features and directed triangle equations

For each triangle block `l in {6,7}`, use its ordinary current State-EV features:

```text
zbar_ij = LN64(shared_pair_stem_ij)
xbar_i  = LN128(native_pre_attention_query_i)
g_ij = E_l zbar_ij
p_i  = P_l xbar_i
q_j  = Q_l xbar_j
r_ij = GELU(LN32(g_ij + p_i + q_j + p_i * q_j))
```

Preserve the reference normalization order, including both the native block
pre-attention normalization and the state module's normalization. Do not change
the original State-EV feature or its message contraction. The extra normalization
below is a separate view for the triangle branch:

```text
u_ij = LayerNorm(r_ij, affine=False, eps=1e-5)
a_ij = tanh(A_l u_ij)
b_ij = tanh(B_l u_ij)
gate_ij = sigmoid(C_l u_ij + c_l)

numerator_ij = sum_{k distinct from i,j} nu_ik * (a_ik * b_kj)
mean_ij = numerator_ij / denominator_ij       # only when denominator > 0
t_ij = gate_ij * mean_ij
```

`*` is a channelwise product, not a dot product. The 16 channels can learn
different combinations of relationship evidence. The factors and gate are
head-shared within a block, while the eventual output projections are head-specific.
All projections are independent between the two triangle blocks.

Mask `a`, `b`, and `gate` to zero for invalid or diagonal triangle pairs, without
altering diagonal features used by original State-EV. In particular, `b_jj=0`
helps enforce the exclusion of `k=j`, but the denominator must still exclude
that particle: zeroing a contribution is not the same as removing it from a mean.

Define `t_ij=0` for invalid targets, diagonal targets, or zero denominator.
Use a safe denominator of one on the zero branch before masking; do not compute
`0/0` and hide it afterward. Every nonzero `t_ij` corresponds to three distinct
valid particles. No gate-dependent denominator or additional softmax is used.

All factors and target gates in a block come from **the same pre-update states**.
First compute the necessary projections, then perform triangle mixing. Do not
update a pair in place and let a later chunk read the updated version. Block 7
recomputes its own projections from its incoming states; no block-6 pair cache
is its scientific input. The new information reaches it through particle updates.

### Added message and native integration

Let `alpha_hij` be the original unaveraged, post-dropout attention weights:

```text
pooled_triangle_hi = sum_j alpha_hij * t_ij
triangle_hi = T_lh pooled_triangle_hi          # 16 -> 16 for each of 8 heads

joined_extra = concat_heads((ordinary_EV + original_State_EV) + triangle)
attention_output = original_native_attention_output
                 + masked_linear(joined_extra, native_out_projection_weight)
```

The extra linear path uses the native output weight **without adding its bias
again**. Retain the reference valid-receiver mask, head scaling, attention-output
normalization, dropout, residuals, FFN, and all class-attention behavior.
Native MHA is called once per block; all existing messages and the triangle
message use that one native dropout sample. The triangle branch adds no extra
dropout draw and changes no Q/K or physics-bias logits directly.

## 5. Shape and memory contract

`B` is local batch size, `N<=128` the native trimmed length, `H=8`, and `d=16`.

| Quantity | Shape |
| --- | --- |
| Valid particle mask | `[B,N]` |
| Neighbor weights | `[B,N,N]`, forward-local, non-differentiable |
| Existing normalized physics pairs | `[B,N,N,64]` |
| Original State-EV feature chunk | `[B,C,N,32]`, initially `C=64` |
| Each narrow projection cache `a,b,gate` | `[B,N,N,16]` |
| Original attention weights | `[B,8,N,N]` |
| Ordinary-row candidate slots | `[B,N,8]` plus validity/weights; safe dummy slots allowed |
| Mixed triangle tile | `[B,Q,J,16]` |
| Intermediary gather tile | `[B,Q,K_tile,J,16]` |
| Pooled triangle output | `[B,8,N,16]` |
| Output projection parameters per triangle block | `[8,16,16]` |

First engineering settings: feature chunk `C=64`, triangle receiver tile `Q=16`,
sender tile `J=64`, and intermediary tile `K_tile=4`. Tile sizes are execution
choices, not changes to the graph or architecture; any adjustment needs numerical
parity and a new recorded benchmark configuration.

For `B=256,N=128`, one FP32 narrow projection cache occupies **256 MiB**.
Three caches are **768 MiB per triangle block**, before gradients, concatenation
temporaries, and the unchanged base activations. Both blocks' graphs can remain
live in training; deleting a Python reference does not free tensors retained
for backward. Do not present 768 MiB as the full added peak.

A `[256,16,4,64,16]` FP32 gather tile is **64 MiB**. A full
`[256,128,8,128,16]` gather tensor would be **2 GiB**, and a dense
`[256,128,128,128,16]` triplet tensor would be **32 GiB**. Neither full tensor
belongs in the implementation, including transiently or only in backward.

The first version deliberately stages three narrow caches because different
target pairs need access to other pairs' features. Unlike ordinary State-EV,
this access cannot be provided by pretending every receiver chunk is independent.
Do not retain the full width-32 `r` tensor in addition to these caches.

## 6. Efficient, differentiable two-stage implementation

### Stage A: original message plus narrow pair projections

At each of the two affected blocks, call native MHA once and obtain its original
weights. In pure query-chunk functions:

1. Read the existing normalized-pair cache and endpoint projections explicitly.
2. Compute `r_ij` exactly once for the chunk.
3. Compute the original State-EV message with unchanged arithmetic.
4. Compute the three new narrow projections from a separate `u_ij` view.
5. Apply triangle masks and return old-message pieces plus `a,b,gate` and
   preactivation health flags.

Concatenate the narrow projections along the receiver axis, preserving both
directed pair axes. Concatenation is differentiable; do not fill a detached
preallocated cache with values under `no_grad`. Every later triangle contribution
must backpropagate to its two contributing edges and its target gate.

Keep new caches as local tensors returned/passed explicitly, never module buffers,
cross-batch state, or disk files. A dense all-pair projection cache is not the
same as dense all-triplet materialization.

### Stage B: bounded sparse interaction tiles

For an ordinary receiver/sender tile, initialize a connected zero numerator
and gather the up-to-eight candidate intermediaries in tiles of four:

```text
numerator_tile += sum_over_k_tile(
    eligible_weight_ijk * a_ik * b_kj
)
t_tile = gate_ij * numerator_tile / safe_denominator_ij
t_tile = apply_target_and_denominator_mask(t_tile)
pooled_triangle_hi += sum_over_j_tile(alpha_hij * t_tile)
```

Carry out the reductions in FP32. Apply `T_lh` after pooling; a linear projection
commutes with the sum, so there is no need to build a head-specific pair-value
tensor `[B,H,N,N,16]`. This is the same pooling-before-projection principle as
the existing EV implementation.

Use flat pair/batch indices or an audited equivalent for gathers. Avoid expanding
the full `b` cache across all receiver/candidate axes before indexing: a view
can be cheap in forward but cause huge gradient intermediates in backward.
Verify that repeated references to a pair accumulate all of its gradients;
do not deduplicate occurrences that have different target weights.

### Boundary-tie rows

Rows with more than eight nonzero neighbor weights use a separate packed-row
path. Process only those rows, with bounded receiver/sender/intermediary tiles,
and retain all fractional weights. Do not increase the candidate dimension for
every normal row because one row has a large tie group.

A first bound is at most 256 packed receiver rows, 64 senders, and four
intermediaries per fallback tile. Continue until all nonzero candidates in
those rows have contributed. Bound allocations, not the mathematical set of
contributors. No random truncation, hidden tie cap, or skipping such jets.

Record tie handling and any dynamic-index compaction/synchronization overhead.
Neighborhood construction and row packing happen once per forward, not once
per triangle block or during backward. Candidate-list storage order can affect
rounding, but must never change membership or weights. Permutation tests use
declared numerical tolerances rather than bitwise promises.

### Dense contraction oracle and possible engineering fallback

For small tests, a useful exact alternative is a per-channel matrix product:

```text
numerator_ijc = sum_k (nu_ik * a_ikc) * b_kjc
```

With `nu_ii=0`, `b_jjc=0`, and the correct target/denominator masks, this is the
same scientific model, including cutoff ties. It does **not** require an explicit
triplet tensor, but a dense matrix multiply still does cubic arithmetic.

Dense GEMM may be competitive with many small irregular gathers on some GPUs.
Use it as the mathematical oracle and, if profiling warrants, a separately
measured same-model backend candidate. Do not silently switch the production
backend or keep claiming sparse FLOPs for a dense implementation. Backend and
reduction order belong in the admitted runtime identity.

## 7. Numerical stability, initialization, and autograd

### Precision

Preserve the base's FP16-autocast training and FP32 master parameters.
The new `A/B/C` dense projections follow autocast. New LayerNorm, `tanh`, sigmoid,
projection-cache storage, intermediary products/reductions, normalization by
neighbor mass, and triangle attention pooling use FP32 with autocast disabled.
The small final `T` contraction follows the declared autocast policy and its
result is cast to the original State-EV message dtype before the addition.
FP32 validation uses no autocast; pure FP64 tests retain double precision.

Do not change the old message's precision because the new branch uses FP32
accumulation. Check finite projection preactivations before bounded nonlinearities
could hide infinities; monitor intermediates and outputs as well. No `nan_to_num`,
arbitrary score clipping, or silent conversion of a failure into a zero message.
Collect health flags on device and coordinate failure across ranks outside pure
chunks. Retain the established AMP recovery policy for gradient overflow.

### Initialization

Build the fully seeded reference first. Use isolated per-layer RNG seeds derived
from `(seed, layer, "triangle-v1")` for the new parameters, without advancing
the base's initialization/dropout streams:

- Xavier-uniform gain 1 for `A`, `B`, and `C` weights.
- `A` and `B` have no bias; `C` bias is zero.
- All entries of the head-specific output `T` are zero.
- No zero scalar gate in addition to `T`; no new dropout or auxiliary loss.

At initialization, the complete model has the base function. `T` can receive
gradients on a nondegenerate batch; upstream triangle parameters initially have
zero gradients through `T` and should activate after it updates. Do not bypass
the branch when `T` happens to be zero, freeze its factor projections, or apply
an extra learning-rate multiplier.

For one-/two-particle jets the triangle message is identically zero, while all
base computations remain active. Use masked, graph-connected calculations and
safe dummy indices rather than dropping new parameters from a rank's graph.
Test batches/ranks with no valid triangles under `find_unused_parameters=False`.
Reject truly empty jets consistently before distributed execution proceeds.

### Recomputation and determinism

Use pure non-reentrant checkpointed functions for Stage A and Stage B. Pass all
graph-connected caches, masks, neighbor weights/indices, and native attention
weights needed by recomputation explicitly. Never read a mutable module cache
after the outer forward clears it. Preserve gradients through all cached factors.

Do not checkpoint MHA, trimmers, BatchNorm updates, graph construction, or a
function that appends to external state. `preserve_rng_state=False` is appropriate
only for deterministic, randomness-free chunk functions. PyTorch describes the
recomputation-state risk in its
[checkpoint documentation](https://docs.pytorch.org/docs/2.5/checkpoint.html).

Repeated-index GPU gather backward can involve accumulation-order sensitivity.
Measure uninterrupted-repeat behavior before interpreting resume differences.
The implementation must meet the existing restart contract; do not weaken it
because the new gather path is inconvenient. If necessary, select an audited
fixed-order reduction or the same-model dense-contraction backend and benchmark
again. Any change to deterministic runtime flags must be recorded and audited
for its effects on the base as well as the new branch.

## 8. Integration boundaries and expected parameter count

Use a new package, provisionally `part_ev_triangles`, with an instance-local
forward adapter. Keep all original modules/parameter names and base buffers.
Register each new module exactly once under a clearly identifiable namespace
in blocks 6 and 7. Do not patch installed Weaver or modify old model globals.

Recommended forward order:

1. Validate inputs; run the native trimmer once, retaining the matching vectors,
   features, and valid mask.
2. Construct and freeze forward-local neighbor weights/indices from those vectors.
3. Run the original particle embedding and pair encoder once; bind the original
   eight layerwise biases and shared pair stem.
4. Run blocks 0 through 5 unchanged.
5. Run each of blocks 6 and 7 with its own current-state Stage A, Stage B, and
   additive triangle output. Preserve the exact native particle-block operation order.
6. Run the two original class-attention blocks, normalization, and classifier.
7. Clear existing pair bindings/cache in success and exception paths. Do not
   serialize or preserve a neighborhood or projection cache across forwards.

The [explicit pair-memory forward adapter](../part_ev_pair_memory/model.py) is
an integration precedent, not a dependency on its memory model. A new adapter
must authenticate the pinned native source and preserve native `c_attn` layout,
residual scaling, normalization, dropout, and output-projection semantics exactly.
Reject unsupported extra-pair/export modes instead of returning baseline outputs.

| Added component per triangle block | Parameters |
| --- | ---: |
| `A`, bias-free `32 -> 16` | 512 |
| `B`, bias-free `32 -> 16` | 512 |
| `C`, `32 -> 16` plus bias | 528 |
| Eight `16 -> 16` output projections | 2,048 |
| **Per block** | **3,600** |
| **Two blocks** | **7,200** |

Expected model total: **2,334,662 parameters**, approximately **0.309%** above
the base. This is an analytic count, not a measured implementation result.
Assert it after construction, preserve the base subset exactly, and audit all
new parameters' optimizer membership. The base count check must still pass
before adding the new modules. No weights are shared between the two triangle
blocks merely to force the count to match.

## 9. Correctness tests and meaningful failure cases

### Geometry and selection

- Compare wrapped rapidity/azimuth distances with a simple CPU oracle, including
  azimuth values on opposite sides of the wrap boundary.
- Check invalid/self exclusion, zero padding, correct row mass, fewer than eight
  neighbors, exactly eight, and ordinary untied cutoffs.
- Check a tie group entirely inside the selected set and a tie group crossing
  the eighth-neighbor cutoff. Verify the fractional weights exactly as specified.
- Test coincident coordinates, nonidentical features at identical distances,
  identical particles, and a jet with every eligible distance tied.
- Jointly permute particle axes; the neighbor-weight matrix must permute with
  them, rather than selecting different tied particles by position.
- Verify construction uses the post-trimming arrays and runs once, with no labels,
  validation-dependent choices, added files, or gradient through geometry selection.

### Mathematical oracle

Implement a tiny explicit loop over valid `i,j,k`, with all three distinct, and
the declared `nu` weights. Compare it with dense per-channel contraction and
the tiled sparse/tie backends. Match intermediate means, gated pair features,
pooled messages, outputs, and gradients with respect to factors, gates, native
attention weights, and projection parameters.

Tests must catch:

- Accidentally using `b_jk` instead of `b_kj`; construct deliberately directed
  inputs so an incorrect transpose cannot pass through symmetry.
- Keeping `k=j` in the denominator after zeroing its numerator, including when
  `nu_ij` is fractional at a cutoff tie.
- Reusing another layer's factor cache or mixing updated and pre-update pairs.
- Losing duplicate-reference gradient contributions during gather/scatter.
- Normalizing by eight instead of the actual eligible neighbor mass.
- Treating missing triangles as zeros included in a nonzero denominator.

Use FP64 gradient checks and nonzero output projections. With test-only neighbor
budget `>= N-1`, the sparse formula should match the dense all-eligible-triangle
formula. This is an oracle configuration, not an unannounced production variant.
Cover non-divisor tile sizes, chunk boundaries, mixed valid lengths, one-/two-
particle cases, and rows routed through the tie fallback.

Declare FP64/FP32/AMP tolerances before admission. Existing FP32 targets of
`rtol=1e-5, atol=1e-6` for outputs and `rtol=1e-4, atol=1e-5` for gradients are
reasonable starting points; compare like precision paths. Diagnose a failure
before changing a tolerance. FP16 arithmetic need not meet an FP64 threshold.

### Full model, DDP, and restart

- Compare the candidate at `T=0` with the base in logits, base gradients, buffers,
  and RNG state. Repeat with original State-EV outputs activated and layerwise
  bias heads made different; otherwise preservation tests are too weak.
- Activate the triangle outputs and verify every new projection learns, gradients
  reach contributing pair features, and original paths remain trainable.
- Compare checkpointed/uncheckpointed execution, two consecutive outstanding
  forwards followed by backward, and recovery from an injected forward exception.
- Count native trimmer, pair encoder, MHA, dropout, and BN calls/updates; recomputation
  must not add stochastic draws or running-statistic updates.
- Test eval-mode permutation invariance with trimming disabled and all inputs
  permuted together, including exact cutoff ties. Training dropout is stochastic;
  require matched masks for operation-level equivalence rather than arbitrary
  identical outputs from reordered stochastic forwards.
- Test two DDP ranks with unequal lengths, one rank with no valid triangles,
  and only one rank using the tie fallback. New parameters must remain connected.
- Compare uninterrupted repeats and resumed runs across optimizer/Lookahead and
  validation boundaries. Round-trip model, optimizer slow/fast state, scaler,
  per-rank RNG/trimmer, counters, and exact data-replay metadata.

## 10. Benchmark: prove the actual cost, including the tie path

Normal sparse interaction work is approximately `O(B*N^2*K*d)`, plus pair
projections and attention pooling, with `K=8,d=16` in two blocks. At `N=128`,
the intermediary count is about one sixteenth of a dense triangle contraction
for untied rows. This is not a whole-model speedup estimate. Boundary ties can
increase actual nonzero support up to `N-1`, and a dense backend performs cubic
arithmetic regardless of sparsity in its weights.

Before production, an independent benchmark must measure the candidate and
unchanged Layerwise State-EV on the intended two-A100 runtime. Separate:

- Geometry/cutoff construction and candidate packing.
- New projections, interaction gathers/reductions, and output projections.
- Full forward/backward, including scatter/recomputation cost.
- Native-length training, full-128 training, and adversarial full-128 tie cases.
- Allocated/reserved GPU peaks, host memory, cold start, and repeated fetch windows.
- FP32 streaming validation, fixed-length inference, and batch-one latency.
- Checkpoint write/load, active-path AMP health, and distributed resume.

Measure tie-row incidence, maximum and percentile nonzero neighbor counts, and
the time consumed by the tie path on real training/validation streams. If it is
frequent or disproportionately slow, report that result before committing to
production; do not silently truncate the graph. Benchmark synthetic worst cases
without pretending their frequency predicts the real-data average.

Start with 20 warm-up and 4,200 measured training steps with repeated fetch
windows, following the existing extension protocol, plus at least 128,000 events
of streaming validation. Report the actual sample sizes and whether targets
completed. Benchmark weights never initialize production. Exercise nonzero
triangle outputs, not a special zero-output fast path.

Use the existing 90% GPU-memory headroom policy and AMP calibration followed
by confirmed successful updates; reject nonfinite forward state and sustained
skipping beyond the inherited 5% ceiling. Two GPUs do not establish that
single-GPU evaluation at batch 512 will fit: measure that evaluation path too.

Request CPU preparation for 2 hours, the GPU benchmark for **5 hours 30 minutes**,
and a CPU summary for 15 minutes. Preliminary requests stay below six hours.
Set an internal deadline before Slurm termination and report incomplete evidence
honestly. A scheduler `COMPLETED` state alone does not authorize training.

If cost is unacceptable, first examine same-model tiling, selected-factor
projection reuse, or a measured dense-GEMM backend. Do not call reduced triangle
width, fewer active blocks, smaller neighbor budget, reduced precision caches,
or different tie semantics an invisible performance optimization. Architectural
changes need a new model identity; backend/numerical changes need fresh parity
and benchmark admission. Do not promise unchanged accuracy after a model change.

## 11. Training, comparison, and diagnostics

The first task starts fresh at seed 101 with the authenticated full
`ranger_full_v1` recipe, not a shortened schedule or trained donor checkpoint:

| Setting | Required first arm |
| --- | --- |
| Data | Official 100M training jets and 5M validation pool; unchanged preprocessing |
| Exposure | 1,000,000 update attempts, global batch 512; nominal 512M exposures |
| Optimizer | Existing complete-state Weaver RAdam plus Lookahead |
| RAdam | Betas `(0.95,0.999)`, epsilon `1e-5`, weight decay zero |
| Lookahead | Period 6 successful optimizer steps, mixing coefficient 0.5 |
| Learning rate | Existing function/schedule from `1e-3` to `1e-5`; no new warm-up or multiplier |
| Clipping / early stopping | Unchanged: none / disabled |
| Training runtime | Two same-node A100s; local batch 256; SyncBatchNorm; one worker per rank |
| Validation | 50 checks at 20,000-attempt intervals; unchanged 1.28M-event streaming procedure |
| Selection | Highest validation accuracy, first tie retained |
| Final evaluation | Selected checkpoint on full 5M validation, not the independent 20M test set |

Freeze effective architecture and runtime separately from inherited descriptive
recipe fields: legacy `shared` or `native_single_gpu` labels do not describe
this layerwise/DDP model. Do not rewrite donor hashes to hide a mismatch.
Import the existing LR function rather than guessing its boundary indexing.
Track successful updates and AMP skips as well as attempted updates.

Compare curves at equal update cutoffs and report selected-checkpoint accuracy,
macro AUC, classwise rejection at declared efficiencies, and QCD passing counts.
The full validation population overlaps model selection. A single seed is
exploratory; confirm a promising effect with matched seeds before claiming
reproducibility or evaluating the independent test population.

Audit GPU count, local/global batches, normalization, precision, data order,
optimizer, initialization, and restart history before reusing a control.
Historical one-GPU ParT or a GH200/native-BN State-EV run is not a runtime-matched
architecture-only control for this two-A100/SyncBatchNorm arm. A new control
training requires explicit approval.

On an isolated fixed diagnostic population, record per triangle block:

- Factor and gate preactivation RMS; tanh/sigmoid saturation fractions.
- Triangle output norm relative to original State-EV and ordinary EV.
- Output/factor/gate parameter and gradient norms from normal training updates.
- Effective neighborhood masses/counts, zero-denominator fraction, and tie use.
- Post-training sensitivity to disabling triangles or perturbing intermediary
  relations, labeled as sensitivity checks rather than independently trained controls.

Keep diagnostics small and away from live training RNG/data position and BN
statistics. Never log the full triplet tensor. Large branch norms or visually
plausible intermediaries do not prove a physics mechanism.

For an encouraging result, separately approved controls could test an equally
sized pair-only extra message, a different fixed neighborhood, or more matched
seeds. These distinguish relational structure, extra capacity, locality, and
seed variation. Post-training message removal alone cannot make that distinction.

## 12. Independent implementation and queue-readiness scope

Proposed deliverables, not existing commands:

```text
part_ev_triangles/{model,neighbors,protocol,benchmark}.py
scripts/run_part_ev_triangles.py
scripts/submit_part_ev_triangles.py
sbatch/run_part_ev_triangles.sh
sbatch/submit_part_ev_triangles.sh
tests/test_part_ev_triangles_{model,neighbors,contraction,benchmark,launcher}.py
```

Use audited common recipe/data/checkpoint utilities, an isolated source snapshot,
and a distinct campaign directory. Record architecture, tie rule, sparse/dense
backend, tile sizes, precision, source/environment hashes, and verified train/val
inventories. Resolve stable package identities rather than treating lazy import
changes as real installations, but never bypass genuine compatibility checks.

This plan does not submit jobs, cancel queued experiments, mutate frozen sources,
change environments, or make the current bias/multiplication/memory campaigns
depend on this one. Use the verified SPORC tier3 account/partition settings;
live limits and GPU availability must be checked at eventual submission.

After benchmark review, production intent is **one long training allocation**,
not a fixed three-day segment chain. Project walltime from sustained training,
all 50 validations, and checkpoint I/O with the existing explicit 30% planning
margin. Queue delays, contention, and restart replay remain separate risks.
Do not assume twelve days or lower the scientific budget to fit a request.

Retain atomic full-state restart checkpoints every 1,000 attempts and at
validation/stop boundaries, safe selected-weight retention, a live-writer lock,
and strict completion gates. Evaluation and report depend only on this model's
successful training/evaluation, not on unrelated experiments. A failure must
not authorize evaluation of an incomplete budget.

Account for snapshots, full optimizer/slow-weight state, atomic-write temporary
copies, selected candidates, and evaluation outputs across all admitted campaigns.
Triangle caches remain GPU/host activations only, never a persistent disk cache.
The new raw FP32 weights are only 28,800 bytes, which is not a storage guarantee
for the campaign. Never delete data or old results automatically.

## 13. Milestones and acceptance criteria

1. **Reference audit:** pin source/runtime/data, effective model config, base
   initialization, recipe, and parameter inventory.
2. **Graph oracle:** implement and test rapidity geometry, fractional cutoff ties,
   target exclusion, row mass, and particle-order behavior.
3. **Triangle oracle:** explicit loops and per-channel contraction agree in
   outputs and gradients for all mask, direction, and tie cases.
4. **Model integration:** all eight original State-EV blocks remain; only two
   added messages change behavior; zero-output and active-output tests pass.
5. **Bounded execution:** staged caches and tiled sparse/tie paths agree with
   the oracle, including gradients, recomputation, and allocated backward memory.
6. **Runtime safety:** DDP no-triangle ranks, AMP, outstanding-forward cleanup,
   repeat/resume behavior, immutable manifests, and launcher dry runs pass.
7. **Benchmark queue readiness:** independent under-six-hour prepare/benchmark/
   summary workflow, readable reports, and package README with exact commands.
8. **Production readiness after review:** passed authenticated measurements,
   realistic walltime/storage budgets, model-local dependency chain, guarded
   restart, and explicit reviewed submission instructions.

Expected parameter count is **2,334,662**. Every added parameter must be registered
once, be included in the optimizer, and have verified trainability. Prove the
same-state directed path calculation and tie semantics before optimizing them.
CPU mathematical checks do not establish GPU performance, benchmark completion
does not establish accuracy, and preserving the base as a special case does not
guarantee training finds a better solution.

## 14. Code entry points

- [Full Layerwise State-EV assembly](model.py).
- [Original optimized state chunks and pair cache](../part_ev_state_speed/model.py).
- [Current-state feature and attention/EV integration](../part_ev_message_study/model.py).
- [Layerwise pair-bias binding and masks](../part_layerwise_screen/model.py).
- [Pinned native model identity and input configuration](../part_ev_fullscale/model.py).
- [Explicit instance-local forward precedent](../part_ev_pair_memory/model.py).
- [Complete-state Ranger optimizer](../part_ev_fullscale/optimizer.py).
- [Full recipe source](../part_ev_state_fullscale/protocol.py).
- [Existing two-A100 benchmark pattern](../part_ev_state_bias/protocol.py).
- [Existing production restart/admission reference](../part_ev_state_production/README.md).

**Implementation principle:** keep the proven particle and pair-message paths
intact, add an explicit and differentiable way for related pairs to interact,
and make correctness, degeneracies, and real execution cost visible before
committing to long training.
