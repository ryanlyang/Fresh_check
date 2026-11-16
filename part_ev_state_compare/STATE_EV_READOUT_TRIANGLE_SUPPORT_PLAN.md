# Three additions to full all-eight Layerwise State-EV

Recorded: 2026-11-16. Research and implementation-design plan only.
The pair-readout arm now has an [independent implementation and benchmark
launcher](../part_ev_pair_readout/README.md). GPU admission and accuracy remain
unmeasured; no jobs were queued by that implementation. The other two arms
remain design plans.

Detailed implementation plans:

- Idea A: [Direct pair-to-classifier readout](DIRECT_PAIR_CLASSIFIER_READOUT_PLAN.md).
- Idea B: [Sparse triangle-aware updates](TRIANGLE_AWARE_UPDATES_PLAN.md).
- Idea C: [Support-aware aggregation](SUPPORT_AWARE_AGGREGATION_PLAN.md).

They fix initialization, precision, integration, tests, and admission details.
Only idea A has been implemented so far; ideas B and C remain documentation only.

## 1. The starting point: yes, all three build on Layerwise State-EV

Keep the successful **noncompact, standard-relations, all-eight Layerwise
State-EV** model intact. Each first experiment adds exactly one new mechanism.
Do not substitute ordinary EV, shared-bias State-EV, the last-four model, or
the compact architecture.

The base has **2,327,462 parameters**: eight particle blocks, width-32 State-EV
in every particle block, a shared physics-pair stem with independent layerwise
attention-bias heads, ordinary EV, and the existing class-attention readout.
Retain its same-math caching, chunking, and activation-checkpointing speedups.

The motivating seed-101 run reached **86.3151% training-time validation accuracy
at check 49 of 50**. This was a two-A100/SyncBatchNorm run, not an independent
test result or a multi-seed estimate. A later model needs a matched comparison,
not merely a number above this historical maximum.

| Proposed arm | Addition to unchanged Layerwise State-EV | Main question |
| --- | --- | --- |
| Pair readout | A small learned pair summary enters the final classifier | Are useful relationships being lost before classification? |
| Sparse triangles | Late-block pair messages consult other pairs through a third particle | Does explicit relationship-to-relationship reasoning help? |
| Support-aware aggregation | An extra message retains learned content and its amount of support | Does normalized averaging discard useful evidence strength? |

These are **new, separate ideas**, not renamed versions of the existing
[state-conditioned bias](STATE_CONDITIONED_ATTENTION_BIAS_PLAN.md),
[relation-state multiplication](RELATION_STATE_MULTIPLICATION_PLAN.md), or
[evolving pair memory](EVOLVING_PAIR_MEMORY_PLAN.md) experiments. Do not bundle
those mechanisms into these first arms. Combinations can be proposed later.

## 2. What we already have, and what these additions could change

Let `z_ij` be the shared physics-pair embedding, and let `x_i^l` and `x_j^l`
denote the particle states supplied to the existing state-message module in
particle block `l`. Its width-32 pair-state features are schematically:

```text
g_ij = E_l LN64(z_ij)
p_i  = P_l LN128(x_i^l)
q_j  = Q_l LN128(x_j^l)
r_ij = GELU(LN32(g_ij + p_i + q_j + p_i * q_j))
```

`*` means elementwise multiplication. Preserve the actual reference's
normalization order, including block normalization before this module.
The existing State-EV message uses an attention-weighted sum of `r_ij` and
head-specific output projections. The classifier subsequently reads particle
embeddings through the class-attention blocks; it does not directly pool the
pair-state tensor.

The particle states already contain context from other particles. Consequently,
the base is not blind to third particles, relationships, or multiplicity.
The hypotheses concern **more direct information paths and useful inductive
biases**, not an assertion that the base cannot represent these functions.

The reason to hope for a meaningful gain is that these additions change what
information is preserved or combined, rather than only increasing width.
However, none has an established expected accuracy gain. Redundant features,
optimization difficulties, and added capacity without useful structure could
all produce small, zero, or negative improvements.

## 3. Idea A: direct pair-to-classifier readout

### Concept

Give the classifier both a particle summary and a relationship summary.
Today, useful pair information must first affect particle embeddings and then
survive the class-token aggregation. A direct pair summary gives that information
another route to the prediction.

For example, relationships involving energetic constituents might collectively
indicate a useful radiation pattern even if no individual particle embedding
retains a clean summary of it. This is an illustrative hypothesis, not a claim
that learned summaries will correspond to identifiable physical objects.

### First design

Use **two learned pooling queries**, reading the existing width-32 pair-state
features from the **last particle block only**. These are the features computed
from states entering that block, not a new recomputation from its output states.
Keep both existing class-attention blocks unchanged.

For each valid directed off-diagonal pair:

```text
u_ij       = LN32(r_ij)
key_ij     = W_key u_ij                         # 16 channels
value_ij   = W_value u_ij                       # 32 channels
score_aij  = dot(query_a, key_ij) / sqrt(16)     # a = 0, 1
weight_aij = softmax_over_valid_pairs(score_aij)
summary_a  = sum_ij weight_aij * value_ij
```

The queries are learned parameters, not labels, predefined subjets, or extra
input particles. Exclude diagonal pairs from this new readout so that it focuses
on relationships; retain diagonal handling in all original branches unchanged.
If there are fewer than two valid particles, define the pair summary as zero
without evaluating an all-masked softmax.

Let `h_cls` be the existing normalized 128-channel class-token representation:

```text
combined = concat(h_cls, summary_0, summary_1)   # 192 channels
delta    = W_out GELU(W_in combined + b_in) + b_out
h_final  = h_cls + delta
logits   = existing_classifier(h_final)
```

Use a 64-channel hidden layer. Initialize only `W_out` and `b_out` to zero;
initialize the queries, key/value projections, and hidden projection normally
with isolated RNG streams. This initially recovers the base function while
allowing the new output projection to receive gradients. The hidden projection
lets pair and class-token evidence interact, rather than merely adding two
independent sets of class scores. No additional loss or dropout is needed in
the first version.

This is inspired by learned attention pooling in
[Set Transformer](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf), applied
here to State-EV pair features. It is not a claim that this exact jet adaptation
has already been validated or is novel in every component.

### Why it could give a substantial gain

If relationship information is useful but incompletely retained by the current
particle-to-class-token route, this addition addresses that bottleneck directly.
It also supplies a shorter gradient path from classification to the final pair
features. Two summaries can retain different aspects of the pair population,
rather than forcing a single average to contain everything.

That is the strongest argument for this arm: it can help even if the existing
State-EV feature extractor is already good. It could also fail because the
class token already retains everything useful, the two queries become redundant,
or the new branch acts mostly as an extra classifier MLP.

### Efficient implementation and diagnostics

- Reuse final-block pair features as they are computed. Do not add a second
  eight-block pair encoder or retain all layers' pair activations.
- Pool globally across all valid pairs, not with separate softmaxes per chunk.
  A differentiable, numerically stable running maximum/denominator/weighted-sum
  reduction can merge chunk statistics without keeping the full score tensor.
  Chunk merging must agree with a dense reference in forward and backward.
- Return summaries explicitly through the forward path. Do not rely on mutable
  hook side effects that activation checkpointing might execute a second time.
- Two queries attend to `N^2` pairs: the pooling term is quadratic, not the
  quartic cost of full self-attention among `N^2` pair tokens. Projection and
  backward costs still need measurement.
- Monitor summary similarity, pooling concentration, new-branch gradient norms,
  and the scale of `delta` relative to `h_cls`. Attention visualizations are not
  causal explanations or proof of physical specialization.
- A later control can keep an equally sized classifier MLP but remove pair
  inputs. This distinguishes extra classifier capacity from pair access; it
  would require a separately approved training run.

## 4. Idea B: sparse triangle-aware pair messages

### Concept

For a target relationship `i-j`, consult relationships `i-k` and `k-j` involving
a third real particle `k`. The same target pair can belong to different larger
structures depending on those other two relationships.

This differs from evolving pair memory: memory retains a particular pair's
history across layers; triangles explicitly combine different pairs within
a layer. It also differs from the earlier multiplication proposal, which
modulates a pair using its two endpoint states rather than other pair features.

Triangle interactions in
[AlphaFold's pair representation](https://www.nature.com/articles/s41586-021-03819-2)
provide a precedent for this kind of edge-to-edge reasoning. The sparse,
directed path-style construction below is our proposed adaptation, not a copy
of the full Evoformer or evidence of an expected jet-tagging gain.

### First design: limited extra work, full base model retained

- Add the new branch only in particle blocks **6 and 7**, using zero-based
  numbering: the final two of eight blocks.
- Preserve original State-EV at width 32 in **all eight** blocks. This is not
  the previously tested last-four State-EV approximation.
- Use triangle width **16** and a neighbor budget of **eight** per receiver,
  normally at most eight actual intermediaries. The detailed plan shares weight
  across cutoff ties, which can require more contributing particles and extra
  work. Use independent learned projections in the two blocks.
- Select candidates once per forward pass, after trimming/reordering, using
  nearest valid particles in the reference's rapidity-azimuth geometry, with
  wrapped azimuth. This is a sparse graph choice, not the separate Selected
  Relations feature set. No truth information or new dataset fields are used.

Let `K_i` contain up to eight valid neighbors of `i`, excluding `i` itself.
For each target `i-j`, remove `j` from its eligible intermediaries without
backfilling. Require three distinct valid particles. For the ordinary, untied
case, define:

```text
u_ij = LN32(r_ij)
a_ik = tanh(A_l u_ik)                          # 16 channels
b_kj = tanh(B_l u_kj)                          # 16 channels
gate_ij = sigmoid(C_l u_ij + c_l)              # 16 channels

t_ij = gate_ij * mean_{k in K_i, k != j}(a_ik * b_kj)
extra_message_hi = T_lh sum_j alpha_hij * t_ij
```

At cutoff ties, the [detailed plan](TRIANGLE_AWARE_UPDATES_PLAN.md) replaces this
hard set by fractional neighbor weights and the mean by its weighted equivalent.
It preserves permutation symmetry but can exceed eight actual contributors;
the worst-case work and tie-path benchmark must be reported explicitly.

Define `t_ij=0` when `i=j` or there are no eligible intermediaries. Normalizing
by the actual eligible count avoids making this arm an accidental support-count
experiment. The narrow bounded factors help control the scale of the branch.

Add `extra_message` alongside, not instead of, the existing value, ordinary EV,
and State-EV messages. Use the same native attention weights/dropout realization
as those branches. Initialize the new head-specific `T_lh` projections to zero,
but not `A_l`, `B_l`, or `C_l`. A separate output path avoids relying on the
base State-EV output projection to activate the new branch.

All three pair features in a triangle must be computed from the same
pre-update layer states. Do not use partially updated pairs or let chunk order
determine which version of a pair another pair sees. There is no persistent
pair-memory recurrence in this arm and no change to the attention logits.

### Why it could give a substantial gain

The opportunity is to recognize relational configurations explicitly rather
than compressing every relationship into endpoint states before combining them.
That could help distinguish patterns with similar individual pairs but different
arrangements of surrounding constituents. The target-pair gate lets the model
decide which kinds of third-particle evidence matter for a particular relation.

It is the most structurally ambitious of these three ideas, but also the least
certain in cost-benefit terms. Endpoint context might already suffice; local
candidate selection can miss wide-angle structure; extra products and gathers
can be costly; and triangle features might learn redundant geometric facts.
No specific physical decay topology is guaranteed to benefit.

### Avoiding a runtime explosion

Dense triangles require roughly `O(N^3 * d)` work. Restricting the intermediary
set gives `O(N^2 * K * d)` work for the ordinary untied added interaction, plus
quadratic feature projections. At `N=128` and `K=8`, this is approximately 16 times fewer
intermediary contributions than dense triangles **for that branch**. It is not
a 16-fold whole-model speedup or a promise of negligible overhead. Symmetric
cutoff-tie handling can have more nonzero neighbors; its pathological case is
cubic and must not be hidden behind this ordinary-case complexity statement.

Implementation rules:

1. Keep the original dense particle attention. Only the new triangle branch is
   sparse; do not remove ordinary pair messages to fit the budget.
2. Compute narrow pair projections in chunks, then gather intermediary features
   in bounded query/candidate tiles. Never materialize `[B,N,N,N,d]`, or a
   full `[B,N,N,K,d]` tensor merely because `K` is smaller.
3. Account honestly for the intermediate pair-projection storage. Existing
   State-EV consumes features chunk by chunk; access to another pair's features
   is not free. A staged narrow projection cache or recomputation is required.
   All such caches are per-forward activations, not persistent disk files.
4. Activation checkpointing and chunking reduce retained memory, not cubic
   arithmetic. Fixed intermediary count is what reduces the arithmetic order.
5. Avoid rebuilding neighborhoods in each block. Use the detailed plan's
   permutation-consistent weight-sharing at the physical-distance cutoff, not
   padded positions or input indices; test exact-distance ties and exchangeable
   identical particles explicitly, and measure the extra tie-path work.
6. First validate against a small explicit loop over the same sparse graph.
   With all eligible intermediaries admitted, also compare against a dense
   triangle reference. Check gradients, masks, and both directed orientations.

Benchmark before deciding whether two late blocks and width 16 are worthwhile.
Changing to one block, a smaller width, or different candidates is a declared
architecture variant, not a silent same-math optimization.

## 5. Idea C: support-aware aggregation

### Concept

Preserve both **what the supporting pair features say** and **how much support
there is**. A normalized average of one copy of a feature vector and ten copies
of that same vector can be identical. The rest of a transformer may still infer
multiplicity; this example concerns information lost by the averaging operation,
not a theorem that ParT cannot count.

Do not sum ordinary softmax attention weights to estimate support: before
attention dropout they sum to one over valid keys. Use separate unnormalized
learned gates, while leaving all original attention messages unchanged.

The general motivation is related to combining aggregation statistics and degree
information in [Principal Neighbourhood Aggregation](https://arxiv.org/abs/2004.05718).
The proposal here is a small extra State-EV branch, not the full PNA architecture.

### First design

Add a branch in each of the eight particle blocks. Use **four learned support
channels** and a shared **16-channel content projection** per block. The channels
can learn different kinds of support; do not assign them physics labels.

For valid off-diagonal neighbors of receiver `i`, and support channel `a`:

```text
u_ij   = LN32(r_ij)
s_aij  = sigmoid(w_la dot u_ij + b_la)
v_ij   = V_l u_ij                              # 16 channels

count_ai    = sum_j s_aij
content_ai  = sum_j s_aij * v_ij / max(count_ai, epsilon)
fraction_ai = count_ai / max(number_of_valid_nonself_neighbors_i, 1)
features_i  = concat_a(content_ai, log1p(count_ai), fraction_ai)
extra_i     = O_l GELU(H_l features_i + b_l)
```

With four support channels, `features_i` has 72 channels. Use hidden width 32
and a zero-initialized, bias-free output projection into the existing 128-channel message
space, reshaped into its eight head slots before the common attention output
projection. Keep the original message/residual path, attention, and State-EV
unchanged. Initialize the other projections normally and gate biases to zero.

Use FP32 gate/count accumulation and `epsilon=1e-6`; define all new features and
the added message as zero when no nonself neighbor is valid. Apply the valid
receiver mask after the output projection as well. The new branch excludes
self-pairs, but does not change their treatment in the base.

The count is a **soft support score**, not a number of subjets, a probability
of a physical hypothesis, or a calibrated confidence estimate. `log1p` limits
scale growth without erasing the count, while the support fraction helps
distinguish selective evidence from merely having more constituents. Do not
apply a normalization across these summary features that would silently erase
the count information we deliberately introduced.

### Why it could give a substantial gain

Several moderately informative relationships might jointly matter more than
one isolated instance, even when their average content is similar. Retaining
both content and support gives the next particle update a direct way to use
that distinction. Separate support channels allow different patterns to have
different amounts of evidence.

The main risk is that the gates learn almost constant scores, making the branch
mostly a particle-multiplicity feature. That might improve validation without
delivering the proposed relationship-specific benefit, and could be sensitive
to fragmentation, detector modeling, or domain shifts.

Measure gate variation, count distributions, saturation, and accuracy/rejection
within constituent-multiplicity bins. A later control using only raw neighbor
counts would test whether learned support is actually necessary. A matched
content-only branch would test the additional value of its support statistics.
Neither control is automatically authorized by this plan.

### Cost and implementation

The [detailed implementation plan](SUPPORT_AWARE_AGGREGATION_PLAN.md) fixes the
masking, feature layout, precision, and adapter design. Its analytic addition is
56,608 parameters, giving an expected total of 2,384,070; implementation must
verify that count and benchmark the actual overhead.

The branch reuses `r_ij` while its chunk is live. Gate sums and weighted content
sums can be accumulated without retaining all pair features. There is no new
third-particle dimension: the added pair processing remains quadratic, with
small widths, plus a particle-level MLP.

That makes it a reasonable low-overhead candidate, not a measured cheap model.
Eight extra content projections, backward recomputation, and GPU kernel traffic
still cost time. Compare a dense reference with the chunked reductions, including
FP32 accumulation under AMP and the no-neighbor case. This is distinct from the
older pair-variance experiment: it retains unnormalized learned support, not
only a second moment under normalized attention.

## 6. Initialization, correctness, and implementation boundaries

For every arm:

- Build from `build_model('state_layerwise', ...)` in
  [the current assembly](model.py). Reuse
  [the optimized state feature path](../part_ev_state_speed/model.py) and
  [the existing message integration](../part_ev_message_study/model.py).
- Put new implementations in separate modules/campaigns. Do not alter frozen
  source snapshots, old campaign metadata, running jobs, datasets, or shared
  environments. A mutable checkout must not become a queued job's source.
- Preserve base parameter initialization and RNG progression. Seed new modules
  independently, and verify identical base state dictionaries at initialization.
- Use zero-initialized final residual projections, not an all-zero branch.
  At initialization, upstream new parameters can legitimately have zero gradient
  until their output projection learns. Test that output gradients are nonzero,
  then test upstream gradients after a controlled nonzero activation/update.
- Verify base-function equivalence with the new output disabled, including masks,
  variable lengths, native dropout, and the reference's diagonal conventions.
  Do not demand bitwise equality between genuinely different precision paths.
- Verify active forward/backward correctness against a simple reference; cover
  chunk boundaries, padding, one-/two-particle edge cases, and particle-order
  equivariance/invariance. Handle invalid empty inputs consistently with the base.
- Pass graph-connected tensors explicitly through recomputed functions. No
  detached feature caches, stale cross-batch state, or hidden forward side effects.
- Test single-GPU and intended DDP execution, AMP health, optimizer inclusion,
  checkpoint round trips, and uninterrupted-versus-resumed training behavior.
- Count actual trainable parameters and measure FLOPs with declared conventions,
  sequence lengths, and forward/backward scope. The only verified count in this
  plan is the base count; new counts and runtime ratios require implementation.

## 7. Evaluation and execution plan

Suggested first scientific arms are Layerwise State-EV plus pair readout,
plus sparse triangles, and plus support-aware aggregation, trained separately
from fresh initialization. Reuse a control only after auditing its recipe and
runtime; a new control run needs explicit approval.

Keep the existing full Ranger recipe: **1,000,000 update attempts**, global batch
512, the official 100M training pool, and **50 validation checks every 20,000
updates** using the existing 1.28M-event streaming-validation procedure. These
are validation checks, not 50 full dataset epochs. Log attempted and successful
updates and AMP skips. Start with seed 101 only if that exploratory choice is
confirmed when implementation/submission is requested.

Select the checkpoint by highest validation accuracy with the existing tie
rule. Compare complete curves at equal update cutoffs and report AUC and
classwise QCD rejection at the accuracy-selected checkpoint. Evaluate the final
selected checkpoint on the full 5M validation pool; this overlaps selection
data and is not independent testing. Keep the 20M test split untouched during
development. Include QCD passing counts when presenting rejection changes.

For architecture claims, match GPU count, global/local batches, normalization,
precision, data loading, optimizer, schedule, and initialization protocol.
The historical State-EV run used two A100s/SyncBatchNorm; a one-GH200/native-BN
run is not a runtime-identical architecture-only contrast. Disclose segmented
restart history and other unavoidable differences instead of attributing them
to the model. Use matched additional seeds before claiming a reproducible gain.

The prior SPORC intent is a useful starting point: two same-node A100s and one
long production allocation per model, with safe full-state restart checkpoints.
It is not a queue command or a fixed walltime decision. Benchmark each arm first
with requested walltime **below six hours**, on its intended training runtime.
Measure sustained real-data training, all validation overheads, maximum-length
memory, checkpoint cost, FP32 inference throughput, and batch-one latency.

Project production walltime from those measurements plus an explicit margin;
check live scheduler/account limits before submission. Each model's evaluation
and report should depend only on its own successful training. Do not change
or cancel the existing bias, multiplication, or memory jobs for this work.

Persistent storage should consist of ordinary snapshots, compact diagnostics,
full-state restart/selected checkpoints, and audited evaluation outputs.
**Do not persist pair, triangle, or support activations to disk.** Account for
all concurrently admitted campaigns before launch; do not assume that a small
parameter increase makes storage automatically safe, and do not delete data or
older artifacts automatically.

## 8. Priority and what would count as success

1. **Pair readout first:** my preferred first experiment because it addresses a
   plausible classification bottleneck with one late, relatively small addition.
2. **Support-aware aggregation next:** another targeted information-preservation
   change, with multiplicity-only explanations explicitly checked.
3. **Sparse triangles:** the most ambitious new interaction and the highest
   compute/integration risk. Its benchmark is particularly important.

This ordering is an engineering judgment, not a predicted accuracy ranking.
All three preserve the complete base model, but preserving it as a special
case does not guarantee that training finds a better solution.

A convincing result would be improved validation accuracy and useful classwise
rejection, confirmed across matched seeds, with a documented inference/training
cost. A meaningful rejection gain can matter even if aggregate accuracy changes
modestly. Conversely, a single favorable maximum or more parameters is not
evidence for the proposed mechanism.

Only after independent results should we consider combinations. Pair readout
could expose useful triangle-derived information to the classifier, and support
statistics could complement both; their gains might also overlap or interfere.
No combined model or additional control training is included automatically.

**Bottom line:** these are three ways to add capabilities to the same full
Layerwise State-EV model: read relationships more directly, combine relationships
more explicitly, and retain how much evidence supports a relationship pattern.
They provide concrete reasons to test for substantial gains, not a promise that
any one of them will deliver a particular accuracy increase.
