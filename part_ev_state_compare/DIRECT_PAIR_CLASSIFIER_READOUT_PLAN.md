# Direct pair-to-classifier readout on full Layerwise State-EV

Recorded: 2026-11-16. Detailed first-design and implementation plan.
**Status: implemented as an independent model and benchmark launcher; local CPU
tests pass. GPU admission and accuracy are unmeasured. No production submission.**

See the [implementation and queue instructions](../part_ev_pair_readout/README.md).

This is the detailed specification for idea A in the
[three-addition overview](STATE_EV_READOUT_TRIANGLE_SUPPORT_PLAN.md).
The narrow implementation decisions below refine that overview. They do not
authorize changing existing experiments or implementing the other two ideas.

## 1. Objective and scientific hypothesis

Add a small, trainable readout of the final particle block's pair-state features
to the existing jet-level classifier. Preserve the complete noncompact,
standard-relations, all-eight Layerwise State-EV backbone and its original
particle/class-token classification route.

The current model uses pair features to update particle embeddings, then reads
those embeddings with two class-attention blocks. The proposed addition lets
the final classifier also see a learned summary of the relationships directly.

The hypothesis is an **information-access bottleneck**, not a claim that the
base cannot represent relationships. A relationship can be useful for deciding
the jet class without being fully preserved through both particle-message
aggregation and class-token aggregation. Giving the classifier direct access
could recover that evidence and provide a shorter gradient route to the final
pair-state features.

Two learned summaries may retain complementary patterns. A small nonlinear
fusion module lets their usefulness depend on the existing class-token context.
The summaries are not predefined subjets, physical labels, or guaranteed
interpretable structures. The model learns them solely through the unchanged
classification loss.

This is a plausible route to a meaningful improvement with limited additional
capacity. It does not establish an expected accuracy increase: the original
readout could already preserve the useful information, both summaries could
become redundant, or extra classifier capacity could explain any improvement.

### Reference and comparison target

The base is `state_layerwise` from [the existing assembly](model.py):

- Eight particle blocks, eight attention heads, particle width 128.
- Standard physics relations, one shared width-64 pair stem, and eight
  independent layerwise bias heads; not eight independent pair encoders.
- Ordinary EV plus width-32 State-EV in **all eight** particle blocks.
- The unchanged two class-attention blocks, final class LayerNorm, and classifier.
- Existing pair-normalization caching, query chunking, and recomputation.
- **2,327,462 trainable parameters.**

Its historical seed-101 two-A100/SyncBatchNorm full-schedule run reached
**86.3151% training-time validation accuracy at check 49 of 50**. This motivates
the experiment; it is not a test result or a fixed performance threshold that
establishes an architectural gain by itself.

The first arm contains no selected relations, state-conditioned logit bias,
explicit relation-state multiplication, evolving pair memory, triangle branch,
support-count branch, compaction, distillation, or warm-started training.

## 2. Locked first-design choices

| Component | First implementation |
| --- | --- |
| Source features | Existing directed width-32 State-EV features in particle block 7, zero-based |
| Source time | States entering the final particle block's attention, not states after that block |
| Pooling domain | All valid ordered pairs with `i != j` within each jet |
| Learned pooling queries | Two, each width 16 |
| Key projection | Bias-free `32 -> 16`, shared by both queries |
| Value projection | Bias-free `32 -> 32`, shared by both queries |
| New input normalization | Non-affine channelwise LayerNorm, epsilon `1e-5`, on a separate readout view of `r_ij` |
| Pool normalization | One softmax per query over the whole valid pair set |
| Fusion input | Existing normalized class token plus both ordered summaries: width 192 |
| Fusion MLP | `192 -> 64 -> 128`, GELU, biases in both linear layers |
| Fusion placement | Residual addition after existing class LayerNorm, before unchanged classifier |
| Identity initialization | Final fusion weight and bias zero; all earlier readout weights nonzero |
| New dropout/loss terms | None |
| New learned temperature/gate | None; fixed score scale `1/sqrt(16)` |
| Query chunk | 64 receiver particles, matching the optimized base |
| Empty pair set | Zero summaries and zero added classifier correction |
| First scientific task | Fresh seed 101, full matched Ranger recipe |
| First target runtime | SPORC tier3, two same-node A100s, DDP/SyncBatchNorm |

These dimensions are reasoned starting choices, not validated optima. Two
queries offer more than one summary without introducing a large readout stack.
The key width limits scoring cost; the value width retains the existing state
feature width. The fusion MLP allows class-dependent use of pair evidence.

Do not add extra normalizations after pooling or after the residual addition,
specialize queries by class, impose a diversity loss, or freeze the backbone
in this first version. Those would be separately declared experiments.

## 3. Exact features, masks, and shapes

Notation: `B` is local batch size, `N <= 128` is the native post-trimming
sequence length, `C <= 64` is a receiver chunk, and `A=2` is the query count.

| Tensor | Shape |
| --- | --- |
| Valid particles | `[B, N]` |
| Native attention query | `[N, B, 128]` |
| Normalized physics-pair cache | `[B, N, N, 64]` |
| Existing pair-state features for a chunk | `[B, C, N, 32]` |
| New keys / values | `[B, C, N, 16]` / `[B, C, N, 32]` |
| New scores within a chunk | `[B, 2, C, N]` |
| Final pair summaries | `[B, 2, 32]` |
| Existing class representation / fused representation | `[B, 128]` |
| Final logits | `[B, 10]` |

Use the existing final-block state features, in their exact arithmetic order:

```text
zbar_ij = LN64(shared_physics_pair_stem_ij)
xbar_i  = LN128(query_i)                 # query already has native pre-attention norm
g_ij    = E_7 zbar_ij
p_i     = P_7 xbar_i
q_j     = Q_7 xbar_j
r_ij    = GELU(LN32(g_ij + p_i + q_j + p_i * q_j))
```

Preserve the existing FP32 normalization helpers, autocast behavior before
normalization, and valid-pair masking. Do not remove the extra state-module
normalization because the native block has already normalized its inputs.
The new readout's normalization must not overwrite `r_ij` used by ordinary
State-EV or change the base message contraction.

Define a separate mask:

```text
readout_valid_ij = valid_i AND valid_j AND (i != j)
has_pairs_b = number_of_valid_particles_b >= 2
```

The base's diagonal pairs remain unchanged. Only the new global readout excludes
them. Include both `i-j` and `j-i`: even if raw physics relations are symmetric,
the state feature is directed. Do not triangular-pack, symmetrize, or halve its
pair set. The count of eligible ordered pairs is `n_valid * (n_valid - 1)`.

All masks and features must use the **same native post-permutation/truncation
particle axes**. There is one native trimming call per forward, not a separate
trimmer for the readout. No batch-global or cross-jet pooling is permitted.

## 4. Readout equations and classifier fusion

For each valid pair and query `a`:

```text
u_ij    = LayerNorm(r_ij, affine=False, eps=1e-5)
k_ij    = W_key u_ij
v_ij    = W_value u_ij
s_aij   = dot(query_a, k_ij) / sqrt(16)
w_aij   = exp(s_aij) / sum_{valid (p,q)} exp(s_apq)
summary_a = sum_{valid (i,j)} w_aij * v_ij
```

Implement the softmax stably as specified below, not using raw exponentials.
Its normalization is across both particle axes. It is not a per-receiver
attention distribution and does not reuse the original particle attention
weights: these two queries learn which relationships matter to the global
decision independently of which relationships matter to particle updates.

For a jet with no eligible pairs, both summaries are zero. Do not evaluate an
all-`-inf` softmax. Reject a completely empty jet consistently with the existing
experiment input contract; a one-particle jet is valid and has no readout pairs.

Let `h` be the original `mod.norm(cls_tokens).squeeze(0)` after both original
class-attention blocks:

```text
fused_input = concat(h, summary_0, summary_1)   # [B, 192]
hidden      = GELU(W_in fused_input + b_in)   # [B, 64]
delta       = W_out hidden + b_out           # [B, 128]
delta       = delta * has_pairs[:, None]
logits      = original_fc(h + delta)
```

The empty-pair correction mask refines the overview's edge-case behavior: a jet
without relationships should not acquire a class-token-only correction through
this branch. Apply it as a graph-connected tensor mask, not a rank-dependent
Python bypass. Do not add `delta` to `h` in place; `h` is also a fusion input.

Keep the original classifier parameters and output contract. Training/evaluation
drivers receive logits; their existing metric code computes probabilities.
Do not introduce a second softmax before cross-entropy. An unsupported feature-
only or export mode must fail clearly rather than silently bypassing the readout.

This is learned-query pooling inspired by
[Set Transformer, section 3.2](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf),
not a full Set Transformer block: there is no pair-token self-attention or new
attention stack between summaries. The proposed fusion MLP supplies their
nonlinear interaction with the class representation.

## 5. Initialization and trainability

### Parameter initialization

Construct the complete reference model under the existing caller's global seed
first. The builder's `seed` argument alone does not seed every native module.
Create new parameters afterward inside an isolated CPU RNG context with a
stable seed derived from `(experiment_seed, "pair-readout-v1")`.

- Sample each query independently from a standard normal distribution, then
  rescale its initial L2 norm to `sqrt(16)`. Do not constrain its norm during
  training. This breaks query symmetry without prescribing different roles.
- Initialize `W_key`, `W_value`, and `W_in` with Xavier uniform, gain 1.
- Initialize `b_in=0`, `W_out=0`, and `b_out=0`.
- There are no affine parameters in the new LayerNorm.
- Leave all original parameters and normalization buffers untouched.

Compare the full base subset of state dictionaries, including buffers, and the
caller RNG states against a reference-only construction. Register each new
module exactly once. A duplicate registration or a second hidden copy of the
backbone is a correctness failure, even if `parameters()` deduplicates it.

### Expected learning sequence

At initialization, `delta=0`, so the complete model reproduces the base function.
On a nondegenerate batch, the final fusion projection can receive gradients
immediately. Earlier new readout parameters initially have zero loss gradients
through the zero projection; they should become active after it updates.

This short activation sequence is intentional. Do not zero both the scoring
path and the output projection, attach another zero scalar gate, or bypass the
branch whenever its output weights are zero. Do not demand nonzero score
gradients for a synthetic example whose values are all identical.

All new parameters must remain in the autograd graph and the optimizer,
including on a rank containing only one-particle jets. Distinguish zero-valued
gradients from unused parameters. Test with `find_unused_parameters=False`.
Use the reference Ranger settings for the new parameters; no extra learning-rate
multiplier, optimizer, weight decay, or warm-up schedule in this arm.

Once active, classification gradients flow into both the original class-token
route and the pair feature extractor. Do not detach summaries, particle states,
or pair features to make memory usage smaller. Training still starts fresh,
not from the checkpoint that achieved 86.3151%.

## 6. Precision and globally correct chunked pooling

### Precision policy

Retain the base numerical/runtime policy. New key/value and fusion linear
layers follow the surrounding FP16 autocast during training, with FP32 master
parameters. The new normalization, query-key dot products, scaled scores,
softmax exponentials, denominators, weighted-value reductions, and chunk merges
run in FP32 with autocast disabled. Cast projected keys/values explicitly for
these reductions; merely calling `.float()` inside an enabled autocast region
does not define the intended matrix-multiply precision.

The reference class vector and pooled summaries are combined without an
unnecessary downcast; fusion linears then follow autocast. Cast the final
correction to `h.dtype` before the residual addition. FP32 validation/inference
uses no autocast. Preserve float64 in pure mathematical tests instead of forcing
them to FP32. Keep TF32 and cuDNN benchmarking disabled as in the reference.

Detect nonfinite valid feature/projection/score/summary/fusion values. Do not
turn them into finite predictions with clipping or `nan_to_num`. Aggregate
health flags on device and use coordinated rank handling; avoid one host sync
per chunk or an exception on one rank while others enter a collective.

### Chunk statistics

Within each receiver chunk and for each jet/query, flatten the two pair axes
conceptually and compute these small sufficient statistics:

```text
present_c = any readout-valid pair in chunk c
m_c = maximum valid score, or 0 if not present_c
e_pair = exp(score_pair - m_c) for valid pairs, else 0
d_c = sum_pair e_pair
z_c = sum_pair e_pair * value_pair             # 32 channels
```

Mask invalid scores before exponentiation; do not let padding enter a maximum
or denominator. All-empty statistics are `(present=False, m=0, d=0, z=0)`.
Use safe finite operands in unselected tensor branches too: `where` does not
make an overflowing exponential or `-inf - -inf` safe merely by hiding its
result. The chunk still executes a connected computation when its mask is empty.

### Merge rule

For two disjoint sets of chunk statistics, choose `m` as the maximum of the
present sets' maxima, the present set's maximum if only one is present, and
zero if neither is present. Then:

```text
shift_A = where(present_A, m_A - m, 0)
shift_B = where(present_B, m_B - m, 0)
scale_A = exp(shift_A) * present_A
scale_B = exp(shift_B) * present_B
d = scale_A * d_A + scale_B * d_B
z = scale_A * z_A + scale_B * z_B
present = present_A OR present_B
summary = z / where(present, d, 1)
```

Merge in increasing receiver-chunk order for reproducible arithmetic. The
result is the same global softmax-weighted summary in exact arithmetic.
Do not average independently normalized chunk summaries, normalize by the number
of pairs a second time, or normalize separately for each receiver.

Maxima used solely as numerical shifts may be detached in this implementation:
the common shift cancels from the final normalized ratio. Keep all scores,
denominators, numerators, and values otherwise differentiable, and verify this
gradient claim against the dense softmax oracle, including ties. Changing the
merge order is mathematically equivalent but not necessarily bitwise identical.

The normalization strategy follows the general online-softmax idea described by
[Milakov and Gimelshein](https://arxiv.org/abs/1805.02867); the weighted-vector
merge here must be independently tested. That paper's speed measurements are
not predictions for this model.

## 7. Integration without duplicate feature computation or mutable readout hooks

Implement an independent package, provisionally `part_ev_pair_readout`, with
an instance-local forward specialization. Preserve original parameter names
and modules. Do not patch installed Weaver, change global class methods, or
modify old scientific modules to make this arm work.

The current code has no explicit pair-summary output from a particle block.
The implementation must add one locally rather than storing live summaries in
a hook or a module's `last_summary` attribute.

### Recommended forward sequence

1. Validate input support and run the native trimmer exactly once.
2. Run the original particle embedding and shared pair encoder exactly once.
   Preserve the reference's pair-encoder BatchNorm population and all eight
   layerwise bias bindings.
3. Run particle blocks 0 through 6 through their unchanged forward paths.
4. Dispatch block 7 through a small audited helper that returns both its normal
   particle output and a `[B,2,32]` pair summary.
5. Run the two original class-attention blocks and original final normalization.
6. Fuse the local pair summary with `h`, and call the unchanged classifier.
7. Clear the existing pair bindings/cache on success and exception. No new
   readout state survives the forward call.

The final-block helper must call native MHA **once**, retaining its original
layerwise bias and its one attention-dropout realization. Compute ordinary EV
exactly as before. In the final State-EV chunk function, compute `r_ij` once and
return both the original state-message contribution and the new pooling
statistics. The readout uses its own scores, but does not alter or resample the
native attention weights used by the original messages.

Complete the particle block with the exact native head scaling, attention
output projection, normalization, dropout, residual additions, and FFN order.
Do not simplify the pinned `c_attn` reshape/einsum sequence or apply a projection
bias twice. There is no modification of the particle output by the readout in
this first arm; its contribution enters only at classifier fusion.

Proposed explicit interfaces, not existing APIs:

```text
joint_state_chunk(weights, normalized_pairs, p_chunk, q, base_mask,
                  readout_mask, readout_parameters)
    -> state_message_chunk, present, maximum, denominator, numerator, health

final_particle_block_with_readout(block, x, padding_mask, readout)
    -> updated_particles, merged_pair_summary, health

readout.fuse(h_cls, pair_summary, has_pairs)
    -> h_final
```

Register readout parameters under one dedicated namespace such as
`mod.pair_readout.*`; pass references/tensors to helpers without registering the
same submodule again. Reuse the final state's existing `receiver`, `sender`,
`edge`, `output`, and pair cache. A source-pinned instance specialization like
[the pair-memory forward adapter](../part_ev_pair_memory/model.py) is a useful
integration precedent, but do not import its memory mechanism into this model.

### Activation checkpointing

Keep the joint chunk computation pure and use non-reentrant checkpointing.
The native MHA/dropout and all BatchNorm modules stay outside this recomputed
region. Its explicit tensor inputs include the normalized pair slice, endpoint
projections, masks, and native weights needed by the old message. It returns
both old-message tensors and pooling statistics through autograd.

The existing `preserve_rng_state=False` choice is valid here only because this
chunk contains no randomness. Do not consult a mutable cache during backward:
the outer forward will already have cleared it. Do not append summaries to an
external list or log counters inside recomputation. Ordinary local collection
of the chunk's returned tensors in the outer forward is safe.

PyTorch explicitly warns that a recomputed function behaving differently can
produce incorrect gradients; see its
[checkpointing documentation](https://docs.pytorch.org/docs/2.5/checkpoint.html).
Verify behavior in the pinned deployment version rather than changing packages
to match a documentation example.

## 8. Parameter budget, compute, and storage

The locked design adds the following trainable parameters:

| Component | Calculation | Added parameters |
| --- | --- | ---: |
| Two learned queries | `2 * 16` | 32 |
| Bias-free keys | `16 * 32` | 512 |
| Bias-free values | `32 * 32` | 1,024 |
| Fusion hidden layer | `64 * 192 + 64` | 12,352 |
| Fusion output layer | `128 * 64 + 128` | 8,320 |
| **Total addition** | | **22,240** |

Total: **2,349,702**, approximately **0.956% more than Layerwise State-EV**.
Construction and local tests now confirm the analytic design count, base
parameter subset, and optimizer membership. The inherited base builder's count assertion must
run before the addition, not be weakened to accept any count.

New pair processing remains quadratic in particle count. It includes one
readout-only normalization, key/value projections, and two global summaries
in the final block. There is no self-attention among all `N^2` pairs, no triangle
dimension, and no second particle backbone. The fusion MLP is per jet.

Use chunking/recomputation to avoid retaining the full new pair activation
stack. A full `[256,128,128,32]` FP32 tensor alone is **512 MiB**, before saved
backward tensors. Small parameter count does not imply negligible activation
memory or runtime. The base's existing pair cache is still present and must be
included in peak measurements.

Start with the ordinary PyTorch operations and explicit global-pooling merge.
Do not introduce compilation, custom CUDA, a new dependency, or algebraic
reparameterization to get the first benchmark through. Later optimizations
need nonzero-output forward/backward parity and fresh benchmark evidence.

Persistent artifacts are ordinary snapshots, checkpoints, compact logs, and
evaluation outputs. Never write pair features or pooling activations to disk.
The extra FP32 weights contain 88,960 bytes before serialization overhead, but
restart states, retained selected candidates, atomic-write temporary copies,
and prediction outputs dominate campaign planning. Calculate a complete
concurrent storage budget rather than inferring safety from the weight count.

## 9. Required tests before a GPU benchmark

### A. Dense mathematical oracle

Implement a simple small-input oracle that explicitly builds all directed valid
pair features and uses a single masked softmax over flattened pair indices.
The optimized implementation must agree in summaries, fused logits, input
gradients, and all parameter gradients.

Cover:

- Chunk sizes 1, a non-divisor of `N`, 64, and `>= N`; sequence lengths crossing
  a chunk boundary; unequal valid lengths and noncontiguous masks.
- One valid particle, two valid particles, a chunk with no eligible pairs, and
  batch members with different eligible-pair counts. Entirely empty jets reject.
- Equal scores, large positive/negative finite scores, and distinct values.
  Confirm global-softmax shift invariance and test a case where averaging
  chunk-local summaries gives a different, deliberately wrong result.
- Directed features for which `r_ij != r_ji`; diagonal and padded-feature
  perturbations must not change the isolated readout's answer.
- Float64 gradient checks on scores, values, queries, and fusion. Test the
  detached numerical-shift implementation against a fully differentiable oracle.

Use declared FP64/FP32 tolerance policies rather than blanket bitwise claims.
A reasonable initial FP32 target is forward `rtol=1e-5, atol=1e-6` and gradients
`rtol=1e-4, atol=1e-5`, consistent with existing extension tests. AMP needs a
separate, predeclared policy comparing like precision paths. Do not loosen a
failed tolerance without diagnosing its cause and recording the decision.

### B. Full-model preservation and active behavior

- Construct reference and candidate with identical base initialization. With
  fusion output zero, compare logits, base gradients, buffers, and RNG behavior.
- Repeat with nonzero original State-EV outputs and differentiated layerwise
  bias heads; a test with all extra state messages inactive is insufficient.
- Activate the readout with a controlled nonzero output projection, then compare
  dense versus chunked execution, checkpointing on/off, and mixed precision.
- Confirm direct readout gradients reach final-block state projections and the
  shared pair stem. Confirm existing gradients still reach all particle blocks,
  original EV, layerwise bias heads, class blocks, and the original classifier.
- Check output-projection learning at initialization and upstream new gradients
  after a successful update. Test on informative, nondegenerate examples.
- Test particle-permutation invariance in eval mode with trimming disabled.
  Compare only jointly permuted particle/pair axes; stochastic training dropout
  and trimming do not imply identical outputs under arbitrary reordered inputs.
- Verify no new cross-jet dependence beyond the reference's existing training
  normalization behavior. Test eval-mode batch independence.
- Count native trimmer, pair encoder, and MHA calls; confirm one of each intended
  call, no extra dropout draws, and no additional BatchNorm updates in backward.
- Inject an exception and run a subsequent forward to check cleanup. Run two
  outstanding forwards followed by backward to catch stale summary/cache reuse.

### C. Distributed and restart integrity

Test two ranks with unequal native trimmed lengths and a rank whose whole batch
has no readout pairs. All new parameters must participate in DDP's autograd
graph without turning on unused-parameter discovery to conceal a bug.

Round-trip the full model, Ranger inner state and Lookahead slow weights/counter,
GradScaler, rank RNG, native trimmer state, update/check counters, and exact
stream-replay information. Compare uninterrupted and resumed update sequences,
including a Lookahead boundary and a validation boundary. Record repeatability
before interpreting a resume discrepancy as checkpoint corruption.

No pair summaries belong in restart state; each is recomputed from the current
jet batch. Parameter registration order and optimizer membership must be stable
across construction and resume.

## 10. Frozen science and diagnostic reporting

Retain the complete `ranger_full_v1` numerical recipe, not the short screening
schedule. Read it from the authenticated reference and verify these essentials:

| Setting | Required first arm |
| --- | --- |
| Training pool | Official 100M jets, unchanged preprocessing and maximum 128 particles |
| Update budget | 1,000,000 update attempts, global batch 512; nominal 512M exposures |
| Optimizer | Pinned Weaver RAdam plus existing complete-state Lookahead wrapper |
| RAdam | Betas `(0.95,0.999)`, epsilon `1e-5`, weight decay 0 |
| Lookahead | Every 6 successful optimizer steps, mixing coefficient 0.5 |
| Learning rate | Existing schedule from `1e-3` to `1e-5`, 15 decays at reference checks 35 through 49 |
| Clipping / early stopping | None / disabled, as in the reference |
| Validation | 50 checks, every 20,000 attempts; unchanged continuous 1.28M-event procedure |
| Checkpoint selection | Highest validation accuracy, first tie retained |
| Full evaluation | Selected checkpoint on official 5M validation, no independent test access |
| Seed | Fresh 101; extra seeds require a later approved comparison |

Import and verify the existing learning-rate function; do not reimplement its
boundary indexing from the table. Distinguish the legacy recipe's descriptive
fields from the effective architecture/runtime: the actual model has layerwise
bias and the intended training uses SyncBatchNorm. Record those effective
settings explicitly without silently altering donor hashes or claiming a
legacy `shared`/`native_single_gpu` string describes the deployed architecture.

Log attempted and successful updates, skipped AMP updates, learning rate,
validation progress, and accuracy/AUC at the accuracy-selected checkpoint.
Training checks are not full dataset epochs. Full-5M validation overlaps model
selection; do not label it an independent test. Keep the 20M test split untouched.

On a small fixed diagnostic population in isolated eval mode, collect:

- Pooling entropy normalized by `log(valid_pair_count)` where defined; maximum
  pair weight and effective support `1 / sum(w^2)` per query.
- Similarity of the two summaries and their weight distributions.
- Key/value/summary RMS, query norms, and `||delta|| / max(||h||, epsilon)`.
- New-parameter gradient norms from normal training steps, without extra
  optimizer steps or diagnostic backward passes into the live training state.
- A readout-disabled evaluation and controlled pair-summary perturbations,
  labeled as post-training sensitivity checks, not independent training controls.

Do not run these expensive diagnostics every update, dump full pair tensors,
modify live BN statistics, or consume the training RNG/data stream. A concentrated
query is not automatically good; a large correction is not evidence of accuracy
gain; an attention map is not a causal physics interpretation.

## 11. Benchmark and production admission

### Independent benchmark-only campaign

Proposed new package/scripts should be isolated from current launchers:

```text
part_ev_pair_readout/{model,protocol,benchmark}.py
scripts/run_part_ev_pair_readout.py
scripts/submit_part_ev_pair_readout.py
sbatch/run_part_ev_pair_readout.sh
sbatch/submit_part_ev_pair_readout.sh
tests/test_part_ev_pair_readout_{model,pooling,benchmark,launcher}.py
```

These deliverables are now implemented; see the package README above. Prefer
audited common utilities over copying a complete trainer. Do not expand an old
production launcher's accepted architectures merely to bypass new admission.

Use a new campaign directory with a hashed source snapshot, effective model
config, exact reference identity, environment identity, train/validation inventory,
and immutable recipe/runtime contract. Preserve existing campaigns and pending
jobs. Stable environment inventories must not depend on lazy additions to
`sys.path`; a genuine mismatch requires diagnosis, not a blanket ignore rule.

Initial resource intent:

- CPU preparation: 2 hours; GPU benchmark: **5 hours 30 minutes**; CPU summary:
  15 minutes. All preliminary requests remain below six hours.
- GPU benchmark: two same-node A100s, DDP/SyncBatchNorm, local batch 256,
  global batch 512, one training worker per rank, reference rank-zero validation
  batch 512 and two validation workers; 8 CPUs and 64 GiB host memory initially.
- Use the verified SPORC partition/account configuration. Check live GPU type,
  QoS/resource limits, and free storage before submitting; do not silently
  substitute Tigris or a different normalization/runtime.

Measure the candidate and unchanged Layerwise State-EV on the same allocation
under comparable settings. Report cold-start overhead separately and record
measurement order/cache effects. Aim for the existing sustained protocol of
20 warm-up and 4,200 measured steps with repeated fetch windows, rather than a
handful of cached batches. Use separate model instances/streams; benchmark
weights are disposable and never initialize production.

Benchmark coverage must include native-length training, full-128-particle
forward/backward memory, AMP recovery and successful updates, FP32 streaming
validation, fixed-length inference, batch-one latency, checkpoint I/O, and a
restart smoke test. Exercise both identity-initialized and active nonzero
readout paths; zero output must never select a cheaper bypass.

Record allocated and reserved peaks on each device with a declared headroom
guard, initially the existing 90% policy. Allow the established short scaler
calibration, then require finite forward values and confirmed successful
updates; fail sustained skipping beyond the inherited 5% ceiling. Do not turn
off AMP, change batch size, or suppress a health failure to secure a pass.

Enforce an internal deadline before Slurm termination. Incomplete measurements
produce an explicit incomplete/failed admission, never extrapolated success.
After source, model, precision, or runtime changes, obtain fresh benchmark
evidence. Wall-clock projections must include all 50 validations and checkpoint
I/O, plus an explicit planning margin such as the existing 30%; queue delays,
restart replay, and contention are separate uncertainties.

### Later production path

Production implementation/submission is a separate approved step after review.
Intent is one long allocation for this model, not a prearranged three-day chain.
Derive the request from measurements and live scheduler limits rather than
assuming twelve days. Full-state checkpoints are still required every 1,000
attempts and at validation/stop boundaries; keep selected weights safe through
atomic writes and retain the original completion-budget checks.

Use a model-local `train -> full-validation evaluate -> report` dependency
chain. It must not wait for unrelated triangle/support or existing extension
campaigns. Evaluate only after verified full training and validation completion.
Do not create an unapproved cross-cluster duplicate, automatic retry loop, or
cleanup action. Explicitly budget other admitted campaigns and temporary
checkpoint/prediction writes before any new submission.

## 12. Evidence for a useful gain and follow-up controls

The primary comparison is full Layerwise State-EV with versus without this
readout, at matched update budgets and accuracy-based selection. Compare curves
at equal check cutoffs as well as selected checkpoints. Report accuracy, macro
AUC, per-class QCD rejection at declared efficiencies, passing counts, parameter
counts, training/inference throughput, and measured memory.

Audit training data, optimizer, global/local batches, precision, normalization,
initialization, and restart histories before reusing the historical control.
One GH200/native BatchNorm versus two A100s/SyncBatchNorm is not an architecture-
only comparison. A new matched control is scientifically valuable if the old one
does not match, but needs explicit approval rather than silent submission.

For a promising result, subsequent separately approved controls can distinguish:

1. **Extra classifier capacity:** a comparable classifier MLP that reads only
   the class token, with its actual parameter/FLOP mismatch reported.
2. **Learned selection versus generic pair access:** uniform pooling of the
   same final pair features, preserving a comparable fusion stage.
3. **Seed effects:** matched additional seeds and eventual independent testing.

Removing the readout from an already trained model is useful sensitivity
evidence, but not a substitute for these independently trained controls. Do not
claim a mechanism solely from a single favorable maximum, attention pattern,
or post-training ablation. A meaningful classwise rejection improvement can be
valuable even when aggregate accuracy moves modestly.

Only consider four queries, earlier/multiple-layer readouts, class-conditioned
queries, different fusion widths, or combinations with other mechanisms after
this locked first design has an interpretable result. Such changes require new
model identities and benchmarks. There is no promised accuracy target or
claim that extra expressivity guarantees improved optimization/generalization.

## 13. Implementation milestones and acceptance checklist

1. **Reference audit:** pin the actual backbone/source, effective runtime,
   preprocessing, initialization, recipe, and expected parameter inventory.
2. **Dense oracle:** implement and test masks, global pooling, empty-pair
   handling, fusion, and gradients on small FP64/FP32 examples.
3. **Model integration:** add the instance-local final-block/forward adapter,
   retain all base modules, and prove identity-path and active-path correctness.
4. **Chunked optimization:** jointly compute old state messages and new pooling
   statistics, merge globally, and verify forward/backward/checkpoint parity.
5. **Runtime integrity:** test DDP, AMP, multiple forward lifetimes, complete-state
   restart, source locking, storage accounting, and launcher dry runs.
6. **Benchmark queue readiness:** provide an independent prepare/benchmark/summary
   launcher, readable evidence schema, under-six-hour preliminary limits, and
   clear commands in a package README. No production submission in this stage.
7. **Production readiness after review:** authenticate passed evidence, freeze
   the full training identity, implement the model-local chain and guarded resume,
   and provide explicit reviewed queue commands without altering existing jobs.

The implementation is ready for benchmarking only when its expected total is
**2,349,702**, all base parameters/buffers and numerical pathways are accounted
for, the active readout has a verified gradient path, masks and chunk reductions
are correct, and restart/launch safety checks pass. CPU tests do not substitute
for measured GPU admission, and benchmark completion does not establish accuracy.

## 14. Code entry points

- [Full Layerwise State-EV assembly and base counts](model.py).
- [Same-math state feature cache and query chunks](../part_ev_state_speed/model.py).
- [State features and native-attention/EV integration](../part_ev_message_study/model.py).
- [Layerwise bias wrappers and pair-context binding](../part_layerwise_screen/model.py).
- [Pinned Weaver identity, reference config, and outer model wrapper](../part_ev_fullscale/model.py).
- [Instance-local explicit forward precedent, without reusing its memory mechanism](../part_ev_pair_memory/model.py).
- [Complete-state Ranger optimizer](../part_ev_fullscale/optimizer.py).
- [Full recipe and learning-rate source](../part_ev_state_fullscale/protocol.py).
- [Existing independent two-A100 benchmark contract](../part_ev_state_bias/protocol.py).
- [Existing production restart/admission reference](../part_ev_state_production/README.md).

**Implementation principle:** expose the existing learned relationships to the
classifier with a small, well-trained, numerically sound new path, while retaining
the entire successful Layerwise State-EV model and a fair way to measure what
the addition actually contributes.
