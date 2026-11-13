# Evolving pair memory across layers

Recorded: 2026-11-13. Detailed first-design and implementation plan.
Planning only: no production implementation, benchmark admission, or jobs.

## Recommendation and motivation

Build **gated, multi-timescale pair memory alongside the existing State-EV
message**, retaining noncompact, standard-relations, all-eight layerwise
State-EV at state width 32. Use a separate 32-channel memory for each directed
particle pair, updated across depth and read through the existing attention.

Current State-EV asks: "Given what these particles look like now, what message
should their relation contribute?" Memory adds: "What have earlier layers
learned about this particular pair, and should newer evidence replace it?"
Particle states already carry history, but that history has been aggregated
over neighbors. The hypothesis is that a dedicated pair-specific history can
preserve information that is awkward to retain in one vector per particle.

This is my preferred first design for balancing accuracy potential, optimization
stability, and feasible execution. It is not a prediction of a large gain:
the extra history could be redundant, poorly aligned across layers, or costly
without helping classification.

The reference has **2,327,462 parameters** and reached **86.3151% training-time
validation accuracy** in seed 101 with two A100s/SyncBatchNorm. Keep its existing
current-state path intact. Do not replace it with memory, shrink it, switch to
last-four blocks, introduce selected relations, or combine the other two ideas
in this first experiment.

This refines the earlier sketch in two ways: replace an unconstrained residual
memory sum with a bounded retain/write update, and give memory its own additive
value readout instead of inserting it inside the original State-EV features.
The latter preserves the original message and provides a direct learning path
for the new readout even while the original State-EV output starts at zero.

## 1. Fixed first-design choices

| Component | First implementation |
| --- | --- |
| Particle backbone | Existing eight particle blocks and two unchanged class blocks |
| Physics inputs | Existing standard relations and shared 64-channel pair stem |
| Attention bias | Existing independent layerwise physics-bias heads |
| Current-state message | Existing width-32 State-EV in all eight blocks, unchanged |
| Pair memory | Directed, width 32, shared across heads but not across pairs |
| Writer | Layer-specific `32 -> 32` projection and bounded candidate |
| Recurrence | Learned channelwise retain/write gate; two initial retention scales |
| Reader | Layer-specific, head-specific linear value readout, initialized to zero |
| Lifetime | One jet forward pass; differentiable across layers, reset between passes |
| First training arm | Fresh seed 101, unchanged full Ranger schedule and two-A100 runtime |

Width 32 is the accuracy-first starting point, not a benchmark-tuned choice
between 16 and 32. A width reduction after a memory failure would be a separately
declared model, not an invisible implementation optimization.

## 2. Exact read/write ordering

Number particle blocks `l=0,...,7`. Let `m^l` be the history available when block
`l` starts. Memory is indexed by receiver `i`, sender `j`, and channel `d`.
The dense shape is `[local_batch, N, N, 32]`.

| Block | History read into particle messages | History written for the next block |
| --- | --- | --- |
| 0 | None: there is no earlier pair history | Initialize `m^1` from its current State-EV observation |
| 1 through 6 | Read incoming `m^l` | Gated update to `m^(l+1)` |
| 7 | Read `m^7` | None: no later particle block would consume it |

Thus there are **seven writers, six learned update gates, and seven readers**.
Do not allocate an unused first reader or last writer. All eight blocks retain
their ordinary EV and current-state State-EV messages.

Both the current observation and the attention weights use states entering the
block. Read incoming memory, not the candidate just written by this same block.
Particle updates and the new memory then advance together. This has no circular
dependency, and the last block still sees fresh states through ordinary State-EV.

Memory remains directed even if the raw physics relations are symmetric:
`m_ij` need not equal `m_ji`. Keep valid diagonal pairs consistently with the
reference; zero every pair involving padding after initialization and updates.
Do not triangular-pack or mirror memory tensors.

## 3. Writer: a fresh observation anchored to physics

Reuse the existing graph-connected State-EV features, not a duplicate pair MLP:

```text
g_ij^l = E_l LN64(z_ij)
p_i^l  = P_l LN128(x_i^l)
q_j^l  = Q_l LN128(x_j^l)
r_ij^l = GELU(LN32(g_ij^l + p_i^l + q_j^l + p_i^l * q_j^l))

o_ij^l = LN32(r_ij^l)
c_ij^l = tanh(W_l o_ij^l + b_l)       # writers l=0,...,6
```

Here `x` denotes the states presented to the existing state-message module,
including the reference block's pre-attention normalization. Preserve that
normalization order; this notation does not authorize removing a normalization.
All new LayerNorms are non-affine, channelwise, epsilon `1e-5`.

The original `r` still feeds its original State-EV output without modification.
The extra normalization and projection apply only to the memory candidate.
Each writer has independent weights to translate that layer's feature basis
into the jointly learned memory coordinates. Initialize `W_l` with Xavier
uniform, gain 1, and `b_l=0`, using isolated per-layer initialization seeds.

Every observation still includes the same physics anchor `z_ij`; later writers
also see contextualized endpoint states. Do not multiply writes by attention
weights: a currently low-attention pair should be able to preserve information
for a later layer. Other particles affect this history through endpoint states,
not through a new cubic-cost triplet operation.

For block 0, set `m^1=c^0`, then apply the pair mask. A learned gate against an
empty initial memory adds little value; start with a full first observation.

## 4. Recurrence: learn what to keep and what to replace

For writers `l=1,...,6`, use four learned 32-element vectors per layer:

```text
u_ij^l = sigmoid(a_l * c_ij^l + b_gate_l * m_ij^l
                 + d_l * (c_ij^l * m_ij^l) + beta_l)

m_ij^(l+1) = (1 - u_ij^l) * m_ij^l + u_ij^l * c_ij^l
```

All products here are elementwise. **`u` is the write fraction**: zero retains
history; one replaces it. Candidate, prior memory, and their interaction let the
gate learn whether a channel should accept new evidence. This is a lightweight
GRU-inspired accumulator, not a full GRU with a reset gate and dense recurrent
matrices. Its memory/candidate product is not the separate proposed modulation
of physics relations by particle states.

Initialize `a_l=b_gate_l=d_l=0`. Initialize gate biases as follows:

- Channels 0--15: `beta=-log(4)`, giving write fraction 0.2 and retention 0.8.
- Channels 16--31: `beta=0`, giving write fraction 0.5 and retention 0.5.

These provide slower and faster initial histories over a short eight-block
network. They are learned biases, not fixed groups or a new regularizer.
Avoid extremely slow initial gates such as 0.99 retention, which could make
later observations almost irrelevant before the model has learned anything.

The convex mixture of bounded candidates keeps each valid memory channel in
`[-1,1]`, up to rounding. That bounds activations, **not gradients or logits**.
Do not LayerNorm the carried memory after each update: doing so would discard
part of the retain/write amplitude information. Cross-channel mixing happens
in the writers and readers; gates remain inexpensive and channelwise.

## 5. Reader: history supplies an additional value message

For reader blocks `l=1,...,7`, let `alpha^l` be the existing native per-head
attention weights, including the reference's attention dropout:

```text
pooled_memory[b,h,i,d] = sum_j alpha^l[b,h,i,j] * m^l[b,i,j,d]
history_value[b,h,i,c] = sum_d V_l[h,c,d] * pooled_memory[b,h,i,d]

extra_heads = ordinary_EV_heads + current_State_EV_heads + history_value
```

`V_l` has shape `[8,16,32]` and no bias. Keep the existing operation order for
the first two contributions, then add history. Join heads and apply the native
output-projection weight once to this extra message, without adding its bias
again; add it to the unchanged ordinary attention output as the reference does.
Preserve the subsequent head scaling, norms, residuals, FFN, and dropout.

**Pool before projecting.** A linear reader commutes with weighted aggregation;
there is no need for a `[B,H,N,N,16]` history-value tensor. Reuse native attention
weights rather than computing a second attention, changing softmax temperature,
renormalizing dropout weights, or adding a memory-derived bias in this arm.

This leaves attention's formula unchanged, not its eventual learned weights:
memory changes particle states, which can alter subsequent Q/K attention.

## 6. Initialization and gradient flow

Initialize every `V_l` to zero; keep writers nonzero as above. With matched old
parameters, the new message is initially zero, recovering the original model's
function subject to execution-path numerical parity checks. Preserve existing
parameter and data/dropout RNG streams when constructing new modules.

On a nondegenerate fixture:

1. First backward: readers can receive nonzero gradients because their incoming
   memories are nonzero. Writer/gate gradients are initially zero through the
   zero readers; that is expected.
2. After a successful reader update: gradients reach writers and update gates,
   including earlier writers through later readers and the carried memory.
3. Gradients must traverse all layers normally. No detached recurrent state,
   truncated backpropagation, or `no_grad` memory construction during training.

Do not add a second zero scalar multiplier, skip the branch when `V=0`, or put
the reader behind the original zero-initialized State-EV output projection.
Those changes can defeat the intended learning path. AMP-skipped steps do not
count as successful updates for the activation test.

## 7. Implementation boundaries and efficient execution

Use a new experimental package and frozen campaign snapshot, provisionally
`part_ev_pair_memory` / `STATE_MEMORY_EV_LAYERWISE`. These are proposed names,
not existing launchable components. Leave current campaigns and installed
Weaver sources untouched.

Start from [the current model assembly](model.py),
[FastStateMessage](../part_ev_state_speed/model.py),
[message/output integration](../part_ev_message_study/model.py), and
[layerwise bias](../part_layerwise_screen/model.py).

Unlike the other two ideas, this needs **explicit cross-block activation flow**.
Create a private, source-audited forward/block adapter that carries `(x,m)` while
reusing the native submodules and their operation order. Avoid a mutable
`self.current_memory` read by hooks during backward. Trim/permutate particles
once before creating memory, then use the same axes/masks throughout. Class
attention and final classification stay unchanged and do not carry pair memory.

Execution plan:

- Normalize the shared pair stem once per forward, keeping its autograd graph.
  Compute each block's endpoint projections once, as in the current speedup.
- Start with query chunks of 64. Inside each pure chunk, reuse `r` for the old
  state output and candidate; read incoming memory and return outgoing memory
  explicitly alongside message contributions. Do not recompute native attention.
- Use non-reentrant activation checkpointing for these deterministic chunk
  functions. Pass graph-connected incoming memory, weights, masks, and features
  explicitly; bind the correct layer module, avoiding late-bound loop closures.
- No in-place memory update or detached preallocated copy. Concatenation of
  returned chunks must retain gradients. Do not stash all candidates/gates for
  diagnostics, and do not recompute BatchNorm inside a checkpointed function.
- Retain the reference AMP policy for dense projections and pooling. Compute
  new normalization, candidate `tanh`, gate logits/sigmoid, and the recurrent
  mixture in FP32; store carried memory in FP32. Make casts explicit at the
  message boundary. Do not silently enable TF32, BF16, compilation, or dropout.
- Require finite preactivations too: bounded `tanh` is not permission to conceal
  overflow. Diagnostic float64 fixtures may use float64 throughout.
- Each DDP rank owns its own jet memories. Synchronize model gradients and use
  the existing SyncBatchNorm, not cross-rank pair-memory communication.
- No persistent memory buffer or disk cache. Forward locals can be released;
  backward uses explicit saved arguments. Reset on every new batch/forward,
  including after exceptions, validation, and resumed execution.

Checkpointed recomputation must reproduce the forward function and its RNG
behavior. `preserve_rng_state=False` is appropriate only for the pure chunks
without random operations, not native attention/dropout or an entire block.
See the [PyTorch checkpoint documentation](https://docs.pytorch.org/docs/2.12/checkpoint.html);
verify behavior on the actual pinned cluster version, not just local PyTorch.

## 8. Parameter, compute, and storage budget

Expected additions for this exact design:

| Component | Parameters |
| --- | ---: |
| Seven candidate projections, each `32*32 + 32` | 7,392 |
| Six gates, each four 32-element vectors | 768 |
| Seven readers, each `8*16*32` | 28,672 |
| **Added** | **36,832** |
| **Total including reference** | **2,364,294** |

This is approximately **1.58% more parameters**, not a matched-capacity control.
Assert counts and that every new parameter belongs to the optimizer/Lookahead.

Additional leading forward work per jet includes `7*N^2*32^2` writer MACs,
`7*8*N^2*32` reader-pooling MACs, and `7*8*N*32*16` reader-projection MACs, plus
normalization and elementwise gate/nonlinear operations. One multiply-add is
two FLOPs under that convention; these terms are not the total model FLOPs or
a runtime estimate. Backpropagation and checkpoint recomputation add work.

**Activation memory is the real constraint.** At local batch 256, padded length
128, and FP32, one `[256,128,128,32]` memory is **0.5 GiB**. Seven boundary
values alone are **3.5 GiB**. This excludes their gradients, checkpoint inputs,
temporary copies, attention tensors, the original State-EV path, and workspaces.
Query chunking reduces temporary tensors, not the full quadratic memory carry.
Do not claim constant-memory training or infer fit from the parameter count.

Under no-grad inference, only current/next memory and chunk temporaries need
remain live; do not retain a list of all layers' histories. Inference still has
the extra arithmetic. Benchmark throughput and latency separately.

The added FP32 weights are only about 0.14 MiB; optimizer/slow-weight state adds
more, but pair activations are never written to persistent storage. Budget
ordinary checkpoints, prediction outputs, and existing campaigns separately.
No cleanup or deletion is implied by this plan.

## 9. Tests required before GPU admission

1. **Pure recurrence oracle:** small unchunked forward/backward versus chunked
   and checkpointed implementations with nonzero readers; include chunk edges.
2. **Baseline identity:** zero-reader output and old-parameter gradient parity
   against the unchanged model, including trained-like nonzero original State-EV
   outputs. Fresh zero State-EV alone could hide an integration bug. Check BN,
   trimmer, dropout RNG, output projection, and parameter-copy coverage as well.
3. **Learning path:** reader-first activation, then finite writer/gate gradients
   after successful updates. A final-layer reader must reach the first writer.
   No permanently unused parameters hidden by permissive DDP settings.
4. **Actual history dependence:** hold later observations, current states,
   geometry, and attention fixed; change an earlier observation and verify that
   a later read changes. This distinguishes memory from a stateless adapter.
5. **Gate semantics:** update zero retains, update one overwrites; first write
   initializes; final block does not write. Check bounds and finite gradients.
6. **Pair correctness:** simultaneous particle permutation on both pair axes,
   directed asymmetry, padding isolation, valid diagonals, unequal lengths and
   one-particle inputs. Handle/reject empty inputs consistently with the base.
7. **Lifetime:** consecutive unrelated forwards, backward after forward cleanup,
   exception recovery, and train/eval transitions cannot leak jet memory. It
   must not appear as a persistent activation in model checkpoints.
8. **Distributed/restart:** two-GPU updates and saved/reloaded continuation across
   a Lookahead boundary; restore full optimizer/scaler/RNG/data-stream state.
   Preserve the existing exact replay contract, including replay runtime.

Use exact equality where supported, otherwise predeclared justified tolerances
for FP32/AMP. A failing comparison triggers diagnosis, not automatic relaxation.

Planning-stage sanity check: an in-memory CPU float64 toy on local PyTorch
2.12.1 passed the 36,832-parameter count, zero readout, reader-first gradients,
subsequent gradient coverage, bounded/masked memory, chunked forward/backward
agreement, history sensitivity, and final-reader-to-first-writer gradient checks.
This was **not** an integrated ParT test, checkpoint-recomputation test, CUDA/DDP
benchmark, or accuracy experiment. Implement the above as persistent tests later.

## 10. Diagnostics and benchmark-to-production path

Collect small, detached scalar summaries at sparse intervals: per-layer write
fractions and saturation, memory/candidate RMS, update size, readout norm relative
to current State-EV, and writer/reader gradient norms. Never log full pair
tensors. Diagnostics must not advance training RNG or normalization state.
Nonzero memory/readouts are evidence of activity, not evidence of useful gains.

Benchmark on **two same-node A100s, local batch 256, global batch 512**, with
the reference workers/numerics. Requested diagnostic walltime stays under six
hours. Exercise both initial and active readers, sustained real-data training,
maximum 128-particle forward/backward, validation, checkpoint I/O, and resume.
Measure both ranks' allocated/reserved memory and AMP recovery. Target at least
10% allocation headroom on the actual GPUs, with reserved memory/workspace
behavior also reviewed; one successful short batch is insufficient admission.

If memory is tight, first test smaller query chunks with oracle parity. Do not
silently detach history, reduce width, sparsify pairs, change batch/BatchNorm,
or drop blocks. If the declared model still does not fit, stop and choose a
new explicitly documented variant rather than launch an unstable run.

Production follows the [shared full-schedule contract](STATE_EV_NEXT_IDEAS.md#shared-training-and-comparison-contract):
fresh seed 101; 1M update attempts; 50 checks at 20k intervals; unchanged Ranger
recipe, data population, checkpoint selection, and full-5M final evaluation.
Keep the independent test set untouched. No warm start from the 86.3151% model.

Use **one long SPORC training job**, not planned three-day segments. Verify the
live requested partition/account. Twelve days is provisional; choose walltime
from measured training plus all validations/checkpointing and explicit margin.
Retain regular atomic latest/full-state and accuracy-selected checkpoints.
Only this model's successful completion unlocks its evaluation and report.

The main control is unchanged all-eight layerwise State-EV, with a provenance
audit before reusing historical results. Its two-GPU recipe and segmented
restart history must be disclosed. Any new control training needs approval.

Compare equal-update progress and accuracy-selected full-5M accuracy, AUC, and
classwise rejection with underlying acceptance counts. A later history-disabled
evaluation can show sensitivity, but is a distribution-shift ablation, not proof
that recurrence caused a gain. If this arm wins, a separately approved trained
stateless/capacity control and matched seeds should test that interpretation.
Do not declare a breakthrough from the best result among many single-seed tries.

## 11. Why this design, and what is deliberately deferred

- **Not an unbounded residual memory:** a retain/write mixture makes forgetting
  explicit and controls channel amplitude without overwriting the proven path.
- **Not a full GRU or a deep pair MLP:** reuse existing pair/state observations;
  add one small candidate projection and cheap gates at quadratic pair scale.
- **Not a static cache:** each write depends on the current contextualized states
  and each gate can depend on the previous memory; history affects later states.
- **Not a replacement for State-EV:** current evidence remains available directly
  even if historical evidence is stale or unhelpful.
- **Not a simultaneous routing experiment:** memory-derived attention bias and
  explicit relation-state modulation remain separate plans. Combining them
  later requires individual evidence and a new controlled comparison.

Relevant precedents are evolving edge channels in
[EGT](https://arxiv.org/abs/2108.03348), jointly updated node/pair representations
in [GRIT](https://proceedings.mlr.press/v202/ma23c.html), and gated recurrent
retention in [Cho et al.](https://arxiv.org/abs/1406.1078). They motivate carrying
and selectively updating pair information; they do not establish this design's
novelty, JetClass gains, or inherited theoretical guarantees.

The implementable hypothesis is precise: **preserve today's strong current-state
message, and add a learnable history of each pair without forcing the classifier
to use it at initialization**. GPU feasibility and classification benefit remain
to be measured. See the [three-idea index](STATE_EV_NEXT_IDEAS.md) for the separate
[attention-bias](STATE_CONDITIONED_ATTENTION_BIAS_PLAN.md) and
[multiplication](RELATION_STATE_MULTIPLICATION_PLAN.md) experiments.
