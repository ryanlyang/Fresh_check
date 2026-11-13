# State-conditioned physics attention bias

Recorded: 2026-11-13. Model and independent SPORC benchmark implemented in
[`part_ev_state_bias`](../part_ev_state_bias/README.md). Benchmark queue commands
are in that README. No remote jobs submitted; production admission remains
conditional on the new GPU evidence and a subsequent production launcher.

Base: noncompact, standard-relations, all-eight layerwise State-EV, width 32,
2,327,462 parameters. The motivating seed-101 two-A100/SyncBatchNorm run reached
86.3151% training-time validation accuracy at check 49. This is not independent
test evidence. See the [shared context and notation](STATE_EV_NEXT_IDEAS.md).

**Question:** Can the same pair geometry deserve different attention depending
on what the network currently understands about the two particles?

## Recommended first experiment: accuracy-focused, not compact

The aim is to improve the final accuracy-selected checkpoint of the successful
all-eight layerwise State-EV model. The following are our preferred first-design
choices, not experimentally established optimal hyperparameters or a promise
to exceed 86.3151%. Preserve the successful model's capacity and full training
budget; use implementation optimizations only when they preserve the intended
function and gradients.

| Decision | First version | Reason |
| --- | --- | --- |
| Backbone and relations | Existing noncompact ParT, standard physics relations | Isolate the new mechanism; no selected-relations or compact changes |
| State-EV | All eight particle blocks, width 32 | Retain the successful message branch rather than remove capacity |
| Existing physics bias | Keep all eight independent layerwise heads and shared pair stem | Add adaptive behavior without discarding the learned geometry pathway |
| New bias features | Existing nonlinear, directed 32-channel state-pair features | Directly connect the relation and both endpoint states without another pair encoder |
| New bias heads | Independent, bias-free `32 -> 8` projection in each block | Different layers and heads can learn different corrections |
| Correction strength | `tanh(B_l r_ij^l)`, fixed bound of one logit unit | A nonzero learning signal at initialization with bounded changes to attention |
| Initialization | All new `B_l` weights zero; retain existing initialization | Start from the corresponding original model function, not arbitrary altered attention |
| Which messages change | Native values, ordinary EV, and State-EV share corrected weights | Test a coherent state-conditioned attention mechanism |
| Training | Fresh seed 101, unchanged full Ranger recipe | Keep the architecture change separate from optimizer or exposure changes |

The bound of one is a concrete starting choice, not a known accuracy optimum.
Do not add a learned temperature, a separately tuned learning rate, extra
dropout, new normalization, a private pair encoder, or ideas 2/3 to this first
arm. If diagnostics later motivate a different bound or design, record it as
a new experiment rather than silently changing this one.

## Exact computation and intended benefit

Use the existing feature calculation in the [shared model description](STATE_EV_NEXT_IDEAS.md#what-the-current-model-already-does), including its
normalization, GELU, receiver/sender projections, and valid-pair masking. Let
`B_l` have shape `[8, 32]`. For each valid pair:

```text
delta_b_ij^(l,h) = tanh((B_l r_ij^l)_h)

score_ij^(l,h) = dot(q_i^(l,h), k_j^(l,h)) / sqrt(d)
                + b_lh(z_ij)
                + delta_b_ij^(l,h)

alpha_ij^(l,h) = softmax_over_valid_j(score_ij^(l,h))
```

Native attention applies its existing dropout to `alpha` once during training;
the native value path and both EV paths use the resulting same weights. In
evaluation, dropout is disabled as before. Retain existing output projections,
residual connections, and downstream block operations.

The physics-bias head currently depends only on the pair embedding. The new
term allows its contribution to change with the particles' evolving states.
Q/K attention is already state-dependent, so this is specifically a direct
**state-by-physics-relation interaction**, not the first use of particle state
in attention. A receiver-only additive offset would cancel under row softmax;
keep the nonlinear pair-and-endpoint interaction rather than replacing it with
such an offset.

Sharing `r_ij^l` means routing and message content both train the state-feature
extractor. Do not detach either path or freeze the backbone. This is intentional,
but competing gradients are a risk to inspect rather than assume beneficial.

The addition is **2,048 parameters** (`8 blocks * 8 heads * 32 features`), giving
an expected total of **2,329,510**. This assumes no additional learned gates,
biases, or normalization parameters. Assert this contract when implementing.

Construct everything from the states entering the current block. No current
block output is needed to compute its attention, so there is no circular
dependency. New corrections are directed even though the standard physics
pair features are symmetric: never generate them by triangular mirroring.

This is not the old `EV_ROUTING` experiment. That used a geometry-only
correction and changed only EV's aggregation weights. This proposal conditions
the correction on endpoint states and changes the full attention operation.

## Integration into the current code

Implement a separate experimental package/driver, with its own campaign schema
and source snapshot. Compose or adapt the current all-eight layerwise model
without changing the implementations or frozen snapshots used by old jobs.
The implementation uses `part_ev_state_bias` and `STATE_BIAS_EV_LAYERWISE`.
The benchmark submitter snapshots committed code into a new campaign directory.

Important insertion point: `LayerwiseAttention.forward` currently supplies its
bound layer-specific bias as `attn_mask` to the inner `MessageAttention`.
The new inner attention implementation must add the dynamic correction to that
received mask **before its one native MHA call**. Adding it outside the wrapper
and then letting the wrapper replace the mask would silently discard it.

Forward plan for each particle block:

1. Receive the original particle states, shared pair stem, validity mask, and
   that layer's existing physics-bias mask.
2. Normalize/project states with the existing state branch. Compute pair-state
   features in query chunks and project them to eight bounded bias corrections.
3. Assemble the correction in native attention-mask layout `[batch*heads,N,N]`.
   Add it to the layer bias without mutating the original bound tensor in place.
   Preserve key-padding exclusion and self-pair behavior. Set invalid-pair
   corrections to zero; do not replace valid native masks or introduce fully
   masked query rows that create NaNs.
4. Call the same native attention implementation once, requesting per-head
   weights. Its ordinary output already includes the corrected attention.
5. Aggregate ordinary EV and the unchanged state-message branch with those
   returned weights. Add their output-projected contribution exactly once,
   without adding the native output-projection bias a second time.
6. Clear forward-local bindings/caches on both success and exceptions, retaining
   only explicit tensors needed by autograd/checkpoint recomputation.

Preserve native Q/K score scaling, dropout/RNG behavior, AMP policy, and mask
dtype compatibility. Use stable evaluation of the new bounded correction,
then cast/merge deliberately; do not accidentally promote or replace the
entire native attention computation as an unrelated change. No FlashAttention
or other kernel rewrite is bundled into this first architecture experiment.

## Memory-conscious implementation without reducing model capacity

Adding only 2,048 weights does **not** imply negligible compute or activation
memory: the state-pair features are now needed before and after softmax.

- Preserve the existing forward-local, graph-connected normalized-pair cache.
- Start with query chunks of 64, as in the optimized all-eight reference.
- Checkpoint the pure feature-to-bias chunk function. Pass all graph-connected
  inputs explicitly, without consulting a mutable cache during backward.
- Retain the scalar correction per head, not the complete 32-channel pair-state
  tensor for every layer. Recompute the same state features in the existing
  post-attention message chunks as necessary. Both branches' gradients must
  reach their shared projections.
- Do not wrap the dropout-bearing MHA call in a checkpoint region with RNG
  preservation disabled. The new feature chunks should contain no dropout or
  mutable BatchNorm state.
- Keep the original full state width, eight blocks, global batch, and particle
  handling. If memory is tight, first evaluate function-preserving chunking
  and recomputation options rather than quietly shrinking the model.

A straightforward unchunked implementation on small fixtures is the numerical
oracle for the chunked implementation. Measure the extra feature-computation
cost; do not describe recomputation as free or claim a speedup before profiling.

## Initialization and correctness checks

Create new parameters without advancing the existing model/data/dropout RNG
streams, for example by isolating construction RNG. New production runs start
from the normal seeded model initialization, not the trained 86.3151% weights
or benchmark weights. Zero output heads preserve the initial function, not
the subsequent optimization trajectory.

Do not skip the new branch during training merely because `B_l` is all zero:
that would prevent learning its first update. Do not add a second zero-valued
trainable multiplier that also blocks its gradient.

Required tests before GPU admission:

1. **Identity insertion:** with `B_l=0`, match the unchanged model's output and
   existing-parameter gradients under controlled RNG, padding, and dropout.
   Require exact equality where the execution path permits it; otherwise use
   predeclared, justified FP32/AMP tolerances, not tolerances fitted to a failure.
2. **Learning path:** each new head receives finite, nonzero gradients on a
   nondegenerate fixture. Once the appropriate output heads are nonzero, verify
   gradients reach receiver, sender, and edge projections from the new branch.
   Existing zero-initialized state-output heads need not give all hidden
   projections nonzero gradients on the very first update.
3. **Nonzero equivalence:** compare outputs and gradients against the small
   unchunked oracle, including backward checkpoint recomputation and query
   lengths crossing chunk boundaries.
4. **Attention semantics:** verify that native values and both EV contributions
   use the corrected weights; preserve the original single dropout application.
5. **Structure:** test directed/asymmetric state pairs, permutation equivariance,
   valid self-pairs, padded particles, and trimming alignment. Define behavior
   for unsupported empty jets explicitly instead of allowing silent NaNs.
6. **Training infrastructure:** test two-rank synchronization, new-parameter
   optimizer/Lookahead coverage, checkpoint restore, scaler and RNG restoration,
   and restart equivalence through a Lookahead synchronization boundary.

## Full training and comparison contract

- **Site/runtime:** submit from `sporcsubmit`, using two A100s on one node,
  DDP/SyncBatchNorm, global batch 512 (256 per rank), and the reference worker,
  numerical, and data-stream settings. Start with its eight CPUs and 64 GiB host
  memory; verify adequacy in the benchmark.
- **Recipe:** copy and freeze `ranger_full_v1`, including its exact LR schedule,
  RAdam/Lookahead settings, AMP policy, and checkpoint-selection rule. Do not
  introduce AdamW, altered weight decay, or a new warmup for this arm.
- **Exposure:** 1,000,000 update attempts, nominally 512M jet exposures from the
  official 100M training pool. Report AMP-skipped updates separately.
- **Validation:** retain all 50 checks, every 20,000 updates, using the existing
  1.28M-event streaming-validation procedure. These are not full dataset epochs.
  Select highest accuracy, retaining the first check in a tie.
- **Final evaluation:** run the selected checkpoint on the official full 5M
  validation population, recording accuracy, macro AUC, and classwise QCD
  rejection with acceptance counts. This overlaps selection validation and is
  not an independent test. Do not open the 20M test population for development.
- **Control:** compare against the unchanged all-eight layerwise State-EV, not
  just ordinary ParT. Reuse the seed-101 two-A100 reference only after auditing
  recipe, population, environment, initialization, and runtime compatibility.
  Its segmented restart history must remain disclosed. If compatibility is
  insufficient, propose a fresh matched control and obtain approval for that
  extra run; do not silently add it.

Compare full learning curves, matched-update checkpoints, and best accuracy
over the same 50 selection opportunities. Treat single-seed improvements as
exploratory, especially after repeated model selection on this validation set.
Do not increase the exposure or select a different checkpoint for each metric
to make the new model appear stronger.

## Benchmark and long-job launch plan

1. Implement and pass the local tests, then freeze a new independent snapshot
   and environment evidence. No modifications to existing campaigns or shared
   environments are required or authorized by this plan.
2. Queue a short correctness/throughput benchmark on the intended two-A100
   runtime, with requested walltime **under six hours**. Measure sustained
   real-data training across multiple fetch windows, maximum-length memory,
   FP32 validation throughput, checkpoint time, AMP health, and restart behavior.
   Test both zero-initialized and nonzero active correction paths.
3. Project the full update budget plus all 50 validations and checkpoint I/O,
   then add a declared safety margin. Include inference throughput and memory
   in the report. Do not equate FLOPs, parameter count, or allocated GPU count
   with measured wall-clock performance.
4. Review the evidence before production. **Twelve days (288 hours) is a
   provisional walltime allowance, not a runtime prediction.** The previous
   model's reported estimate was approximately 146 hours training, 19 hours
   validation, and 216 hours including checkpoint costs and planning margin;
   none of those measurements establishes this new branch's speed.
5. Use **one long training allocation**, not predetermined three-day segments.
   Verify live account/QoS, partition limits, reservations, and available GPU
   resources. The requested workflow is SPORC production (`tier3` historically;
   current documentation calls it `sporc`). Resolve this explicitly rather than
   silently routing to another partition. Long requests may still wait longer
   for scheduling; a free-looking queue is not a start-time guarantee.
6. Attach this model's full-validation job to **its own successful training**,
   followed by its own report. Do not make it wait for the other proposed models
   or an all-model completion gate. Any cross-model comparison report is separate.

Continue atomic restart checkpoints every 1,000 steps and at validation/stop
boundaries, preserving full optimizer/Lookahead/scaler, counters, per-rank RNG,
trimmer, and data-replay information. Keep committed selected-model candidates
safe during interrupted writes. A long allocation still needs recovery support;
unplanned failures must not be labeled successful or authorize evaluation of
an incomplete training budget. Do not automatically submit a new segment chain.

Check persistent storage before admission using projected restart, selected
checkpoint, prediction, and snapshot sizes. State-pair activations stay in
memory/recomputation, not a disk cache. Never log full pair tensors or delete
old campaign artifacts automatically.

## Diagnostics, interpretation, and scope boundary

On a small fixed diagnostic sample, record per-layer/head correction RMS and
maximum magnitude, fraction near tanh saturation, changes in pre-dropout
attention entropy/distributions, new-head gradient norms, and state-message
norms. Collect diagnostics without perturbing training RNG, BatchNorm state,
or data position; do not perform expensive diagnostic passes every update.

Watch for an unused correction, early saturation, attention collapse, nonfinite
gradients, or throughput/memory regressions. These diagnose implementation and
optimization problems; they are not grounds to claim an accuracy improvement.
Do not change the bound or recipe mid-run based on validation results.

Implementation is conceptually a small additional bias head but requires careful
attention integration, precision, recomputation, and distributed-resume testing.
The first arm stands alone: **no explicit relation-state multiplication, evolving
pair memory, selected relations, or compaction is bundled into it**. See the
[idea index](STATE_EV_NEXT_IDEAS.md) for the two separate plans. This document
does not itself submit benchmarks or authorize production jobs. The independent
implementation and current queue instructions are linked above.

## Code entry points

- [Current layerwise State-EV assembly](model.py)
- [Optimized state-message implementation](../part_ev_state_speed/model.py)
- [State features and message integration](../part_ev_message_study/model.py)
- [Layerwise bias wrapper](../part_layerwise_screen/model.py)
- [Reference two-GPU training workflow](../part_ev_layerwise_ddp/README.md)
