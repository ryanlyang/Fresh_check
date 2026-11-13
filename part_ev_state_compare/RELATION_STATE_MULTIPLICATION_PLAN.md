# Explicit relation-state multiplication inside State-EV

Recorded: 2026-11-13. Proposed first implementation; no code or jobs submitted.

## Objective and fixed starting point

Test whether an explicit multiplicative interaction improves the final
accuracy-selected checkpoint of **noncompact, standard-relations, all-eight
layerwise State-EV**, keeping its state width of 32 and the full training recipe.
The reference has 2,327,462 parameters and reached 86.3151% training-time
validation accuracy in seed 101 on two A100s with SyncBatchNorm.

These are promising design choices, not proven optimal settings or a prediction
of a particular accuracy gain. Current State-EV already combines relations and
particle states nonlinearly; this proposal changes how easily the model can
express their interaction, not what input information it receives.

## Core idea: particle states modulate relation channels

The original state-message features are:

```text
g_ij = E_l LN(z_ij)          # 64-channel pair embedding -> 32 channels
p_i  = P_l LN(x_i^l)        # receiver particle state -> 32 channels
q_j  = Q_l LN(x_j^l)        # sender particle state -> 32 channels
h_ij = g_ij + p_i + q_j + p_i * q_j
r_ij = GELU(LN(h_ij))
```

Here `p` and `q` are the State-EV endpoint projections, not native attention's
Q/K projections. All `*` operations below are elementwise.

Add two independent bias-free `32 -> 32` matrices per particle block:

```text
gamma_ij = tanh(A_l p_i + C_l q_j)
h_new_ij = h_ij + g_ij * gamma_ij
r_new_ij = GELU(LN(h_new_ij))
```

Algebraically the relation contribution becomes
`g_ij * (1 + gamma_ij)`. Each relation channel has a state-dependent multiplier
between zero and two. Compute the original `h_ij` in its original operation
order and then add the residual for a clean zero-modulation comparison.

Interpretation: what the network currently believes about these particles can
directly suppress or amplify components of their learned relation representation.
The modulation acts on the projected learned embedding, not directly on raw
physical quantities such as momentum or angle.

## Recommended design choices

- **Channelwise, not one scalar per pair.** The 32 channels can change relative
  to each other, rather than merely rescale a vector. LayerNorm can remove much
  of a uniform rescaling of its entire input; changing the channel composition
  gives this mechanism a more direct effect. A scalar applied only to `g`
  would not necessarily vanish, but is a less expressive first choice.
- **Distinct receiver and sender matrices.** Keep endpoint roles separate;
  directed messages need not behave the same under `i <-> j`.
- **Independent matrices in all eight blocks.** Early and late particle states
  can support different interpretations. Retain width 32 in every state block.
- **Bounded residual modulation.** Use `tanh` with fixed unit amplitude, not an
  unconstrained multiplier, learned temperature, or additional trainable gate.
  Zero means the original relation contribution, not removing it.
- **Preserve the existing message.** Keep `p_i + q_j + p_i*q_j`, the existing
  normalization/GELU, and the head-specific state-output projection. Do not
  replace the branch with an entirely new MLP or gate its entire final output.
- **No attention-bias extension in this arm.** Retain existing Q/K attention,
  layerwise physics bias, ordinary EV, dropout, and native value aggregation.
  Later attention weights can still change indirectly as updated particle
  representations propagate; their formula is not modified.
- **Shared across heads, not across layers.** Each layer's 32-channel modulation
  feeds its existing eight head-specific state-output projections. Per-head
  pair gates would add complexity and pair-tensor cost; defer that variant.
- **No compaction, selected relations, private pair encoder, pair memory,
  optimizer change, or extra training exposure.** Test this one change first.

Expected addition: `8 * 2 * 32 * 32 = 16,384` parameters.
Expected total: **2,343,846**, approximately 0.70% above the reference.
Assert these counts in implementation; this is not a parameter-matched control.

The bounded range cannot reverse a relation channel's sign or amplify its
coefficient beyond two through this gate alone. That is a deliberate first-run
constraint, not evidence that the range maximizes accuracy. Saturation or an
unused gate should be diagnosed, not used to silently retune a running model.

## Identity initialization and the learning path

Initialize both `A_l` and `C_l` to zero, with no biases. At initialization,
`gamma=0` and the current State-EV feature calculation is recovered. Preserve
the original model initialization and RNG streams by isolating construction
of the new modules. Train fresh with seed 101; do not load the trained 86.3151%
checkpoint or benchmark weights.

There is an important difference from the attention-bias extension: **the
existing State-EV output projection itself starts at zero**. Consequently:

1. The first backward pass can give the state-output projection a nonzero
   gradient while the new modulation matrices still have zero gradients.
2. Once the state-output projection becomes nonzero, gradients can reach the
   modulation matrices through the message path.
3. This is expected delayed activation, not a permanently dead double-zero
   adapter. Test it over successful optimizer updates, accounting for AMP skips.

Also test with a deliberately nonzero state-output projection: even with
`A=C=0`, both modulation matrices should then receive finite nonzero gradients
on a suitable nondegenerate fixture. Do not add another zero-valued multiplier,
freeze the state output, or skip the gate computation when its weights are zero.

## Implementation and efficient execution

Use a new experimental model/package and campaign snapshot. Compose the
existing layerwise model and replace only its state-message modules with the
extended implementation. Keep old model files, checkpoints, and frozen jobs
unchanged. A provisional label is `STATE_MULTIPLY_EV_LAYERWISE`; it is not an
implemented run ID yet.

The relevant starting point is
[FastStateMessage](../part_ev_state_speed/model.py), with its existing pair cache,
query chunking, and checkpointed aggregation.

Forward strategy:

1. Normalize/project current particle states once as before.
2. Precompute `a_i=A_l p_i` and `c_j=C_l q_j` **per particle** before forming
   pairs. Do not repeat a dense projection independently for every pair.
3. For each query chunk, calculate the original `g` and `h`, broadcast
   `a_i+c_j`, apply `tanh`, and add `g*gamma` before the existing LN/GELU.
4. Mask invalid pair features, pool them with the unchanged native attention
   weights, and apply the existing head-specific state-output projection.
5. Keep ordinary EV and native attention/output-projection behavior unchanged.

This adds per-particle projections of order `N*32^2` and pairwise elementwise
operations of order `N^2*32`, not a new `N^2*32^2` dense pair MLP. Retain
query chunks of 64 initially, checkpoint the pure chunk function, and pass
graph-connected arguments explicitly. Never detach endpoint states or cache
them across jets. Do not store full gate tensors for all layers or write them
to disk.

Preserve reference FP32 normalization, AMP policy, mask/trimming alignment, and
dropout RNG behavior. Check the new gate/product's numerical range and make
casts explicit. Compare the chunked implementation with an unchunked small-input
oracle before calling any optimization equivalent. Small parameter growth is
not a guarantee of negligible runtime or memory growth.

## Acceptance tests and diagnostics

Before any production submission:

- Zero-modulation output and existing-parameter gradient parity with the
  unchanged model, including a trained-like nonzero state-output projection.
- Exact equality where the execution path permits it; otherwise predeclared
  justified FP32/AMP tolerances, not relaxed thresholds after a failure.
- The delayed-gradient checks above, plus coverage of every layer's new matrices
  in the optimizer, Lookahead slow weights, and checkpoint state.
- Nonzero-gate forward/backward agreement between chunked/checkpointed and
  unchunked implementations, across query-chunk boundaries.
- Finite gradients, directed-pair behavior, consistent trimming, valid self-pairs,
  padding isolation, and permutation equivariance.
- Two-GPU synchronization and uninterrupted/resumed updates across a Lookahead
  boundary, preserving scaler, RNG, and data-stream state.
- Fixed-input checks that this arm does not alter native attention or ordinary
  EV calculations when supplied the same states and unchanged shared weights.

On a small fixed diagnostic sample, record per-layer gate mean/spread, saturation
fraction, modulation-to-original-preactivation norm, resulting state-message
norm, and new-parameter gradient norms. A nonzero gate does not prove useful
classification behavior. Preserve RNG and normalization state during diagnostics.

Main risks are tanh saturation, redundancy with the current nonlinear state
branch, and learning extra capacity rather than a useful multiplicative
structure. Evaluate those risks alongside validation accuracy and rejection,
not just whether the gate becomes active.

## Training, benchmarking, and comparison

Use the common [full-schedule comparison contract](STATE_EV_NEXT_IDEAS.md#shared-training-and-comparison-contract):
fresh seed 101; full Ranger recipe; 1,000,000 update attempts; global batch 512;
all 50 training-time validation checks; two A100s on one SPORC node with DDP and
SyncBatchNorm. Do not shorten training to fit a desired walltime or alter the
optimizer specifically for this branch.

First run a short correctness/throughput benchmark with a requested walltime
under six hours. Exercise both identity and active modulation, sustained real
data, maximum-length memory, FP32 validation, checkpointing, and restart.

Then choose one long training allocation from the measured full-runtime estimate
plus margin. Twelve days is a provisional allowance, not a measured prediction.
No planned three-day segments. Keep regular resumable checkpoints, and attach
this model's evaluation and report only to its own successful upstream jobs.

Primary control: unchanged all-eight layerwise State-EV under matching runtime
and protocol. Audit whether the historical reference can be reused; request
approval for any additional control training rather than adding it silently.
Preserve the selection rule and compare equal-update progress and the same
number of validation opportunities. Final full-5M metrics use the accuracy-selected
checkpoint; independent test data stay untouched during development.

## Later combination, not part of this first arm

If individual results justify it, feed `r_new` into both the value projection
and the [state-conditioned attention-bias head](STATE_CONDITIONED_ATTENTION_BIAS_PLAN.md).
Compare that combined model against each individual extension and the same
control; improvement over the control alone would not establish synergy.

This document does not expand the [pair-memory plan](EVOLVING_PAIR_MEMORY_PLAN.md),
implement code, change existing campaigns, or authorize job submissions.
