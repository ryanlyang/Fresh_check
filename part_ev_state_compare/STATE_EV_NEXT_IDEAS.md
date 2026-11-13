# Three extensions of all-eight layerwise State-EV

Recorded: 2026-11-13. Research-plan index; attention-bias and relation-state
multiplication models/benchmarks implemented as separate campaigns.

## Separate idea documents

| Idea | Document | Status |
| --- | --- | --- |
| State-conditioned physics attention bias | [Attention-bias plan](STATE_CONDITIONED_ATTENTION_BIAS_PLAN.md) | [Independent model and SPORC benchmark](../part_ev_state_bias/README.md); GPU evidence pending |
| Explicit relation-state multiplication | [Multiplication plan](RELATION_STATE_MULTIPLICATION_PLAN.md) | [Independent model and SPORC benchmark](../part_ev_state_multiply/README.md); GPU evidence pending |
| Evolving pair memory across layers | [Pair-memory plan](EVOLVING_PAIR_MEMORY_PLAN.md) | Detailed gated-memory design, implementation safeguards, and benchmark plan |

The detailed attention-bias section previously in this file has moved to its
own document. No design decisions were changed by that move. Each extension is
an independent first experiment, not three simultaneous changes to one model.

## Starting point and scope

Build on **noncompact, standard-relations, all-eight layerwise State-EV**,
state width 32 and 2,327,462 parameters. Retain its existing same-math speedups;
do not substitute the compact or last-four model.

The motivating seed-101 two-A100/SyncBatchNorm run reached **86.3151% training-time
validation accuracy at check 49**. That is not an independent test result or a
promise of reproducible improvement. Historical ordinary ParT controls used a
different runtime, so those contrasts do not isolate architecture alone.

The plans aim to preserve the successful architecture while exploring specific
extensions. Potential gains are hypotheses, not predictions or established
optimal settings. Existing jobs, snapshots, environments, and data remain untouched.

## What the current model already does

Let `z_ij` be the shared 64-channel physics-pair embedding and `x_i^l`, `x_j^l`
the particle states entering block `l`.

- Ordinary Q/K attention already depends on the current particle states.
- Each layer's physics-bias head reads `z_ij`, not the current particle states.
- Ordinary EV adds a pair-derived value message.
- State-EV adds a nonlinear message using both endpoint states and `z_ij`.

The state-message features are schematically:

```text
g = E_l LN(z_ij)
p = P_l LN(x_i^l)
q = Q_l LN(x_j^l)
r_ij^l = GELU(LN(g + p + q + p * q))
```

Here `*` is elementwise multiplication. Each head projects the
attention-weighted sum of `r_ij^l` into an added value message. Later-layer
particle states already contain information gathered from other particles.

## Shared training and comparison contract

- Fresh initialization, initially seed 101; no warm start from trained reference
  or benchmark weights. Preserve existing RNG streams when adding parameters.
- Unchanged full `ranger_full_v1`: 1,000,000 update attempts, global batch 512,
  nominally 512M exposures from the official 100M training pool. Track AMP skips.
- Fifty validation checks at 20,000-update intervals, retaining the existing
  1.28M-event streaming-validation procedure. These are not full dataset epochs.
- Highest-accuracy checkpoint selection with the first tie retained. Report
  accuracy, AUC, and classwise rejection at that checkpoint rather than selecting
  a different checkpoint for each metric.
- Full-5M validation evaluation after training; it overlaps selection validation.
  Keep the independent 20M test population untouched during model development.
- Compare primarily with unchanged all-eight layerwise State-EV, auditing the
  recipe, data, runtime, environment, and initialization before reusing a control.
  Disclose the reference's segmented restart history. Obtain approval for any
  additional matched control training instead of adding it silently.
- Single-seed results are exploratory; confirm promising effects with matched
  seeds and eventual independent testing. A later combination of ideas 1 and 2
  is separate from their individual tests, not automatically part of the campaign.

## Shared execution intent

Submit from `sporcsubmit`, using two A100s on the same node, DDP/SyncBatchNorm,
global batch 512 (256 per GPU), and the reference data-worker/numerical settings.
Verify the live production partition and account limits: `tier3` is the historical
name, while current RIT documentation calls it `sporc`; do not silently reroute.

Benchmark each architecture first with requested walltime under six hours.
Measure sustained training, validation, maximum-length memory, checkpointing,
AMP health, and restart behavior. A small parameter increase does not establish
a small runtime increase.

Production intent is **one long allocation per model**, not a scheduled chain
of three-day training segments. Twelve days is a provisional allowance subject
to each model's benchmark and live scheduler limits, not a runtime guarantee.
Retain atomic latest/full-state checkpoints and safe selected candidates.
Do not weaken restart or completion checks to avoid another benchmark.

Each model's evaluation depends only on its own successful training, followed
by its own report. Other models must not block its results. Check persistent
storage before admission, never cache pair activations to disk, and do not
automatically delete old artifacts.

## Code entry points

- [Current layerwise State-EV assembly](model.py)
- [Optimized state-message implementation](../part_ev_state_speed/model.py)
- [Original state features, message integration, and EV-only routing](../part_ev_message_study/model.py)
- [Existing layerwise physics-bias heads](../part_layerwise_screen/model.py)
- [Two-GPU runtime of the reference](../part_ev_layerwise_ddp/README.md)

These documents record the designs. The first two independent implementations
are linked above; benchmark submission and production admission remain separate
steps. Pair memory remains a plan, not part of either implemented model.
