# Privileged Per-Particle View Distillation

Status: proposed implementation plan for a separate research branch. This
protocol does not alter or invalidate the running prediction-anchored bridge
campaign.

The canonical pilot name is `particle_view_500k_v1`.

## 1. Purpose

This project replaces microscopic offline residual targets with a small,
task-aware contextual view attached to each real HLT particle.

For every HLT particle, a frozen offline ParT supplies contextual offline
particle embeddings. The HLT particle acts as a query into that offline
particle set. A learned cross-attention adapter compresses the resulting
message into a one-, two-, four-, or eight-dimensional privileged particle
view. A privileged HLT consumer learns how to use those views. An HLT-only
particle-view predictor then reconstructs them and is fine-tuned through the
exact frozen consumer with tagger-useful logit distillation.

The deployable graph is:

```text
HLT particles
    -> HLT-only particle-view predictor
    -> predicted view attached to each HLT particle
    -> frozen selected HLT view consumer
    -> tag logits
```

The deployable graph never loads offline particles, the offline teacher, the
privileged view generator, true particle views, or cached teacher logits.

The plan is designed to answer four questions:

1. Can a small per-particle view derived from a strong offline classifier give
   an HLT consumer a meaningful privileged ceiling?
2. How much of that view can be inferred from HLT alone?
3. Does tagger-useful KD recover more privileged gain than representation
   regression alone?
4. Does privileged view supervision beat an equal-data, equal-capacity
   HLT-only network trained without privileged targets?

## 2. Central hypothesis

The existing physical residual fields ask the predictor to recover many
microscopic quantities whose exact values and particle correspondence may be
unavailable from HLT. The shuffle study suggests that the useful signal is
event-specific and signed, but is only partly dependent on exact particle
assignment and barely dependent on the identity of the current nested radius
blocks.

The particle-view hypothesis is different:

> The recoverable privileged signal is a low-dimensional, particle-indexed
> contextual message describing how each HLT particle's interpretation of the
> jet would change if offline information were available.

This target is still grounded in the particle representation used by ParT:

- every output belongs to a real HLT particle;
- permutation of HLT particles permutes the outputs;
- permutation of offline particles leaves each output unchanged;
- the consumer reasons with the fields through ordinary particle embeddings
  and attention;
- no global class logit is broadcast as a fake particle feature.

## 3. Locked terminology and model graph

The following names are used throughout the implementation.

### `A0_view`

The matched HLT-only ParT baseline. It trains on the exact canonical training
jet identities and labels used by every other trainable component.

### `A0_view_long_deploy`

The same architecture and training jets as `A0_view`, trained for the exact
number of CE-bearing optimizer updates applied to weights retained in one
selected deployable particle-view path. It is trained only after the selected
path's actual update ledger is frozen.

### `A0_view_total_label_budget`

The same architecture and training jets as `A0_view`, trained for the total
label-bearing optimizer-step budget of one selected scientific pipeline,
including the offline teacher and oracle-view discovery. It is deliberately
conservative and is reported separately from the deployable-path update
control. It is trained only after the selected pipeline's actual
label-exposure ledger is frozen.

### `Toff_view`

A strong offline-input ParT trained on the canonical training jets. It is
frozen before any privileged particle-view target is generated.

### `Gview`

The privileged particle-view generator. It receives HLT particle queries and
the frozen contextual offline particle embeddings from `Toff_view`. It
produces a small view vector for every HLT particle without hard particle
matching.

### `Cview_discovery`

A provisional consumer trained jointly with a candidate `Gview` in the raw,
bounded discovery coordinate system. It exists only to shape and measure the
candidate oracle view. It is never deployed and is not the registered clean
consumer.

### `Cview_probe`

A fixed-recipe candidate consumer reinitialized from `A0_view` after a
candidate generator is frozen and its provisional normalizer is fit. It sees
normalized coordinates from epoch zero and is used only for fair target
recoverability ranking.

### `Pview_probe`

A fixed-capacity HLT-only recovery probe trained for every frozen target
candidate. All candidates use the same probe architecture, optimizer, update
budget, and representation-only objective. Its predicted-view gain through a
candidate consumer is the primary target-selection quantity.

### `Cview_clean`

The final registered clean privileged consumer initialized from `A0_view`.
It is trained from epoch zero in the frozen, normalized coordinate system of
the selected `Gview`. It consumes true views and measures the maximum value of
the selected view definition.

### `Pview_0`

The HLT-only view predictor after representation warm-up and before
tagger-useful distillation.

### `Cview_robust`

A robust consumer initialized from `Cview_clean` and trained with a locked
mixture of true, `Pview_0`-predicted, perturbed, and missing views. It learns
how much to trust imperfect deployable views.

### `Pview`

The final HLT-only particle-view predictor after frozen-consumer KD.

### `Dview`

The deployable bundle consisting only of `Pview` and the selected frozen
consumer.

### Same-consumer recovery

Every recovery number compares the same frozen consumer:

```text
privileged logits = Cview(HLT, true_view)
predicted logits  = Cview(HLT, predicted_view)
zero-view logits  = Cview(HLT, zero_view)
```

No recovery loss may compare logits from two independently trained consumers.

## 4. Non-goals of the first pilot

The first pilot does not:

- claim that the learned view is a physical particle match;
- infer or persist an offline-to-HLT one-to-one correspondence;
- replace the final HLT-only deployment requirement;
- run a rho/alpha staircase;
- select a model using final-test results;
- interpret raw attention weights as unique physical explanations;
- require the learned view coordinates to have independent semantic names;
- combine the particle-view campaign with the running residual-field campaign;
- use more labeled training jets for `Pview` than for `A0_view`;
- stop the execution graph because a scientific quality metric is weak.

Scientific quality failures are reported as warnings. Only integrity,
provenance, numerical-finiteness, resource, or deployment-contract failures
stop jobs.

## 5. Single-training-pool data contract

The pilot deliberately uses one training pool rather than separate consumer
and distillation partitions.

### Canonical `particle_view_500k_v1` inventory

| Logical split | Count | Purpose |
|---|---:|---|
| `train` | 500,000 | all model fitting and all privileged-target fitting |
| `model_val_stop` | 75,000 | epoch selection and early stopping only |
| `model_val_select` | 75,000 | target, architecture, and loss ranking |
| `stack_val` | 150,000 | sealed confirmation and fusion fitting/evaluation |
| `final_test` | 150,000 | one-time HLT-only final evaluation |

The source-manifest adapter may alias an existing 500k `model_train` split to
logical `train` and deterministically partition an existing 150k `model_val`
into the two validation children above. The resulting identity and child
manifest hashes are immutable. It must not silently concatenate, truncate,
resample, or replace the training identities.

Every one of the following uses the exact same `train` identity hash:

- `A0_view`;
- `A0_view_long_deploy`;
- `A0_view_total_label_budget`;
- `Toff_view`;
- `Gview`;
- `Cview_discovery`;
- `Cview_probe`;
- `Pview_probe`;
- `Cview_clean`;
- `Pview_0`;
- `Cview_robust`;
- every final `Pview` objective;
- every equal-capacity direct control.

There is no `train_consumer`, `train_distill`, fold assignment, cross-fit
teacher, or post-cache refit. The exact frozen checkpoints used to create
targets are the checkpoints used live during their corresponding loss.

Using the same examples is intentional. The scientific comparison asks
whether privileged supervision extracts more deployable value from the same
labeled sample. Generalization is assessed on `model_val_stop`,
`model_val_select`, `stack_val`, and `final_test`.

### Validation and fusion use

`model_val_stop` is used only for:

- early stopping;
- checkpoint selection;
- fitting nonlabel calibration choices explicitly authorized by this plan.

`model_val_select` is used for:

- target-definition ranking;
- architecture and loss ranking;
- full development metrics;
- preliminary substitution and recovery comparisons.

No optimizer step is selected by `model_val_select`. Every candidate first
selects its epoch using only `model_val_stop`, reloads that checkpoint, and is
then evaluated once on `model_val_select`.

`stack_val` remains sealed until a configuration is selected on
`model_val_select`. After confirmation, a deterministic 75k/75k split may be
used only for fusion-fit and fusion-evaluation. That split does not alter the
single model training pool.

`final_test` is HLT-only for every deployable row. Offline inputs on
`final_test` are forbidden by the execution contract.

### Same-data overfitting audit

Because target generation and reconstruction use the same training jets, each
stage must report:

- training and `model_val_stop` view losses;
- training and `model_val_stop` frozen-consumer KD;
- train/validation cosine and relational agreement gaps;
- train/`model_val_select` deployable gains;
- effective parameter count and labeled CE update count.

No arbitrary "rewind two epochs" rule is used. Checkpoints are selected by the
locked validation rule. Large train/validation gaps trigger a quality warning
and capacity/regularization ablations, not an informal checkpoint choice.

## 6. Offline teacher `Toff_view`

`Toff_view` is a matched offline ParT trained on offline particles from
`train`. Its class order, preprocessing, masks, maximum particle count, and
label contract must match `A0_view`.

### Locked matched-ParT recipe

The baseline recipe is an implementation contract, not a placeholder to be
chosen during the campaign. `A0_view` and matched `Toff_view` use:

- the repository's 17-dimensional `PF_FEATURE_NAMES` preprocessing in its
  registered order:
  `part_pt_log, part_e_log, part_logptrel, part_logerel, part_deltaR,
  part_charge, part_isChargedHadron, part_isNeutralHadron, part_isPhoton,
  part_isElectron, part_isMuon, part_d0, part_d0err, part_dz, part_dzerr,
  part_deta, part_dphi`;
- at most 128 particles, the repository padding mask, and the same
  train-fitted preprocessing constants;
- `ParticleTransformer` with input width 17, pair-input width 4,
  `embed_dims=[128, 512, 128]`,
  `pair_embed_dims=[64, 64, 64]`, eight attention heads, eight particle
  blocks, two class-attention blocks, GELU, class-token pooling, final
  LayerNorm, an empty extra-FC list, one linear 10-class head, and no
  classifier dropout;
- the repository's four-vector-derived ParT pair features with
  `pair_input_dim=4`, `use_pre_activation_pair=False`,
  `block_params=None`, `trim=True`, and `for_inference=False`;
- AdamW with learning rate `3e-4`, weight decay `1e-4`,
  betas `(0.9, 0.999)`, gradient-norm clipping at `1.0`, AMP, and effective
  batch size 128;
- 2,000 linear warm-up updates followed by cosine decay to `3e-6`;
- at most 40 epochs and patience eight on `model_val_stop`;
- checkpoint selection by the deterministic rule below; and
- replica seeds 101, 202, and 303. Seed 101 supplies the canonical target
  discovery taps, while all confirmed comparisons use matched seeds.

Both models are initialized from scratch by the exact repository
`ParticleTransformer` constructor after setting Python, NumPy, CPU Torch, and
all CUDA RNGs to the replica seed. The initialization implementation and
library versions are registered; no pretrained or partially copied weights
are permitted in these baseline/teacher rows.

The HLT and offline instances are structurally identical. They differ only
in their registered particle source, source-specific train-fitted
preprocessing statistics, and resulting checkpoint. `Toff_view` must not be
made larger merely because offline inputs are available.

The registered larger-teacher ablation is also exact:
`embed_dims=[192, 768, 192]`, `pair_embed_dims=[96, 96, 96]`, eight heads,
12 particle blocks, three class-attention blocks, effective batch size 128
using batch 64 with two-step gradient accumulation, and otherwise the same
optimizer, schedule, validation, mask, pair-feature, and seed contract.
Existing checkpoints must reproduce one of these serialized recipes exactly
or be marked as a nonselectable provenance diagnostic. Every listed field,
including preprocessing constants and update counts, is serialized and
hashed; environment or command-line defaults may not silently override it.

The canonical teacher checkpoint is selected on `model_val_stop` by:

1. highest accuracy within an absolute tolerance of `1e-4`;
2. lowest cross-entropy within that pool;
3. lower ECE;
4. earliest epoch.

The selected checkpoint is frozen permanently. It is not refit after
selection.

### Contextual token tap

The canonical target source is the penultimate particle-transformer block
before class pooling. The exact canonical tensor is the particle-token tensor
returned after both attention and feed-forward residual additions of that
block, before the next transformer block and before pooling. Teacher dropout
is disabled. The target generator applies its own registered input
LayerNorm; it does not reuse or invent a teacher-side normalization layer.
The tapped tensor has shape:

```text
[batch, offline_particles, teacher_width]
```

The class token, pooled jet embedding, classifier logits, labels, and global
loss values are not included in the offline memory presented to `Gview`.

The following tap choices are ablations:

- raw offline particle embedding;
- middle transformer block;
- penultimate transformer block, canonical;
- final particle block before pooling;
- learned weighted mixture of the last three frozen blocks.

The offline teacher remains frozen for every canonical and selectable run.
Jointly updating it is an explicitly nonselectable diagnostic because it
destroys target stationarity.

### Offline-teacher source screen

Because the teacher disappears at inference, the target screen includes:

- the matched offline ParT, canonical;
- a larger offline ParT trained on the same `train` identities;
- the strongest existing provenance-compatible offline checkpoint;
- a learned mixture of contextual tokens from two frozen teachers.

Every teacher source must use the same class order and paired jet identities.
An existing checkpoint is selectable only when its registered training
identity hash equals canonical `train`; otherwise it is an explicitly
nonselectable stronger-teacher diagnostic.
Teacher choice is ranked by preliminary predicted-view gain, not offline
accuracy or clean oracle ceiling.

## 7. Matching-free privileged target generator `Gview`

For HLT jet \(X^H\), offline jet \(X^O\), and HLT particle \(i\):

\[
u_i^H
=
\operatorname{A0ViewTokens}_{\ell_q}(X^H)_i
\]

\[
q_i = Q(u_i^H)
\]

\[
h_j^O = \operatorname{ToffViewTokens}(X^O)_j
\]

\[
a_{ij}^{(m)}
=
\operatorname{softmax}_{j}
\left(
\frac{Q_m(q_i)K_m(h_j^O)^\top}{\sqrt{d_h}}
+
b_m(p_{ij})
\right)
\]

\[
c_i
=
\operatorname{Concat}_{m}
\sum_j a_{ij}^{(m)}V_m(h_j^O)
\]

\[
\tilde f_i=\tanh(B(c_i))
\]

\[
f_i^*
=
\tilde f_i
-
\operatorname{mean}_{k\in\mathrm{valid\ HLT}}\tilde f_k
\]

where \(p_{ij}\) contains physically grounded pair relations between HLT
anchor \(i\) and offline token \(j\):

- \(\Delta\eta\);
- wrapped \(\Delta\phi\);
- \(\log(\Delta R+\epsilon)\);
- relative log-\(p_T\);
- optional pair-mass or \(k_T\)-like features already supported by the local
  ParT utilities.

There is no nearest-neighbor assignment, Hungarian matching, transport plan,
or hard radius cutoff.

The canonical HLT query source is the penultimate frozen-`A0_view` particle
block. Its exact tap location matches the offline tap contract: after both
residual additions, before the next block and before pooling, with dropout
disabled. `Gview` applies its own registered input LayerNorm. `A0_view`
weights never receive gradients while targets are generated.

The query-source screen includes:

- raw HLT particle features;
- the frozen initial HLT particle embedding;
- the frozen middle `A0_view` particle block;
- the frozen penultimate `A0_view` particle block, canonical;
- a learned mixture of the final three frozen `A0_view` particle blocks.

The HLT and offline tap layer identities, tensor locations, dropout modes, and
input-normalization hashes are part of the view-coordinate binding.

### Canonical generator profile

- two cross-attention blocks;
- pre-layer normalization;
- width 160;
- eight attention heads;
- feed-forward expansion factor four;
- one learned null/offline-absence token;
- unrestricted attention with learned geometric pair bias;
- bounded `tanh` coordinates before centering;
- masked mean-centering across valid HLT particles;
- registered post-centering raw bounds `[-2, 2]`;
- four-dimensional bottleneck;
- whole-coordinate, per-event dropout probability `0.10` during
  discovery-consumer training;
- zero-mean Gaussian coordinate noise with raw standard deviation `0.05`
  during discovery-consumer training;
- zero padding on invalid HLT particles;
- deterministic evaluation with no dropout.

The null token gives an HLT query a valid destination when no offline
particle-specific structure is relevant. Null-attention fraction is reported.

Regularization is applied in this exact order: `tanh`, masked centering,
whole-coordinate per-event dropout with a mask of shape `[B,1,d]`, elementwise
Gaussian noise on valid entries, masked re-centering, then invalid-particle
zeroing. Dropping one coordinate therefore drops it for every particle in
that jet and cannot create a particle-index shortcut. Evaluation omits
dropout and noise. Since each pre-centered value lies in `[-1,1]`, the
registered deterministic post-centering range is `[-2,2]`.

### Why mean-center the view

Without a constraint, `Gview` could repeat a global offline class code on every
HLT particle. The canonical target subtracts the masked per-jet mean after the
bottleneck. This makes the exported view exactly zero-mean across valid
particles and forces the visible field to describe particle-relative context
rather than a replicated global logit.

Uncentered views remain an ablation. They are never silently treated as the
canonical particle-view definition. Centering is not claimed to remove every
possible class code: covariance, norms, and particle patterns can still carry
high bandwidth. The bounded coordinates and explicit rate/covariance budget
below are therefore mandatory.

### Permutation contract

Tests must verify:

- permuting HLT particles permutes `f*` in exactly the same way;
- permuting offline particles leaves `f*` unchanged within tolerance;
- padding never affects valid views;
- no target lookup depends on offline particle index;
- every target row is indexed by an HLT jet identity and HLT particle index.

## 8. Learning a useful privileged view

Candidate `Gview` and `Cview_discovery` models are trained jointly while
`Toff_view` and the canonical `A0_view` query tap remain frozen. The discovery
consumer is initialized from `A0_view`, and every new view-dependent path is
attached through a zero-initialized residual scale so epoch-zero logits equal
`A0_view` logits. View adapters and gates themselves use the nonzero,
gradient-reachable initialization specified in Section 9.

The oracle-stage objective is:

\[
L_\mathrm{oracle}
=
L_\mathrm{CE}(z_\mathrm{clean},y)
+
\lambda_\mathrm{offKD}
L_\mathrm{KD}(z_\mathrm{clean},z_\mathrm{off};T=2)
+
\lambda_\mathrm{floor}L_\mathrm{variance\ floor}
+
\lambda_\mathrm{rate}L_\mathrm{rate}
+
\lambda_\mathrm{cov}L_\mathrm{covariance}.
\]

Canonical weights:

- CE: `1.0`;
- offline-teacher KD: `0.5`;
- variance-floor regularization: `0.01`;
- rate regularization: `0.02`;
- covariance regularization: `0.01`.

For the raw centered view values over valid particles in a minibatch, let
\(\Sigma\) be the coordinate covariance and \(v_d=\Sigma_{dd}\). The locked
regularizers are:

\[
L_\mathrm{variance\ floor}
=
\frac{1}{d}\sum_d\max(0,0.02-v_d)
\]

\[
L_\mathrm{rate}
=
\max\left(0,\operatorname{tr}(\Sigma)-0.50d\right)^2
\]

\[
L_\mathrm{covariance}
=
\frac{1}{d(d-1)}
\sum_{r\ne s}\Sigma_{rs}^2.
\]

For `d=1`, `L_covariance` is defined to be exactly zero rather than dividing
by an empty off-diagonal count.

The variance floor only prevents total collapse. The rate term places an
upper information budget on particle-pattern variance. The covariance term
discourages redundant coordinates. No loss simply rewards particles for
being different.

The offline-teacher KD weight is screened at `0.0`, `0.25`, `0.5`, and `1.0`.
True-view evaluations additionally report deterministic 4-bit and 8-bit
per-coordinate quantization diagnostics using train-fitted symmetric ranges.
Quantized views are diagnostics unless explicitly registered as an ablation.

The canonical bottleneck is four float channels per valid HLT particle.
Bottleneck width is part of the view-definition hash.

### Two-pass coordinate and consumer contract

Discovery coordinates are provisional. No normalizer is added to a consumer
after it has been trained.

For every candidate target:

1. train `Gview + Cview_discovery` in the raw bounded coordinate system;
2. freeze the candidate `Gview`;
3. generate train statistics and fit a candidate normalizer;
4. reinitialize `Cview_probe` from the exact `A0_view` checkpoint;
5. train `Cview_probe` from epoch zero using only normalized and clipped true
   views;
6. train the fixed-capacity `Pview_probe` in that exact coordinate system;
7. select the probe checkpoint on `model_val_stop`;
8. evaluate true, predicted, and zero views through the exact same
   `Cview_probe` once on `model_val_select`.

All discovery consumers use AdamW, learning rate `3e-4`, weight decay
`1e-4`, batch size 128, gradient clipping `1.0`, and at most 30 epochs with
patience six on `model_val_stop`. Every `Cview_probe` uses the same optimizer
and a fixed 12-epoch training budget; its checkpoint is selected on
`model_val_stop`, but the budget is not shortened for apparently easy target
definitions.

After target selection, repeat the boundary deliberately:

1. freeze and register the selected `Gview`;
2. fit and register the final normalizer on `train`;
3. write the view-coordinate binding;
4. publish and audit the canonical float32 selected-view caches;
5. reinitialize the final `Cview_clean` from `A0_view`;
6. train it from epoch zero using only the canonical cached coordinates.

Final `Cview_clean` uses the same optimizer family, at most 40 epochs, and
patience eight on `model_val_stop`.

The registered clean consumer never sees raw discovery coordinates and never
has a coordinate transformation inserted after training.

### Fixed-capacity recovery probe

`Pview_probe` is a width-96, three-block ordinary HLT particle transformer
with the same input preprocessing for every target candidate. It trains for a
fixed eight epochs using Huber, cosine, and relational view losses only. It
does not receive labels, consumer gradients, CE, or KD. Its purpose is to
measure intrinsic HLT recoverability with a bounded, identical estimator.

Target candidates are ranked on `model_val_select` by:

1. highest preliminary predicted-view gain through `Cview_probe`;
2. recovery status and recovered fraction under the total-order rule in
   Section 18;
3. highest oracle gain;
4. lowest predicted-view cross-entropy;
5. smaller bottleneck width;
6. lexicographically smaller run ID; and
7. lexicographically smaller registered target ID.

Scientific metrics never determine whether the next jobs are submitted.
Every structurally selectable target remains selectable even when its oracle
gain is small or negative. Oracle gain below `0.0050` absolute accuracy emits
the prominent `WARN_ORACLE_GAIN_BELOW_005` record; nonpositive oracle gain
additionally emits `WARN_ORACLE_GAIN_NONPOSITIVE`. The deterministic ranking
still chooses a numerical winner and all downstream architecture, loss,
seed, and control jobs continue. These warnings constrain the eventual
scientific claim, not campaign execution.

### Recoverability-co-designed target

At least one first-pilot target uses recoverability co-design. Split the
generator into a rich context `Rview` and small projection `Bview`:

```text
frozen offline contextual memory
        -> frozen rich per-particle context Rview
        -> trainable 1/2/4/8-dimensional projection Bview
        -> target view

HLT-only fixed-capacity probe
        -> predicted target view
```

`Rview` is the 160-dimensional, layer-normalized output of the canonical two
cross-attention blocks, before the bottleneck. It is trained first with a
provisional four-dimensional head `Bseed` and `Cview_discovery` using the
ordinary discovery objective. Thus there is never an undefined phase in
which rich context is trained without a bottleneck or consumer. Its
checkpoint is selected on `model_val_stop` by highest oracle accuracy, then
lowest oracle CE, then earliest epoch, and is frozen permanently. `Bseed` and
that provisional consumer are discarded before co-design.

`Rview` itself is LayerNorm-normalized but is neither `tanh`-bounded nor
particle-mean-centered; it is an internal rich feature and is never cached as
the target coordinate or exposed to a consumer. Bounds, centering, train-fit
normalization, and coordinate hashing apply to `Bview(Rview)`.

The trainable co-design projection is:

```text
LayerNorm(160) -> Linear(160,160) -> GELU -> Linear(160,d)
-> tanh -> masked mean-center
```

It obeys the same `[-2,2]` post-centering bounds, per-event
whole-coordinate dropout, elementwise noise, re-centering, invalid-zeroing,
rate, covariance, and normalization contracts as every other candidate.
Before final normalization, co-design uses raw bounded coordinates.

Co-design initializes one `Cview_discovery` from `A0_view` and one
`Pview_probe` from its registered random initialization immediately before
cycle 1; `Bview` uses Xavier-uniform linear weights and zero biases under seed
101. Consumer adapters use the Section 9 initialization. All three persist
across cycles and are not reinitialized. Co-design then runs exactly 12
alternating cycles with no performance-based early termination:

1. run 2,000 probe optimizer steps, updating only the co-design
   `Pview_probe`;
2. run 500 projection/consumer optimizer steps, updating only `Bview` and a
   persistent `Cview_discovery`, while `Rview` and the probe are frozen.

This is exactly four probe steps per projection/consumer step. The probe uses
AdamW at `3e-4`, weight decay `1e-4`; `Bview` uses AdamW at `1e-4`, weight
decay `1e-5`; the discovery consumer uses AdamW at `3e-5`, weight decay
`1e-4`. All use effective batch size 128 and gradient clipping `1.0`.
Projection steps use the canonical oracle CE/KD, rate, covariance, and
variance-floor terms plus probe agreement
`0.5 L_Huber + 0.1 L_rel`. Dropout/noise is applied only to the consumer
input; the frozen probe-agreement target is the deterministic centered view.

All 12 cycle checkpoints are retained as compact metrics plus the best
checkpoint. The selected cycle has highest predicted-view accuracy through
its exact consumer on `model_val_stop`, then recovery status/fraction under
the Section 18 total order, highest oracle accuracy, lowest CE, and earliest
cycle. Selection does not shorten the cycles or stop other runs.

The rich offline context never moves during co-design. After co-design,
freeze the selected projection, fit its provisional normalizer, reinitialize
`Cview_probe` from `A0_view`, and train a fresh independent `Pview_probe`
from scratch with the ordinary candidate budgets. The co-design probe cannot
grade its own target.

The `Rview` registration, provisional-head registration, every cycle's
projection/consumer/probe hashes, optimizer-step ledger, selected-cycle
record, final projection registration, normalizer, fresh probe, and
coordinate binding are explicit artifacts. Only the final frozen projection
enters the coordinate binding.

### Oracle utility

\[
G_\mathrm{oracle}
=
\operatorname{Acc}
\left(Cview(\mathrm{HLT},f^*)\right)
-
\operatorname{Acc}
\left(Cview(\mathrm{HLT},0)\right).
\]

The report also includes the difference from independently trained
`A0_view`, but same-consumer zero-view gain is the primary view-ceiling
diagnostic.

## 9. Privileged consumer architecture

The canonical consumer starts from the selected `A0_view` checkpoint. It uses
the view through both a token path and a relational attention path.

`Cview_clean` receives only true privileged views, but its training applies
whole-coordinate per-event dropout `0.05` and elementwise zero-mean Gaussian
noise with standardized sigma `0.02`. The order is normalized clipping,
dropout with mask `[B,1,d]`, noise on valid entries, masked re-centering, and
invalid-zeroing. Evaluation uses the exact unperturbed true view. In this plan,
"clean" distinguishes true-view training from the `Pview_0`/error/missing
mixture used by `Cview_robust`; it does not mean absence of ordinary
regularization.

The unperturbed source before these training-only augmentations is always the
canonical float32 selected-view cache. `Cview_clean` never mixes live
generator outputs with cached coordinates.

### Token injection

\[
h_i \leftarrow h_i
+\alpha_\mathrm{token}g_iM_\mathrm{view}(f_i),
\qquad
g_i=\operatorname{sigmoid}(s_i).
\]

`M_view` is a two-layer adapter with normally initialized weights.
`s_i` is produced by a two-layer gate from the HLT embedding and view;
its final bias and weights are initialized to zero, so `g_i=0.5` at epoch
zero. The bounded scalar residual scale is:

\[
\alpha_\mathrm{token}
=\tanh(a_\mathrm{token}),
\qquad a_\mathrm{token}=0\ \text{at initialization}.
\]

The canonical injection point is after the first particle embedding/local
block, not directly in raw physical-feature space.

### Pair-bias injection

\[
b_{ij}^{view}
=
M_\mathrm{pair}
\left[
f_i,\,
f_j,\,
f_i-f_j,\,
f_i\odot f_j
\right].
\]

The resulting scalar per head is added to the ordinary ParT pairwise attention
bias after multiplication by
`alpha_pair * sqrt(g_i*g_j)`, where:

\[
\alpha_\mathrm{pair}
=\tanh(a_\mathrm{pair}),
\qquad a_\mathrm{pair}=0\ \text{at initialization}.
\]

Both effective scales are therefore bounded in `[-1,1]`. This lets the view
change both what a particle represents and how particles relate without
compensating for small trust gates through an unbounded residual scale.

Both adapter paths use ordinary Xavier initialization; only the two residual
scales are zero. Epoch-zero logits therefore equal `A0_view` exactly, the
residual scales receive gradients on the first optimizer step, and adapter,
gate, and generator gradients must be nonzero by the second optimizer step.
The test suite explicitly checks this two-step gradient reachability so a
zero-initialized multiplication cannot silently freeze the view path.

The canonical trust regularizer is:

\[
L_\mathrm{trust}
=
\operatorname{mean}_{i\in\mathrm{valid}}g_i.
\]

It weakly prefers the smallest trusted correction sufficient for the task.
Its weight is `0.01`; gate mean, 1st/50th/99th percentiles, and fractions
below `0.01` or above `0.99` are reported. Fixed-trust ablations set
`g_i=1` and `L_trust=0`.

Reports also include both raw and effective residual scales, the masked RMS
and 99th-percentile norms of
`alpha_token*g_i*M_view(f_i)`, the per-head RMS/99th-percentile effective
view pair bias, and their joint distribution against gate percentiles.

### Consumer controls

The campaign includes:

- raw-feature concatenation only;
- embedding residual only;
- pair-bias only;
- embedding residual plus pair bias, canonical;
- injection after block 0, block 1, and the midpoint;
- fixed trust of one;
- learned bounded trust, canonical;
- view dropout without a robust predicted-view mixture;
- a global pooled/offline-logit broadcast diagnostic, nonselectable.

## 10. Robust consumer construction

`Cview_clean` measures the clean oracle ceiling. Deployment distillation uses
both clean and robust consumers as paired diagnostics, with the robust
consumer canonical for breadth.

First train `Pview_0` using view-representation supervision. Freeze that exact
checkpoint. Then initialize `Cview_robust` from `Cview_clean` and train with
the following per-batch view-source mixture:

Every true view and `true_view - Pview_0(HLT)` residual in this stage comes
from the canonical float32 selected-view cache; live `Gview` regeneration is
forbidden.

| Source | Probability |
|---|---:|
| true `Gview` view | 0.30 |
| frozen `Pview_0` prediction | 0.45 |
| `Pview_0` prediction plus sampled validation-calibrated error | 0.15 |
| zero/missing view | 0.10 |

The error sampler is fit only on `train` prediction residuals and its scale is
calibrated conservatively against `model_val_stop`. It preserves complete
eventwise particle/dimension correlation rather than drawing independent
Gaussian coordinates.

The calibration procedure is:

1. store bounded residual samples
   `true_view - Pview_0(HLT)` from `train`;
2. measure per-dimension norms, covariance spectra, and 50th/90th/95th/99th
   eventwise residual-norm quantiles on `train` and `model_val_stop`;
3. choose one deterministic scalar inflation factor equal to the maximum
   held-out/train ratio among the 90th, 95th, and 99th quantiles, clipped to
   `[1.0, 2.0]`;
4. sample whole masked residual events from `train`, preserving
   particle-to-particle and dimension-to-dimension correlations;
5. multiply by the registered inflation factor and apply it to a current
   prediction;
6. include predictions from the final three `Pview_0` snapshots and
   Monte-Carlo predictor dropout in equal proportions.

The sampler records its residual source identities, snapshots, dropout seed,
quantiles, covariance diagnostics, inflation factor, coordinate binding, and
content hash. If `model_val_stop` tails are more than twice the train tails,
the factor saturates at two and emits a predictor-overfit warning.

The robust-consumer registration additionally binds the exact three snapshot
checkpoint hashes, Monte-Carlo dropout mode and probability, snapshot/dropout
mixing proportions, residual-event sampling algorithm and RNG seed, source
split identity, residual mask schema, inflation computation version,
mixture-source probabilities, `Pview_0` architecture/config hash, consumer
initialization hash, optimizer/schedule, and selected epoch. A robust
checkpoint whose sampler or mixture registration differs is incompatible
even if its tensor shapes match.

Robust training does not update `Pview_0`, `Gview`, `Toff_view`, or
`A0_view`. The selected `Cview_robust` checkpoint is frozen before final
predictor KD.

The clean consumer remains a required paired run for:

- the canonical full predictor;
- the simple particle-only predictor;
- representation-only training;
- the final selected deployable predictor.

## 11. Canonical HLT-only predictor architecture

The strongest canonical predictor is
`PVA3_hierarchical_particle_query_decoder`.

### 11.1 Input representation

The predictor consumes only cached HLT tokens, masks, HLT particle
four-vector-derived pair features, and allowed HLT preprocessing constants.
It cannot load:

- offline tokens;
- offline teacher embeddings;
- true views;
- oracle logits at inference;
- class labels at inference.

### 11.2 Local geometric particle encoder

- particle width 192;
- three pre-norm local geometric attention/message blocks;
- ParT-compatible pairwise geometric bias;
- residual MLP expansion factor four;
- dropout `0.05`;
- valid-mask preservation after every block.

The local stage builds neighborhood and prong-role evidence before global
context is compressed.

### 11.3 Hierarchical HLT region representation

Particles are softly pooled into a deterministic, sequential hierarchy:

```text
16 fine region slots -> 8 intermediate slots -> 4 coarse slots
```

The 8-token level is pooled from the 16 fine tokens, and the 4-token level is
pooled from the 8 intermediate tokens. The three levels are not independently
pooled from particles.

At the first particle-to-16 assignment there is not yet a physical slot
position, so the initial assignment is deliberately embedding-only. For
particle \(i\), fine slot \(r\), mask \(m_i\), particle embedding \(h_i\),
and learned slot seed \(e_r\):

\[
s^{(0)}_{ir}
=
M_\mathrm{assign}
\left(h_i,e_r\right)
\]

\[
a^{(0)}_{ir}
=
\operatorname{softmax}_{r}(s^{(0)}_{ir}).
\]

These initial assignments produce provisional fine-slot four-vectors and
centroids. One and only one geometric refinement pass then computes the
ordinary registered pair features \(p^{(0)}_{ir}\) from particle \(i\) to
that provisional centroid:

\[
s^{(1)}_{ir}=M_\mathrm{refine}(h_i,e_r,p^{(0)}_{ir}),
\qquad
a_{ir}=\operatorname{softmax}_r(s^{(1)}_{ir}).
\]

All remaining quantities use \(a_{ir}\). There are no invented initial slot
coordinates and no iterative convergence criterion.

Let \(w_i=p_{T,i}/\sum_{k\in valid}p_{T,k}\) be the normalized transverse
weight used only for embedding aggregation. Then:

\[
W_r
=
\sum_i m_i a_{ir}w_i
\]

\[
h_r
=
\frac{\sum_i m_i a_{ir}w_i h_i}
{\max(W_r,\epsilon)}.
\]

The numerical denominator uses `epsilon=1e-8`; individual `w_i` values are
not clamped. Physical four-vectors use the assignment exactly once and do
not receive this additional normalized-\(p_T\) weight:

\[
P_r^\mu=\sum_i m_i a_{ir}P_i^\mu.
\]

Thus a particle's physical momentum is neither normalized away nor counted
twice. Region \(p_T,\eta,\phi,m\) are derived from \(P_r^\mu\), with wrapped
angles computed from summed Cartesian momentum. These four-vectors and the
inherited occupancy enter the ordinary ParT-style region pair bias.

For the sequential 16-to-8 assignment, let
\(B_{rs}=\operatorname{softmax}_s M_{16\to8}(h_r,e_s,p_{rs})\).
For the 8-to-4 assignment, let
\(C_{st}=\operatorname{softmax}_t M_{8\to4}(h_s,e_t,p_{st})\).
The higher physical vectors and transverse masses are:

\[
P_s^\mu=\sum_r B_{rs}P_r^\mu,\quad
W_s=\sum_r B_{rs}W_r,\qquad
P_t^\mu=\sum_s C_{st}P_s^\mu,\quad
W_t=\sum_s C_{st}W_s.
\]

Higher-level embeddings use their inherited \(W\) values once in the same
normalized weighted-average equation. They never multiply \(P^\mu\) by
\(W\) again.

The effective occupancy is:

\[
n_r^\mathrm{eff}=\sum_i m_i a_{ir}.
\]

A region is considered empty when either
`n_eff < 1e-3` or inherited HLT transverse-weight mass is below `1e-5`.
Empty regions use a learned level-specific state, zero four-vector, and an
explicit empty mask in attention.

Slot collapse is monitored with assignment entropy, maximum slot mass, empty
rate, and:

\[
L_\mathrm{balance}
=
\frac{1}{R}\sum_r
\left(
\frac{W_r}{\sum_s W_s+\epsilon}
-
\frac{1}{R}
\right)^2.
\]

The canonical balance weight is `0.01`; it is a weak collapse guard, not a
requirement that physical jets divide uniformly. A no-balance ablation is
required. It is active with the same weight during representation warm-up,
frozen-consumer distillation, CE-only schedule controls, and optional joint
fine-tuning. It is evaluated at all three hierarchy transitions and averaged
equally. It is an architecture regularizer, not a scientific gate.

Assignments depend only on HLT embeddings and HLT geometry. The construction
is permutation-equivariant and masked, and no region can bypass the particle
mask.

The hierarchy relation used by global attention is soft, not a hard
parent/child label. For tokens at the same level it is the identity matrix;
fine-to-intermediate relation is `B`, intermediate-to-coarse is `C`, and
fine-to-coarse is matrix product `B @ C`. Reverse-direction features use the
corresponding transpose. The pair-bias MLP receives the directed relation
probability, its complement, source/destination levels, and the geometric
features. Empty rows are zeroed. This exact matrix construction is part of
the architecture hash.

### 11.4 Global region transformer

Four full-attention transformer blocks operate over the 28 multiscale region
tokens. Scale embeddings distinguish fine, intermediate, and coarse tokens.
Cross-scale attention is unrestricted. Its pair bias uses:

- wrapped centroid \(\Delta\eta,\Delta\phi,\Delta R\);
- relative region log-\(p_T\);
- region invariant-mass pair features;
- hierarchy levels and the soft directed hierarchy-relation probabilities;
- empty-region masks.

### 11.5 Particle-query decoder

Every original HLT particle retains a query identity. Two cross-attention
decoder blocks let each particle query the globally contextualized region
tokens:

\[
u_i =
\operatorname{Decoder}
\left(q_i^\mathrm{particle},H^\mathrm{regions}\right).
\]

This is the defining architectural component: global reasoning is returned as
a different contextual message for every real HLT particle.

Particle-to-region decoder attention includes a learned pair bias from the
particle and region four-vectors, wrapped angular displacement, relative
log-\(p_T\), hierarchy level, region occupancy, and empty mask. It does not
reuse the offline geometric bias or access offline coordinates.

### 11.6 Particle refinement and output

One full particle-attention refinement block operates on the concatenated
local and decoded representations. Independent heads predict:

- normalized view mean;
- bounded log variance;
- optional internal trust score.

The public predicted particle view has exactly the selected bottleneck width.
Uncertainty and trust are internal unless a registered ablation exposes them
to the consumer.

### 11.7 Resource controls

Every architecture records:

- total and trainable parameters;
- forward FLOPs at 128 valid particles;
- peak training and inference memory;
- batch-1 and campaign-batch latency as diagnostics;
- whether predictor and consumer weights are shared.

The canonical first pilot uses separate predictor and consumer networks to
maximize scientific clarity. Shared-backbone deployment is a later
compression experiment.

## 12. View normalization and immutable coordinates

Each frozen target candidate receives a provisional per-dimension normalizer
fit on `train` for its recovery-probe evaluation. After target selection, the
selected generator is reloaded, its train statistics are recomputed, and a
final immutable normalizer is registered before `Cview_clean` training:

- masked mean;
- masked standard deviation with floor `1e-4`;
- fixed standardized clipping threshold `[-6, 6]`.

The canonical representation uses standardized coordinates clipped to
`[-6, 6]` only after recording preclip statistics. Invalid particles are
exactly zero.

The normalizer hash, `Gview` checkpoint hash, offline-teacher checkpoint hash,
HLT-query checkpoint and tap hash, offline tap-layer identity, bottleneck
width, centering policy, bounded-coordinate policy, quantization ranges, and
pair-feature schema jointly identify the target coordinate system. The
binding also fixes canonical selected-view materialization to little-endian
float32, the exact normalization/clipping/invalid-zeroing order, and the
`1e-6` live-publication audit tolerance.

No run may use targets from one coordinate system with a consumer or predictor
registered against another.

## 13. Predictor losses

### Whitened view regression

For standardized residual \(e=\hat f-f^*\), the primary representation
Huber uses `beta=0.1`:

\[
L_\mathrm{Huber}
=
\operatorname{event\ mean}
\left[
\operatorname{valid\ coordinate\ mean}
\rho_{0.1}(e)
\right],
\]

where \(\rho\) is the piecewise Huber function defined under the uncertainty
loss. Padded particles are excluded. Each eligible event is averaged over
its valid particle-coordinate entries first, then eligible events receive
equal weight. An event with no valid particle-coordinate entry contributes
zero and is excluded from the event denominator; if the whole minibatch is
empty, `L_Huber=0`.

### Directional agreement

\[
L_\mathrm{cos}
=
1-\operatorname{cosine}(\hat f_i,f_i^*)
\]

for target rows with `||f_i*||_2 >= 1e-6`. Cosine denominators are clamped
to `1e-6`. Degenerate target rows are excluded rather than assigned a
direction; if a minibatch contains none, `L_cos=0`.

### Relational view loss

\[
\bar f_i = \frac{f_i}{\max(\lVert f_i\rVert_2,10^{-6})},
\qquad
S(F)_{ij}=\bar f_i^\top\bar f_j,
\]

\[
L_\mathrm{rel}^{(e)}
=
\frac{1}{N_e(N_e-1)}
\sum_{\substack{i\ne j\\i,j\in valid(e)}}
\left|S(\hat F)_{ij}-S(F^*)_{ij}\right|.
\]

The diagonal and all padded pairs are excluded. The loss is first averaged
within each event, then averaged over eligible events so jets with many
particles do not dominate. Events with fewer than two valid particles
contribute exactly zero and are excluded from the eligible-event
denominator. If no event is eligible, `L_rel=0`. This preserves which HLT
particles have similar contextual views even when individual coordinates are
imperfect.

### Uncertainty loss

For standardized residual \(e=\hat f-f^*\), define Huber penalty with
`beta=0.1`:

\[
\rho_\beta(e)=
\begin{cases}
\frac{e^2}{2\beta},&|e|\le\beta\\
|e|-\frac{\beta}{2},&|e|>\beta.
\end{cases}
\]

The uncertainty head predicts log variance
\(\ell=\operatorname{clip}(\ell_\mathrm{raw},-6,3)\), and:

\[
L_\mathrm{unc}
=
\operatorname{masked\ mean}
\left[
\frac{1}{2}\exp(-\ell)\rho_{0.1}(e)+\frac{1}{2}\ell
\right].
\]

The mask covers valid particles and coordinates. This optional
heteroscedastic Huber likelihood is calibrated on `model_val_stop`; the
selected calibration is evaluated once on `model_val_select`. Reports
include interval coverage, NLL, error-versus-variance rank correlation, and
fractions at both log-variance bounds.

### Frozen-consumer KD

\[
L_\mathrm{KD}
=
T^2
\operatorname{KL}
\left(
\operatorname{softmax}(z^*/T)
\;\|\;
\operatorname{softmax}(\hat z/T)
\right)
\]

where:

```text
z*   = frozen_Cview(HLT, true_view)
zhat = frozen_Cview(HLT, predicted_view)
```

`true_view` is always read from the canonical float32 selected-view cache,
and cached `z*` logits bind that cache plus the exact consumer checkpoint.
The canonical temperature is `2.0`.

### Event cross-entropy

\[
L_\mathrm{CE}
=
\operatorname{CE}(\hat z,y).
\]

CE is optional and always reported separately because it can turn the view
channel into a task-latent control signal rather than a reconstruction.

### Canonical phase-two objective

\[
L_\mathrm{primary}
=
1.0L_\mathrm{KD}
+
0.35L_\mathrm{Huber}
+
0.10L_\mathrm{cos}
+
0.10L_\mathrm{rel}
+
0.15L_\mathrm{CE}
+
0.01L_\mathrm{trust}.
\]

The representation anchor remains nonzero throughout distillation.

## 14. Training schedule

### Stage A: matched baselines

Train and freeze:

- `A0_view`;
- `Toff_view`;
- the predeclared-canonical-architecture parameter-matched direct HLT
  control; and
- the predeclared-canonical-architecture forward-FLOP-matched direct HLT
  control.

Stage A does not train `A0_view_long_deploy`,
`A0_view_total_label_budget`, or eventual-selected-bundle direct controls,
because their authentic budgets and target capacity do not exist yet.

### Stage B: oracle view discovery

For every target-definition ablation:

1. initialize the consumer from `A0_view`;
2. normally initialize view adapters/gates and zero only their residual
   scales;
3. train `Gview` and `Cview_discovery`;
4. select their epoch on `model_val_stop`;
5. freeze `Gview` and fit its candidate normalizer on `train`;
6. reinitialize and train `Cview_probe` in normalized coordinates;
7. train the fixed eight-epoch `Pview_probe`;
8. select all epochs on `model_val_stop`;
9. report oracle, predicted, and recovered gains once on
   `model_val_select`;
10. rank target candidates primarily by predicted-view gain;
11. persist only the selected checkpoints and lightweight metrics.

After selection, register the target coordinate binding, publish the
canonical float32 selected-view caches, audit them, reinitialize
`Cview_clean` from `A0_view`, and train the final clean consumer only from
those cached values. The canonical target definition is frozen before the
broad predictor sweep.

### Stage C: predictor representation warm-up

Train `Pview_0` for a fixed four complete epochs using:

```text
1.0 Huber + 0.25 cosine + 0.15 relational + 0.05 uncertainty
```

Warm-up never stops early. The fixed budget makes every objective enter the
frozen-consumer stage from an equivalent state.

### Stage D: robust consumer

Build and freeze `Cview_robust` from the selected `Pview_0` using the locked
view-source mixture in Section 10.

### Stage E: predictor distillation

Train for at most 40 epochs with:

- AdamW;
- learning rate `3e-4`;
- weight decay `1e-4`;
- gradient norm clipping at `1.0`;
- cosine decay after a short linear warm-up;
- mixed precision when numerically valid;
- early-stop patience eight on `model_val_stop` deployable accuracy.

Checkpoint selection is:

1. highest `model_val_stop` deployable accuracy within `1e-4`;
2. lowest cross-entropy within that pool;
3. recovery status and fraction under the Section 18 total-order rule; and
4. earliest epoch.

No KD agreement or view MSE is allowed to select a less accurate checkpoint
outside the accuracy tolerance. The selected checkpoint is then evaluated
once on `model_val_select` for configuration ranking.

### Stage F: optional HLT-only joint fine-tuning

The frozen-consumer result remains the primary causal experiment. A separate
selectable performance branch copies the selected predictor and consumer and
jointly fine-tunes only those deployable HLT-only weights:

```text
frozen oracle target network:
    selected_frozen_Cview(HLT, true_view)

live deployable network:
    trainable_consumer_copy(HLT, trainable_Pview(HLT))
```

The joint objective uses oracle-logit KD `1.0`, view anchoring `0.25`, CE
`0.10`, and trust regularization `0.01`. It uses learning rate `3e-5`, at most
eight epochs, and patience three on `model_val_stop`. The frozen oracle target
network never changes.

This branch is reported as `Dview_joint`, not as same-consumer recovery,
because the live consumer has changed. It competes with the frozen bundle on
`model_val_select` and must be compared against an identical predictor-plus-
consumer architecture trained with the same freeze/unfreeze schedule using CE
only.

### Stage G: selected-path fairness closure

After `model_val_select` has fixed the privileged-scientific winner and the
best pre-Stage-G eligible HLT-only deployable model, write an immutable
`selected_path_fairness_ledger.json` with one entry per distinct winner
configuration and per-replica subentries for seeds 101, 202, and 303. Each
subentry resolves:

- clean versus robust consumer;
- frozen versus jointly tuned deployment;
- actual completed CE-bearing and label-bearing optimizer updates in every
  retained stage;
- selected predictor and consumer parameter counts;
- measured selected-bundle forward FLOPs at 128 valid particles; and
- the selected architecture/configuration hashes.

Only after that ledger exists, train the matched-seed A0-long controls for
each winner-replica subentry and the three-seed direct controls for each
distinct winner configuration:

- `A0_view_long_deploy` to the exact retained-deployable-path CE-update
  budget;
- `A0_view_total_label_budget` to the exact selected-pipeline
  label-bearing-update budget;
- a direct HLT control matched to the eventual selected bundle's deployed
  parameter count; and
- a direct HLT control matched to the eventual selected bundle's measured
  forward FLOPs.

Each A0-long trajectory is uninterrupted to the exact required optimizer
update. It registers both the checkpoint at exactly that update and the best
`model_val_stop` checkpoint observed at or before that update, using the
ordinary A0 checkpoint rule. The conservative scientific comparison uses
the better validation-selected checkpoint; the exact-update checkpoint is
reported alongside it so overtraining cannot make the control artificially
weak.

The deterministic direct-control registry searches its predeclared
HLT-only depth/width grid and minimizes relative mismatch, first for the
requested quantity and then for the other quantity, with smaller models and
lexicographic config ID as ties. For quantity \(q\):

\[
\operatorname{relative\ error}(q)
=
\frac{|q_\mathrm{control}-q_\mathrm{target}|}
{\max(q_\mathrm{target},1)}.
\]

The unrounded integer parameter/FLOP totals are converted to float64 for this
calculation. A result is labeled parameter-matched only when relative
parameter error is at most `0.05`, and FLOP-matched only when relative
forward-FLOP error is at most `0.10`, inclusively.

Deployed parameter count is the number of unique learned scalar parameter
elements in the exported inference graph, including frozen parameters and
deduplicating shared storage; nonlearned buffers and preprocessing constants
are excluded.

The locked FLOP protocol uses:

- the exact exported predictor-plus-consumer inference graph;
- batch size one, exactly 128 valid particles, an all-valid mask, and the
  registered 17-feature/four-vector input schema;
- the persisted `flop_fixture_v1` input (seed `44_017`) with content hash,
  finite deterministic features/momenta, and
  `energy=sqrt(px^2+py^2+pz^2+1)` for every particle;
- evaluation mode, float32 execution, dropout disabled, and no AMP or
  compiler fusion;
- the common boundary from preprocessed HLT tensors through final class
  logits, including in-graph pair features, hierarchy, view prediction,
  normalization, consumer attention, and classification;
- exclusion only of disk I/O, cache decoding, raw-source preprocessing, and
  argmax/reporting;
- one multiply plus one add counted as two FLOPs, with reductions and
  elementary arithmetic/transcendentals counted by the versioned
  operator-level counter; and
- one registered FLOP-counter implementation/version hash for every candidate
  and target, with unsupported operators rejected during preflight.

The counter is the local semantic counter `particle_view_flops_v1`, applied
to the unfused exported graph. Matrix multiplication with shapes
`[m,k] x [k,n]` costs `2*m*k*n`; a following bias costs `m*n`.
Elementwise add/subtract/multiply/divide, comparison, `exp`, `log`, `sqrt`,
`erf`, `tanh`, and sigmoid each cost one FLOP per output element. A reduction
over length `n` costs `n-1` additions, and softmax over length `n` costs
`n` exponentials, `n-1` additions, and `n` divisions. LayerNorm, GELU,
attention, pair-feature construction, and normalization are decomposed into
those same semantic operations. Indexing, masking assignment, reshape,
transpose, concatenation, and memory movement cost zero. The counter never
uses wall-clock kernels or hardware-specific fused-operation counts.

Exact integer totals and per-operator breakdowns are stored before relative
error is computed. If no registry candidate meets a tolerance, the nearest
control still trains and the campaign continues, but it receives
`WARN_CONTROL_MATCH_TOLERANCE` and cannot support the corresponding fairness
claim.

If the privileged and pre-Stage-G deployable winners are identical, these
jobs are deduplicated by fairness-entry hash. Otherwise each winner receives
its own matched-seed A0-long and selected-bundle direct controls. Both the
Stage-A canonical-architecture matches and Stage-G
eventual-selected-bundle matches
are retained and reported. Stage-G controls finish before sealed `stack_val`
reporting is finalized; no stack metric is used to construct, tune, or select
them.

After their checkpoints have been selected exclusively on `model_val_stop`,
all three seeds of the Stage-G A0-long controls, selected-bundle direct
controls, and the privileged winner's exact schedule/architecture-matched
CE-only comparator are explicitly authorized for one `stack_val` evaluation.
The Stage-G controls never receive a selected-view cache. None of these
comparator authorizations grants `final_test` access; a CE-only model receives
final-test access only if it was independently selected as the pre-Stage-G
deployable winner. The primary Stage-G comparison uses:

- the `best_model_val_stop_within_budget` checkpoint for each A0-long
  replica;
- the ordinary `model_val_stop`-selected checkpoint for each direct-control
  and CE-only replica; and
- three-seed mean `stack_val` accuracy, with per-seed values and sample
  standard deviation also reported.

The privileged winner "beats" a Stage-G control only when its three-seed mean
`stack_val` accuracy is strictly greater. Exact-update A0 checkpoints are
secondary rows and cannot replace the conservative best-within-budget
comparison. These extra stack authorizations do not admit Stage-G controls to
winner selection, fusion fitting, or final-test evaluation.

### Exact fairness and label-exposure ledger

Every run records optimizer steps, labeled examples processed, CE-bearing
steps, teacher-KD steps, view-supervision steps, and measured training FLOPs
by stage.

For each distinct winner family, `A0_view_long_deploy` matches the sum of
CE-bearing optimizer steps applied to weights that remain in that selected
deployable path:

- baseline initialization/fine-tuning counted once;
- clean or robust consumer CE steps;
- predictor-distillation CE steps;
- optional joint-fine-tuning CE steps.

Its paired `A0_view_total_label_budget` additionally matches label-bearing
steps used to train:

- `Toff_view`;
- `Cview_discovery`;
- the final clean and robust consumers;
- every selected-path CE-bearing predictor stage.

Its budget is computed from that winner's logical pipeline, not the entire
ablation search. Each A0-long control uses the same train identities and class
sampling as its compared path. Registrations include the winner-family name
and fairness-entry hash.

The identical CE-only predictor-plus-consumer control is the primary fairness
control: it uses the same component graph, initialization, stage boundaries,
freeze/unfreeze schedule, optimizer-step counts, and hyperparameter-trial
budget, with privileged targets/KD removed. Parameter- and FLOP-matched direct
HLT controls receive the same registered eight-trial optimization grid as the
selectable canonical privileged family:

```text
learning_rate in {1e-4, 3e-4}
weight_decay in {1e-5, 1e-4}
dropout in {0.05, 0.10}
```

Architecture, target, and loss ablations are scientific rows rather than
hidden hyperparameter trials and are itemized separately. The campaign report
includes total search GPU-hours separately from selected-pipeline training
FLOPs.

## 15. Pilot ablation campaign

The campaign intentionally favors broad evidence. Every declared single-seed
screen row and every predeclared three-seed confirmation row is submitted
regardless of earlier metrics. Screens control the preregistered amount of
replication and persistent storage; they are not quality gates. Numerical
ranking may add the declared best-row confirmation replicas, even when the
best row is scientifically weak.

### 15.1 Target-source and target-generator screen

Single-seed full-validation rows:

| ID | Change | Selection status |
|---|---|---|
| `VGEN_TAP_RAW` | raw offline particle embedding | selectable |
| `VGEN_TAP_MID` | middle offline transformer block | selectable |
| `VGEN_TAP_PENULT` | penultimate block, canonical | canonical/selectable |
| `VGEN_TAP_FINAL` | final particle block | selectable |
| `VGEN_TAP_MIX3` | learned mixture of final three blocks | selectable |
| `VGEN_QUERY_RAW` | raw HLT query features | selectable |
| `VGEN_QUERY_EMBED` | frozen initial A0 embedding | selectable |
| `VGEN_QUERY_MID` | frozen middle A0 block | selectable |
| `VGEN_QUERY_PENULT` | frozen penultimate A0 block, canonical | canonical/selectable |
| `VGEN_QUERY_MIX3` | learned mixture of final three frozen A0 blocks | selectable |
| `VGEN_XATTN1` | one cross-attention block | selectable |
| `VGEN_XATTN2` | two blocks, canonical | canonical/selectable |
| `VGEN_XATTN4` | four blocks | selectable |
| `VGEN_NO_PAIR` | no geometric pair bias | selectable |
| `VGEN_PAIR` | learned pair bias, canonical | canonical/selectable |
| `VGEN_LOCAL02` | hard local radius control | diagnostic/nonselectable |
| `VGEN_LOCAL04` | wider local radius control | diagnostic/nonselectable |
| `VGEN_NO_NULL` | remove null token | selectable |
| `VGEN_NULL` | null token, canonical | canonical/selectable |
| `VGEN_CENTERED` | masked mean-centered, canonical | canonical/selectable |
| `VGEN_UNCENTERED` | uncentered view | diagnostic/nonselectable |
| `VGEN_DIM1` | one-dimensional bottleneck | selectable |
| `VGEN_DIM2` | two-dimensional bottleneck | selectable |
| `VGEN_DIM4` | four-dimensional bottleneck, canonical | canonical/selectable |
| `VGEN_DIM8` | eight-dimensional bottleneck | selectable |
| `VGEN_KD000` | oracle offline-KD weight `0.0` | selectable |
| `VGEN_KD025` | oracle offline-KD weight `0.25` | selectable |
| `VGEN_KD050` | oracle offline-KD weight `0.5`, canonical | canonical/selectable |
| `VGEN_KD100` | oracle offline-KD weight `1.0` | selectable |
| `VGEN_NO_RATE` | remove rate/covariance budget | diagnostic/nonselectable |
| `VGEN_RECODESIGN` | frozen-rich-context recoverability co-design | selectable |
| `VGEN_TEACHER_LARGE` | larger matched offline teacher | selectable |
| `VGEN_TEACHER_EXISTING` | best existing offline teacher | selectable only with exact train/recipe provenance; otherwise diagnostic |
| `VGEN_TEACHER_MIX2` | two-teacher contextual-token mixture | selectable only when both teachers are provenance-compatible |
| `VGEN_MEMORY_HLT` | replace offline memory with frozen `A0_view` HLT contextual tokens | mandatory pre-Stage-G performance control; deployment eligible, privileged-claim ineligible |
| `VGEN_MEMORY_HLT_SELFMASK` | HLT memory with the same-particle key/value masked | single-seed diagnostic/nonselectable |

Every row runs the complete two-pass candidate normalizer, `Cview_probe`, and
fixed-capacity `Pview_probe` procedure. The top two target definitions by
preliminary predicted-view gain, using the deterministic target-selection
hierarchy in Section 8, proceed to the main predictor screen. The canonical
four-dimensional centered penultimate-tap target proceeds regardless, so
selection cannot erase the predeclared hypothesis. Exactly two ranked targets
are forwarded even if every gain is negative; there is no minimum-performance
gate.

`VGEN_MEMORY_HLT` uses the same generator width, cross-attention blocks,
bottleneck, centering, rate budget, consumer, probe, optimizer, and update
budgets as the canonical row. Its queries and memory come from the exact
frozen `A0_view` checkpoint, and its pair features are HLT-to-HLT geometry.
It tests whether the apparent benefit comes from the learned per-particle
generator interface rather than offline privilege. It always runs seeds 101,
202, and 303 and is carried through the matching architecture and loss
confirmation, regardless of its seed-101 result. Competitiveness within
`0.0010` of the selected offline-memory target is reported as a prominent
control warning, not used to decide whether its remaining replicas run.

`VGEN_MEMORY_HLT_SELFMASK` uses the same seed-101 recipe but masks the
query's exact same-particle HLT memory entry before softmax, while retaining
the learned null token. It distinguishes generic HLT contextual-memory value
from a trivial self-copy path. It runs the complete candidate
consumer/recovery-probe procedure, but remains diagnostic and does not enter
either winner family.

### 15.2 Consumer interface screen

| ID | Interface | Selection status |
|---|---|---|
| `C_RAWCAT` | concatenate at raw input | selectable |
| `C_EMBED` | embedding residual only | selectable |
| `C_PAIR` | view-derived pair bias only | selectable |
| `C_EMBED_PAIR` | embedding residual plus pair bias, canonical | canonical/selectable |
| `C_INJECT0` | injection immediately after embedding | selectable |
| `C_INJECT1` | injection after first block, canonical | canonical/selectable |
| `C_INJECTMID` | midpoint injection | selectable |
| `C_FIXED_TRUST` | no learned trust | selectable |
| `C_GATED_TRUST` | bounded learned trust, canonical | canonical/selectable |
| `C_CLEAN` | clean true-view training | selectable |
| `C_DROPOUT` | clean plus view dropout | selectable |
| `C_ROBUST_MIX` | predicted/error/missing mixture, canonical | canonical/selectable |

### 15.3 Predictor architecture screen

| ID | Architecture | Selection status |
|---|---|---|
| `P_C0_PARTICLE` | particle-only residual MLP/control | selectable |
| `P_PART_BASIC` | ordinary particle transformer | selectable |
| `P_LOCAL` | local geometric blocks only | selectable |
| `P_LOCAL_GLOBAL` | local plus full particle global attention | selectable |
| `P_HIER_NO_DECODER` | hierarchical regions with pooled broadcast | selectable |
| `P_HIER_DECODER` | hierarchy plus particle-query decoder | selectable |
| `P_HIER_DECODER_REFINE` | full canonical model | canonical/selectable |
| `P_NO_PAIR_BIAS` | canonical without pair geometry | selectable |
| `P_NO_REFINEMENT` | canonical without final particle refinement | selectable |
| `P_REGIONS_8_4` | smaller 8/4 hierarchy | selectable |
| `P_REGIONS_16_8_4` | canonical hierarchy | canonical/selectable |
| `P_REGIONS_16_8_4_2` | extra two-token global scale | selectable |
| `P_NO_BALANCE` | canonical hierarchy without collapse guard | selectable |
| `P_WIDTH128` | width 128 | selectable |
| `P_WIDTH192` | width 192, canonical | canonical/selectable |
| `P_WIDTH256` | width 256 | selectable |
| `P_SHARED_CONSUMER_STEM` | shared frozen HLT stem | diagnostic/nonselectable |

### 15.4 Loss screen

| ID | Objective | Selection status |
|---|---|---|
| `L_VIEW` | view regression only | privileged/pre-Stage-G eligible |
| `L_VIEW_COS` | view plus cosine | privileged/pre-Stage-G eligible |
| `L_VIEW_REL` | view plus relational | privileged/pre-Stage-G eligible |
| `L_VIEW_ALL` | full representation warm-up objective | privileged/pre-Stage-G eligible |
| `L_KD` | KD only | privileged/pre-Stage-G eligible |
| `L_KD_VIEW` | KD plus view anchor | privileged/pre-Stage-G eligible |
| `L_KD_VIEW_REL` | KD plus view and relational anchors | privileged/pre-Stage-G eligible |
| `L_KD_CE` | KD plus CE | privileged/pre-Stage-G eligible |
| `L_CE` | CE only | pre-Stage-G performance-control eligible; privileged-claim ineligible |
| `L_CE_VIEW` | CE plus privileged view | privileged/pre-Stage-G eligible |
| `L_PRIMARY` | canonical KD/view/relational/small-CE objective | canonical privileged/pre-Stage-G eligible |
| `L_PRIMARY_NO_CE` | canonical without CE | privileged/pre-Stage-G eligible |
| `L_PRIMARY_CE05` | CE weight `0.05` | privileged/pre-Stage-G eligible |
| `L_PRIMARY_CE15` | CE weight `0.15`, canonical | canonical privileged/pre-Stage-G eligible |
| `L_PRIMARY_CE35` | CE weight `0.35` | privileged/pre-Stage-G eligible |
| `L_PRIMARY_NO_TRUST` | canonical with `trust_weight=0` | privileged/pre-Stage-G eligible |
| `L_UNCERTAINTY` | canonical plus uncertainty calibration | privileged/pre-Stage-G eligible |

`L_PRIMARY_NO_TRUST` leaves the frozen consumer's learned gates and bounded
residual scales intact but removes `L_trust` from predictor optimization. It
tests whether encouraging small trust during distillation opposes KD by
rewarding predicted views the consumer ignores.

### 15.5 Privileged-supervision and capacity controls

These controls are mandatory:

- identical predictor and consumer trained end-to-end with CE only;
- identical predictor architecture trained with HLT self-distillation but no
  offline target;
- Stage-A parameter/FLOP controls matched to the predeclared canonical
  bundle;
- Stage-G parameter/FLOP controls matched to the eventual selected bundle,
  using the `5%`/`10%` tolerances;
- deeper direct HLT ParT matching total predictor-plus-consumer depth;
- post-selection `A0_view_long_deploy`, with exact-update and
  best-validation-within-budget checkpoints;
- post-selection `A0_view_total_label_budget`, with exact-update and
  best-within-budget checkpoints;
- predictor initialized randomly versus HLT-A0 warm-started;
- frozen random view generator;
- offline global logits broadcast to each particle, nonselectable;
- raw offline features cross-attended without the offline classifier;
- offline classifier KD directly into an HLT ParT without particle views;
- jointly fine-tuned `Dview_joint`;
- identical joint predictor-plus-consumer schedule with CE only.

These rows determine whether any gain comes from privileged particle views,
extra labeled optimization, additional capacity, ordinary KD, or generic
model diversity.

### 15.6 Negative and structural controls

Evaluate the exact frozen consumer with:

- zero view;
- same-norm random view;
- sign-reversed view;
- event-shuffled view with no fixed points;
- particle-shuffled view within each jet;
- view-dimension permutation;
- independently shuffled dimensions;
- per-jet mean view broadcast to every particle;
- per-particle view with the jet mean removed, canonical;
- frozen uncontextualized HLT-query embedding;
- true view on only the highest-\(p_T\) 25%, 50%, and 75% of particles;
- predicted view on only those same subsets;
- masked null-token-only view.

Permutation identity rates, padding preservation, class balance, and
per-class gains are reported.

### 15.7 Focused interactions

The following interactions run even if their individual row is not the
single-seed leader:

- bottleneck width x representation/KD loss;
- clean versus robust consumer x canonical predictor;
- embedding-only versus embedding-plus-pair-bias x canonical predictor;
- penultimate versus mixed-three tap x canonical predictor;
- plain ParT versus full hierarchical decoder x primary/no-CE/CE-only;
- centered versus uncentered target x dimension 2/4/8;
- uncertainty off/on x clean/robust consumer;
- standard versus recoverability-co-designed target x dimension 2/4/8;
- frozen-consumer versus joint HLT-only fine-tuning x primary/CE-only.

### 15.8 Seed policy

Broad screens use seed 101. Then:

- the canonical predeclared configuration;
- the best architecture configuration;
- the best no-CE configuration;
- the best small-CE configuration;
- the CE-only upper-bound diagnostic;
- representation-only;
- direct capacity/FLOP controls;
- the best alternative target definition;
- the recoverability-co-designed target;
- the mandatory HLT-memory generator control;
- `Dview_joint` and its schedule-matched CE-only control

run seeds 101, 202, and 303.

Configuration and winner-family selection uses the complete Section 18
ordering. Within each selected configuration, the retained representative is
the median-accuracy seed, with lower cross-entropy and then lower seed ID as
deterministic ties.

All three replicas of each distinct selected winner configuration are then
evaluated on sealed `stack_val`, together with the exact matched CE-only and
Stage-G fairness comparators authorized in Section 14. The report shows every
replica, the three-seed mean and median, and paired gains against same-seed A0
replicas. `stack_val` cannot change a selected configuration or
representative seed. Only each preselected winner-family median
representative is authorized for the one-time HLT-only `final_test`; the
other replicas and comparator-only rows never access it.

## 16. Campaign size and execution profile

The registry generator must enumerate every row and record whether it is:

- selectable;
- diagnostic;
- single-seed screen;
- three-seed confirmation;
- clean-consumer paired;
- robust-consumer paired;
- final-test eligible.

The expected first campaign is approximately:

- 36 target-generator/query/teacher/control rows;
- 12 consumer rows;
- 17 predictor architecture rows;
- 17 loss rows;
- at least 16 privileged/capacity/fairness controls, with selected-path
  controls deduplicated only when winner hashes match;
- 13 negative/structural controls;
- at least 14 focused interaction rows after Cartesian expansion;
- up to 39 three-seed confirmation replicas, counting the 13 declared
  configuration roles across three seeds before identity deduplication.

Shared artifacts reduce the actual number of expensive full trainings:
negative controls are evaluations, compatible loss rows share warm-starts, and
target screens reuse the frozen offline token stream.

The generated registry, not this approximate inventory, is authoritative.
The preflight must reconcile declared, generated, selectable, diagnostic, and
seed-expanded counts before submission.

## 17. Metrics

### 17.1 Classification

Report on allowed splits:

- accuracy;
- cross-entropy;
- macro per-class accuracy;
- one-vs-rest AUC;
- background rejection at locked signal efficiencies;
- confusion matrix;
- ECE and Brier score;
- per-class efficiency;
- paired event-level win/loss/tie counts;
- per-seed paired accuracy-gain distributions;
- paired event bootstrap confidence intervals;
- McNemar-style discordance statistics.

For every post-selection A0-long control, the table contains separate rows
for `exact_matched_update` and `best_model_val_stop_within_budget`, including
their update indices. Primary conservative comparisons use the latter; the
former is never substituted silently.

Classification calibration and seed stability use these locked definitions:

- ECE is top-label ECE with 15 equal-width confidence bins on `[0,1]`,
  left-closed/right-open except the final closed bin; empty bins contribute
  zero weight.
- Multiclass Brier score is the event mean of
  `sum_class (probability - one_hot_label)^2`.
- Three-seed calibration values are arithmetic means of the three per-seed
  metrics.
- Seed variability is the sample standard deviation (`ddof=1`) of the three
  accuracies. Reports also show the minimum seed accuracy and minimum
  same-seed gain over A0.

Seed variability is diagnostic and cannot by itself satisfy a success
criterion.

Paired bootstrap uses 10,000 deterministic event-resampling replicates,
stratified by class, with seed `917_301`. Reports include the two-sided 95%
percentile interval for every primary gain. McNemar reports the exact
two-sided binomial p-value from events where only one of the paired models is
correct.

### 17.2 View recovery

Report:

- masked Huber/MSE and normalized MSE;
- per-dimension Pearson and Spearman correlations;
- particlewise cosine;
- eventwise cosine;
- relational Gram-matrix error;
- centered kernel alignment;
- predicted and target view norms;
- target and predicted between-particle variance;
- between-event versus within-event variance;
- uncertainty calibration;
- null-attention fraction;
- trust saturation;
- bounded token/pair residual scales and effective correction/bias norms
  versus trust percentiles;
- invalid-particle maximum absolute output.

### 17.3 Gain definitions

\[
G_\mathrm{oracle}
=
\operatorname{Acc}(C(HLT,f^*))
-
\operatorname{Acc}(C(HLT,0))
\]

\[
G_\mathrm{pred}
=
\operatorname{Acc}(C(HLT,\hat f))
-
\operatorname{Acc}(C(HLT,0))
\]

\[
R_\mathrm{view}
=
\frac{G_\mathrm{pred}}{G_\mathrm{oracle}}
\]

when the oracle denominator is positive.

When `G_oracle <= 0`, either gain is nonfinite, or a row has no defined
same-consumer counterfactual, `R_view` is stored as JSON `null` with
`recovery_status="undefined"`; it is never encoded as zero, infinity, or an
inherited value. `Dview_joint` has undefined recovery because its live
consumer changed and never inherits recovery from its frozen parent. A
three-seed mean recovery is defined only when all three replica recoveries
are finite with positive oracle denominators; otherwise the aggregate
recovery is undefined.

Also report gain over independently trained `A0_view`, but never mix that
quantity with same-consumer recovery.

### 17.4 Transformer diagnostics

Report:

- cross-attention entropy by head and layer;
- fraction of attention assigned to the null token;
- effective number of offline tokens attended per HLT query;
- attention-weight distance distributions;
- view similarity versus HLT geometric distance;
- region occupancy and empty-region rate;
- decoder attention across hierarchy scales;
- ablation sensitivity by particle \(p_T\) rank and region.

Attention diagnostics are descriptive. They are not treated as unique
physical explanations.

## 18. Selection, confirmation, and continuation

Scientific performance rules never cancel downstream jobs. They produce
quality warnings.

Every node writes `quality_warnings.jsonl`; aggregation writes
`quality_warning_summary.json` and `quality_warning_summary.md` at the
campaign root. Each warning contains:

- warning code and severity;
- graph node, configuration, seed, and split;
- observed value, reference value, and declared warning threshold;
- plain-language interpretation and suggested diagnostic;
- hashes/paths of the supporting metric artifacts; and
- UTC timestamp and source commit.

The summary begins with a high-visibility count by code and hyperlinks every
warning to its configuration report. Required codes include weak or
nonpositive oracle gain, weak/nonpositive recovery, large train-validation
gap, robust-tail saturation, negative stack gain, failed material-effect
threshold, calibration degradation, trust/slot saturation, and suspiciously
strong shuffled or HLT-memory controls, selected-control match-tolerance
misses, and post-selection controls outperforming their matched winner.

Scientific warning emission always exits zero and is never referenced by an
`afterok` dependency, selector eligibility predicate, or submission
condition. A selector must choose the deterministic best finite row even if
all rows carry warnings. The graph generator submits the full predeclared
downstream campaign before performance is known; dynamically added
confirmation rows are triggered by numerical ranking, not by a minimum
quality bar. Weak results may limit claims, but they never strand GPU jobs.

Hard fail-closed conditions are limited to:

- split/provenance mismatch;
- class-order or preprocessing mismatch;
- nonfinite targets, logits, losses, or gradients;
- invalid-particle leakage;
- forbidden offline dependency in a deployable bundle;
- target/consumer coordinate mismatch;
- stale teacher, normalizer, or checkpoint hashes;
- final-test access before selection and confirmation;
- storage-budget or resource-contract violations.

These are correctness or execution-integrity failures: continuing would use
the wrong data/model or could not execute safely. Ordinary low accuracy,
negative gain, weak recovery, poor calibration, a failed confidence
interval, or an unexpectedly strong control is never a hard failure.

### Winner families and total ordering

The campaign records three distinct outcomes:

1. `selected_privileged_scientific_model`: best deployable row trained with
   genuine offline-memory particle-view supervision and eligible for a
   privileged-learning claim. HLT-memory, CE-only, global-broadcast, random,
   and other diagnostic/control rows are excluded.
2. `selected_pre_stage_g_hlt_deployable_model`: best deployment-valid
   HLT-only row among the selection-eligible rows available before Stage G.
   It may be the privileged winner, `Dview_joint`, a CE-only model, or an
   HLT-memory model.
3. `best_diagnostic_control`: numerically strongest diagnostic/nonselectable
   row, reported without authorizing it as a deployment or privileged claim.

An HLT-memory or CE-only row may therefore become the final pre-Stage-G
performance bundle when deployment-valid, but it cannot become
`selected_privileged_scientific_model` or support a privileged-learning
claim. `Dview_joint` is privileged-training eligible and
pre-Stage-G-performance eligible, but its recovery remains undefined and the
frozen-consumer winner is still reported as the primary causal result.

Post-selection A0-long and selected-capacity/FLOP controls are confirmatory
comparators, not candidates in either winner selector: admitting them would
make the selected-bundle matching target circular. They never trigger a
second selection/matching loop. If one outperforms a winner, the report emits
`WARN_POSTSELECTION_CONTROL_WINS` and limits the corresponding scientific or
capacity claim while leaving the preselected bundle unchanged. The winner
summary always includes
`post_stage_g_control_numerically_better: true|false`, the strongest control
ID, and its signed three-seed mean `stack_val` accuracy difference relative
to each preselected winner.

Within each eligible family, configurations use this complete ordering:

1. highest three-seed mean `model_val_select` deployable accuracy;
2. highest three-seed median accuracy;
3. lowest mean cross-entropy;
4. recovery status: a finite three-seed recovery sorts before undefined
   recovery, then higher finite recovery wins;
5. fewer deployed parameters;
6. lexicographically smaller registered run ID; and
7. lexicographically smaller configuration ID.

Recovery is consulted only after the first three keys tie. If both rows have
undefined recovery, key 4 is skipped. It is never imputed from a parent
model. Single-seed target ranking uses the analogous rule: finite
positive-denominator recovery sorts before undefined recovery, higher finite
recovery wins, and if both are undefined the selector advances directly to
oracle gain. These rules produce a winner even if every row has negative
oracle or deployable gain.

All three preregistered replicas of the selected privileged configuration,
selected pre-Stage-G deployable configuration, and explicitly authorized
Stage-G fairness/CE-only controls are evaluated once on sealed `stack_val`,
as specified in Sections 14 and 15.8; identical artifacts are deduplicated
by hash. A negative stack-validation gain is reported, prevents a
strong-support claim, and does not retroactively replace either selected
model with a model chosen on `stack_val`.

Final-test evaluation occurs only after the preselected median representative
of each distinct winner family and its fusion recipes are frozen. Only those
preauthorized representatives access `final_test`, and reports never confuse
a pre-Stage-G performance/control winner with the privileged-scientific
winner. Stage-G fairness controls are never final-test authorized.

## 19. Fusion evaluation

The particle-view model may learn information complementary to ordinary HLT
ParT. The fusion report includes:

- `A0_view` alone;
- selected `Dview` alone;
- logit average `A0_view + Dview`;
- learned linear logit fusion fit on the stack-validation fit half;
- predetermined independent-seed `A0_view + A0_view` fusion control;
- capacity-matched direct-ParT fusion control;
- optional fusion with the existing P7b deployable model if provenance and
  class order match.

Fusion is evaluated on the held-out stack-validation half before its single
HLT-only final-test evaluation.

The `A0 + A0` control is mandatory. It uses predeclared independent
checkpoint pairs `(101,202)`, `(202,303)`, and `(303,101)`; a checkpoint is
never fused with itself and no pair is selected from stack performance.
Results are reported per pair and as their mean. This distinguishes
particle-view complementarity from ordinary independent-seed ensemble
diversity.

## 20. Deployment contract

A deployable bundle contains:

- bundle kind: frozen-consumer `Dview` or jointly tuned `Dview_joint`;
- HLT-only predictor config and checkpoint;
- selected clean or robust consumer config and checkpoint;
- HLT preprocessing/schema hashes;
- view normalizer;
- bottleneck width and coordinate hash;
- class order;
- mask and maximum-particle contract;
- resource profile;
- source commit and artifact hashes.

It must not contain or require:

- offline particles;
- `Toff_view`;
- `Gview`;
- true views;
- cached privileged logits;
- training labels;
- oracle attention maps;
- an offline normalizer not already materialized as the deployable view
  normalizer.

A reload test runs the exported bundle in a fresh process with only HLT input
paths visible and compares logits against the pre-export model.

Reports strictly separate:

- offline/oracle diagnostics;
- privileged true-view diagnostics;
- frozen-consumer HLT-only deployable results;
- jointly fine-tuned HLT-only deployable results;
- pre-Stage-G HLT-only performance-control results, including CE-only or
  HLT-memory winners, plus explicit post-Stage-G comparator deltas;
- fusion/ensemble results.

## 21. Storage and RAM policy

Offline contextual token tensors are never persisted for the full dataset.
They are generated once per staged batch in RAM from the frozen offline
teacher and consumed by `Gview`.

For target-generator packs on a 500 GB node, the canonical execution mode is
allocation-local teacher-tap staging:

1. open the offline source once;
2. run the selected frozen teacher once over the required split;
3. store the contextual particle tap as float16 in node RAM with its mask and
   jet identities;
4. hash the staged logical contents;
5. reuse the same read-only RAM tensor across target variants and epochs in
   that allocation;
6. release it when the allocation exits.

The staging representation is deterministic. Teacher forward output is
materialized in float32, nonfinite values are rejected, and valid token
entries are cast once to IEEE-754 binary16 using round-to-nearest,
ties-to-even with stochastic rounding disabled. Masks stay boolean and jet
identities stay their registered integer dtype. The logical content hash is
over dtype/shape metadata plus canonical little-endian binary16 bytes,
boolean-mask bytes, and identity bytes. Every consumer of the staged tensor
promotes those same binary16 values; it may not recompute a float32 tap and
call it equivalent.

For 500k jets, 128 particles, and teacher width 160, the raw float16 tap is
approximately 19.1 GiB before masks and identities, comfortably within the
500 GB allocation. Two-teacher mixtures stage both taps only after the
preflight memory calculation passes.

The staged tensor is never written to persistent campaign storage. A required
equivalence test compares the staged tensor with a fresh live teacher tap
after applying the exact same float32-to-binary16 conversion. It requires
bitwise-equal valid binary16 values and exact masks/identities; the separately
reported float32-versus-binary16 absolute error must be finite and no greater
than half the spacing between the two adjacent finite binary16 values that
bracket the float32 source value, with exact midpoint ties resolved to the
even binary16 significand. Target-generator
packs fail closed if the staged source, teacher, tap, or identity hash differs
from the reservation.

For the selected target definition, the publisher must persist one canonical
normalized-view cache for every downstream-authorized split:

```text
little-endian float32 view: [jets, max_particles, d]
bool mask
jet identity arrays
content/provenance metadata
```

The normal training publication covers `train`, `model_val_stop`, and
`model_val_select`. A sealed `stack_val` cache is published only after stack
authorization and only for oracle/diagnostic confirmation. No selected-view
cache is ever published for `final_test`.

The canonical materialization order is: deterministic float32 `Gview`
forward, raw masked centering, registered float32 normalizer, standardized
clipping to `[-6,6]`, invalid-particle zeroing, conversion to canonical
little-endian float32 with no reduced-precision intermediate, then hashing
and writing. Nonfinite values fail before publication. The content hash
covers dtype, byte order, shape, clipping/conversion policy, view bytes, mask
bytes, ordered jet identities, coordinate-binding hash, and split hash.

`Cview_clean`, every `Pview` representation target, robust-consumer residual
construction, frozen-consumer KD target generation, oracle evaluation, and
reloaded training/evaluation must read these exact float32 values. None may
silently regenerate live views or substitute float16. The cache is the
downstream source of truth, while `Gview` remains its authenticated
provenance parent.

A deterministic publication audit regenerates a registered sample from
`Gview`, applies the identical operation order, and requires exact
masks/identities plus maximum absolute view error `<=1e-6`; the cache
round-trip itself must be bitwise exact. A mismatch is an integrity failure,
not a quality warning.

At 500k jets, 128 particles, and `d=4`, the dense float32 view payload is
approximately 0.95 GiB before compression. `d=8` is approximately 1.91 GiB
before metadata and compression. Preflight accounts for every authorized
split, but only the one selected target definition is persisted.

Unselected target variants are RAM-only and retain metrics plus the best
checkpoint. Selected frozen-consumer target logits are materialized once as
canonical little-endian float32 arrays with their own locked binding and are
reused by every compatible KD row; no dense offline embeddings are cached.

Retention policy:

- retain all JSON metrics and registries;
- retain all three-seed confirmed checkpoints;
- retain canonical and selected screen checkpoints;
- delete unselected optimizer states after metrics are finalized;
- retain at most one best checkpoint per unconfirmed run;
- retain no batchwise attention tensors beyond bounded diagnostic samples.

Preflight measures actual source-cache and checkpoint sizes and reserves a
campaign budget before submission.

## 22. Provenance and deterministic artifacts

Required immutable artifacts include:

- unified split manifest;
- `A0_view` registration;
- `A0_view_long_deploy` registration;
- `A0_view_total_label_budget` registration;
- `Toff_view` registration;
- offline and HLT query token-tap specifications;
- HLT-memory control tap specification and registration;
- `Gview` checkpoint registration;
- `Rview`, co-design cycle, and final `Bview` registrations where applicable;
- view-coordinate binding;
- canonical selected-view float32 cache manifests by authorized split;
- clean-consumer registration;
- `Pview_0` registration;
- robust-consumer registration;
- target-logit binding;
- campaign registry;
- resource reservations;
- privileged-scientific, pre-Stage-G-deployable, and diagnostic winner
  records;
- selected-path fairness ledger and post-selection control registrations;
- confirmation record;
- per-node quality-warning ledgers and aggregate warning summary;
- deployed bundle manifest;
- final report manifest;
- label-exposure and training-FLOP ledger.

The lineage direction is strictly:

```text
source/split registrations
        -> A0 and offline-teacher registrations
        -> HLT/offline tap specifications
        -> Gview checkpoint registration
        -> view normalizer
        -> view-coordinate binding
        -> canonical selected-view float32 cache
        -> clean-consumer registration
        -> Pview_0 and robust-consumer registrations
        -> target-logit binding
        -> final predictor/deployment registrations
```

The view-coordinate binding binds:

- source manifest hash;
- exact `train` identity hash;
- HLT and offline source hashes;
- `A0_view` checkpoint, config, query tap, and input-normalization hashes;
- offline teacher checkpoint and config hashes;
- offline tap layer and exact tensor location;
- cross-attention config;
- pair-feature schema;
- centering policy;
- bounded-coordinate and rate-budget policy;
- null-token policy;
- bottleneck width;
- generator checkpoint;
- normalizer hash;
- selected-view dtype/byte order;
- materialization, clipping, invalid-zeroing, and conversion order; and
- live-publication audit tolerance.

The coordinate binding never includes a consumer checkpoint. The clean
consumer registration references the coordinate binding. Predictor and robust
consumer registrations reference both the coordinate binding and their exact
parent consumer. A target-logit binding references the coordinate binding,
canonical selected-view cache manifest, exact consumer checkpoint, and target
source.

Teacher and generator checkpoints are hashed before the coordinate binding is
written. Bindings are hashed before caches are created. Cache metadata records
the binding hash, split hash, and canonical logical-content hash. All
downstream true-view consumers reference the cache manifest rather than a
fresh generator forward. No artifact includes its own content hash as an input
parent, and no child checkpoint appears in an ancestor binding.

## 23. Tigris execution contract

Every Slurm script must:

```bash
export PYTHONNOUSERSITE=1
```

and use:

```text
account = reu-aisocial
```

The conda environment is `atlas_kd_tigris`, activated through the configured
Miniforge profile rather than a stale `miniconda3` executable.

Jobs are grouped into:

1. source and single-pool manifest validation;
2. matched A0/offline-teacher and canonical direct-control training;
3. target-generator screens;
4. canonical view publication;
5. predictor warm-up and robust-consumer construction;
6. predictor/loss/architecture packs;
7. three-seed model-validation confirmation and numerical winner selection;
8. selected-path fairness-ledger publication and post-selection A0/direct
   controls;
9. sealed stack-validation and fusion;
10. aggregate/report/export/reload and final-test authorization;
11. one-time HLT-only final test for the preauthorized representative
    bundles.

Dependency recovery accepts existing completed job IDs by logical graph node.
Scientific warnings do not create `DependencyNeverSatisfied` chains.

## 24. Required implementation surfaces

### Python modules

Proposed modules under
`teacher_logit_reco/local_particle_residual_field/particle_view/`:

- `contracts.py` - schemas, hashes, split and deployment contracts;
- `offline_teacher.py` - teacher registration and contextual token taps;
- `target_generator.py` - matching-free HLT-query/offline-memory attention;
- `recovery_probe.py` - fixed-capacity target recovery and co-design;
- `view_cache.py` - bounded compact selected-view caching;
- `consumer.py` - clean/robust view consumer and injection paths;
- `predictor.py` - hierarchical particle-query decoder;
- `losses.py` - representation, relational, KD, CE, and trust losses;
- `train_oracle.py` - target-generator and clean-consumer training;
- `train_predictor.py` - warm-up and frozen-consumer distillation;
- `joint_finetune.py` - optional HLT-only predictor/consumer fine-tuning;
- `robust_consumer.py` - predicted/error/missing mixture training;
- `controls.py` - shuffle, sign, random, broadcast, and capacity controls;
- `metrics.py` - classification, view, attention, and recovery reports;
- `registry.py` - ablation inventory and seed expansion;
- `selection.py` - numerical selection and representative retention;
- `deployment.py` - HLT-only export and reload audit;
- `fusion.py` - stack-validation fusion and A0+A0 controls;
- `storage.py` - measured reservations and eviction policy.

The new package may reuse stable ParT, pair-feature, metric, and provenance
utilities, but it must not silently import residual-field semantics.

### Command-line entry points

Proposed scripts:

- `prepare_particle_view_campaign.py`;
- `train_particle_view_baseline.py`;
- `train_particle_view_offline_teacher.py`;
- `train_privileged_particle_view_oracle.py`;
- `train_particle_view_recovery_probe.py`;
- `publish_particle_view_targets.py`;
- `train_particle_view_predictor.py`;
- `train_particle_view_robust_consumer.py`;
- `finetune_particle_view_deployable.py`;
- `evaluate_particle_view_controls.py`;
- `select_particle_view_configuration.py`;
- `confirm_particle_view_deployable.py`;
- `run_particle_view_fusion.py`;
- `export_particle_view_bundle.py`;
- `report_particle_view_campaign.py`.

### Slurm entry points

Proposed scripts:

- `run_prepare_particle_view_campaign.sh`;
- `run_train_particle_view_baselines.sh`;
- `run_train_particle_view_oracle.sh`;
- `run_particle_view_predictor_pack.sh`;
- `run_particle_view_confirmation.sh`;
- `run_particle_view_fusion.sh`;
- `run_particle_view_export_report.sh`;
- `submit_particle_view_full_pilot.sh`.

## 25. Required tests

### Data and provenance

- all trainable components use the same `train` identity hash;
- no training child partition exists;
- `model_val_stop`, `model_val_select`, `stack_val`, and `final_test` remain
  disjoint;
- stale source, teacher, tap, generator, consumer, or normalizer hashes fail;
- coordinate binding excludes consumer checkpoints and lineage is acyclic;
- canonical selected-view caches are little-endian float32, bind the exact
  operation order and split, and every downstream true-view consumer
  authenticates their manifest;
- consumer registration references the exact coordinate binding;
- final-test offline access fails before model execution.

### Offline token tap

- token tap excludes class/pooled tokens and logits;
- tap masks and ordering match offline particle inputs;
- HLT and offline tap locations are after block residuals and before pooling;
- query-source permutations and learned-mixture hashes are deterministic;
- frozen teacher weights never receive gradients;
- teacher reload reproduces token embeddings and logits.
- locked base and large teacher recipes reject any optimizer, schedule,
  architecture, tap, preprocessing, mask, or update-count drift.

### Matching-free target generator

- HLT permutation equivariance;
- offline permutation invariance;
- no offline-index lookup or one-to-one matching artifact;
- null-token behavior;
- padding invariance;
- geometric pair-bias shape and wrapping;
- bounded coordinates respect the declared range;
- variance-floor, rate, and covariance losses match hand calculations;
- centered views have zero masked per-jet mean within tolerance;
- 4-bit and 8-bit quantization ranges are train-fitted and deterministic;
- live and allocation-RAM teacher taps agree within tolerance;
- deterministic evaluation reproduces target hashes.
- `VGEN_MEMORY_HLT` contains no offline source and matches the canonical
  generator/probe budget exactly.
- the self-masked HLT-memory diagnostic masks only the exact same-particle
  key/value before softmax and preserves null/padding behavior;

### Consumer

- zero-scaled view paths reproduce `A0_view` logits;
- provisional discovery consumers cannot be registered for deployment;
- the final clean consumer is reinitialized after normalizer registration;
- raw and normalized coordinate consumers cannot be interchanged;
- token and pair-bias injections obey masks;
- trust bounds hold;
- residual scales receive gradients on step one and the adapters/gates receive
  gradients by step two while epoch-zero logits remain exactly A0;
- token and pair residual scales remain in `[-1,1]`, and effective
  correction/pair-bias diagnostics match hand calculations;
- zero, random, sign, event-shuffle, and particle-shuffle controls are valid;
- clean and robust checkpoints bind the correct target coordinate system.

### Predictor architecture

- exact output shapes for dimensions 1, 2, 4, and 8;
- invalid outputs are exactly zero;
- slot softmax normalizes over destination regions;
- 8/4 levels are derived sequentially from 16/8 parents;
- region four-vectors, centroids, occupancy, empty thresholds, and balance loss
  match miniature hand calculations;
- the first assignment is embedding-only, its sole refinement uses provisional
  centroids, and physical four-vectors never receive a second `pT` weight;
- soft `B`, `C`, and `B @ C` hierarchy-relation matrices match hand
  calculations and obey masks;
- particle permutation equivariance;
- region-token permutation behavior;
- decoder returns one view per original HLT particle;
- uncertainty bounds and deterministic inference;
- architecture and resource hashes change on material config changes.

### Losses and training

- representation, cosine, relational, uncertainty, KD, CE, and trust losses
  match hand-computed miniature examples;
- primary Huber uses beta `0.1`, event-balanced masking, and the locked empty
  behavior;
- relational loss excludes diagonals/padding, is event-balanced, and returns
  zero for a single-particle event; cosine degeneracy and log-variance bounds
  match the locked definitions;
- every target candidate uses the identical fixed-capacity probe and budget;
- target selection prioritizes predicted-view gain and emits, but never gates
  on, oracle-gain warnings;
- co-design freezes rich context and grades the result with a fresh probe;
- KD uses the exact same frozen consumer on both sides;
- warm-up has a fixed four-epoch budget;
- robust-consumer mixture proportions are deterministic under a seed;
- predictor is frozen during robust-consumer construction;
- robust residual sampling preserves whole-event correlation and matches
  registered held-out tail inflation;
- robust-consumer reload rejects any snapshot, dropout, sampler, seed,
  mixture, source-split, or optimizer lineage drift;
- consumer is frozen during final predictor distillation;
- joint fine-tuning keeps its oracle target network frozen and is never
  reported as same-consumer recovery;
- CE-only joint control matches the complete freeze/unfreeze schedule;
- CE-only cannot be mislabeled as privileged representation training;
- checkpoint selection uses only `model_val_stop`, and configuration ranking
  uses only `model_val_select`;
- finite and undefined recovery rows have a deterministic total order,
  `Dview_joint` never inherits recovery, and complete ties resolve by
  parameter count/run/config IDs;
- fairness ledgers reconcile CE steps, label exposure, FLOPs, and search
  trials;
- selected-path A0/direct controls cannot start before the immutable fairness
  ledger exists, exact-update and best-within-budget A0 checkpoints are both
  registered, and direct-match tolerances are audited.
- relative parameter/FLOP errors and the `flop_fixture_v1` operator counts
  match hand calculations and use the registered counter hash;

### Deployment and reports

- exported bundle loads with offline paths removed;
- deployable outputs require only HLT;
- export/reload logits match;
- oracle and deployable rows are separated;
- final-test rows are HLT-only;
- fusion controls include A0+A0;
- paired bootstrap and McNemar results are deterministic;
- all selected replicas access stack validation, only the preselected median
  replica accesses final test, and A0+A0 uses only the predetermined
  independent-seed pairs;
- Stage-G A0/direct and matched CE-only comparators are stack-authorized with
  the locked checkpoints/three-seed mean but remain final-test forbidden
  unless independently selected before Stage G;
- privileged-scientific, pre-Stage-G-deployable, and diagnostic winners remain
  distinct in selection, export, final-test authorization, and reporting;
- ECE/Brier definitions, comparable-accuracy tolerance, calibration
  precedence, seed sample deviation, and post-control-winner boolean/delta
  match miniature deterministic examples;
- strong-support claims enforce the material-effect threshold;
- every scientific threshold emits a structured warning with exit code zero,
  and no warning can block, cancel, or make an `afterok` dependency
  unsatisfiable;
- registry counts reconcile with generated jobs;
- Tigris scripts export `PYTHONNOUSERSITE=1`;
- Tigris account is exactly `reu-aisocial`.

## 26. Expected failure modes and interpretations

### Large offline ceiling, near-zero predicted recovery

The view generator encoded inaccessible offline detail. Reduce bottleneck
width, strengthen centering/norm constraints, increase representation
anchoring, or explicitly co-design for recoverability. Do not simply enlarge
the predictor.

### Small oracle ceiling

The target tap, bottleneck, or consumer interface is not useful. Compare
penultimate/mixed taps, view dimension, and token-plus-pair injection before
rejecting particle views.

### CE-only wins while view recovery collapses

The system is using the field channel as an auxiliary HLT computation rather
than reconstructing privileged context. Treat CE-only as a capacity/task
upper bound and prioritize KD plus representation anchors.

### Representation metrics improve without classification gain

The target includes predictable components that the consumer does not need.
Increase tagger-useful KD or redesign the target bottleneck rather than
optimizing MSE further.

### Uncentered view greatly outperforms centered view

The generator may be broadcasting global offline class information. Report it
as a diagnostic and compare against direct offline-logit KD and the explicit
global-broadcast control.

### Particle shuffle retains most gain

The consumer is using the within-jet distribution more than exact particle
alignment. Compare pair-bias-only, mean-broadcast, and relational-loss
ablations. This does not automatically invalidate the view.

### Direct matched ParT matches the privileged model

The gain comes from architecture or compute rather than privileged
supervision. The particle-view interface may still be useful operationally,
but no privileged-learning claim is supported.

### `A0 + Dview` fusion matches `A0 + A0`

Complementarity is ordinary model diversity. Do not attribute it specifically
to particle-view reasoning.

### Robust consumer beats clean consumer with true views

Noise/dropout acts as regularization. Keep clean and robust ceilings separate
and do not relabel robust performance as the clean oracle ceiling.

## 27. Implementation order

Implementation is divided into ten similarly sized, independently testable
steps.

### Step 1 of 10: contracts and unified training manifest

Implement the package skeleton, single-`train` manifest adapter, split audits,
deterministic 75k/75k stop/select validation children, content hashes,
coordinate-binding schema, registry schema, deployment schema, label-exposure
ledger, and miniature fixtures. Prove that no consumer/distillation training
partition or cross-fit identity enters the graph.

### Step 2 of 10: matched HLT and offline teachers

Implement `A0_view`, predeclared-canonical capacity/FLOP direct controls, and
the fully locked base/large ParT recipes,
matched/larger/existing offline teachers, frozen contextual HLT and offline
token taps, checkpoint registration, token shape/mask audits, and
deterministic reload tests.

### Step 3 of 10: matching-free privileged view generator

Implement contextual HLT query taps, offline-memory cross-attention, geometric
pair bias, null token, bounded/centered bottlenecks 1/2/4/8, exact rate and
covariance constraints, permutation tests, RAM teacher-tap staging,
bit-exact float16 conversion/equivalence, the mandatory self-inclusive and
self-masked HLT-memory controls, and candidate target provenance.

### Step 4 of 10: clean consumer and oracle-view discovery

Implement zero-scaled, gradient-reachable A0 warm-start, provisional
discovery consumers, two-pass normalization, fixed-capacity recovery probes,
recoverability co-design, final clean-consumer reinitialization, token and pair-bias
injection, canonical float32 selected-view publication, offline-KD/rate
screens, target ranking by predicted gain, and scientific warning reports.

### Step 5 of 10: canonical hierarchical particle-query predictor

Implement local geometric blocks, the exact sequential 16/8/4 assignment and
single centroid-refinement pass, non-double-weighted four-vector equations,
soft hierarchy-relation matrices, occupancy/empty/collapse contracts, global
geometric region attention, particle-region decoder bias, final refinement,
mean/uncertainty heads, masks, resource profiling, and architecture controls.

### Step 6 of 10: representation training and robust consumer

Implement fixed-budget view warm-up, Huber/cosine/relational/uncertainty
losses, `Pview_0` registration, held-out-tail-calibrated correlated error
sampling, snapshot/dropout predictions, deterministic robust-view mixtures,
robust-consumer training, and paired clean/robust evaluation.

### Step 7 of 10: frozen-consumer distillation and loss campaign

Implement exact same-consumer target/live forwards, KD/CE/representation/trust
objectives, checkpoint selection on `model_val_stop`, ranking on
`model_val_select`, loss screens, target x loss interactions, optional
HLT-only joint fine-tuning, schedule-matched CE controls, train/validation
generalization reports, and teacher-independent deployment audits.

### Step 8 of 10: controls, seeds, selection, and fusion

Implement structural shuffles, random/sign/broadcast controls, capacity and
ordinary-KD controls, seed expansion, numerical aggregation, median-replica
retention, total ordering for undefined recovery, separate privileged and
pre-Stage-G deployable winners, selected-path fairness-ledger publication,
post-selection A0-long and selected-capacity/FLOP controls, paired bootstrap/McNemar
reports, sealed stack-validation, A0+A0 fusion, and optional P7b fusion.

### Step 9 of 10: storage, provenance, export, and reports

Implement RAM staging, bounded diagnostics, measured reservations, eviction,
canonical float32 selected-view cache validation, acyclic
coordinate/cache/consumer/logit lineage validation, fairness-budget
accounting, HLT-only frozen and jointly tuned bundle export/reload,
oracle/deployable report separation, and final-test authorization.

### Step 10 of 10: production Slurm graph and rehearsal

Implement Tigris scripts, full campaign registry reconciliation, logical-node
dependency recovery, `PYTHONNOUSERSITE=1`, `reu-aisocial`, dry-run/print-only
modes, structured non-gating quality warnings, miniature filesystem
rehearsal, clean-start submission test, warning-continuation recovery test,
and one-command full-pilot submission.

## 28. Definition of pilot success

The privileged particle-view idea is considered strongly supported if the
three-seed `selected_privileged_scientific_model`:

1. has positive same-consumer clean privileged oracle gain in at least two of
   three seeds and mean oracle gain of at least `0.0020` on
   `model_val_select`; `0.0050` remains a desired-ceiling warning threshold,
   not an execution or selectability floor;
2. has positive HLT-only predicted-view gain over both its zero-view endpoint
   and matched `A0_view` in at least two of three seeds;
3. confirms at least `0.0020` absolute accuracy gain, or 0.20 percentage
   points, over matched `A0_view` on sealed `stack_val`;
4. has a positive lower bound on the deterministic 95% paired bootstrap
   interval on `stack_val` and exact McNemar two-sided `p < 0.05`;
5. beats both A0-long budget controls and the selected-bundle
   capacity/FLOP-matched direct HLT controls registered against its own
   privileged-winner fairness entry on three-seed mean `stack_val` accuracy,
   using the primary checkpoints specified in Stage G;
6. either has strictly higher three-seed mean `stack_val` accuracy than the
   identical CE-only particle-view control, or has comparable accuracy
   (`abs(mean_accuracy_privileged - mean_accuracy_CE) <= 0.0010`) and
   materially better calibration under this precedence rule:
   mean ECE lower by at least `0.0020`; otherwise, when absolute mean-ECE
   difference is below `0.0020`, mean Brier score lower by at least `0.0020`.
   Seed sample standard deviation and worst-seed accuracy/gain are reported
   diagnostics and cannot satisfy this criterion;
7. retains a positive and reported fraction of same-consumer privileged gain;
8. exports and reloads without any offline dependency;
9. preserves a positive gain on the one-time HLT-only final test.

A positive gain that fails the material-effect or paired-statistical criteria
is reported as promising but does not satisfy the strong-support definition.
Final-test statistics are confirmatory descriptions and do not alter the
preselected winner records.

Failure of any success criterion emits a warning and changes only the wording
of the scientific conclusion. It never changes a job exit code, cancels a
queued node, invalidates an otherwise finite checkpoint, or prevents the
remaining pilot rows from running.

The pilot is still informative if only a subset holds:

- a large oracle ceiling with low recovery identifies a target
  recoverability problem;
- a small oracle ceiling rejects the selected view definition;
- CE-only improvement without privileged advantage identifies an architecture
  effect;
- strong fusion without standalone gain identifies complementary
  representations;
- a narrow bottleneck outperforming a wide one supports the
  low-dimensional-view hypothesis.

## 29. Follow-up only after the pilot

If the first pilot succeeds, later work may test:

- semantic particle-role decoders attached to learned views;
- low-rank view-derived attention coordinates;
- shared predictor/consumer backbones;
- predictor compression and trigger-latency optimization;
- high-data 3M/3M-equivalent training with one unified 6M pool;
- transfer to alternative HLT degradation profiles;
- fusion with residual fields and existing P7b models;
- multi-layer view targets and progressive view curricula.

These extensions must not be added to the first pilot before the canonical
matching-free particle-view hypothesis is tested cleanly.
