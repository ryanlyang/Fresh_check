# EdgeValue context and relation-gating screen on SPORC

Date: 2026-10-16  
Status: implementation available in `part_ev_screen/`; see its [queue guide](../part_ev_screen/README.md). Local tests cover the implementation; remote GPU probes remain required before production. No jobs have been submitted by the implementation task.  
Source inspected: repository commit `6a977730f5447e2f5df787c42ddefe5db5892798`.

## 1. Objective and boundaries

Run seven single-seed training experiments to investigate two complementary extensions of ParT-EV, plus an optimizer-recipe comparison:

1. **Jet-conditioned EdgeValue:** use the current jet representation to change the relation-derived additive value message.
2. **Relation-gated particle values:** use each pair representation to change which channels of the sending particle's value are transmitted.
3. **Their combination:** enable both mechanisms, without giving the relation gate an additional jet-context input.
4. **AdamW controls:** compare ordinary ParT and existing ParT-EV under the previously agreed AdamW recipe.

All seven runs use seed **101**, the official **100M training population**, **200,000 training steps**, and global batch **512**. This is a development screen, not a final multi-seed or full-budget result. Run on SPORC GPU compute nodes, submitted from `sporcsubmit`; use the account's available production partition, resolving the requested `tier3` name against the live configuration.

Grouped/subjet attention belongs to a separate experiment and is out of scope. No selected PT/TRACK/REGION feature families, layerwise scalar-bias ablations, plain-attention ablations, or parameter-matched extra runs are added to this seven-run screen. Ordinary ParT here **includes its standard shared pairwise attention bias**.

The implementation choices below are a concrete, economical first version, not a claim that these widths, gates, or pooling choices are empirically optimal. Freeze them before inspecting screen results; changes after that require a new versioned experiment.

## 2. Exact run registry

Task identity must include the optimizer recipe, not just architecture and seed; otherwise the AdamW controls would overwrite the original-recipe controls.

| Array index | Run ID | Architecture | Recipe | Seed | Jet-conditioned EV | Relation-gated values |
|---|---|---|---|---:|:---:|:---:|
| 0 | `PART_RANGER` | `part` | `ranger_short_v1` | 101 | No | No |
| 1 | `EV_RANGER` | `part_ev` | `ranger_short_v1` | 101 | No | No |
| 2 | `EV_CONTEXT_RANGER` | `part_ev_context` | `ranger_short_v1` | 101 | Yes | No |
| 3 | `EV_GATE_RANGER` | `part_ev_relation_gate` | `ranger_short_v1` | 101 | No | Yes |
| 4 | `EV_CONTEXT_GATE_RANGER` | `part_ev_context_relation_gate` | `ranger_short_v1` | 101 | Yes | Yes |
| 5 | `PART_ADAMW` | `part` | `adamw_short_v1` | 101 | No | No |
| 6 | `EV_ADAMW` | `part_ev` | `adamw_short_v1` | 101 | No | No |

The first five runs isolate architecture changes within one recipe. The last two test whether the EV benefit depends on the optimizer recipe. They do not test the new context/gating mechanisms with AdamW.

Retrain both architecture controls on SPORC under this reduced budget. Historical longer-budget Tigris checkpoints are contextual information, not interchangeable controls.

## 3. Shared backbone and data

Reuse the pinned reference configuration in `part_ev_fullscale/model.py`:

- Weaver `0.4.17`; verify its ParticleTransformer source SHA-256:
  `246be11c969fabf87a94fa2f68522da82a7f002069b370f4e4d7e3030fab1ce2`.
- Official `particle_transformer/data/JetClass/JetClass_full.yaml`, unchanged.
- 17 particle input features, ten classes, maximum 128 particles.
- Particle embedding widths `[128, 512, 128]`; model width 128.
- Eight heads, head width 16, eight particle-attention blocks, two class-attention blocks.
- Native particle FFN width 512; native normalizations, head/residual scaling, dropout, sequence trimming, padding, and class readout.
- Standard four-channel kinematic pair input and native pair embedding widths `[64, 64, 64]`.
- One shared pairwise scalar-bias encoder reused by the particle blocks. The new modules do not replace or directly modulate that bias path.
- Ordinary unweighted cross-entropy, with no new label smoothing, augmentation, or class weighting.
- One GPU per run, batch 512, two data-loader workers, unchanged continuous ROOT streaming and strict file-read checks.
- FP32 parameters with FP16 autocast and gradient scaling; TF32 off and cuDNN benchmarking off, matching the existing protocol.

Data location to verify, not assume present:

```text
/home/ryreu/atlas/datasets/JetClass/Pythia/train_100M
/home/ryreu/atlas/datasets/JetClass/Pythia/val_5M
```

Verify 1,000 training ROOT files and 50 validation ROOT files, their expected names, and 100,000 events per file. Reuse the staged data in place; do not copy it into every campaign or make a large token cache. The official data populations are documented in the [ParT repository](https://github.com/jet-universe/particle_transformer).

### Exposure accounting

| Quantity | Existing full-budget recipe | This screen |
|---|---:|---:|
| Training population | 100,000,000 | 100,000,000 |
| Training steps | 1,000,000 | 200,000 |
| Global batch | 512 | 512 |
| Processed training jet exposures | 512,000,000 | 102,400,000 |
| Checkpoint-validation interval | 20,000 steps | 20,000 steps |
| Validation checks | 50 | 10 |

The screen has approximately 1.024 population-equivalents of exposure, not a 20M training subset. Continuous sampling does not guarantee exactly one visit per event. Keep iterators alive across validation checks; do not restart the first data range each time.

As in the existing trainer, a step is a consumed training batch/attempted optimizer update. AMP can skip an update after detecting overflow. Log attempted steps, successful updates, and skipped updates separately; do not silently extend only one run's exposure budget to compensate.

## 4. Exact model definitions

### 4.1 Notation and the existing EV control

For one jet at particle-attention layer `l`:

- `x_i^l` is the 128-dimensional pre-attention-normalized particle representation passed into native MHA.
- `v_j^(l,h)` is the ordinary projected value for head `h`, including the native value-projection bias.
- `e_ij` is the shared 64-dimensional pair stem already used by existing EV, after native pair packing/scattering and masking.
- `a_ij^(l,h)` is the native per-head attention weight, **after attention dropout**, including the standard scalar pair bias and key mask.
- `U^(l,h)` is the existing EV projection, with shape `[16,64]` for each head.

Suppress layer/head indices where unambiguous. Existing EV computes the head output

$$
o_i = \sum_j a_{ij}\left(v_j + Ue_{ij}\right).
$$

Heads are concatenated and passed through the native output projection, followed by the unchanged rest of the block. Implementation currently computes native attention output first and adds the projected EV correction. Keep that approach. Add the native output-projection **bias only once**.

Existing EV projections remain Xavier-initialized, not zero-initialized. The additions below start as no-ops relative to that existing EV model.

### 4.2 Jet-conditioned EV: a feature-wise multiplicative context gate

Use a masked mean of the current layer's particle representations, followed by parameter-free LayerNorm:

$$
c^l = \operatorname{LN}_{128}\left(
\frac{\sum_i m_i x_i^l}{\sum_i m_i}
\right),
\qquad
s^l = 1 + \tanh\left(W_2^l\,\operatorname{GELU}(W_1^l c^l+b_1^l)+b_2^l\right).
$$

Exact specification:

- `m_i` is the validity mask after the native trimmer's permutation/truncation.
- Pool separately within each jet, in FP32. Reject a jet with no real particles.
- `LayerNorm(128, elementwise_affine=False, eps=1e-5)`.
- Context MLP: `Linear(128,32) -> GELU -> Linear(32,64)`; both linears have biases; no added dropout or batch normalization.
- Separate context MLP in each of the eight particle blocks; one 64-channel gate shared by the eight heads within a block.
- Initialize `W_1` with Xavier uniform and `b_1=0`; initialize `W_2=0`, `b_2=0`.
- Consequently `s=1` at initialization, with channel scales mathematically in `(0,2)` thereafter (floating-point saturation can reach the endpoints).
- Do not detach `c`, `s`, or the pair stem. Classification gradients train the entire path.

The conditioned additive message is

$$
F^{l,h}(e_{ij},c^l)=U^{l,h}\left(e_{ij}\odot s^l\right).
$$

This is a deliberately restricted, efficient realization of `EV = F(pair, jet)`, not a general per-pair concatenation MLP. It introduces a genuine multiplicative interaction between the pair and the jet, rather than adding the same global vector to every edge.

**Efficient computation:** because `s` is common to all pairs in a jet at that layer,

$$
\sum_j a_{ij} U(e_{ij}\odot s)
=U\left[\left(\sum_j a_{ij}e_{ij}\right)\odot s\right].
$$

Apply the gate after the existing attention-weighted pair aggregation. There is no need to materialize another full pair-value tensor. This identity holds for the actual post-dropout weights too; it does not require their row sums to equal one.

The context uses representations entering the block, not its future output or final classifier prediction. It includes only the valid particles available after trimming. Recompute it at each layer. It is a compressed summary, not lossless access to the entire jet. This design is related to established feature-wise conditioning, not a claim to invent that principle; see [FiLM](https://arxiv.org/abs/1709.07871) and [global-conditioned edge updates in graph networks](https://arxiv.org/abs/1806.01261).

### 4.3 Relation-gated particle values: bounded rank-four channel gating

This mechanism changes the particle value `v_j`, **not** the raw pair variables, scalar attention bias, or additive EV projection.

Use a low-rank channel gate to avoid storing `[batch,heads,N,N,head_dim]` activations. Set the gate rank to **R=4** for this first screen:

$$
\bar e_{ij}=\operatorname{LN}_{64}(e_{ij}),\qquad
t_{ijhr}=\tanh\left(A_{hr}\bar e_{ij}+b_{hr}\right),
\qquad B_{hrd}=\tanh(\Theta_{hrd}),
$$

$$
g_{ijhd}=1+\frac{1}{R}\sum_{r=1}^{R}t_{ijhr}B_{hrd}.
$$

Exact specification, separately in each particle block:

- `LayerNorm(64, elementwise_affine=False, eps=1e-5)` on the pair channel dimension; this normalization is private to the gate branch, leaving the existing EV and bias branches unchanged.
- A linear map `64 -> heads * R = 32`, with bias; reshape outputs to eight heads and four gate coefficients.
- Initialize `A` with Xavier uniform, `b=0`.
- A learned tensor `Theta` of shape `[8,4,16]`, initialized to zero.
- Apply `tanh` to coefficients and basis as in the equations, with the fixed `1/R` normalization. Each effective value-channel gate lies mathematically in `(0,2)`.
- No extra scalar strength parameter, dropout, dependence on `c`, or dependence on the current node representations in this first version.
- Explicitly zero the gate coefficients for invalid pairs after normalization/projection; padded zeros must not become learned pseudo-particles through biases.

At initialization `Theta=0`, so `g=1` and this arm initially matches existing EV. Do not initialize both `A` and `Theta` to zero: that would kill the bilinear branch's learning signal. `Theta` can learn on the first step; upstream gate parameters can begin learning once it departs from zero.

The head output in this arm is

$$
o_{ihd}=\sum_j a_{ijh}\left(g_{ijhd}v_{jhd}+[U^h e_{ij}]_d\right).
$$

Compute only the correction to native attention:

$$
\Delta o_{ihd}^{\mathrm{gate}}
=\frac{1}{R}\sum_r B_{hrd}
\left[\sum_j a_{ijh}t_{ijhr}v_{jhd}\right].
$$

This is an exact factorization of the specified gate, not an approximation to its forward calculation. Rank four does restrict the gate's expressiveness compared with an unrestricted channel-wise network.

Implementation requirements:

- Extract values using the native MHA value-projection weights and bias; do not add a second learned value projection.
- Reuse the **same native post-dropout attention weights** for ordinary values, EV, and gate corrections. Do not draw a second attention-dropout mask or apply another softmax to the gated weights.
- Contract per rank/head using batched matrix multiplication/einsum. Avoid hidden broadcast products with an extra head-dimension axis on both particle axes.
- The explicit coefficient tensor is at most `[batch,8,4,N,N]`; use rank/head chunking or activation recomputation if the measured training peak requires it. Autograd-saved tensors must be included in the memory measurement.
- Combine this correction with the EV correction before the native output projection, excluding an additional output bias. Mask invalid query corrections and preserve the native residual/normalization sequence.

Unlike a scalar head gate, different value channels can change differently. It is still ordinary pairwise attention with a modified message, not grouped or three-way attention.

### 4.4 Combined model

Enable exactly the two modules above:

$$
o_i^{l,h}=\sum_j a_{ij}^{l,h}
\left[
g_{ij}^{l,h}(e_{ij})\odot v_j^{l,h}
+U^{l,h}\left(e_{ij}\odot s^l(c^l)\right)
\right].
$$

Do not feed context into `g` and do not feed context-modulated pair embeddings into its input. Otherwise the combined arm would change more than turning on the two separately tested mechanisms. The modules can still interact through learned particle states and the shared pair stem during training.

### 4.5 Initialization, symmetry, and parameter accounting

Construct the complete existing ParT/EV backbone first. Add extension modules afterward using isolated, deterministic RNG streams keyed by `(seed, layer, mechanism)`. Restore the training RNG after extension initialization. The context module must initialize identically in context-only and combined arms; the gate module must initialize identically in gate-only and combined arms. Shared native parameters and EV projections must match across the appropriate controls.

The pair stem and gate are symmetric under swapping the pair, but the delivered messages need not be: attention weights and sending-particle values are directed. Context pooling must be permutation invariant; particle outputs must be permutation equivariant under consistent input/mask permutations.

Expected counts from the specified dimensions, to verify in executable tests:

| Architecture | Expected trainable parameters | Added over existing EV |
|---|---:|---:|
| Ordinary ParT | 2,143,486 | Not applicable |
| Existing EV | 2,209,022 | 0 |
| EV + context | 2,258,942 | 49,920 |
| EV + relation gate | 2,229,758 | 20,736 |
| EV + both | 2,279,678 | 70,656 |

Context increment: `8 * ((128*32+32) + (32*64+64))`. Gate increment: `8 * ((64*32+32) + 8*4*16)`. AdamW does not alter architecture counts. These seven runs do not isolate capacity from mechanism; a promising result may warrant a later matched-capacity control.

## 5. Exact reduced-budget optimizer recipes

Use the same 200,000 consumed batches and ten checkpoint-validation opportunities for every run. No early stopping and no extra updates for a weak result.

### 5.1 `ranger_short_v1`

- Pinned Weaver RAdam: LR `1e-3`, betas `(0.95,0.999)`, epsilon `1e-5`, weight decay `0`.
- Lookahead: `k=6`, `alpha=0.5`, including correct slow-weight/counter checkpointing.
- No gradient clipping and no added LR warmup.
- Compress the existing fifteen-step late decay by five, independently of validation frequency.

For zero-based step `s` in `0..199999`, set the LR **before** the update:

```python
milestones = tuple(range(140_000, 200_000, 4_000))  # 15 milestones
decays = sum(s >= boundary for boundary in milestones)
lr = 1e-3 * (0.01 ** (decays / 15))
```

Updates 1 through 140,000 use the initial LR. Update 140,001 receives the first decay. Update 196,001 receives the fifteenth decay; the last 4,000 updates use `1e-5`. Keep validation at steps 20,000, 40,000, ..., 200,000; do not conflate LR milestones with ten validation-check indices.

### 5.2 `adamw_short_v1`

- AdamW: peak LR `1e-3`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `1e-4`.
- Decay matrix/convolution weights and EV projection tensors. Exclude biases, normalization parameters, scalar/vector scales, and the class token, using the existing alias-safe parameter grouping.
- No Lookahead.
- Global gradient-norm clipping at `1.0`, after AMP unscale and before the attempted update.
- Linear warmup over 10,000 updates, then cosine decay to `1e-5` at update 200,000.

For one-based update ordinal `t=s+1`:

```python
if t <= 10_000:
    lr = 1e-3 * t / 10_000
else:
    p = (t - 10_000) / 190_000
    lr = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + cos(pi * p))
```

This differs from the Ranger recipe in optimizer, Lookahead, weight decay, betas/epsilon, LR schedule, and clipping. Describe findings as **optimizer-recipe sensitivity**, not an isolated causal effect of AdamW or weight decay.

## 6. Validation and measurements

### 6.1 During training: retain the existing checkpoint-selection protocol

- Use the separate continuous validation stream drawn from the official 5M validation population.
- Evaluate 1,280,000 validation events after each 20,000 training steps, for ten checks.
- Preserve deterministic loader settings and matching validation-stream progression across all seven runs. Record batch/event fingerprints at agreed checkpoints to verify alignment, not merely equal event counts.
- Choose the highest validation-accuracy checkpoint, retaining the earlier checkpoint on an exact tie.
- Record step, processed exposures, accuracy, mean cross-entropy, event/class counts, and validation duration at every check.
- Add macro AUC recording (OVO primary, OVR secondary) at each check. The current trainer does **not** record AUC; this is new instrumentation, not an existing artifact to assume is available. Do not use AUC to select a different checkpoint.
- Calculate AUC over the whole check, not a mean of batch AUCs. Release temporary check predictions after metrics are committed; do not retain all ten sets by default.

The 1.28M events at successive checks need not be identical. Comparing each model's best scalar value can involve different validation subsets. Therefore it is not sufficient as the sole final screen comparison.

### 6.2 Common post-training validation evaluation

After all seven runs finish, freeze their selected checkpoint hashes and evaluate those checkpoints on **all 5,000,000 official validation events exactly once each**, in the same deterministic order.

This is seven evaluation jobs, not seven additional training runs. It gives a common population for the screen's primary comparison. It remains development validation, since the population has been used during checkpoint selection; do not describe it as an untouched final test.

Implement a finite validation reader explicitly. The existing `build_loader('val', ...)` is infinite and shuffled; the existing test driver enforces 20M test-specific names/counts. Do not misuse either by relabeling validation files as test files. Require 500,000 events of each of the ten classes, keep the last partial batch, and use the official preprocessing with native evaluation trimming warmed consistently.

Report:

- Accuracy and cross-entropy.
- Macro AUC OVO, with OVR separately labeled.
- Per-class QCD rejection at signal efficiencies 30% and 50% for Hbb, Hcc, Hgg, H4q, Wqq, Zqq, and Tbqq.
- Hqql at 99% and Tbl at 99.5%, following the existing metric convention.
- Passing QCD counts and achieved signal efficiencies alongside rejection values.

Reuse the definitions in `part_ev_fullscale/metrics.py`: rank by signal-minus-QCD logits, include threshold ties, and calculate population metrics rather than averages over ROOT files. Zero accepted QCD means an undefined finite empirical rejection; retain counts and mark it, never invent a finite value or silently discard it.

Store FP32 logits and labels in a deterministic file/entry order with an identity manifest, using file-wise atomic progress/resume metadata. Seven sets of `[5M,10]` FP32 logits require approximately **1.4 GB decimal** before labels and metadata. A shared identity manifest avoids repeating large string identifiers per model. Do not save all training-batch outputs or generate a new ROOT/token cache.

The 20M test set is not needed, read, uploaded, or queued for this screen. A later confirmatory multi-seed/full-budget experiment can use it under a separately fixed evaluation plan.

## 7. Comparisons and interpretation

Use the common 5M-validation results for the final screen table. Let `A(run)` be accuracy on that population, in percentage units, so subtraction gives percentage points.

| Question | Contrast |
|---|---|
| Existing EV gain under Ranger | `A(EV_RANGER) - A(PART_RANGER)` |
| Jet-context increment | `A(EV_CONTEXT_RANGER) - A(EV_RANGER)` |
| Relation-gate increment | `A(EV_GATE_RANGER) - A(EV_RANGER)` |
| Combined increment | `A(EV_CONTEXT_GATE_RANGER) - A(EV_RANGER)` |
| Context benefit with gating present | `A(EV_CONTEXT_GATE_RANGER) - A(EV_GATE_RANGER)` |
| Gate benefit with context present | `A(EV_CONTEXT_GATE_RANGER) - A(EV_CONTEXT_RANGER)` |
| Factorial interaction on the accuracy scale | `A(EV_CONTEXT_GATE_RANGER) - A(EV_CONTEXT_RANGER) - A(EV_GATE_RANGER) + A(EV_RANGER)` |
| Existing EV gain under AdamW | `A(EV_ADAMW) - A(PART_ADAMW)` |
| Optimizer-recipe effect on the EV gain | `[A(EV_ADAMW)-A(PART_ADAMW)] - [A(EV_RANGER)-A(PART_RANGER)]` |

Repeat relevant contrasts for macro AUC; report rejection values and same-class ratios without treating them as additive accuracy effects. A positive factorial interaction is descriptive on that metric scale, not proof of a mechanism or statistical significance.

The report must include all seven outcomes, including failures, and show:

1. Matched-step validation curves and selected checkpoint steps.
2. The common-population accuracy/AUC table and specified contrasts.
3. Per-class rejection/count tables.
4. Parameter counts, peak GPU memory, sustained throughput, training/evaluation time, and AMP diagnostics.
5. Context/gate summaries: average scale, variation across jets/pairs, saturation frequency, and correction norm relative to ordinary attention output. Collect on a fixed small diagnostic validation sample, not every training step.

One seed provides no estimate of between-seed variation. Event-level precision does not replace seed replication. There is no promised minimum gain or automatic winner threshold: prioritize reproducible-looking accuracy/AUC improvement, useful class-specific rejection changes, stability, and cost together. Expand promising comparisons to matched additional seeds, then the original full exposure budget. Do not claim a shortened-budget ranking establishes the fully trained ranking.

## 8. SPORC environment and scheduling

### 8.1 Platform preparation

The current full-scale worker and environment setup explicitly require Tigris `aarch64`. Changing only `--partition` is insufficient. Add an isolated SPORC worker/environment rather than disabling that protection on historical campaigns.

- Submission host: `sporcsubmit.rc.rit.edu`; training executes on allocated GPU nodes, not the login host.
- Account requested: `reu-aisocial`, verified with live `my-accounts` output.
- Requested partition: `tier3` if still accepted for the account; otherwise use its supported production name `sporc`. RIT documents the rename in its [Slurm guide](https://research-computing.git-pages.rit.edu/docs/slurm_quick_start_tutorial.html). Check live partition/GPU constraints before fixing the launcher settings; do not silently change an explicit partition request.
- Suggested new environment path: `/home/ryreu/miniconda3/envs/part_ev_screen_sporc`. The existing `atlas_kd_sporc` environment is only a possible donor, not evidence of compatibility. Never mutate a donor or an environment serving other jobs.
- Require `x86_64`, CUDA-enabled PyTorch, the pinned Weaver version/source, and official YAML. Resolve compatible dependency builds and freeze exact installed versions before preparing the campaign; `requirements-part-fullscale.txt` alone is not a complete lockfile.
- Do not run ARM binaries or reuse the ARM environment. All seven runs and their evaluations use the same new SPORC environment and chosen GPU type.
- Start by targeting one A100 per run if that is available on the account's production partition. Verify actual GPU memory/driver compatibility. Do not mix A100/H100/etc. within the screen without explicitly revising and documenting the runtime contract.
- No extension compilation or Ninja dependency should be needed for the proposed pure-PyTorch modules. Do not import the old relational-particle-transformer campaign backend or its custom extension pipeline.

Same-platform reference parity, numerical checks, and input correctness are required. Bitwise reproduction of historical Tigris predictions is not a prerequisite for a new, independently controlled SPORC training campaign. If comparing platforms later, label that as a separate portability experiment.

### 8.2 Small checks before production

Keep each pre-production Slurm walltime **under six hours**; target at most two hours per check:

1. CPU preparation: environment/source/data inventory and local semantic tests; two-hour allocation.
2. GPU smoke/benchmark: cover all seven `(architecture, recipe)` tasks with separate temporary models. Use a short benchmark array, each task limited to two hours, rather than one long serial job.
3. Each benchmark measures warmup plus at least 200 timed training steps, forward/backward/optimizer operation, several actual data-fetch windows, and validation/evaluation memory. Include small finite-gradient/numerical checks at representative initial and peak learning rates; optimizer-state maturation is not reproduced by simply assigning a later LR.
4. Benchmark all architectures at full 128-particle length as a memory stress case in addition to the native-trimmed throughput probe. No production weights/checkpoints are created from benchmark models.

A failed or incomplete benchmark cannot authorize training. Frozen pass checks include finite outputs/losses, no unexplained nonfinite gradients, at least one successful optimizer update, and no more than 5% skipped updates in the measured window; document the observed counts even when passing. Passing a short check is not evidence of convergence stability.

Initial resource request for probes: one GPU, eight CPUs, 64 GiB host memory, two data workers. This is a proposal to measure, not a claim it will fit. If memory is insufficient, inspect the peak, fix avoidable intermediates or raise the allocation, then repeat the probes. Do not silently shrink the batch/particle limit or introduce gradient accumulation, which would also affect native BatchNorm behavior.

Set production walltime from measured sustained throughput, ten validation passes including AUC cost, initialization/checkpoint overhead, and a stated safety margin (initial planning margin 30%). Validation and restart I/O can prevent a fivefold exposure reduction from being a fivefold wall-clock reduction. Do not infer SPORC speed from the GH200 measurements.

### 8.3 Job graph and isolation

```text
prepare
   -> benchmark array [0..6]
   -> train array [0..6]
   -> common-validation evaluation array [0..6]
   -> report
```

Use `afterok` dependencies within the SPORC Slurm cluster. Review the benchmark before authorizing/submitting production. Proposed initial training concurrency is **2**; adjust after resource/I/O measurements, without changing model or data recipes. Every task requests one GPU. Additional evaluation/report tasks do not change the count of seven training runs.

Suggested new campaign root:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/ev_context_gate_200k_<UTC_TIMESTAMP>
```

Freeze committed source, the seven-task registry, per-task recipes, dataset manifest, exact environment, GPU/batch policy, and architecture hyperparameters. Use run directories such as `runs/EV_ADAMW/seed_101`; do not key them by architecture alone. Record submissions and prevent duplicate live tasks. Source/recipe changes require a new campaign; unrelated repository pulls must not affect pending/running jobs.

Retain best checkpoints and atomic resumable checkpoints with model, optimizer (including Lookahead state), scaler, model/data RNG state, trimmer counter, consumed training/validation cursors, LR position, and diagnostic counters. Resume the same task through deterministic data replay. Signal-based checkpointing should use the existing safe batch-boundary design. Never delete raw data, another campaign, or checkpoints automatically.

## 9. Recommended code organization

The implementation uses a separate screen package/launcher while reusing tested reference components. The following paths are implemented; use the linked queue guide rather than treating this specification as a shell script:

| Proposed component | Responsibility |
|---|---|
| `part_ev_screen/protocol.py` | Immutable seven-task registry, both shortened schedules, contracts and unique run IDs |
| `part_ev_screen/model.py` | Wrap existing EV; context and rank-four relation-gate modules; counted architectures |
| `part_ev_screen/data.py` | Reuse training/validation streams; explicit finite 5M validation reader and identity manifests |
| `part_ev_screen/reporting.py` | Seven-run results, factorial/optimizer contrasts, progress plots and cost diagnostics |
| `scripts/run_part_ev_screen.py` | Prepare, benchmark, train, evaluate-validation, report phases |
| `scripts/setup_part_ev_screen_sporc_env.sh` | Dedicated x86 environment creation and verification; no in-place donor upgrade |
| `sbatch/run_part_ev_screen_sporc.sh` | Frozen-source x86 GPU worker |
| `sbatch/submit_part_ev_screen_sporc.sh` | Resource checks, dependency graph, source freeze, resume and dry-run behavior |
| `tests/test_part_ev_screen_*.py` | Protocol, math, initialization, loader, checkpoint, launcher and reporting coverage |

Reuse `part_ev_fullscale/optimizer.py`, checkpoint helpers, strict ROOT validation/preprocessing, and population metric definitions where compatible. Build `part` and `part_ev` controls with their existing builders, then extend the EV model for the new arms.

Do **not** monkey-patch `part_ev_fullscale.protocol.RECIPE` globally. Several existing helpers read that constant directly; the old campaign schema also requires a single recipe and a different task list. New screen code must consume explicit per-task recipes. Do not repurpose the existing final-test driver as the screen's validation driver.

Keep the historical full-scale `part`, `part_ev`, plain, AdamW-pilot, and parameter-matched contracts backward compatible. This document does not authorize altering their frozen metadata or source trees.

## 10. Required tests and acceptance checklist

### Mathematical and model tests

- [ ] Context modulation after aggregation matches a small explicitly materialized pair-value reference in outputs and gradients.
- [ ] Rank-four gate contraction matches the explicit `g_ijhd * v_jhd` calculation in outputs and gradients; test nonzero random gates, not only identity initialization.
- [ ] The combined formula matches a direct small-tensor reference.
- [ ] At identity initialization, all three extensions reproduce existing EV with matching weights, masks, and dropout RNG. Test evaluation and training modes using explicit numerical tolerances.
- [ ] Zero EV plus identity value gate reproduces native ParT; nonzero gates still act when EV is disabled in a unit-test fixture.
- [ ] All native initial weights and existing EV projections match the proper controls; extension initialization does not advance the training RNG.
- [ ] Gate/context output stages receive nonzero finite first-step gradients; upstream zero-blocked parameters receive gradients after the output stage changes. No permanently dead branch.
- [ ] Changing context changes EV messages after a nonzero context head is installed; identity initialization must not be mistaken for a disconnected path.
- [ ] Padding does not affect new context/gate calculations; no gate message is sent by padded keys. Do not assert that changing padding leaves native training BatchNorm statistics unchanged, since native pair embedding has its own padding behavior.
- [ ] Permutation tests consistently permute all particle inputs, pair tensors, and masks; check in evaluation mode to avoid confusing random dropout/trimming with a symmetry violation.
- [ ] Value projection includes native biases exactly once; attention dropout realization and output projection are shared correctly.
- [ ] Actual trainable parameter counts match the table; no unused trainable modules are registered.

### Protocol, runtime, and reporting tests

- [ ] Exactly seven unique run IDs, all seed 101; exactly five Ranger and two AdamW tasks.
- [ ] Exactly 200,000 consumed training batches and 102.4M exposures; ten checks at the specified boundaries.
- [ ] LR boundary tests cover `s=0,139999,140000,195999,196000,199999` for Ranger and `t=1,10000,10001,200000` for AdamW.
- [ ] Preserve optimizer grouping, AMP-unscale-before-clipping, correct Lookahead state, and counted skipped updates.
- [ ] Train/validation data stream fingerprints match across architecture and optimizer arms. Resume matches an uninterrupted small reference run including stream position and trimmer state.
- [ ] Finite common validation visits exactly 5M unique file/entry identities, with 500,000 per class and no dropped tail; no test files are accessed.
- [ ] Metric fixtures verify OVO/OVR labeling, tied thresholds, zero-QCD handling, and population rather than batch-averaged AUC.
- [ ] Reports distinguish checkpoint-selection metrics from common-validation metrics, missing/failed runs from zeros, and single-seed evidence from seed statistics.
- [ ] Launcher dry-run creates no jobs/files, validates x86/partition/account/GPU choices, freezes source, constructs only same-cluster dependencies, and blocks duplicate writers.
- [ ] Seven GPU benchmark tasks pass on the chosen SPORC environment before production; measure both native-trimmed throughput and full-length peak memory.
- [ ] Existing full-scale tests still pass if shared code is changed.

## 11. Completion and follow-up

The screen is complete when all seven training runs finish their fixed budget, all seven selected checkpoints are evaluated on the common 5M validation population, and a report contains the registered contrasts, class metrics, and resource/stability measurements.

If a mechanism looks promising, repeat the relevant controls and variants with seeds 202 and 303 before treating small differences as robust. A full-budget follow-up is required to establish that gains persist beyond short-exposure training. If AdamW increases the EV-minus-ParT gap, testing context/gating under AdamW is a subsequent experiment, not an unreported change to this one.

No new-model benefit, novelty claim, or final test improvement is assumed in advance. Background for the unchanged baseline is the [Particle Transformer paper](https://arxiv.org/abs/2202.03772); feature-wise/global conditioning has established antecedents linked above. The contribution being screened is the utility and interaction of these particular mechanisms in this controlled ParT setting.
