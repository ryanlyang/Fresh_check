# Seven-run full-schedule ParT / State-EV campaign on Tigris

Plan dated 2026-11-05. **Planning document only: no new models, launchers, jobs,
or benchmark results are created by this document.**

Implementation status (2026-11-05): the seven model definitions and benchmark-only
Tigris workflow now live in [part_selected_fullscale](../part_selected_fullscale/README.md).
All seven parameter counts below have been confirmed by construction, including
the two originally starred compositions. Remote numerical/performance results,
full production orchestration, and final reused-control comparability audits
remain pending. The new launcher cannot submit production training.

## 1. Agreed scope

Train seven new seed-101 models on Tigris, with one GH200 per training run and
the full Ranger training schedule. Reuse the completed baseline shared-bias
ParT result; do not retrain it. Reuse the separately queued standard-relation,
layerwise all-eight State-EV campaign when its results become available.

The scientific questions are:

1. Can ordinary ParT capacity account for the gains of State-EV?
2. How do shared and layerwise attention bias affect State-EV?
3. What does state conditioning add beyond ordinary EV with layerwise bias?
4. Does the PT/TRACK/REGION relation setup help the baseline and the proposed
   model, and does State-EV still help against a selected-relations capacity control?

The third capacity control **also has selected relations**, but retains shared
attention bias, ordinary ParT value messages, and no state conditioning. Thus
it is a wider version of Selected ParT, not a standard-input model with a larger
parameter budget and not a layerwise-bias model.

The latest decision is to benchmark the selected models before choosing their
production walltimes. This supersedes the earlier suggestion of proceeding
with startup checks alone for those runs. There are now **three selected models
to benchmark**, not two, because the selected capacity control is an additional
selected workload. Benchmark updates do not count as production training.

## 2. Exactly seven new training runs

The IDs below are implemented in the new benchmark campaign. Production training
submission remains pending benchmark review. All runs use seed 101 and fresh
model/optimizer state.

| Index | Proposed run ID | Relations | Attention bias | Value mechanism | Particle FFN width | Expected parameters |
|---:|---|---|---|---|---:|---:|
| 0 | `PART_MATCH_STATE_SHARED` | Standard | Shared | Ordinary ParT; no EV/state | 599 | 2,323,750 |
| 1 | `PART_MATCH_STATE_LAYERWISE` | Standard | Shared | Ordinary ParT; no EV/state | 601 | 2,327,894 |
| 2 | `SELECTED_PART_MATCH_STATE_LAYERWISE` | Selected | Shared | Ordinary ParT; no EV/state | 601 | 2,337,814* |
| 3 | `STATE_EV_SHARED` | Standard | Shared | Optimized State-EV, all eight blocks | 512 | 2,323,710 |
| 4 | `EV_LAYERWISE` | Standard | Layerwise | Ordinary EV; no state | 512 | 2,212,774 |
| 5 | `SELECTED_PART_SHARED` | Selected | Shared | Ordinary ParT; no EV/state | 512 | 2,153,406 |
| 6 | `SELECTED_STATE_EV_LAYERWISE` | Selected | Layerwise | Optimized State-EV, all eight blocks | 512 | 2,337,382* |

Standard means the four native ParT pair descriptors. Selected means those same
four descriptors plus the learned PT, TRACK, and REGION embeddings. The particle
embedding still receives its original 17 features.

Shared/layerwise describes **scalar attention-bias heads**, not the EV value
projections. EV projections remain independent per particle block. All State-EV
arms use all eight particle blocks, state width 32, and the existing optimized
query-chunk size 64; they are not late-four or rank-16 variants.

Counts have now been confirmed by constructing all seven models. Asterisks mark
the two compositions that were initially estimates; their implemented counts
match those estimates, using the existing 9,920-parameter selected-relation
addition with no extra adapters. Preparation rechecks every count before
freezing the campaign. If a composition changes that accounting, recompute its
capacity target and document the change **before training**, not after inspecting
results.

### Capacity-matching rule

Use the nearest uniform integer hidden width in the eight particle FFNs. Keep
the native Linear/LayerNorm/Linear structure, eight particle blocks, class FFNs
at width 512, attention heads, embeddings, and classifier unchanged. Do not add
unused parameters just to manufacture an exact count.

For the pinned backbone, increasing each particle FFN by one unit adds:

```text
8 blocks * (128 input weights + 1 bias + 2 LayerNorm parameters + 128 output weights)
    = 2,072 parameters.

P_standard(w) = 2,143,486 + 2,072 * (w - 512)
P_selected(w) = 2,153,406 + 2,072 * (w - 512)
```

| Capacity control | Matching target | Target count | Control count | Control minus target |
|---|---|---:|---:|---:|
| Standard shared ParT, width 599 | Standard shared all-eight State-EV | 2,323,710 | 2,323,750 | +40 (+0.0017%) |
| Standard shared ParT, width 601 | Standard layerwise all-eight State-EV | 2,327,462 | 2,327,894 | +432 (+0.0186%) |
| Selected shared ParT, width 601 | Selected layerwise all-eight State-EV | 2,337,382* | 2,337,814* | +432 (+0.0185%) |

These are **near-parameter-matched controls**, not exactly equal counts or
compute-matched controls. Parameter location and FLOPs differ. The two standard
capacity controls are deliberately very similar (width 599 versus 601); retain
both as requested, but do not expect that small width difference itself to be a
large scientific effect.

Sources: [native FFN widening/count rule](../part_ev_fullscale/model.py),
[existing State-EV and width-601 control](../part_ev_state_compare/model.py),
[selected counts](../part_selected_screen/protocol.py), and
[ordinary layerwise EV](../part_layerwise_screen/model.py).

## 3. Reused controls: no extra baseline training

| Reference | Role | Treatment |
|---|---|---|
| Standard shared ParT, seed 101, full schedule | Baseline for ordinary capacity and selected-relation gains | Reuse completed Tigris training and available evaluation artifacts |
| Standard shared ordinary EV, seed 101, full schedule | Context for ordinary EV and layerwise EV effects | Reuse if its compatible full-schedule artifacts are available; do not launch another training |
| Standard layerwise all-eight State-EV, seed 101, full schedule | Target for run 1 and direct comparison for runs 3, 4, and 6 | Reuse the independently queued one-GH200 Tigris campaign |

Known starting locations to inspect, not substitutes for an artifact audit:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/part_fullscale/part_ev_100m_20261013T044756Z
/home/ryreu/atlas/Fresh_check/checkpoints/part_fullscale/part_ev_100m_20261013T044756Z_validation_20261018T224308Z
/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_tigris_segments/benchmark_723341
```

Resolve models by task, recipe, seed, checkpoint identity, and population rather
than by directory name alone. The first campaign uses its older `part`/`part_ev`
run layout; build a read-only adapter instead of assuming the new run-ID layout.
Validate the full update/check budget, accuracy-selection rule, data inventory,
preprocessing, normalization where applicable, precision, and software/runtime
compatibility. Record hashes of the selected checkpoints and metric artifacts.

The selected screens and their SPORC donors are useful implementation references,
but their **200k-step results are not full-schedule controls**. Likewise, the
two-A100 SyncBatchNorm State-EV run is not a runtime-matched replacement for the
one-GH200 target. Do not silently mix them or choose the nicer accuracy.

Reuse metrics only when they refer to the same full-5M validation population and
accuracy-selected checkpoint. If a required evaluation is missing, report it as
pending and identify the evaluation-only work needed; do not automatically rerun
baseline training or inference. If compatibility cannot be established, disclose
the mismatch and resolve it before calling a comparison matched.

Do not cancel or alter the queued Tigris/SPORC State-EV chains, selected recovery
jobs, or their frozen source. New training need not wait for a reused control to
finish, but final contrasts must wait for its compatible evaluation artifacts.

## 4. Fixed scientific recipe

| Setting | Decision |
|---|---|
| Seed | 101 for each of the seven new runs |
| Data | Official JetClass 100M training pool; 5M validation population |
| Training budget | 1,000,000 attempted updates at global batch 512: 512M exposures |
| Validation during training | 50 checks, one per 20,000 updates; 1.28M streaming validation events per check |
| Optimizer | Existing pinned RAdam + Lookahead implementation, called Ranger here |
| RAdam settings | LR starts at `1e-3`; betas `(0.95, 0.999)`; epsilon `1e-5`; weight decay 0 |
| Lookahead | `k=6`, `alpha=0.5`, including restartable slow weights and counter |
| LR schedule | Full reference schedule: 15 decays at zero-based steps 700,000, 720,000, ..., 980,000, reaching `1e-5` |
| Other optimization | No new warmup, clipping, AdamW, accumulation, or early stopping |
| Hardware | One GH200; single-GPU native BatchNorm; no DDP/SyncBatchNorm |
| Numerical policy | Existing FP16 training/FP32 master weights and GradScaler; FP32 validation; TF32 off |
| Data behavior | Native particle permutation/trimming, at most 128 particles, original loader ordering and two workers |
| Checkpoint selection | Highest check-validation accuracy; first exact tie retained |
| Final evaluation | Accuracy-selected checkpoint on all 5M validation jets |

Use the full schedule in [part_ev_fullscale/protocol.py](../part_ev_fullscale/protocol.py),
with explicit architecture metadata and single-GPU settings. Do not accidentally
inherit `ranger_short_v1` from a selected-screen launcher, the width-544 ordinary-EV
capacity recipe, or the default three-seed task list.

The first decay affects update 700,001 because LR is set from the zero-indexed
step before the update. The budget corresponds to about 5.12 pool-size exposures;
the stream does not imply exact coverage once per epoch. The 50 checks are not
50 full dataset passes and are not 50 independent validation populations.

Freeze the existing FP16 scaler policy; log scale changes and skipped updates.
Expected startup calibration skips are not permission to ignore sustained
nonfinite gradients or count skipped updates as successful optimizer steps.

Reuse the existing ARM environment if it satisfies the selected backend and
baseline contracts. Build native extensions for that environment without
upgrading packages used by running jobs. Any environment incompatibility needs
an explicit decision, not an invisible switch to a different numerical setup.

The independent 20M test set is out of scope. No test-based configuration or
checkpoint selection, downloads, or inference are included in this plan.

## 5. Comparisons and interpretation

| Comparison | Question / limitation |
|---|---|
| Run 3 vs reused baseline ParT | Gain from shared-bias all-eight State-EV |
| Run 3 vs run 0 | State-EV versus similar parameter budget in ordinary FFNs, both shared bias |
| Reused layerwise State-EV vs run 1 | Layerwise State-EV package versus a shared-bias capacity control |
| Reused layerwise State-EV vs run 3 | Layerwise versus shared attention bias with State-EV fixed |
| Reused layerwise State-EV vs run 4 | State conditioning added to ordinary EV with layerwise bias fixed; parameter count also increases |
| Run 5 vs reused baseline ParT | Effect of the selected-relation setup on ordinary shared-bias ParT |
| Run 6 vs reused layerwise State-EV | Effect of the selected-relation setup on layerwise State-EV |
| Run 6 vs run 2 | Selected layerwise State-EV versus selected shared-bias ParT at similar total capacity |
| Run 2 vs run 5 | Effect of extra FFN capacity with selected relations fixed |

Runs 1 and 2 intentionally keep shared bias even though their matching targets
use layerwise bias. Their comparisons test the complete mechanism against
ordinary extra capacity, not state conditioning alone. No selected layerwise
ordinary-EV run is added in this seven-run budget, so there is no direct
state-only selected-relations ablation here.

The selected implementation also changes pair processing from native symmetric
triangular packing to dense directed pairs and the associated pair BatchNorm
population. Describe gains as effects of the **selected-relation setup**, not
proof that a particular added physical descriptor alone caused them.

These are single-seed comparisons. Report all seven outcomes, including negative
ones, without claiming seed-to-seed reproducibility or significance from point
estimates alone. Subsequent seeds are a separate decision, not part of this queue.

## 6. Selected model integration

Use [SELECTED_RELATIONS_GUIDE.md](SELECTED_RELATIONS_GUIDE.md) as the code map.
Preserve the existing definitions, corrected REGION common-ancestor lookup,
track-validity masks, and training-only normalization.

- PT contributes 8 learned channels; TRACK 12; REGION 12. Together with the four
  native descriptors, the pair stem receives 36 channels and remains width 64.
- Transport the four raw track measurements alongside the native loader inputs,
  but pass only the original 17 features into the particle embedding.
- Build relations on the actual post-trimming particle view. Do not replace its
  tree with a precomputed untrimmed-jet tree.
- Reuse and verify the selected reference's training-only normalization if its
  schema and training inventory match. Those constants were fitted on 50,000
  training-stream examples, not validation data and not only 50,000 training
  examples for the classifier. Do not fit a different normalization per arm.
- All three selected arms must use the same relation encoders, constants, dense
  pair normalization policy, backend, and execution flags.

Run 6 must feed the enriched 64-wide stem to **both** the layerwise bias heads
and the optimized State-EV mechanism. It is not enough to supply selected
features only to attention bias while State-EV continues using a separate
standard-only representation. Keep the relation networks and State-EV jointly
trainable, with shared-per-forward pair normalization and correctly cleared
state caches. Backward/checkpoint recomputation must use its explicit saved
tensors, not mutable context cleared after forward.

Construct run 2 from the no-EV, shared-bias selected architecture and widen only
its particle FFNs. The extra budget must not silently widen relation encoders,
add EV branches, duplicate bias heads, or register unused State-EV parameters.

For a zero-state-branch parity check, the new selected State-EV model should
reduce to the existing selected layerwise **ordinary EV** model, not to bare
ParT. Also test nonzero state outputs and gradients through all eight branches;
zero-initialized output projections alone cannot validate the new computation.

## 7. Execution optimizations to start with

The following are implementation candidates, not promised performance gains:

1. Retain the existing memory-efficient selected path: one packed EV stem,
   no unused EV message for bias-only models, sequential raw-family processing,
   and the existing AMP-safe TRACK concatenation handling.
2. Use the separate batched, parallel CPU REGION builder/direct-packing candidate
   from [part_selected_fast](../part_selected_fast/README.md), after same-runtime
   correctness checks. It preserves independent-jet order and per-tree arithmetic.
3. Begin with four tree threads, two loader workers, and sufficient allocated
   CPUs. Cache and `torch.compile` remain off initially.
4. Keep the existing all-eight State-EV optimizations: pair-normalization reuse
   and query-chunk activation checkpointing. Do not reduce to four blocks or
   rank 16 as a hidden runtime fix.

The batched backend still transfers data between CPU and GPU. Dense TRACK and
REGION arithmetic, learned pair encoders, memory traffic, State-EV forward/backward,
and validation remain real costs. A small parameter increase does not establish
a small runtime increase.

The current fast launcher is for four **short-screen** models and has its own
paired/sustained admission gates. Reuse its model/backend/testing components in
a separate full-schedule campaign; do not weaken its old approval checks or
relabel old benchmarks as evidence for a newly combined selected State-EV model.

## 8. Required selected timing benchmarks

### Three workloads, not a proxy from one

Benchmark runs **2, 5, and 6** separately on one GH200 with production settings.
Run 2's widened backbone changes its cost even though its relation construction
matches run 5. Run 6 adds state computation and a distinct memory/backward path.
Historical A100 or 200k-run walltimes cannot substitute for these measurements.

Initial selected resource candidate: **one GH200, 12 CPUs, 128 GiB host RAM,
four tree threads, two data workers**, with cache and compilation off. Verify
visible device memory and actual Slurm/cgroup host limits, not the GPU product
name alone. A thread-count/resource change is a new execution configuration.

Benchmark and preparation jobs stay **strictly below six hours**, normally at
most five hours each, with internal shutdown allowance. They do not train the
production checkpoints. If a required probe cannot finish within its allocation,
record incomplete evidence and stop admission; do not mark a partial run passed
or silently grant a multi-day diagnostic. Further optimization or a revised,
explicitly documented measurement design is then needed.

### Measurements and correctness

- Before timing, verify native-backend versus batched-backend tree topology,
  ties, masks, packed data, and thread-count determinism on the same ARM runtime.
- Check the three actual selected models, including the widened FFNs and the
  nonzero State-EV integration: FP32/FP16 outputs, gradients, BatchNorm buffers,
  trimming, AMP behavior, and optimizer/checkpoint restoration. Retain existing
  numerical tolerances rather than choosing new ones after seeing failures.
- Compare the candidate with the corresponding reference execution on matched
  real batches and model states. Use repeated, alternating-order timing where
  practical. Report actual speed ratios; do not claim a speedup from source
  changes or warm-cache synthetic timings alone.
- Target the existing selected sustained measurement: 20 warmup updates, then
  5,000 measured updates in five 1,000-update windows. Keep fingerprints/sample
  order, record cold startup, and time loader plus forward/backward/optimizer.
- Include full-128-particle memory/backward checks, bounded AMP calibration,
  and subsequent stable updates under the established selected numerical policy.
  The measured skip fraction must remain below 1%; the full-length confirmation
  must pass its no-skip/stability checks. Report calibration skips separately.
- Measure FP32 validation on at least 131,072 events and record both inference
  and metric-aggregation costs. Checkpoint validation and final full validation
  must include macro-AUC calculation, not just forward throughput.
- Measure restart-checkpoint writes and actual checkpoint sizes. Record host
  process/worker RSS and GPU allocated **and reserved** memory throughout phases;
  retain at least the established 10% memory headroom.
- Profile REGION CPU construction/packing, transfers, PT/TRACK/REGION GPU work,
  pair stem, attention/State-EV, and loader wait. Repeat throughput timing without
  detailed profiling overhead; overlapping spans must not be summed as walltime.

Extrapolation from a small validation sample is approximate, especially for AUC
aggregation. When a projected full job approaches the 20-day limit or timing is
unstable, require a longer validation timing sample, ideally one full 1.28M-event
check, plus additional sustained evidence before approval. Keep that diagnostic
bounded rather than claiming the small-sample projection is a guarantee.

The four standard-relation runs need their own bounded model/startup/memory and
restart checks, but not the selected backend's large profiling workflow. Existing
GH200 evidence can inform initial resource estimates only where the configuration
matches; it does not replace checks of a new width or wrapper. Record early
production throughput for every run.

## 9. Walltime decision, including the 20-day ceiling

Estimate the **full million-update run**, not five times a screen's reported GPU
kernel time. Let `t_step` be the slowest sustained native-stream window's seconds
per update, `t_val` a conservative estimate of one 1.28M-event check, and `t_save`
the measured restart-save time:

```text
T_training = T_startup
           + 1,000,000 * t_step
           + 50 * t_val
           + N_saves * t_save
           + measured best-weight-write / finalization allowance

recommended_hours = ceil(1.30 * T_training / 3600)
```

Count writes from the implemented loop: regular latest-state checkpoints,
pre/post-validation saves, and finalization. Until that is finalized, conservatively
budget at least `total_steps / 1000 + 3 * checks + 2` restart-save equivalents and
up to 50 selected-weight writes. Include compilation/build/startup when performed
inside the allocation; list work done by a separate CPU preparation job separately.
Do not double-count time already included in an end-to-end measured window.

Queue delays are excluded. Initial restart replay is zero for fresh runs; a later
resume needs an updated estimate that includes replay. The 30% margin is a
planning allowance, not a statistical confidence bound or protection against
arbitrary contention. Check actual throughput after jobs start concurrently.

| Workload | Planned production allocation |
|---|---|
| Four standard-relation runs | One uninterrupted 216-hour / nine-day job each, provided its evidence supports that limit |
| Three selected-relation runs | One uninterrupted job each; request `max(216, recommended_hours)`, subject to verified partition/account/QOS limits |
| Final full-5M evaluation | Separate dependent job per new model, with its own measured walltime and memory estimate |

The supplied Tigris partition output on 2026-11-05 gives `MaxTime=20-00:00:00`
(480 hours). Re-query the partition and effective account/QOS limits at submission;
the actual allowance may be lower. If a selected run's margin-inclusive estimate
exceeds that allowance, **do not queue it as if it fits**. Profile/optimize and
remeasure. If it still does not fit, ask for a new scheduling/resource decision;
do not silently shorten training, switch hardware, or introduce segmentation.

For scale, 512M exposures alone require about 296 training jets/s to fit 20 days
with zero validation/startup overhead. With 30% headroom, that floor is about
385 jets/s before those other costs. Passing a training-only throughput threshold
therefore does not prove the complete run fits.

The November 10, 08:00-20:00 maintenance reservation shown in the supplied output
prevents a new nine-day request from fitting before it, if unchanged. Record it
as a dated scheduling observation, not a promised post-maintenance start time.

### If the selected runtime remains too high

Use the profile to choose one next change, then recheck and remeasure:

1. Tune the bounded CPU tree thread count and CPU allocation; avoid oversubscribing
   loader and native-library threads.
2. Address measured GPU feature/packing overhead. TRACK fusion/reassociation and
   batched bias heads are candidates, not established equivalent improvements.
3. Test optional REGION compilation or validation-only exact-view caching only
   if their startup cost, hit rate, memory, and real-stream timing justify them.
4. Consider CPU/GPU overlap only with a demonstrated preservation of post-trim
   inputs, RNG order, and restart behavior.

No frozen learned embeddings, approximated trees, deleted relation families,
smaller batch, altered trimming, fewer state blocks, changed precision, or changed
training/validation budget is authorized as an optimization. A scientific change
needs a separate decision. Freeze one admitted selected execution configuration
for all three selected runs before their fresh production starts.

## 10. Submission and recovery design

Implement a new campaign namespace/source snapshot. The workflow should support
benchmark-only submission first, followed by production after resource review:

```text
new campaign preparation + read-only control/normalization registry
    -> three selected correctness/timing benchmarks
    -> selected resource review and frozen execution decision
    -> seven new full-schedule trainings (one uninterrupted allocation each)
    -> checkpoint-completion/selection checks
    -> seven full-5M validation evaluations
    -> comparison report using available, compatible reused controls
```

Different model walltimes require explicit per-run submission or resource-homogeneous
groups, not a single array forced to reserve the slowest model's time for everyone.
Concurrency is an operational setting to confirm at submission, not seven GPUs
per run. It must respect account limits, filesystem contention, and free space.

If desired later, queue downstream jobs in advance behind verified checks using
dependencies; the default benchmark command must not unexpectedly submit full
training. Standard runs may be submitted separately once ready, but no selected
training starts before its own new-model admission. Do not automatically cancel
unrelated jobs or existing controls. No concrete new queue command exists yet.

Each production job performs bounded startup checks and then begins automatically;
there is no manual intervention halfway through a successful job. Probes use
disposable weights/optimizer/data iterators, and production starts from the frozen
seed and stream rather than continuing diagnostic updates.

Keep atomic latest-training-state saves at least every 1,000 updates and around
validation, plus accuracy-selected weights. Preserve optimizer/Lookahead state,
GradScaler, counters, RNG/trimmer state, and stream cursors. Save best candidates
transactionally so interruption cannot corrupt the selection referenced by the
latest restart state. Resume from latest state, never from best weights alone.

Uninterrupted allocation does not mean disabling checkpoints. Provide safe,
explicit resume after interruption; no automatic chain of three-day jobs is
introduced. Data replay and incomplete-validation replay must not duplicate
optimizer updates or reset the LR schedule. Genuine errors remain failures and
block downstream evaluation.

Reuse the staged dataset in place. Do not copy ROOT data, clone environments,
create persistent full-population pair/tree caches, or delete existing artifacts.
Estimate storage from measured checkpoint sizes, atomic replacement headroom,
bounded selected-weight versions, seven evaluation outputs, and concurrency.
Check available storage without changing unrelated files.

## 11. Implementation map and validation checklist

Build in this order:

1. **Models and counts:** explicit widths 599/601, selected capacity control,
   ordinary layerwise EV, and selected layerwise State-EV composition. Use explicit
   per-model configuration rather than mutating historical global width constants.
2. **Full-schedule protocol:** exactly seven seed-101 tasks, correct recipe,
   selected normalization/backend contract, and a read-only existing-control registry.
3. **Selected benchmark path:** same-runtime parity, memory, timing, attribution,
   full-job estimates, and the 480-hour-or-lower admission decision.
4. **Training/evaluation orchestration:** immutable source, per-run walltimes,
   checkpoint/restart, progress/ETA reporting, full-5M metrics, and duplicate-job guards.
5. **Tests and handoff:** local model/protocol/launcher tests, explicit CUDA/native
   tests on Tigris, then reviewed dry-run and queue instructions.

| Existing code | Reuse / adaptation |
|---|---|
| [Native ParT and FFN widening](../part_ev_fullscale/model.py) | Baseline-compatible capacity controls |
| [Full training recipe](../part_ev_fullscale/protocol.py) and [optimizer](../part_ev_fullscale/optimizer.py) | Million-update Ranger schedule and complete optimizer restart |
| [State-EV implementation](../part_ev_message_study/model.py), [optimized state implementation](../part_ev_state_speed/model.py) | All-eight state messages and tested same-math acceleration |
| [Layerwise/state composition](../part_ev_state_compare/model.py) | Standard shared/layerwise State-EV and capacity checks |
| [Layerwise bias wrappers](../part_layerwise_screen/model.py) | Ordinary layerwise EV and scalar bias-head semantics |
| [Selected model](../part_selected_screen/model.py), [features](../part_selected_screen/features.py), [normalization](../part_selected_screen/normalization.py) | Selected inputs, encoders, masks, and frozen training-only constants |
| [Fast selected backend/model](../part_selected_fast/model.py), [trees](../part_selected_fast/trees.py), [parity](../part_selected_fast/parity.py) | Batched trees and equivalent-execution checks |
| [Fast profiling driver](../scripts/profile_part_selected_fast.py) | Paired component/end-to-end timing; adapt to the three actual selected models |
| [Tigris State-EV benchmark](../part_ev_state_tigris/benchmark.py) | Single-GH200 runtime checks and full-budget timing conventions |
| [Tigris continuation engine](../part_ev_tigris_segments/engine.py) | Native full-budget restart/selection logic; do not reuse its fixed 72-hour allocation policy for this campaign |

Acceptance checklist for the future implementation:

- [ ] Exactly seven fresh tasks; no automatic baseline or existing State-EV rerun.
- [ ] Actual parameter counts, active modules, shared-bias control heads, FFN widths,
  and selected count deltas asserted; no registered-but-unused capacity padding.
- [ ] Selected State-EV zero/nonzero-branch forward/backward, all-eight gradients,
  masking, trimming, cache lifetime, and dense-pair normalization tested.
- [ ] Frozen parameter/normalization/source/environment and dataset identities;
  no changes to current campaigns, controls, or user work.
- [ ] Native/fast tree and model equivalence verified on the relevant ARM/GH200 runtime.
- [ ] Three complete selected timing/resource reports; whole-job recommendations
  include 50 validations and fit the requested walltime and effective maximum.
- [ ] LR milestone boundaries, AMP handling, Lookahead, mid-validation recovery,
  full completion, and first-best-check selection tested.
- [ ] Fake-Slurm tests cover model/resource mappings, dependencies, partial submissions,
  failed-check blocking, no duplicate submissions, and read-only dry-run/status.
- [ ] No production launched by benchmark-only mode; no partial benchmark promoted
  to a pass; no hidden optimization or shortened scientific budget.
- [ ] Final handoff includes measured selected runtimes, unresolved limitations,
  exact proposed walltimes, and real commands only after launchers exist.

## 12. Reporting

Report all new models and compatible controls with parameter counts, validation
accuracy, macro OVO AUC, and deltas for the contrasts above. Show learning curves
at equal update counts; do not compare one run's final accuracy to another run's
unfinished best-so-far value without labeling that difference.

Use final full-5M metrics for the main comparison, not a mixture of 1.28M
checkpoint-validation and full-5M numbers. Also report QCD rejection at 50% and
30% signal efficiency for the seven hadronic signal classes, Hqql at 99%, and
Tbl at 99.5%, with accepted/total QCD counts. Zero QCD acceptance has no finite
empirical rejection; retain the count rather than inventing a finite gain.

Include measured training/evaluation walltime, GPU-hours, CPU/RAM allocation,
peak memory, and actual throughput. Keep GPU-resident inference benchmarks
separate from end-to-end selected inference including REGION CPU work/transfers.
Parameter-matched does not mean compute-matched.

Full-5M validation overlaps checkpoint selection and is not an independent test.
No accuracy retention, speedup, or scientific gain is established by this plan;
those are outcomes to measure after implementation and execution.
