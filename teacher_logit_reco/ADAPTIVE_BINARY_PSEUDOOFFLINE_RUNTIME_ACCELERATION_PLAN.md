# Adaptive Binary Pseudo-Offline Runtime Acceleration Plan

## Status

This document is the implementation plan for reducing the wall-clock latency of the
Adaptive Binary Pseudo-Offline Hierarchy (ABPH) reconstructors described in
`ADAPTIVE_BINARY_PSEUDOOFFLINE_HIERARCHY_PLAN.md`.

The scientific plan remains authoritative for the model, objectives, campaign variants,
data boundaries, and final-claim policy. This document changes the training runtime, not
the scientific question.

The immediate target is the HLT-v2 realistic-degradation campaign at strength `2.5`, with:

- pilot: `500k model_train`, `150k model_val`;
- high data: `5M model_train`, `1M model_val`;
- full rollout validation at the existing cadence;
- the same HLT evidence, offline hierarchy targets, reconstruction losses, and model
  capacities as the scientific plan.

The primary objective is to shorten the elapsed time of each individual reconstructor.
The campaign has enough GPUs to keep independent variants independent and concurrent, so
deduplicating work between separate variants is not the priority.

This plan contains two kinds of changes and reports them separately:

- runtime-equivalent changes: redundant-forward removal, exact-effective-batch execution,
  input staging, and distributed optimization/validation;
- an intentionally non-performance-neutral accelerated screening schedule, which changes
  update exposure and the cosine learning-rate trajectory so the pilot answers whether the
  mechanism has useful signal sooner.

The screening schedule is acceptable for mechanism discovery, but a schedule-truncated
result is not evidence that the architecture cannot work. Final performance claims require
the convergence and extension checks in this document.

## Executive Decision

Implement the acceleration in four layers:

1. Replace the accidentally enormous update ceilings with campaign-aware nominal budgets
   and convergence-gated extensions.
2. Remove redundant decoder and renderer forwards while preserving exactly the active
   loss terms and gradients.
3. Improve per-rank throughput with larger phase-specific microbatches, asynchronous
   target prefetch, pinned CPU tensors, and nonblocking host-to-device transfer.
4. Train each reconstructor with four-GPU distributed data parallelism on Tigris, normally
   as four nodes with one GH200 rank per node.

Layer 1 is a deliberate training-policy change. Layers 2-4 are required to preserve the
same global objective up to expected BF16/DDP numerical variation. Runtime reports must
attribute speedup from fewer updates separately from speedup per optimizer update.

Do not reduce validation coverage or frequency. Do not freeze and cache the HLT encoder,
increase learning rates aggressively, reduce hierarchy depth, reduce the number of
hypotheses, weaken matching/projection iterations, or change the model architecture in
the first acceleration campaign.

## Why the Current Runtime Is So Large

The present defaults are maximum optimizer-update counts:

| Stage | Current maximum updates | Effective batch |
|---|---:|---:|
| root | 150,000 | 1,024 |
| each hierarchy depth | 80,000 | 1,024 |
| renderer | 200,000 | 512 |
| distribution | 200,000 | 512 |

For the `500k` pilot, one effective data pass is approximately:

- `500000 / 1024 = 488.3` root/hierarchy updates;
- `500000 / 512 = 976.6` renderer/distribution updates.

The current ceilings therefore permit approximately:

| Stage | Approximate pilot data passes |
|---|---:|
| root | 307 |
| each hierarchy depth | 164 |
| renderer | 205 |
| distribution | 205 |

Those ceilings are not a useful performance-first training schedule. They are large
emergency ceilings that can keep a steadily improving run alive for days or weeks. The
three-day Slurm limit can expire before the intended curriculum finishes.

There is also avoidable work inside an update:

- phase-2 teacher-forced updates construct a rollout even when no rollout loss is active;
- phase 4 renders hypothesis zero once for particle supervision and then reconstructs and
  renders it again while producing the multi-hypothesis distribution sample;
- target shards are loaded synchronously and CPU batches are copied to CUDA without a
  bounded prefetch/pinned-memory path;
- `distributed_world_size` currently participates only in the effective-batch assertion;
  there is no process group, rank-sharded input stream, gradient synchronization, or
  distributed validation.

## Locked Scientific Invariants

Every optimized run must preserve all of the following.

### Model and objective invariants

- The exact resolved variant configuration and parameterization are unchanged.
- The HLT ParT evidence encoder remains trainable according to the existing curriculum.
- Root, hierarchy, rollout-frontier, particle, distribution, calibration, and auxiliary
  losses retain the weights in the scientific plan.
- The constrained compiler and accounting projections remain in the graph.
- The hierarchy remains `1 -> 2 -> 4 -> 8 -> 16 -> 32` where required by the variant.
- Multi-hypothesis models retain the same hypothesis count, latent dimension, and fixed
  evaluation seeds.
- Renderer phase-space and type-assignment iterations are not reduced.
- Exact per-terminal-group PID quota transport, charge projection, N-body projection,
  Hungarian matching, and unbalanced OT may be batched by compatible cardinality. This
  is an execution transformation only: iteration counts, hard assignments, null
  accounting, closure tolerances, and differentiable losses remain unchanged.
- Production training need not materialize per-group diagnostic assignment objects when
  the same aggregate losses, method counts, and scientific report fields are retained.
- BF16 autocast, AdamW group policies, EMA decay, gradient clipping, and the effective
  global batch contracts remain unchanged.

### Validation and selection invariants

- Evaluate the complete `model_val` split every `2,000` global optimizer updates and at
  every curriculum transition.
- Pilot validation must report exactly `150,000` unique jets; high-data validation must
  report exactly `1,000,000` unique jets.
- Validation remains zero-teacher-forcing rollout validation.
- Checkpoint selection remains `model_val.rollout.loss.total`.
- Validation objective weights must match training's resolved loss contract.
- No final-test data or teacher logits may enter reconstructor training or selection.

### Campaign and provenance invariants

- Each variant and seed remains an independently trained run.
- Existing HLT-cache, target-cache, manifest, jet-identity, source-commit, and resolved
  variant hashes remain mandatory.
- Parallelism, world size, rank layout, local batch sizes, accumulation counts, runtime
  contract version, and schedule profile must be written into every config, checkpoint,
  run report, and submission manifest.
- A checkpoint from a different world size or schedule contract is not silently resumed.

## Explicit Non-Goals

The first implementation will not:

- reduce model-validation size or run validation less often;
- replace full validation with a proxy subset;
- increase the new-module learning rate above `3e-4`;
- cache a frozen HLT representation, because the HLT encoder is intentionally updated in
  every phase (`1.0x` in phase 1 and `0.25x` later);
- share checkpoints between scientifically independent variants;
- reduce particle count, hierarchy depth, renderer quality, or stochastic views;
- use final-test results to set runtime or convergence policy;
- enable `torch.compile` or experimental attention kernels in the first production pass.

Those changes may be studied later, but they are not assumed to be performance-neutral.

## Target Runtime Outcome

The expected improvements are deliberately expressed as ranges until the instrumentation
in Step 1 measures the real phase breakdown on Tigris.

| Source of improvement | Expected affected-stage speedup |
|---|---:|
| four-GPU distributed execution | `2.2x-3.6x` |
| accelerated screening exposure | `5x-20x` versus current hard ceilings |
| redundant-forward removal | `1.2x-1.5x` |
| larger microbatch and lower accumulation | `1.2x-2.0x` |
| prefetch/pinned transfer | `1.1x-1.3x` when input-bound |

These factors do not multiply perfectly because validation, synchronization, checkpoint
I/O, and short stages have fixed costs. The practical target is:

- `8x-20x` lower wall time than a realistic current run that early-stops;
- `20x-50x` lower wall time than exhausting the current maximum budgets;
- root-only B variants in roughly `3-10` hours;
- shallow hierarchy variants in roughly `6-20` hours;
- deep hierarchy/renderer/distribution variants in roughly `12-48` hours.

These are engineering targets, not promises. A production reconstructor that still
projects beyond 48 hours after the optimized timing preflight must not simply receive a
larger wall-time request; its measured bottleneck must be reviewed first.

## 1. Runtime Instrumentation and Reference Benchmark

Add a low-overhead timing contract before changing execution.

### Required timing buckets

Record CUDA-event or synchronized wall timings for:

- target-source wait and shard decompression;
- CPU batch assembly;
- pinned-memory staging;
- host-to-device transfer;
- HLT encoding/root compilation;
- teacher-forced hierarchy decoding;
- rollout hierarchy decoding;
- particle rendering and projection;
- matching/loss construction;
- backward pass;
- gradient synchronization;
- optimizer/EMA update;
- full validation;
- checkpoint serialization.

Also record peak allocated and reserved CUDA memory, local batch size, accumulation count,
rank/world size, jets per second, updates per hour, and validation jets per second.

### Output

Write `runtime_profile.json` beside `training_curves.json`. The file must contain:

- the runtime contract version;
- per-stage totals and medians after excluding the first five warm-up updates;
- measured and projected stage completion time;
- GPU memory high-water marks;
- data-wait fraction and communication fraction;
- the exact Slurm allocation and source/config hashes.

Timing must be disabled or sampled sparsely outside a benchmark window so instrumentation
does not become a new bottleneck.

### Reference benchmark

Before optimization, run a fixed 20-update benchmark for one representative root variant
and one deepest renderer/distribution variant on one GH200. Run exactly one fixed,
identity-attested 4,096-jet model-val timing evaluation at update 20. Curriculum
transitions in this timing-only harness reset optimizer/EMA handoff state but do not run
checkpoint-selection validation. The same fixed identities and deterministic seeds are
reused for the DDP4 parity benchmark. This bounded timing evaluation is not a scientific
selection result: every production training run retains complete model-val evaluation at
every real checkpoint-selection boundary.

The deepest single-rank profiler-overhead controls must run sequentially inside the same
Slurm allocation and record an identical job id, host, node list, and matched-pair id.
Comparing independent nodes is forbidden because node and filesystem variation cannot be
attributed to instrumentation. The benchmark deliberately samples every eligible update,
while production samples one update in every 100. Raw dense overhead remains a diagnostic;
the blocking `3%` production-overhead gate uses the pre-registered dense-overhead
projection divided by that immutable production sample interval. Metric/checkpoint parity
and complete timing coverage remain independently blocking.

## 2. Accelerated Screening Curriculum Budgets

This section is not an exact runtime-equivalence change. Reducing a root stage from
`150k` to `12k` updates both reduces optimizer exposure and compresses the cosine schedule:
the old `5%` warmup is `7,500` updates, while the new warmup is `600` updates. The selected
model can therefore differ even when every distributed/runtime implementation detail is
correct.

That trade is intentional for the first screening pilot. The purpose is to determine
whether the hierarchy/reconstruction mechanism shows meaningful model-validation and
tagging signal without waiting for the old emergency ceilings. Reports and comparisons
must call this policy `accelerated_screening_v1`, never `performance_neutral`.

Replace one undifferentiated maximum with three values per stage:

- `nominal_updates`: the serious intended training exposure;
- `extension_updates`: the size of one low-LR continuation block;
- `hard_max_updates`: the fail-closed absolute ceiling.

The cosine schedule reaches its minimum learning-rate fraction at `nominal_updates` and
stays at that minimum during extensions. It must not restart or jump at an extension.

### Pilot profile

| Stage | Nominal | Extension block | Hard maximum | Approx. nominal passes |
|---|---:|---:|---:|---:|
| root | 12,000 | 4,000 | 24,000 | 24.6 |
| each hierarchy depth | 8,000 | 4,000 | 16,000 | 16.4 |
| renderer | 15,000 | 5,000 | 30,000 | 15.4 |
| distribution | 10,000 | 5,000 | 20,000 | 10.2 |

### High-data profile

| Stage | Nominal | Extension block | Hard maximum | Approx. nominal passes |
|---|---:|---:|---:|---:|
| root | 50,000 | 10,000 | 80,000 | 10.2 |
| each hierarchy depth | 40,000 | 10,000 | 60,000 | 8.2 |
| renderer | 80,000 | 20,000 | 120,000 | 8.2 |
| distribution | 50,000 | 10,000 | 80,000 | 5.1 |

The high-data profile intentionally uses fewer passes because each pass contains ten times
as many distinct jets. It still exposes each stage to substantially more unique training
examples than the pilot.

These budgets apply only to a stage that the resolved variant actually trains. Existing
warm-start semantics remain intact: when a C-tier run inherits a selected root, or a D-tier
run inherits selected root/hierarchy modules, its one-update handoff/contract check is not
expanded into a new full root or hierarchy phase. The run report must identify every
stage as `trained`, `warm_started_handoff`, `disabled`, or `oracle` so runtime accounting
cannot make inherited training look like newly optimized exposure.

### Extension rule

At nominal completion, continue by one extension block only when all conditions hold:

1. the best checkpoint occurred in one of the last two full validation evaluations;
2. the three-evaluation robust slope of `model_val.rollout.loss.total` is negative by at
   least `0.2%` of the current score per evaluation interval;
3. no monitored required objective has degraded by more than `5%` from its best value;
4. nonfinite-skip and compiler-failure limits remain clean;
5. the hard maximum has not been reached.

Re-evaluate the same rule after each extension block. Stop and restore the best model-val
checkpoint when the rule fails. This is a deterministic continuation rule, not a
hyperparameter sweep.

The report must distinguish `nominal_completed`, `extended_for_convergence`,
`plateau_stopped`, and `hard_max_reached`. Reaching the hard maximum while the score is
still materially improving sets `schedule_truncated=true`, makes a negative mechanism
conclusion invalid, and blocks automatic high-data promotion.

### Configuration changes

Extend `ReconstructorCurriculumConfig` with explicit per-stage nominal, extension, and hard
maximum values. Retain a compatibility reader for old checkpoints, but never silently
reinterpret an old schedule as the new runtime contract. The training entry point chooses
the pilot or high-data profile from the campaign's immutable split metadata rather than a
free-form operator label.

### Controlled schedule/runtime checks

Before the full optimized pilot, run two representative comparisons:

1. one root variant with the identical shortened schedule and identical global batches on
   one GPU and DDP4;
2. one deepest renderer/distribution variant with the identical shortened schedule and
   identical global batches on one GPU and DDP4.

These comparisons isolate runtime parallelism from the schedule change. Their validation
trajectories need only agree within predeclared BF16/DDP stochastic tolerances; throughput
must be measured independently.

For each representative variant, also run one conservative extension-cap check. Compare
the nominal selected checkpoint with the best checkpoint available at the extension cap.
The decision is pre-registered and deterministic:

- define relative reconstruction improvement as
  `(nominal_best_loss - extension_best_loss) / nominal_best_loss`;
- `model_val.rollout.loss.total` materially improves when that value is at least `0.005`
  (`0.5%` relative);
- evaluate the nominal and extension checkpoints with the same frozen downstream tagger
  recipe, data split, initialization seed, and training budget;
- define tagging gain as model-val accuracy minus the matched A0 HLT ParT model-val
  accuracy;
- classify `positive_signal` when gain is at least `+0.002` absolute accuracy,
  `negative_signal` when gain is at most `-0.002`, and `no_clear_signal` otherwise;
- the tagging conclusion changes when nominal and extension checkpoints occupy different
  categories.

If either the `0.5%` reconstruction threshold is met or the tagging category changes, the
nominal schedule is marked `schedule_truncated=true` and the affected family uses the
extension cap for the screening campaign. Store all raw values, thresholds, categories,
and matched A0 artifact hashes in the comparison report. This is a two-model safety check,
not a broad tuning sweep.

## 3. Exact Redundant-Forward Removal

### Teacher-forced hierarchy updates

Today phase 2 always computes both:

- a teacher-forced hierarchy for group/topology supervision; and
- a rollout hierarchy for frontier alignment.

When `context.mode == "teacher_forced"`, phase 2 does not activate the rollout-frontier
term. In that case, do not construct the unused rollout. Produce only the supervised
decoder output and the required teacher-forced losses.

Compute the rollout when any of these is true:

- `context.mode == "rollout"`;
- the renderer or distribution phase is active;
- an oracle/control variant explicitly requires rollout state;
- validation is active, since validation is always rollout mode.

The required-loss contract must assert that no omitted tensor was needed by the composed
loss.

### Distribution-phase hypothesis zero reuse

Phase 4 currently:

1. rolls out and renders hypothesis zero for particle matching;
2. calls the all-hypothesis deployment path;
3. rolls out and renders hypothesis zero again.

Refactor deployment into:

- `prepare_shared_reconstruction_forward(...)`;
- `rollout_and_render_hypothesis_zero(...)`;
- `rollout_and_render_additional_hypotheses(..., start_index=1)`;
- an assembly function that returns the same ordered multi-hypothesis output schema.

The particle loss and distribution loss must consume the same hypothesis-zero object.
No second HLT encode, root prediction, root compilation, hierarchy-zero rollout, or
hypothesis-zero render may occur in one optimizer microstep.

The reused hypothesis zero must carry the same fixed zero/reference latent and hypothesis
identity that the current all-hypothesis deployment path assigns to index zero. Additional
hypotheses begin at index one and retain their existing deterministic seeds and ordering.

### Equivalence requirements

For fixed weights, inputs, seeds, and dropout state:

- every active loss term must match the reference path within BF16 tolerance;
- gradients for every active parameter group must match within tolerance;
- the compiled root and rendered hypothesis zero must be object-identical within the new
  path, not merely numerically similar;
- call-count tests must prove that omitted forwards do not execute.

## 4. Phase-Specific Microbatch Calibration

Preserve the global effective batches:

- root/hierarchy: `1,024` jets;
- renderer/distribution: `512` jets.

For world size `W`, local batch `B`, and accumulation `A`:

`effective_batch = W * B * A`.

The default four-GPU target is:

| Stage | World size | Local batch/rank | Accumulation | Global effective batch |
|---|---:|---:|---:|---:|
| root/hierarchy | 4 | 128 | 2 | 1,024 |
| renderer/distribution | 4 | 128 | 1 | 512 |

Because renderer variants can have different memory peaks, add a separate memory preflight
that tests complete forward/backward/optimizer steps, not forward inference alone. It tries
candidate local batches in descending order:

- root/hierarchy: `256`, `128`, `64`;
- renderer/distribution: `128`, `64`, `32`.

Production approval requires this preflight to run under the actual requested DDP
topology, normally four Slurm nodes and four NCCL ranks. The measurement must include:

- initialized NCCL process groups and DDP gradient buckets;
- the stage-appropriate active parameter groups and `find_unused_parameters` behavior;
- accumulated gradients, unscaled/clipped gradients, and materialized AdamW state after
  at least one optimizer step;
- online and EMA model states;
- two populated rank-local prefetch buffers and pinned-memory staging;
- the largest applicable hypothesis/rendering path for that phase.

A single-GPU memory preflight may be used for debugging but cannot approve a DDP4
production batch contract.

Select the largest exact-divisor batch that leaves at least `15%` of device memory free at
the measured peak. Write the selected values to a hash-bound
`runtime_batch_contract.json` before training. Production training reads that immutable
contract; it does not catch an OOM and silently change batch size mid-run.

If no candidate satisfies memory and exact effective-batch constraints, preflight fails.

## 5. Rank-Sharded Target Streaming and Transfer Pipeline

### Distributed training stream

Extend `AdaptiveBinaryTargetBatchSource` with `rank` and `world_size` while preserving one
deterministic global sample stream.

For every global microbatch window of `world_size * local_batch_size`:

- all ranks advance the same logical shard/cursor state;
- rank `r` materializes only its non-overlapping slice;
- shard-tail stitching remains active;
- no rank loads or discards another rank's examples;
- the union of rank-local identities equals the global window exactly once.

Shuffling remains deterministic from campaign seed, epoch, and grouping. Checkpoint state
stores the global logical cursor, shard order, epoch, and per-rank batch counters.

Implement an immutable `GlobalBatchPlan` for every optimizer microstep. Its canonical
payload contains:

- runtime contract, split, grouping, epoch, global update, and accumulation index;
- global window start/stop and expected jet count;
- for every rank, an ordered tuple of `(shard_id, local_start, local_stop)` slices;
- the expected rank-local jet count and ordered jet-identity hash;
- the full-window identity hash and plan content hash.

Every rank independently derives the same plan, all-gathers the plan content hash, and
fails before loading data if the hashes differ. A rank materializes only its declared
slices. The union of rank slices must have no overlap, must equal the full global window,
and must preserve shard-tail stitching. Checkpoints persist the last completed plan and
the exact next-plan cursor, so resume cannot repeat or skip an update window. The training
curve records the plan hash for every sampled/profiled update and every checkpoint.

### Distributed validation stream

Partition model validation deterministically into disjoint contiguous index ranges. Each
jet appears on exactly one rank. Unequal final batch counts are permitted because
validation uses the raw model without gradient synchronization inside its batch loop.

Identity coverage uses a canonical range contract rather than attempting to reduce hashes.
For rank `r`, define immutable `[start, stop)` boundaries before evaluation. The rank:

1. verifies that its observed ordered jet identities exactly equal the active cache's
   expected identities for that range;
2. computes `range_hash` from the canonical ordered identities plus split/start/stop;
3. reports `(rank, start, stop, n_jets, range_hash)` to rank zero.

Rank zero sorts rows by `(start, stop, rank)`, verifies complete contiguous coverage with
no overlap or gap, serializes the ordered rows canonically, and hashes that list into
`validation_coverage_hash`. The report stores both the ordered rows and aggregate hash.
The expected ordered-row hash is independently derived from the immutable active cache;
it must match before the validation result is checkpoint-selection eligible. Per-rank
hashes are never added, XORed, or recursively combined as a substitute for this contract.

### Prefetch and pinned memory

Add a bounded CPU-only prefetcher with queue depth two:

- one background worker prepares the next rank-local batch;
- target-shard hash verification remains mandatory;
- nested target tensors are converted to contiguous CPU tensors and pinned;
- transfer uses `non_blocking=True` on a dedicated CUDA stream;
- the compute stream waits on a transfer event immediately before use;
- at most two prefetched batches may be resident per rank.

Exceptions in the worker must be re-raised in the training thread with shard/split/rank
context. Prefetch must preserve exact batch order and resume position.

### Optional cache-format follow-up

If profiling still shows more than `20%` of update time waiting on target decompression,
add a hash-bound training sidecar format with contiguous uncompressed `.npy` arrays per
shard. Do not make this format a prerequisite until the profiler demonstrates that the
bounded prefetcher is insufficient.

## 6. Four-GPU Distributed Training Runtime

### Process topology

On Tigris, use four Slurm nodes with one GH200 process per node unless cluster inspection
shows a supported four-GPU single-node allocation:

```text
nodes=4
ntasks=4
ntasks_per_node=1
gres=gpu:gh200:1 per node
cpus_per_task=16 per rank
account=reu-aisocial
partition=tigris
```

Launch one Python rank per Slurm task with `srun`. Python derives:

- global rank from `RANK` or `SLURM_PROCID`;
- world size from `WORLD_SIZE` or `SLURM_NTASKS`;
- local rank from `LOCAL_RANK` or `SLURM_LOCALID`.

The worker sets `MASTER_ADDR` from the first allocated hostname and a deterministic,
collision-resistant `MASTER_PORT` derived from the Slurm job ID. Use NCCL and enable
asynchronous error handling. `PYTHONNOUSERSITE=1` remains mandatory.

### DDP integration

Add a small `ReconstructorTrainingModule(torch.nn.Module)` whose `forward(batch, context)`
calls the typed `reconstructor_step` and composes the active reconstruction objective
inside `forward`. Wrap that module with `DistributedDataParallel` so the complete
loss-producing forward is visible to DDP.

The DDP boundary must not return `ReconstructorStepResult` or another custom dataclass.
Return a standard nested mapping/tuple structure whose differentiable leaves are tensors:

```text
{
  "total_loss": Tensor,
  "raw_loss_terms": {loss_name: Tensor},
  "weighted_loss_terms": {loss_name: Tensor},
  "finite_check_tensors": (Tensor, ...),
  "batch_size_tensor": Tensor,
}
```

`total_loss` is the only tensor used for the primary backward call. The active raw and
weighted terms remain reachable from the returned standard container so
`find_unused_parameters=True` can traverse the graph correctly and objective-gradient
diagnostics can be computed without calling a second model forward. Non-tensor metrics
are converted to detached CPU metadata only after the DDP-visible tensor output has been
received. Tests must fail if a custom result object crosses the DDP boundary.

The curriculum changes `requires_grad` membership between stages. Therefore:

1. configure phase trainability before wrapping;
2. build the DDP wrapper at the start of each curriculum stage;
3. rebuild it only at a stage transition;
4. use `find_unused_parameters=True` for controls whose resolved path intentionally omits
   a nominally active module;
5. set `broadcast_buffers=False` so an exceptional rank-local model forward cannot strand
   peers in a forward-time buffer broadcast; the ABPH training wrapper must not rely on
   mutable synchronized buffers;
6. use `no_sync()` for every gradient-accumulation microstep except the last.

Do not call reconstructor methods outside the DDP forward for a trainable loss path. That
would bypass reducer bookkeeping.

### Synchronized update semantics

- Each rank processes distinct examples.
- Gradients are averaged by DDP once per global optimizer update.
- Gradient clipping is applied after synchronized, unscaled gradients.
- Every rank performs the same optimizer and EMA update.
- Dropout/random sampling uses `base_seed + rank` for training, while fixed evaluation
  hypotheses retain their locked campaign seeds.

The effective-batch assertion uses observed all-rank jet counts, not the configured world
size alone.

### Ordered failure and nonfinite protocol

No rank may enter a synchronized backward until every rank has completed a successful,
finite forward for that accumulation microstep.

Use this exact order:

1. Every rank prepares its declared batch-plan slice and catches source, identity,
   host-to-device, or input-schema failures. All ranks execute a pre-forward
   `all_reduce(MIN)` readiness flag. No rank calls DDP forward unless every rank is ready.
2. Every ready rank calls the DDP wrapper. It catches compiler, projection, model-forward,
   and loss-composition failures and records `local_forward_ok=0` rather than immediately
   raising.
3. Successful ranks verify `total_loss` and every required finite-check tensor, then set
   `local_forward_ok=1` only when all are finite.
4. All ranks execute an `all_reduce(MIN)` on the forward-success flag before backward.
5. If the global flag is zero, every rank clears gradients and skips the microstep/update
   together. Gather bounded error summaries to rank zero. Structural errors are re-raised
   synchronously on all ranks; allowed nonfinite skips increment the same counter on all
   ranks. Before retrying, every rank enters a barrier, synchronizes its CUDA stream,
   releases its reference to the old DDP wrapper on the same control-flow branch, and
   enters a second barrier. All ranks then rebuild the stage wrapper collectively from the
   unchanged raw model/optimizer state, verify a common parameter-state hash, and enter a
   final barrier before fetching the next batch. Reducer state from an exceptional forward
   is never reused.
6. Only when the global forward flag is one do all ranks enter DDP backward.
7. After the final accumulated backward, unscale gradients locally, compute a local
   gradient-present-and-finite flag, and `all_reduce(MIN)` it.
8. If the global gradient flag is zero, every rank clears gradients and skips the optimizer
   and EMA update. Otherwise all ranks clip, step, update EMA, and advance curriculum.

No checkpoint, evaluation, transition, or source-cursor commit occurs for a globally
skipped update. Injected-failure tests must prove that a failure before backward and a
nonfinite gradient after backward both terminate or recover without a collective hang.

## 7. Full Distributed Validation, EMA, and Checkpointing

### Validation aggregation

All four ranks evaluate their disjoint portion of the full model-validation split under
the EMA weights. Do not flatten every scalar diagnostic and multiply it by `n_jets`.
Define an explicit reduction schema for each reported value:

- `mean`: additive numerator and denominator;
- `sum`: additive value with no division;
- `count`: integer additive count;
- `ratio`: separately reduced numerator and denominator;
- `non_additive_summary`: diagnostic-only values gathered for a declared summary and
  forbidden from checkpoint selection.

Preserve the current selection semantics exactly by accumulating
`selection_numerator += composed_total_loss * batch_size` and
`selection_denominator += batch_size` on each batch, then all-reducing both values. Each
required raw and weighted loss term receives its own explicit numerator/denominator based
on its evaluator contract. Group-normalized, count, calibration, quantile, and ratio
diagnostics may not masquerade as per-jet means.

Rank zero computes final values from the reduced components and broadcasts the selection
decision. The report records each metric's reduction kind and denominator. Only metrics
with a reviewed additive selection contract may participate in checkpoint selection.

Fail validation unless:

- the reduced jet count equals the immutable split size;
- the reduced identity coverage hash/count contract is complete;
- every required metric is finite;
- every rank used zero teacher forcing and the same objective weights;
- selection numerator/denominator and required-loss reduction schemas are present;
- no non-additive diagnostic is marked checkpoint-selection eligible.

This retains full validation every `2,000` global updates while reducing its wall time.

### Checkpoint ownership

Only rank zero writes JSON and checkpoint files. All ranks enter a barrier after each
atomic write before continuing or transitioning stages.

The checkpoint contains:

- online and EMA model states;
- optimizer/scaler/controller states;
- global source cursor and all rank-local source counters;
- per-rank Python/NumPy/Torch/CUDA RNG states;
- world size and rank topology;
- nominal/extension/hard schedule state;
- DDP/runtime contract version;
- batch-contract and runtime-profile hashes.

On resume, rank zero validates and broadcasts the payload. Each rank restores only its own
RNG/source state. A changed world size, batch contract, active cache identity, or schedule
contract is a hard error unless a separately implemented and tested elastic-resume
migration is explicitly requested. Elastic migration is not part of this plan.

## 8. Slurm and Campaign-Orchestration Changes

Extend `SlurmResourceProfile`/`SlurmJobSpec` with explicit distributed fields:

- `nodes`;
- `ntasks`;
- `ntasks_per_node`;
- `gpus_per_node`;
- `distributed_world_size`;
- `launcher`.

Only reconstructor-training jobs in tiers B, C, and D receive the four-GPU profile.
Baselines, taggers, predictions, fusion, diagnostics, and reports keep their current
profiles unless separately justified. This plan addresses reconstructor latency.

The submitter must include the distributed resource request in the saved command manifest
and fail if the worker's observed world size differs from the requested world size.

Add two queue profiles:

- `ABPH_RECONSTRUCTOR_PARALLELISM=single`: one-GPU smoke/debug compatibility;
- `ABPH_RECONSTRUCTOR_PARALLELISM=ddp4`: required production profile.

Production pilot submission defaults to `ddp4` only after the distributed parity and
resume tests pass. High-data submission remains gated by a successful optimized pilot
report and must not be promoted merely because the jobs ran faster.

DDP4 also requires a measured throughput promotion gate on the representative deepest
renderer/distribution stage:

- end-to-end training throughput speedup over the identical one-GPU shortened schedule
  must be at least `1.8x`;
- gradient-synchronization/collective time must remain below `35%` of steady-state update
  time;
- full distributed validation must be faster than one-GPU validation and preserve its
  exact selection result within tolerance;
- no rank may exceed the memory headroom contract.

If the gate fails, production remains on the fastest validated single-GPU profile while
the profiler is used to evaluate DDP2, larger local batches, or communication changes.
The submitter must not default to DDP4 merely because four GPUs were allocated.

## 9. Numerical and Scientific Equivalence Tests

### Unit tests

Add tests for:

- pilot/high-data nominal, extension, and hard-cap schedules;
- extension and plateau decisions from synthetic validation histories;
- exact `0.5%` reconstruction-improvement threshold boundary cases;
- `+0.002/-0.002` tagging-category boundaries and category-change decisions;
- cosine decay reaching its minimum at nominal completion and not restarting;
- exact global effective batch arithmetic for every supported local batch/world size;
- rejection of a single-GPU batch preflight as DDP4 production approval;
- DDP4 memory preflight including optimizer state, gradient buckets, and full prefetch
  occupancy;
- canonical per-update global batch plans with matching all-rank hashes;
- no-overlap/full-union rank-sharded training windows across shard boundaries;
- checkpoint/resume at a shard tail reproducing the exact next batch plan;
- full validation partition coverage, including uneven final batches;
- canonical ordered validation-range hashes and rejection of reordered, overlapping,
  missing, or identity-tampered ranges;
- explicit validation numerator/denominator, sum, count, ratio, and non-additive schemas;
- rejection of a non-additive diagnostic as a checkpoint-selection metric;
- recursive pinning/nonblocking movement of every target tensor;
- teacher-forced phase-2 rollout call elimination;
- phase-4 hypothesis-zero rollout/render reuse;
- standard tensor-only DDP forward output traversal;
- pre-backward forward-failure consensus and post-backward gradient-finite consensus;
- injected forward/compiler failure without a DDP collective hang;
- synchronized old-wrapper teardown, parameter-hash agreement, and collective DDP rebuild;
- rank-zero-only writes and checkpoint barriers;
- resume with all rank RNG/source states;
- hard failure on world-size or runtime-contract mismatch.

### One-GPU versus four-GPU parity smoke

Using the same small immutable sample and dropout disabled:

1. run one global update on one GPU;
2. run the same global batch partitioned over four ranks;
3. compare every raw/weighted loss, active-group gradient, post-step parameter, EMA value,
   and validation aggregate.

Use tight FP32 tolerances for accounting/compiler state and documented BF16 tolerances for
network outputs/gradients. The test must compare the global batch identities exactly.

### Runtime-equivalence and screening smoke

Train one root and one deep renderer/distribution variant with the identical shortened
schedule and fixed global batch plan on one GPU and DDP4. Separately continue each
representative DDP4 run to its conservative extension cap. The optimized path must:

- produce a statistically compatible validation trajectory;
- select a checkpoint with no required-objective regression beyond the predeclared
  stochastic tolerance;
- preserve all compiler and provenance audits;
- show the expected forward-call reduction;
- satisfy the DDP4 `1.8x` throughput and communication-fraction promotion gate;
- show whether nominal completion or the conservative extension changes the mechanism or
  tagging conclusion.

## 10. Rollout Sequence and Acceptance Gates

### Step 10.1: Instrument only

Implement timing without changing schedules or execution. Capture the one-GPU reference
profiles and archive them under `audits/runtime_reference/`.

Acceptance:

- target timing overhead below `3%` outside the sampled benchmark window;
- a hard operational ceiling of `10%` in the deliberately dense, fully instrumented
  20-update reference; missing the `3%` target is recorded but does not block promotion
  when timing coverage and metric/checkpoint parity pass;
- complete phase and validation timing coverage;
- no metric/checkpoint changes.

### Step 10.2: Forward and input-path optimization

Implement redundant-forward removal, microbatch preflight, prefetch, pinning, and
nonblocking transfer on one GPU.

Acceptance:

- numerical/gradient parity tests pass;
- no reduced validation coverage;
- at least `1.3x` measured training throughput on the deep reference variant, or a profiler
  explanation identifies why the expected bottleneck was absent.

### Step 10.3: DDP smoke

Run the one-versus-four-rank parity smoke, a full `150k` distributed model validation, and
a checkpoint/resume test across multiple nodes. Run the same shortened root and deep
schedule on one GPU and DDP4 so schedule effects are held constant.

Acceptance:

- exact, disjoint training identities;
- exactly `150,000` validation identities;
- finite synchronized gradients and identical post-update parameters across ranks;
- successful rank-zero checkpoint and four-rank resume;
- no orphaned or hanging ranks after injected worker failure;
- standard tensor-mapping DDP output passes unused-parameter traversal;
- explicit selection numerators/denominators reproduce the one-GPU validation result;
- representative deep-stage throughput is at least `1.8x` one GPU with communication
  below `35%` of steady-state update time.

### Step 10.4: Accelerated representative pilot

Run one root variant and one deepest renderer/distribution variant with the pilot schedule,
then continue both to their conservative extension cap for the truncation check.

Acceptance:

- full validation every `2,000` global updates and transitions;
- convergence decision recorded and reproducible;
- nominal-versus-extension conclusion recorded without using final test;
- best model-val checkpoint not forced by a wall-time timeout;
- projected wall time below 48 hours for the deepest production reconstructor;
- no scientific/provenance warning in either run report.

### Step 10.5: Full optimized pilot campaign

Queue all planned pilot variants concurrently under the reviewed four-GPU reconstructor
profile. Do not mix old single-GPU runtime checkpoints into new optimized runs.

Acceptance:

- all required variants complete or fail closed with an explicit cause;
- report provenance agrees on runtime, input, HLT, target, and schedule contracts;
- no variant reaches a still-improving hard maximum without a promotion-blocking warning;
- model-selection conclusions use only model validation.

### Step 10.6: High-data approval

Promote only after the pilot report demonstrates:

- stable optimization and compiler behavior;
- plausible reconstruction/rollout metrics;
- no throughput or distributed correctness anomaly;
- no indication that the nominal schedule truncates meaningful improvement.

The high-data schedule is fixed before queueing. Final-test evaluation remains a separate
approved/frozen stage under the scientific plan.

## Implementation Steps

### Step 1: Add runtime telemetry and reference benchmarking

Implement `runtime_profile.json`, sampled timing, memory telemetry, and deterministic
reference benchmark tooling. Add tests that telemetry is complete and does not change the
training result.

### Step 2: Implement campaign-aware convergence schedules

Add nominal/extension/hard budgets, pilot/high-data profiles, non-restarting low-LR
extensions, deterministic continuation decisions, explicit `accelerated_screening_v1`
labeling, truncation warnings, report fields, and checkpoint state.

### Step 3: Eliminate redundant hierarchy and renderer forwards

Skip inactive rollout work in teacher-forced phase 2 and reuse the exact hypothesis-zero
rollout/render in phase 4. Add call-count, loss-parity, and gradient-parity tests.

Replace sequential per-jet/per-terminal-group numerical solves with batched exact solves.
The acceptance benchmark must time every actual D1 curriculum stage with zero profiler
warmup for one-update handoffs. Its wall-time projection is the sum of the real D1 stage
graph (`1` root handoff, five `1`-update hierarchy handoffs, renderer nominal updates,
and distribution nominal updates), plus scaled complete-model-validation events.
Applying one deepest-stage average to a generic stage-budget sum is forbidden.

### Step 4: Implement phase-specific batch preflight

Add full-step memory calibration, exact global-batch arithmetic, immutable batch-contract
artifacts, and fail-closed production loading.

### Step 5: Implement rank-sharded target sources

Add deterministic non-overlapping distributed training windows, disjoint validation
partitions, canonical per-update rank slice plans, all-rank plan-hash agreement,
shard-tail handling, and resumable global cursor/next-plan state.

### Step 6: Add asynchronous prefetch and transfer staging

Implement bounded background batch preparation, recursive pinned memory, dedicated CUDA
transfer stream, nonblocking copies, and worker-error propagation.

### Step 7: Implement four-GPU distributed optimization

Add process-group initialization, stage-aware DDP wrapping, accumulation with `no_sync`,
standard tensor-mapping forward outputs, ordered pre-backward/post-backward consensus,
gradient clipping, optimizer steps, EMA, and distributed runtime metadata.

### Step 8: Implement distributed validation and checkpointing

Add full-split typed numerator/denominator reduction, non-additive diagnostic boundaries,
coverage checks, rank-zero atomic writes, all-rank barriers, per-rank RNG/source state, and
fail-closed resume validation.

### Step 9: Extend Tigris Slurm orchestration

Add four-node reconstructor resources, `srun` launch, account/partition handling, saved
parallelism manifests, single-GPU debug mode, and queue-graph tests.

### Step 10: Run parity, performance, and campaign acceptance gates

Execute the fixed one-versus-four-GPU parity suite, reference throughput comparison,
same-schedule root/deep comparisons, conservative extension checks, DDP4 `1.8x` promotion
gate, full pilot, and only then the approved high-data campaign.

## Files Expected to Change

Primary implementation targets:

- `teacher_logit_reco/adaptive_binary_pseudooffline/training.py`
- `teacher_logit_reco/adaptive_binary_pseudooffline/production.py`
- `teacher_logit_reco/adaptive_binary_pseudooffline/orchestration.py`
- `scripts/train_adaptive_binary_pseudooffline_variant.py`
- `scripts/submit_adaptive_binary_pseudooffline.py`
- `sbatch/run_adaptive_binary_variant.sh`
- `sbatch/submit_adaptive_binary_pseudooffline_tigris.sh`

Likely new modules/scripts:

- `teacher_logit_reco/adaptive_binary_pseudooffline/distributed.py`
- `teacher_logit_reco/adaptive_binary_pseudooffline/runtime_profile.py`
- `scripts/audit_adaptive_binary_runtime.py`

Tests:

- `tests/test_adaptive_binary_pseudooffline_production.py`
- `tests/test_adaptive_binary_pseudooffline_step8_training.py`
- `tests/test_adaptive_binary_pseudooffline_step12_slurm.py`
- new focused distributed/runtime tests where isolation improves clarity.

## Definition of Done

This acceleration plan is implemented only when:

1. all locked scientific invariants are machine-checked;
2. full validation coverage and cadence are unchanged;
3. one-GPU/four-GPU global-update parity passes;
4. the DDP boundary returns a standard differentiable tensor mapping and ordered failure
   consensus cannot deadlock before backward;
5. distributed validation uses explicit reviewed reduction schemas and reproduces the
   one-GPU selection result;
6. distributed checkpoint/resume is deterministic and fail-closed;
7. redundant forward call counts and gradient parity are tested;
8. every training rank sees its audited disjoint slice of the persisted global batch plan
   and the correct global effective batch;
9. DDP4 exceeds the `1.8x` representative deep-stage promotion floor with acceptable
   communication overhead;
10. nominal and extension-cap representative runs establish whether the screening policy
   is truncating improvement using the pre-registered `0.5%` reconstruction and
   `+0.002/-0.002` tagging-category rules;
11. the representative deep pilot completes without a wall-time extension and projects
   below 48 hours;
12. runtime/provenance contracts appear in checkpoints, reports, and submission manifests;
13. DDP4 batch approval includes production-topology gradient buckets, optimizer/EMA
   state, and full prefetch occupancy;
14. validation coverage is attested by canonical ordered range rows and their expected
   active-cache-derived aggregate hash;
15. the complete adaptive-binary test suite, Python compilation, shell syntax checks, and
   `git diff --check` pass;
16. high-data and final-test gates from the scientific plan remain intact.
