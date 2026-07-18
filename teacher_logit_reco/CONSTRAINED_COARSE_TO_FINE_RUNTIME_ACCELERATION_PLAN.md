# Constrained Coarse-To-Fine Runtime Acceleration Plan

## Purpose

This plan accelerates the complete constrained coarse-to-fine (C2F)
pseudo-offline campaign without reducing its scientific scope.

The full pilot and the full high-data campaign retain:

```text
HLT profile: fixed_hlt_v2_realistic
HLT degradation strength: 2.5
all B-tier hierarchy reconstructors
all C-tier particle-slot reconstructors
all planned tagger, prediction, fusion, report, and approval stages
the original split memberships and provenance rules
the deployable HLT-only final-test path
```

This is not a plan to remove difficult runs, turn controls into different
models, weaken matching, or use fewer campaign families. The purpose is to
make each existing run converge in much less wall time while preserving the
objective that makes the comparison scientifically meaningful.

## Why This Is Necessary

The current C-tier pilot measurements show that the execution profile, rather
than a lack of available memory, is the limiting factor:

```text
single-view C variants: about 21 to 30 jets/s
C6 four-view variant: about 13 jets/s
C4 Hungarian variant: about 6 jets/s
```

At 500k model-train jets and 150k model-val jets, a 30-epoch pilot can take
many days for ordinary C variants and weeks for C4. The high-data campaign
would be correspondingly impractical.

The current run profile has several properties that make acceleration
plausible:

```text
model forward: FP32 only
loss and matching: FP32
single-view physical batch size: 16
C6 physical batch size: 8
single-view peak reserved GPU memory: generally below 10 GB
C6 peak reserved GPU memory: about 16 GB
available accelerator class: GH200
```

The plan therefore targets precision, physical batch utilization, input
overlap, exact hard-matching parallelism, and optimization convergence. It
does not change the hierarchy or the particle reconstruction story.

## Non-Negotiable Scientific Contract

Every accelerated candidate must preserve the following unless it is explicitly
registered as a separately named optimization-schedule ablation:

```text
same HLT and offline cache rows
same split manifest and deterministic balanced ordering
same hierarchy target cache and target shard identities
same model architecture and resolved C-tier variant
same slot count, dust policy, matching mode, and loss weights
same Sinkhorn temperature and 30 Sinkhorn iterations
same Hungarian assignment definition for C4
same random seed contract and row-level shuffle semantics
same validation selection metric
same HLT-only deployment and final-test constraints
```

The accelerated execution profile may change numerical rounding in the model
forward or the number of optimizer updates per epoch. These are not hidden.
They must be recorded in reports, benchmarked against FP32, and accepted only
when the non-inferiority gates in this plan pass.

No final-test data may be used in any runtime, schedule, precision, batch-size,
or worker-count decision.

## Design Overview

The accelerated profile has five coordinated pieces:

```text
1. BF16 autocast for model forward computation.
2. Explicit FP32 reconstruction, matching, and accounting loss computation.
3. Per-family physical batch-size calibration.
4. Exact parallel CPU Hungarian assignment for C4.
5. Warmup-plus-cosine learning-rate schedules with a bounded epoch budget.
```

It is intentionally benchmark-gated. The campaign must use the fastest
candidate that satisfies numerical, reconstruction, and stability checks, not
the candidate that merely reports the largest jets/s value.

## Precision Plan

### BF16 Forward, FP32 Objective

Add an explicit precision configuration rather than treating `--amp` as a
generic switch:

```text
precision_mode:
  fp32
  bf16_forward_fp32_loss
  fp16_forward_fp32_loss   # supported only as a diagnostic fallback
```

The intended production acceleration mode is:

```text
bf16_forward_fp32_loss
```

Execution rules:

1. Model parameters remain FP32.
2. AdamW optimizer state remains FP32.
3. The C2F model forward executes under
   `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.
4. Every tensor consumed by hierarchy accounting, slot matching, Sinkhorn,
   Hungarian cost construction, uncertainty NLL, and total-loss aggregation is
   explicitly converted to FP32 inside a disabled autocast region.
5. Gradients flow through the FP32 casts back to the BF16-forward activations.
6. The trainer carries three separate precision states:

   ```text
   autocast_enabled
   autocast_dtype
   grad_scaler_enabled
   ```

   `autocast_enabled` is true for BF16 and FP16 forward modes.
   `grad_scaler_enabled` is true only for FP16. BF16 executes backward,
   unscaling checks, clipping, and optimizer step without a scaler.
7. All finite checks remain mandatory before optimizer step in every mode.

The explicit FP32 conversion matters. Merely exiting an autocast context does
not promote already-produced BF16 tensors. The implementation needs a small
typed helper that creates an FP32 C-tier output view for loss computation while
preserving autograd links to the forward output.

### Precision Diagnostics

For every run, record:

```text
precision_mode
autocast_dtype
model_parameter_dtype
optimizer_state_dtype
loss_dtype
matching_cost_dtype
autocast enabled state
autocast dtype
gradient scaler enabled state
finite-batch and skipped-batch counts
```

The report must distinguish a BF16-forward run from an FP32 baseline. It may
compare them, but may not silently treat them as bit-identical artifacts.

### Deliberate Non-Changes

The initial accelerated profile must not enable TF32 or `torch.compile` by
default. They add additional numerical or runtime-system variables while the
project is already using an explicit native-Triton fallback on the GH200
environment. They can be separate future benchmarks after the BF16 path is
validated.

## Batch-Size Plan

### Motivation

The active single-view jobs use small physical batches despite substantial
memory headroom. Larger batches reduce launch, collation, and per-batch
matching overhead and should improve accelerator utilization.

Increasing physical batch size is not assumed to be scientifically free. It
reduces optimizer steps per data epoch. The selected configuration therefore
requires a learning-rate calibration and validation non-inferiority test.

### Candidate Grid

Benchmark these physical training/evaluation batch pairs:

```text
single-view B/C variants
  reference: 16 / 32
  candidates: 32 / 64, 48 / 96, 64 / 128

C6 four-view variant
  reference: 8 / 16
  candidates: 16 / 32, 24 / 48, 32 / 64

C4 hard-Hungarian variant
  reference: 16 / 32
  candidates: 16 / 32 initially; raise only after the exact matcher
  parallelization is validated
```

Do not assume that the largest batch fits just because the reported peak
reserved memory is small. Measure memory with the full model forward, loss,
backward, optimizer step, and validation batch. A candidate is rejected if
its peak reserved memory exceeds 80 percent of the physical GPU memory or if
it produces any out-of-memory event.

### Shared Optimizer Schedule Calibration

The current constant learning rate is `2e-4` for the reconstructor parameter
group and `0.05 * 2e-4` for the HLT encoder group. Large-batch optimization
must not blindly use linear LR scaling.

Evaluate the following candidate schedule settings across the representative
variants:

```text
peak reconstructor LR: 2e-4, 4e-4, 6e-4
HLT encoder LR: retain a conservative cap of 1e-5 to 2e-5
weight decay: unchanged
gradient clip norm: unchanged
loss weights and matcher hyperparameters: unchanged
```

The calibration selects one shared optimizer schedule for all comparable
C-tier architecture variants:

```text
C0, C1, C2, C3, C4, C5, C6,
C5-B1, C5-B2, C5-B3, C5-no-slot, and Cdirect-unconstrained
```

That shared contract includes peak LR, HLT encoder LR cap, warmup fraction,
minimum LR ratio, maximum epochs, minimum epochs, and early-stop patience.
It prevents an architecture comparison from silently becoming an LR comparison.

C6 may use a different physical batch size because it renders four views and
has a different activation footprint. C4 may use a different Hungarian worker
count. Neither exception permits a variant-specific LR or schedule. All C-tier
variants, including C6, use the same approved peak LR, encoder LR cap, warmup,
cosine floor, maximum epoch budget, minimum-epoch rule, and patience. Any run
that intentionally uses a different optimizer schedule is a separately named
schedule ablation and is excluded from primary C-tier architecture rankings.

If a larger-batch candidate fails, the runner falls back to the lower accepted
batch size while retaining the same approved shared optimizer schedule. It
must never silently substitute a different LR or batch after a failure.

## Input Pipeline Plan

The data path already uses pinned host memory and nonblocking device transfers.
Keep both. The current trainer creates, consumes, and discards one DataLoader
per target shard. Consequently, `persistent_workers=True` cannot preserve
workers across shard boundaries and is not part of the initial accelerated
profile.

The immediate, objective-preserving input-pipeline candidate is:

```text
prefetch_factor=2
deterministic worker initialization derived from run seed, epoch, and shard
```

Apply `prefetch_factor` only when `num_workers > 0`. It is benchmarked, not
assumed beneficial. High-data currently defaults to zero workers for memory
safety, so worker-count selection must include that reference.

A cross-shard persistent-worker loader is an optional later optimization, not
a prerequisite for `accelerated_candidate_v1`. It requires a single
manifest-ordered iterable dataset that opens one target shard at a time while
retaining worker processes across boundaries. It may be implemented only with
tests proving exactly the same row identities, shard order, balanced shuffle,
and epoch seeding as the current loader.

Benchmark worker counts:

```text
0, 4, 8, 12
```

Choose the smallest count whose throughput is within one percent of the best
candidate and whose host-memory measurement remains safe. More workers are
not automatically better, especially for the large high-data target shards.

Every benchmark record must include CPU utilization, host resident memory,
batch loading time, GPU allocated memory, GPU reserved memory, batches/s, and
jets/s.

## Exact C4 Hungarian Acceleration

### Current Bottleneck

C4 computes hard Hungarian assignments by repeatedly moving small pair-cost
matrices to CPU and calling SciPy serially for individual active cells. The
assignment is intentionally non-differentiable, but the selected GPU pair
costs still define the optimized loss. This makes C4 much slower than the
ordered and Sinkhorn variants.

### Required Implementation

Retain the exact assignment objective while changing only its execution:

1. Construct active cost matrices in the original deterministic pair order.
2. Transfer the packed active-cost collection to CPU once per batch or once
   per target-count group, not once per cell.
3. Solve `scipy.optimize.linear_sum_assignment` in a bounded executor using
   candidate worker counts `1, 4, 8, 12` up to `SLURM_CPUS_PER_TASK`.
4. Reassemble assignments in the original pair order regardless of completion
   order.
5. Transfer only the integer row/column assignment indices to GPU.
6. Compute all selected pair components, accounting penalties, and gradients
   on GPU exactly as in the scalar reference path.

The C4 training runner must expose:

```text
CONSTRAINED_C2F_HUNGARIAN_WORKERS
CONSTRAINED_C2F_HUNGARIAN_EXECUTOR=serial|thread
```

The default remains serial until the threaded path passes parity and benchmark
gates. The selected worker count is written to `config.json`, checkpoints,
run reports, training progress, and the final campaign report.

### C4 Parity Requirement

For tensors with unique optima, threaded C4 must exactly match the scalar
reference in:

```text
assignment row/column indices
all named loss components
total loss
parameter gradients
```

For tied costs, tests must compare objective values and deterministic
tie-breaking policy rather than assuming a different valid optimal assignment
has identical index order.

## Compressed Convergence Schedule

### Rationale

The current reconstructor training uses a constant learning rate for a maximum
of 30 epochs with patience 6. That is conservative but poorly matched to a
campaign that needs the complete high-data-like run list in a practical time.

Use a schedule that reaches useful parameters quickly, then decays rather than
spending many late epochs at the same LR:

```text
schedule: linear warmup followed by cosine decay
maximum epochs: 10
minimum epochs before stopping: 5
early-stop patience: 2 or 3 consecutive non-improvements
minimum LR ratio: 0.05 of peak LR
warmup: first 10 percent of planned optimizer steps, capped at one epoch
selection metric: unchanged model_val reconstruction score
```

The schedule operates on optimizer steps, not merely epoch numbers. This is
required because physical batch-size calibration changes the number of updates
in an epoch.

### Schedule Semantics

1. Set LR to a small positive fraction of peak at step zero.
2. Increase linearly to the calibrated peak LR during warmup.
3. Decay smoothly by cosine to `peak_lr * 0.05` over the remaining planned
   step budget.
4. Keep existing parameter-group LR ratios unless the calibration explicitly
   selects an encoder cap.
5. Do not permit early stopping before the minimum epoch count.
6. After the minimum, stop after the selected number of consecutive
   non-improving model-val scores.
7. Save scheduler state in best and last checkpoints so a supported resumed
   job cannot restart its schedule at peak LR.

The proposed accelerated schedule is an optimization configuration, not a
hidden architecture change. It must appear in every report and be the same for
comparable C-tier variants once selected. C6 may use a different physical batch
size, but its warmup/cosine schedule, data exposure rules, and loss definition
remain explicit and comparable.

### Checkpoint and Resume Contract

The accelerated runtime profile must force
`CONSTRAINED_C2F_RECO_SAVE_LAST_CHECKPOINT=1`. The current space-saving default
of omitting `last.pt` is not permitted for a resumable accelerated campaign.

At every completed epoch, `last.pt` must contain:

```text
model state
optimizer state
scheduler state
completed epoch index
global optimizer step
precision/runtime-profile state
Python, NumPy, Torch CPU, and Torch CUDA RNG states
manifest/cache/target/runtime-profile provenance
```

`best_model_val.pt` records the same scheduler/global-step state for audit, but
it is not a resume source. `last.pt` is the only resumable artifact.

Resume is supported only at completed epoch boundaries. A resume command must
restore `last.pt` and continue at the next deterministic epoch seed. It must
hard-fail if the manifest, HLT/offline/target hashes, runtime-profile hash,
precision mode, physical batch size, worker count, model variant, optimizer,
or scheduler settings differ from the checkpoint. A requeue cannot turn an
FP32 or incomplete run into a BF16 accelerated run by resuming it.

## Benchmark and Acceptance Protocol

### Representative Variants

The calibration must cover the important execution families:

```text
C1: ordered deterministic K=16 baseline
C5-B3: full-depth uncertainty decoder and primary D5 source
C6: four-view stochastic decoder
C4: exact hard-Hungarian path
```

These variants cover ordered matching, uncertainty heads, full hierarchy,
multi-view activation pressure, and CPU hard assignment.

### Fixed Calibration Data

Build a deterministic, manifest-bound calibration slice from campaign
`model_train` and `model_val` rows:

```text
calibration model_train: 50,000 balanced jets
calibration model_val: 30,000 balanced jets
no stack or final-test rows
fixed row identifiers written to calibration_manifest.json
```

The calibration slice must use its own derived artifact set:

```text
calibration_manifest.json.gz
calibration_hlt_cache/
calibration_offline_cache/
calibration_targets/
```

These artifacts are built from the calibration manifest with the same raw
dataset, HLT profile, HLT strength, target-builder version, layout, and cache
validation rules as the parent campaign. They are not the full campaign cache
directories paired with a subset manifest. Each derived cache must record and
be validated against the calibration manifest hash, row count, labels, and jet
identity hash before training starts.

The three-epoch calibration is only a numerical and early-learning screen. It
does not certify that a compressed ten-epoch schedule reproduces the eventual
selection behavior of the 30-epoch reference.

### Benchmark Matrix

For each representative variant:

```text
A. Current FP32 reference batch and constant LR.
B. BF16 forward, FP32 loss, reference batch and constant LR.
C. BF16 forward, FP32 loss, candidate batch and constant LR.
D. BF16 forward, FP32 loss, candidate batch and warmup-cosine schedule.
```

For C4, add serial versus threaded Hungarian comparisons at the same precision
and physical batch size before changing batch size.

### Numerical Gate

Candidate B must satisfy the following against A on identical batches:

```text
no nonfinite outputs, losses, gradients, or optimizer states
all named loss components finite
maximum acceptable relative FP32/BF16 forward component deviation: 1 percent
gradient norm remains finite and within a documented tolerance band
same target counts, matched-slot counts, and assignment cardinalities
```

The exact threshold is applied only to stable non-near-zero components. Tiny
denominators use an absolute tolerance recorded alongside the result.

### Three-Epoch Reconstruction Gate

Candidates C and D may advance to full schedule certification only when, after
three fixed calibration epochs:

```text
model_val reconstruction score is no worse than 1 percent relative to A
matched pT, eta, phi, and PID diagnostics are each no worse than 2 percent
hierarchy accounting and parent-child consistency remain within 2 percent
no new nonfinite batches are observed
peak reserved GPU memory remains at or below the configured safety cap
```

If more than one candidate passes, retain the highest-throughput candidate for
the next certification stage. If no candidate passes, retain the current FP32
profile for that execution family.

Passing this three-epoch gate creates the immutable
`accelerated_candidate_v1` profile artifact. That artifact records the chosen
execution settings, calibration evidence, hashes, the code/environment
fingerprint defined below, and the full-pilot contract; it may queue the
complete accelerated pilot, including the C5-B3/C6 runs needed to gather
subsequent certification evidence. It cannot queue high-data or final-test
claim stages.

### Full Ten-Epoch Schedule Certification

Before promoting an accelerated candidate to `accelerated_approved_v1`, perform a full-scale pilot
comparison for the two reconstructor paths that matter most to downstream
tagging:

```text
C5-B3
C6
```

For each path, run the accepted accelerated configuration for all ten planned
epochs on the full pilot manifest and the full pilot HLT/offline/target caches.
Compare it to a full-pilot FP32 reference with the same architecture, data,
seed contract, and validation metric. The reference may be an already-complete
or still-completing `fp32_reference` campaign artifact only if its provenance
and runtime profile validate exactly.

Certification uses a dedicated fixed-horizon mode:

```text
maximum epochs: 10
minimum epochs: 10
early stopping: disabled
```

The runner must execute and report all ten epochs even if the model-val score
plateaus after epoch five. The normal compressed campaign patience rule is not
used for this certification run.

The ten-epoch candidate must report:

```text
per-epoch model-val reconstruction scores through epoch 10
best model-val score through epoch 10
all named reconstruction diagnostics
wall time, optimizer steps, and resource telemetry
```

The ten-epoch certification passes only when the existing
`accelerated_candidate_v1` profile is non-inferior to the FP32 reference at
the same ten-epoch horizon under the documented reconstruction thresholds.
Its report is retained as evidence bound to that candidate profile. It does
not create a new candidate and does not approve high-data or final-test use.

### Final Thirty-Epoch Promotion Gate

`accelerated_approved_v1` requires a separate fixed-horizon, 30-epoch FP32
reference for each of C5-B3 and C6. The references must run all 30 epochs with
early stopping disabled, on the full pilot manifest and the identical
HLT/offline/target cache contract used by the respective accelerated
certification run. Compare each accelerated candidate's selected model-val
checkpoint against the selected checkpoint of its matching FP32 reference on
the same fixed, full-pilot `model_val` rows.

For **each** of C5-B3 and C6, the final promotion comparison passes only when:

```text
best model-val reconstruction score is no worse than 1 percent relative
matched pT, eta, phi, and PID diagnostics are each no worse than 2 percent
hierarchy accounting and parent-child consistency are each no worse than 2 percent
all named loss components and diagnostics are finite
both the accelerated candidate and FP32 reference have zero nonfinite batches
both the accelerated candidate and FP32 reference have zero skipped batches
```

The reconstruction score is minimized, so the first condition is evaluated as
`accelerated_score <= 1.01 * fp32_reference_score` for stable positive
reference scores. Diagnostic quantities are evaluated in their documented
error direction with the corresponding `1.02` relative bound. Near-zero
reference quantities use the same predeclared absolute tolerances as the
three- and ten-epoch gates; the promotion report must record the selected
tolerance and observed value for every such comparison. A completed 30-epoch
reference that fails any one of these conditions does not promote the profile.

Until both paths pass this final gate, the accelerated profile may be used for
a pilot runtime study but is not represented as proven equivalent to the
30-epoch reference.

### Tagger Sanity Gate

Before declaring the accelerated reconstructor profile campaign-ready, perform
two frozen-reconstructor downstream sanity comparisons on the fixed calibration
validation rows:

```text
C5-B3 accelerated candidate versus the matching selected checkpoint from the
fixed-horizon 30-epoch C5-B3 FP32 reference
  use the dual-view tagger path intended for D5.

C6 accelerated candidate versus the matching selected checkpoint from the
fixed-horizon 30-epoch C6 FP32 reference
  use the four-view pseudo-particle tagger path intended for D6.
```

For both paths, evaluate the two frozen reconstructor taggers on exactly the
same fixed `model_val` jet identities and labels. Store per-jet class logits,
per-jet CE, labels, jet-identity hash, and the prediction-array hash for the
candidate and reference before computing the gate. Do not resample, filter, or
retune either tagger after this row set is fixed.

Within each comparison pair, the taggers must share an identical architecture,
initialization checkpoint and seed, training/validation row identities and
order, optimizer and schedule profile, augmentation and shuffle seeds, epoch
budget, model-val selection rule, and stopping rule. The only permitted
difference is the frozen reconstructor checkpoint (accelerated candidate or
its matching selected 30-epoch FP32 reference). This makes the paired
bootstrap a test of reconstructor differences rather than of tagger training
noise.

Use a deterministic, label-stratified paired percentile bootstrap with 10,000
replicates and a recorded seed. Each replicate resamples row indices with
replacement within every class and retains the candidate/reference pairing.
Define the paired differences as:

```text
delta_accuracy = accuracy_accelerated - accuracy_fp32
delta_ce       = mean_ce_accelerated - mean_ce_fp32
```

The one-sided 95 percent upper confidence bound must satisfy both rules for
each C5-B3 and C6 comparison:

```text
upper_bound(-delta_accuracy) <= 0.005    # no more than 0.5 absolute accuracy points worse
upper_bound(delta_ce)        <= 0.010    # no more than 0.010 CE worse
```

The report must include the observed paired differences, both confidence
bounds, bootstrap seed/count, row and prediction hashes, and pass/fail status.
This guards against a reconstructor metric that looks acceptable while
degrading the downstream object that matters: tagging.

## Runtime Profiles

The submitter must select an explicit runtime profile. It may not silently
change a profile based on GPU availability.

```text
fp32_reference
  current batch defaults
  FP32 forward and objective
  constant LR
  serial Hungarian

bf16_calibration
  candidate batch and worker grid
  BF16 forward, FP32 objective
  constant-LR and warmup-cosine benchmark variants
  optional threaded Hungarian benchmark

fp16_diagnostic
  FP16 forward, FP32 objective, GradScaler enabled
  diagnostic fallback only; never selected for primary campaign ranking

accelerated_candidate_v1
  benchmark-approved batch, workers, precision, Hungarian workers,
  one shared C-tier peak LR, warmup, cosine floor, max epochs,
  minimum epochs, and patience
  created only after the three-epoch calibration gate passes
  retains ten-epoch C5-B3/C6 certification evidence as that pilot work completes
  permitted for the full accelerated pilot only
  no high-data submission and no final-test claims

accelerated_approved_v1
  immutable promotion of accelerated_candidate_v1
  requires fixed-horizon C5-B3/C6 certification, both tagger sanity gates,
  and a passed fixed-horizon 30-epoch FP32-reference comparison for both paths
  retains the exact candidate code/environment fingerprint
  permits high-data submission and final-test claim stages
```

Every output root must include the selected profile in its name and metadata,
for example:

```text
constrained_coarse_to_fine_pseudooffline_hltv2_s2p5_pilot_accel_v1_<stamp>
```

Artifacts from `fp32_reference`, `accelerated_candidate_v1`, and
`accelerated_approved_v1` are not interchangeable under `SKIP_EXISTING`.
Reuse validation must require exact equality of the runtime-profile hash in
addition to existing cache, manifest, target, checkpoint, and prediction
hashes.

### Code and Environment Fingerprint

Every candidate and approved profile artifact must be created from a clean
source tree. The artifact records a canonical `code_environment` object and
its SHA256, containing at least:

```text
full Git source commit SHA
source-tree-clean boolean and SHA256 of `git status --porcelain=v1` output
Python implementation and exact version
PyTorch version
PyTorch CUDA runtime version (`torch.version.cuda`)
SciPy version
```

Creation fails when tracked source changes are present. The profile writer
stores the clean status hash rather than silently normalizing a dirty checkout.
The promotion writer must require every candidate, FP32 reference,
certification report, and tagger-sanity report in its closure to carry the
same `code_environment` SHA256. A source or dependency update therefore
requires a new candidate and recertification; it cannot inherit a prior
approval.

### Immutable Promotion Closure

Promotion writes one canonical `accelerated_approved_v1.json` artifact rather
than copying fields into a submitter environment. It must be content-hashed
and bind this complete parent closure:

```text
accelerated_candidate_v1 profile path and SHA256
accelerated_candidate_v1 profile hash
C5-B3 ten-epoch certification report path and SHA256
C6 ten-epoch certification report path and SHA256
C5-B3 tagger-sanity report path and SHA256
C6 tagger-sanity report path and SHA256
C5-B3 fixed-horizon 30-epoch FP32-reference report path and SHA256
C6 fixed-horizon 30-epoch FP32-reference report path and SHA256
the full-pilot manifest, HLT, offline, target-cache, and jet-identity hashes
the candidate code/environment object and SHA256
the resolved C5-B3 and C6 variant identifiers and their FP32/accelerated
checkpoint hashes
the predeclared numerical and bootstrap thresholds, observed values, and
pass/fail decisions
```

The promotion writer must recompute every listed SHA256, verify that every
report names the same parent candidate profile hash and cache/manifest
contract, code/environment fingerprint, and reject a mixed C5-B3/C6 or
mixed-campaign closure. The high-data/final-claim submitters must recompute
this closure and the active code/environment fingerprint before submission;
they may not accept caller overrides for its model, fusion, precision, or
membership contract. A commit, cleanliness, Python, PyTorch, CUDA, or SciPy
mismatch is a hard submission failure.

The canonical high-data and final-claim submitters must accept only an
`accelerated_approved_v1` promotion artifact. There is no default exploratory
high-data bypass. Any intentionally exploratory high-data experiment requires
a separately named stage, an explicit confirmation variable, and reports that
it is not eligible for primary final claims.

## Reporting Requirements

Add a runtime table to the C2F report with one row per reconstructor run:

```text
variant
runtime profile hash
code/environment fingerprint SHA256
precision mode
physical train/eval batch size
worker count and prefetch settings
peak LR and HLT encoder LR
warmup and decay configuration
epochs completed
early-stop reason
optimizer steps completed
jets/s and batches/s
peak allocated and reserved GPU memory
CPU memory and Hungarian worker telemetry
nonfinite/skipped batch counts
best model-val reconstruction score
```

The report must make the accelerated setup auditable. It must not claim that
two variants used identical optimization conditions when their accepted batch
or LR profiles differ.

## Expected Impact

The following are targets, not guarantees:

```text
ordinary ordered/Sinkhorn C variants: 2x to 3x throughput improvement
C6 multi-view: 1.7x to 2.5x throughput improvement
C4 Hungarian: 3x to 5x throughput improvement after exact parallelization
epoch-budget compression: 30 to 10 maximum epochs, subject to non-inferiority
```

For ordinary C variants, a successful combined profile could reduce end-to-end
training time by about 4x to 6x. The full campaign still contains all planned
runs; it simply reaches each decision much sooner.

## Implementation Steps

### Step 1: Add Runtime Profile Configuration

Extend `CoarseToFineTrainConfig`, the CLI, Slurm runner, and submitter with
named runtime profiles, explicit precision mode, AMP dtype, worker settings,
batch settings, and scheduler parameters. Persist a canonical runtime-profile
hash and the clean code/environment fingerprint in every checkpoint and
report.

### Step 2: Implement BF16-Forward/FP32-Loss Execution

Add typed autocast helpers and FP32 output-casting helpers. Keep the complete
hierarchy and slot loss path in FP32. Add tests for finite BF16 forward output,
FP32 loss tensors, correct gradient flow, and FP32-reference component parity.
Represent autocast enablement, autocast dtype, and scaler enablement as three
separate states; enable the scaler only for FP16.

### Step 3: Improve DataLoader Overlap Safely

Add deterministic prefetch support for the existing per-shard loader. Preserve
shard order, row identities, epoch seeding, and balanced shuffle semantics.
Do not claim cross-shard persistence from `persistent_workers` unless a later
single-loader streaming implementation is added and passes identical
membership/order tests.

### Step 4: Parallelize C4 Without Changing Its Objective

Implement packed CPU cost transfer, bounded ordered Hungarian execution, and
GPU-side selected-component loss aggregation. Add scalar-reference tests for
assignments, components, total loss, and gradients, including ragged masks,
empty cells, overfull cells, and tied-cost behavior.

### Step 5: Add Step-Based Warmup/Cosine Scheduling

Implement constant and warmup-cosine schedules, minimum-epoch behavior,
checkpointed scheduler/global-step state, fixed-horizon certification mode,
and clear early-stop reasons. Force epoch-boundary `last.pt` saving for every
accelerated runtime profile and implement fail-closed resume validation.
Retain a constant LR profile as the reference control.

### Step 6: Build the Calibration-Slice Producer

Create the deterministic manifest-bound 50k/30k calibration slice and build
derived HLT, offline, and target caches for that manifest. Write provenance
metadata and a validator that rejects calibration-manifest/cache/target
mismatches.

### Step 7: Build the Throughput Benchmark Runner

Run the A-D benchmark matrix for C1, C5-B3, C6, and C4. Capture the numerical,
reconstruction, resource, and timing telemetry needed for profile selection.

Implementation artifacts:

```text
scripts/build_constrained_coarse_to_fine_runtime_benchmark_plan.py
  Hash-bound A-D matrix, candidate batch/worker sweeps, and C4 serial/threaded
  Hungarian-worker trials.

sbatch/submit_constrained_coarse_to_fine_runtime_benchmarks.sh
  Validates the Step 6 calibration artifacts, submits every matrix member using
  the normal reconstructor runner, and holds one report job behind all members.

scripts/write_constrained_coarse_to_fine_runtime_benchmark_report.py
  Refuses missing/failed/mismatched artifacts and records per-epoch numerical,
  reconstruction, throughput, input-wait, CPU, host-RSS, CUDA allocated, and
  CUDA reserved-memory telemetry for Step 8.
```

### Step 8: Enforce Profile Acceptance and Selection

Implement a report writer that applies the numerical and non-inferiority gates,
selects execution-only settings per family, selects one shared optimizer
schedule for primary C-tier comparisons, and writes an immutable
`accelerated_candidate_v1` artifact immediately after the three-epoch gate.
Bind fixed-horizon ten-epoch C5-B3/C6 certification reports to that existing
candidate. Require two fixed-horizon 30-epoch FP32 comparisons that pass the
explicit 1 percent/2 percent promotion limits, and the hash-closed promotion
artifact, including an exact shared code/environment fingerprint, before
promoting it to `accelerated_approved_v1`.

The three-epoch portion is implemented by
`runtime_selection.py` and
`write_constrained_coarse_to_fine_accelerated_candidate.py`. It rejects an
incomplete benchmark report, a BF16/FP32 mismatch, nonfinite/skipped batches,
missing diagnostic telemetry, an over-cap GPU reservation, inconsistent clean
code environments, or any configuration that cannot select one shared peak LR
across C1, C5-B3, C6, and C4. The output is an immutable
`accelerated_candidate_v1.json` artifact; it grants pilot-only queue rights.
The 10/30-epoch and tagger-sanity evidence is deliberately not treated as
present until Steps 9 and 10 provide those reports and the approved-profile
closure writer consumes them.

### Step 9: Add Downstream Tagger Sanity Validation

Run fixed-slice frozen-reconstructor tagger comparisons for both C5-B3 and C6
against the matching selected checkpoints from their fixed-horizon 30-epoch
FP32 references. Use the intended D5 dual-view and D6 four-view paths,
respectively, with all non-reconstructor tagger inputs and training choices
identical inside each pair. Apply the fixed-row, label-stratified
10,000-replicate paired-bootstrap accuracy/CE gate before promoting the
profile.

### Step 10: Wire Full Pilot and High-Data Submitters

Make the complete pilot and high-data campaign submitters consume an explicit,
validated runtime-profile artifact. `accelerated_candidate_v1` may submit the
full pilot only. High-data and final-claim submitters require
`accelerated_approved_v1`. Preserve the full original run list and dependency
graph. Reject missing, stale, profile-mismatched, or code/environment-mismatched
reuse artifacts.

## Test Plan

The focused test suite must cover:

```text
BF16 forward and FP32 loss dtype boundaries
finite output/loss/gradient checks under all precision modes
fixed-batch FP32/BF16 component and gradient parity
separate autocast dtype and scaler-state behavior
deterministic worker, shard, and row-order behavior
C4 serial/threaded assignment, loss, and gradient parity
batch-profile memory and failure handling
step-based scheduler values and checkpoint restore
minimum-epoch and patience semantics
fixed-horizon ten-epoch certification cannot stop early
last-checkpoint creation, exact epoch-boundary resume, and resume rejection
for runtime-profile or provenance mismatch
runtime-profile hash inclusion in artifact reuse checks
clean-source and code/environment fingerprint capture
rejection of dirty-source candidate creation and changed-code/dependency
high-data submission after approval
derived calibration-cache provenance enforcement
shared optimizer-profile enforcement across primary C-tier variants
three-epoch gate and full ten-epoch schedule-certification logic
candidate lifecycle: three-epoch creation, ten-epoch evidence attachment,
and only 30-epoch/tagger promotion
fixed-horizon 30-epoch C5-B3/C6 promotion limits and rejection behavior
paired-bootstrap tagger sanity acceptance/rejection for C5-B3 and C6
promotion-closure hash recomputation and mixed-artifact rejection
candidate-to-approved promotion and high-data/final-claim submission gate
benchmark acceptance/rejection logic
submitter refusal without an approved profile artifact
```

Run Python compilation, focused unit tests, the C2F report tests, and shell
syntax checks before queueing any calibration or accelerated campaign job.

## Queueing Rule

### Direct Exploratory Pilot Exception

An operator may deliberately skip the calibration/candidate gate for one full
**pilot-only** run by using `bf16_exploratory_pilot_v1`. This profile fixes the
safe BF16-forward/FP32-loss boundary and retains one selected checkpoint per
reconstructor, but creates neither a candidate nor an approved artifact. It
exists for measuring the complete pilot graph's real wall-clock behavior
without a benchmark fan-out.

`bf16_exploratory_pilot_v1` is never eligible for high-data submission or
final-test claims. A later high-data or claim campaign still requires the
candidate and approved-profile pathway below.

Do not cancel a scientifically valid full campaign merely because this plan
exists. The queue state is deliberately two-stage:

```text
accelerated_candidate_v1
  created after the three-epoch numerical/reconstruction gate passes
  may queue the full accelerated pilot campaign
  collects, but is not replaced by, ten-epoch C5-B3/C6 certification evidence
  may not queue high-data or any final-test claim stage

accelerated_approved_v1
  fixed-horizon C5-B3/C6 certification passed
  both downstream tagger sanity gates passed
  both 30-epoch FP32-reference comparisons passed the explicit promotion limits
  immutable promotion closure validates every parent profile/report/cache hash
  may queue the complete high-data and approved final-claim campaign
```

The high-data submitter must hard-fail unless it receives an immutable,
hash-closed `accelerated_approved_v1` artifact whose provenance and
runtime-profile hash and active code/environment fingerprint match the
requested campaign. When approved, it queues the full original high-data run
list exactly as defined by the main C2F plan.
