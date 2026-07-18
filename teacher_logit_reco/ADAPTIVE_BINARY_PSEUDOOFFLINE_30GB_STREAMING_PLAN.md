# Adaptive Binary Pseudo-Offline 30 GB Streaming Campaign Plan

## Status

This document defines the storage-constrained execution path for the adaptive
binary pseudo-offline hierarchy campaign described in:

- `teacher_logit_reco/ADAPTIVE_BINARY_PSEUDOOFFLINE_HIERARCHY_PLAN.md`
- `teacher_logit_reco/ADAPTIVE_BINARY_PSEUDOOFFLINE_RUNTIME_ACCELERATION_PLAN.md`

It changes artifact retention, data movement, and Slurm orchestration only. It
does not change the scientific campaign registry, HLT-v2 degradation strength
2.5, model dimensions, hierarchy depth, number of hypotheses, reconstruction
losses, tagger architecture, validation splits, model-selection metrics, or the
accelerated screening schedule.

The implementation contract name is:

```text
adaptive_binary_pseudooffline_streaming_30gb_v1
```

The environment selector is:

```bash
ABPH_STORAGE_PROFILE=streaming_30gb_v1
```

The existing cache-heavy path may remain available for debugging, but it must
not be the default on research compute with a 30 GB campaign-storage limit.

## 1. Executive Decision

The campaign must never persist hierarchy target caches, pseudo-particle
caches, optimizer checkpoints, and all selected models at the same time without
accounting for their combined size. In the new primary path:

1. hierarchy targets are either a compact lossless transient cache or are built
   rank-locally in RAM;
2. pseudo views are never written to shared persistent storage;
3. frozen taggers generate their pseudo views into rank-local RAM and reuse
   them during that job;
4. joint taggers generate pseudo views in the live differentiable forward;
5. stack/model-validation scoring persists logits, not pseudo representations;
6. full resume and intermediate-stage checkpoints live only in job-local RAM;
7. shared storage retains lightweight selected weights, logits, reports,
   manifests, and provenance receipts;
8. a concurrency-safe quota ledger rejects any write that could make the
   campaign exceed 30,000,000,000 persistent bytes.

The target steady-state retained footprint is at most 20 GB. The remaining 10
GB is mandatory safety headroom for in-flight atomic writes, Slurm logs,
diagnostics, and unexpected but bounded metadata growth.

## 2. Why RAM Is Usable but Not Persistent

RAM belongs to a Slurm allocation. A dependent job may run on a different node,
so no campaign contract may assume that `/dev/shm`, process memory, or a node's
local temporary directory survives after the producing job exits.

Each job therefore owns a private workspace:

```text
/dev/shm/abph-${SLURM_JOB_ID}-${SLURM_PROCID}
```

or a verified memory-backed `SLURM_TMPDIR` when Tigris exposes one. The worker
must verify the filesystem type, available capacity, and cgroup memory limit
before use. A disk-backed directory is not silently accepted as a RAM
workspace.

The workspace is created with mode `0700` and removed by a shell `trap` on
normal exit, failure, cancellation, and Slurm termination signals. RAM use is
part of the Slurm memory request. It is not free memory outside the allocation.

## 3. Locked Scientific Invariants

The storage profile is allowed to change where bytes live, not what the model
sees. The following are non-negotiable:

- all 128 particle slots remain available;
- one mean plus four fixed stochastic hypotheses remain available;
- kT and C/A hierarchy semantics remain unchanged;
- E7 and F0 retain one exact shared compiled root across both hierarchies;
- canonical particle features, side channels, uncertainty, hierarchy ledgers,
  hierarchy hidden states, support features, masks, topology, and parent links
  remain available to every tagger that currently consumes them;
- reconstruction targets remain float32 in the primary storage profile;
- target memberships remain exact;
- no lossy hidden-state quantization is permitted in
  `streaming_30gb_v1`;
- frozen reconstructors run in `eval()` mode with gradients disabled;
- jointly trained reconstructors preserve their complete differentiable path;
- full model-validation coverage and the existing selection metric remain
  unchanged;
- final-test inputs remain HLT-only and require the existing frozen claim
  contract;
- teacher logits and offline inputs are never loaded by final-test workers.

## 4. Storage Classes

Every generated artifact belongs to exactly one class.

### 4.1 Persistent essential artifacts

These may remain after campaign completion:

- split manifest and immutable input audit;
- compact selected checkpoints;
- teacher-logit blocks required by explicit KD ablations;
- model-validation and stack logits;
- fusion parameters;
- run reports, training curves, runtime profiles, and diagnostics summaries;
- quota ledgers, cleanup receipts, and final campaign reports;
- a small deterministic forensic sample.

### 4.2 Shared transient artifacts

These may exist under the campaign root only while named consumers are active:

- compact HLT input shards;
- compact offline input shards used to build targets;
- compact hierarchy target shards;
- atomic checkpoint staging files.

Every shared transient artifact has an owner stage, a complete consumer set, a
content hash, an expected byte count, and a cleanup policy.

### 4.3 Rank-local ephemeral artifacts

These must live in the verified RAM workspace:

- full optimizer/scaler/EMA resume state;
- intermediate curriculum checkpoints;
- rank-local target batches or target shards;
- generated pseudo-view tensors;
- in-memory frozen-tagger training caches;
- temporary prediction batches;
- temporary decompression and byte-unshuffle buffers.

### 4.4 Forbidden persistent artifacts

The 30 GB profile rejects attempts to persist:

- full pseudo-view NPZ caches;
- `slot_hidden` or hierarchy hidden tensors for every campaign jet;
- full `last.pt` checkpoints after job completion;
- duplicate online, EMA, optimizer, and scaler states inside a selected
  deployment checkpoint;
- all historical stage-best checkpoints;
- final-test pseudo caches;
- unbounded debug tensor dumps.

## 5. Hard Quota Contract

The canonical limits are:

```text
ABPH_MAX_PERSISTENT_BYTES = 30_000_000_000
ABPH_TARGET_STEADY_STATE_BYTES = 20_000_000_000
ABPH_MINIMUM_SAFETY_HEADROOM_BYTES = 10_000_000_000
```

The quota covers the campaign root plus campaign-specific mirrored diagnostics
and logs. The pre-existing read-only JetClass ROOT dataset is not counted as a
campaign artifact.

### 5.1 Concurrency-safe reservations

Concurrent Slurm jobs must not independently observe free space and both claim
it. The campaign root contains:

```text
storage/
  storage_contract.json
  quota_ledger.json
  quota_ledger.lock
  reservations/
  cleanup_receipts/
  storage_audits/
```

Before opening a persistent output, a worker acquires an exclusive `flock` on
`quota_ledger.lock` and reserves its declared worst-case bytes. The reservation
contains job ID, run ID, artifact role, destination, expected bytes, expiration
policy, and source-provenance hash. A write may proceed only when:

```text
committed_bytes + active_reserved_bytes + requested_bytes
    <= ABPH_MAX_PERSISTENT_BYTES
```

After atomic commit, the worker replaces the reservation with measured file
sizes and hashes. Failed writes release their reservations after deleting only
their own verified temporary paths. Stale reservations require an audit of the
recorded Slurm job state before release.

### 5.2 No hidden fallback

When a reservation fails, the job exits before producing a partial artifact. It
does not switch from RAM to shared disk, reduce the data split, reduce the
number of hypotheses, change precision, or delete unrelated artifacts.

## 6. Prequeue Storage and RAM Audit

The submission preflight performs measured projection, not a fixed optimistic
constant.

1. Compile targets and pseudo views for a deterministic stratified sample of
   real jets from every class.
2. Encode them with the exact production storage codecs.
3. Measure bytes per jet by artifact family and valid-particle-count decile.
4. Project each campaign wave using the manifest's exact class and split sizes.
5. Add measured selected-checkpoint sizes from representative B, C, D, E, and
   F models, or conservative parameter-count estimates before those models
   exist.
6. Model concurrent reservations and atomic-write duplication.
7. Write `storage/storage_projection.json`.

Submission fails unless both conditions hold:

```text
projected_peak_persistent_bytes <= 24_000_000_000
projected_final_retained_bytes <= 20_000_000_000
```

The 6 GB gap between the projected peak and the absolute 30 GB cap is not
available for planned artifacts.

The RAM preflight runs under the actual Slurm topology. It includes model
weights, optimizer state, DDP buckets, rank-local input shards, pseudo views,
prefetch buffers, and temporary codec buffers. A rank must retain at least 20%
of its allocated memory as headroom after the calibrated working set is live.

## 7. Compact Lossless Target Cache

Reconstructor target generation is expensive and shared by many B/C/D runs. It
should not be repeated by every model when an exact transient cache fits safely.

### 7.1 Training target schema

The compact cache retains all objective inputs:

- root features;
- level features, masks, topology, parent indices, and memberships;
- particle targets and particle mask;
- HLT/offline valid counts and HLT-axis coordinates;
- labels and ordered jet identities needed for source alignment.

It changes only encoding:

- float targets remain float32;
- boolean masks and memberships are packed with `numpy.packbits` along a
  declared axis and unpacked exactly in RAM;
- topology uses int8;
- parent and member indices use the smallest validated signed integer type;
- floating arrays use a reversible byte shuffle before compression;
- shards remain independently hashed and bounded;
- per-node 64-byte diagnostic identities are replaced by shard-level ordered
  target-identity hashes because they are not consumed by any objective;
- a deterministic forensic sample retains the complete identity arrays.

The loader reconstructs the existing in-memory target object exactly. Model and
loss code do not receive a storage-specific object.

### 7.2 Measured mode selection

After building sample shards, orchestration selects exactly one target mode:

`shared_transient_compact`:

- selected only when compact targets plus required HLT shards and all existing
  committed artifacts remain within the 24 GB projected-peak gate;
- built once and shared by B/C/D workers;
- deleted after all named reconstructor consumers have successful artifact
  receipts.

`rank_local_build`:

- selected when the shared cache would exceed the projection gate;
- each DDP rank reads only its assigned manifest ranges and builds bounded
  target shards in RAM;
- rank-local shards may be reused across epochs within that job;
- no target arrays are committed to shared storage.

The selected mode and projection evidence are immutable campaign provenance.
No worker chooses independently.

### 7.3 Offline input lifecycle

Offline inputs are needed only for target construction, A4, oracle diagnostics,
and explicitly declared reconstruction supervision. They are deleted as soon as
all such consumers have succeeded. Deployable tagger and scoring workers cannot
mount the offline-cache path.

## 8. RAM Workspace and Bounded Sources

Each worker creates a `RankLocalWorkspace` with explicit byte ownership:

```text
workspace/
  inputs/
  targets/
  pseudo/
  checkpoints/
  codec_scratch/
  workspace_manifest.json
```

The workspace allocator exposes reservations just like the persistent quota,
but its limit is based on measured available RAM and the Slurm cgroup. It never
allows cache growth to trigger operating-system swapping.

Input and target sources retain the existing global batch plan. Every rank owns
ordered `(shard_id, start, stop)` ranges, with no overlap and full global union.
Resume within a live allocation restores the committed cursor. A restarted
Slurm job starts from its selected persistent checkpoint and deterministically
rebuilds ephemeral data.

## 9. Pseudo Views Are Never Persisted

### 9.1 Consumer-only in-memory schema

The in-memory cache contains exactly the tensors consumed by the hierarchy-aware
taggers.

Global tensors:

- `shared_root_ledger`;
- `hypothesis_latent`;
- `hypothesis_prior_log_prob`.

Per-hierarchy particle tensors:

- `canonical_features`;
- `side_channels`;
- `mask`;
- `group_indices`;
- `uncertainty`.

Per-hierarchy frontier tensors:

- `ledger`;
- `hidden`;
- `support`;
- `uncertainty`;
- `mask`;
- `topology`;
- `parent_indices`.

The tagger does not receive stored `slot_hidden`, separate particle
four-vectors, separate mass, `local_slot_indices`, or
`source_child_indices`. Particle four-vectors are reconstructed through the
existing canonical-feature/ParT input path. Tests must prove that removing each
field does not alter tagger logits or gradients.

All floating tensors remain in their existing inference precision. Integer and
mask tensors use exact compact in-memory dtypes. Lossy int8 hierarchy-state
quantization is outside this plan.

### 9.2 Frozen tagger training

For E-tier and frozen-reconstructor F-tier variants:

1. load the selected reconstructor and set it to `eval()`;
2. disable reconstructor gradients;
3. assign each DDP rank its immutable training and model-validation ranges;
4. generate pseudo views from HLT inputs only;
5. place consumer-only tensors into rank-local RAM;
6. train and validate the tagger from that RAM cache across all epochs;
7. discard the cache when the job exits.

The generator records the selected reconstructor hash, HLT content hash,
hypothesis seed, schema hash, ordered jet-identity range hash, and exact root
identity checks. A cache entry is reused only when all fields match.

### 9.3 Capacity fallback

The preferred mode is `full_rank_cache`: each rank materializes its complete
assigned model-train and model-validation pseudo data once.

If the actual-topology RAM preflight cannot retain 20% headroom, the campaign
uses `bounded_lru`:

- cache size is fixed before training;
- deterministic shards are generated and evicted by LRU;
- eviction never writes to disk;
- a missing shard is regenerated by the frozen reconstructor;
- cache behavior is telemetry, not a training-order input.

If even one production batch cannot fit with 20% headroom, submission fails.

### 9.4 Joint variants

F0 and every variant that updates the reconstructor use the existing single
shared differentiable forward. They do not consume detached RAM pseudo caches.
The CE and reconstruction objectives continue to use the exact same prepared
root/hierarchy forward required by the scientific plan.

## 10. DDP Tagger Runtime

The 30 GB profile extends rank-sharded DDP execution to taggers that use large
RAM-resident pseudo inputs. The default topology is four ranks, one GPU per
node, when the measured promotion gate passes.

The global effective batch, optimizer schedule, validation set, and selection
metric remain equal to the single-rank contract. The distributed runtime uses:

- persisted global batch plans;
- all-rank forward-success consensus before backward;
- bounded process-group abort on rank-local failure;
- typed validation numerator/denominator reductions;
- rank-zero-only persistent writes;
- canonical ordered validation coverage hashes.

Tagger DDP is promoted only after one representative E7/F0 parity run proves
model-validation metric parity and at least a 1.5x wall-time improvement over
single-rank RAM streaming. Failure of the speed gate falls back to single-rank
streaming; it never falls back to persistent pseudo caches.

## 11. Bundled Model-Validation and Stack Scoring

Persisted fusion inputs are logits only. Scoring is grouped by pseudo source:

```text
D1/kT source family
D2/C-A source family
E7 shared-root dual family
independent-root diagnostic family
joint F-tier checkpoint family
```

For each split and source family, a scoring allocation:

1. creates the source pseudo views once in rank-local RAM;
2. verifies source and identity provenance;
3. evaluates every compatible selected tagger checkpoint in bounded model
   bundles;
4. writes only labels, ordered identity references, and float32 logits;
5. records one source-generation hash shared by all members;
6. deletes pseudo tensors before exit.

Model bundles are calibrated against GPU memory. If all taggers cannot be
resident together, the complete rank-local pseudo shard remains in RAM while
taggers are loaded sequentially. The reconstructor is not rerun merely because
tagger weights change.

Stack-train logits may be stored as float32 because their total size is small.
Fusion fitting remains restricted to stack-train, selection to stack-val or the
existing declared model-validation contract, and final-test evaluation remains
locked.

## 12. Compact Checkpoint Contract

### 12.1 Ephemeral resume checkpoint

During one Slurm job, the worker maintains one full resumable checkpoint in its
RAM workspace. It may contain online model weights, optimizer, scaler, EMA,
curriculum state, source cursor, RNG state, and rank-local runtime state.

It is never copied to shared storage under `streaming_30gb_v1`. A preempted or
failed allocation restarts that run from its last persistent selected handoff,
or from the beginning when no handoff exists. Reports state this limitation
explicitly.

### 12.2 Stage handoff

Only one full stage-best checkpoint exists in RAM at a time. After the next
stage loads and verifies it, the previous full stage checkpoint is deleted.
The campaign may persist a lightweight stage handoff only when another Slurm
job, rather than the current process, must continue training from it.

### 12.3 Selected checkpoint

The persistent selected checkpoint contains:

- selected EMA `model_state_dict` only once;
- model construction metadata;
- resolved variant configuration and hash;
- selection metric and model-validation summary;
- source/cache/manifest hashes;
- runtime, storage-profile, and schedule contracts;
- selected checkpoint role and content hash.

It excludes:

- `online_model_state_dict`;
- optimizer and scaler states;
- duplicate `ema_state_dict`;
- rank RNG snapshots;
- complete train-source state;
- temporary DDP state.

Warm-start and prediction loaders must accept the compact selected contract and
must never require optimizer state from a selected model. Existing full
checkpoints remain readable for backward compatibility but are not emitted by
the 30 GB profile.

## 13. Forensic Sample

The campaign keeps a deterministic, class-stratified model-validation sample
of at most 4,096 jets. It may contain:

- full target identities;
- full renderer outputs;
- fields omitted from the consumer-only RAM schema;
- exact accounting residuals;
- cache-vs-streaming equivalence diagnostics.

The sample has its own strict byte reservation and is never silently enlarged.
It supports debugging without retaining complete internal representations for
millions of jets.

## 14. Campaign Waves and Cleanup Barriers

The full pilot runs in ordered storage waves.

### Wave 0: Projection and contracts

- build the split manifest;
- run real-data storage and RAM probes;
- select target mode and tagger RAM mode;
- create the immutable quota ledger and storage contract.

No model job queues if projection fails.

### Wave 1: Inputs and baselines

- build only required transient HLT/offline inputs;
- train A-tier baselines and A4;
- compact their selected checkpoints;
- verify input and checkpoint provenance.

### Wave 2: Targets

- create the compact shared transient target cache when approved, otherwise
  register rank-local target building;
- run the actual target feasibility audit;
- create the bounded forensic sample.

### Wave 3: Reconstructors

- train all B/C/D variants with the accelerated schedule and approved DDP
  contracts;
- persist compact selected checkpoints and reports only;
- run reconstructor diagnostics that require targets;
- produce one success receipt per target consumer.

### Barrier A: Privileged-data cleanup

After all required B/C/D and privileged diagnostic receipts exist:

- delete shared target shards;
- delete offline-cache shards not required by an explicit remaining oracle;
- reconcile the quota ledger;
- write `cleanup_privileged_inputs.json`.

Tagger jobs do not begin before this barrier unless the storage projection proves
that concurrent occupancy remains below the 24 GB planned peak.

### Wave 4: Taggers

- train E/F and neural G variants with rank-local pseudo views;
- persist compact selected checkpoints and reports;
- discard all job-local pseudo data on exit.

### Wave 5: Scoring and fusion

- run source-family bundled model-validation and stack scoring;
- persist logits only;
- fit and evaluate G-tier fusions;
- produce diagnostics from logits and the bounded forensic sample.

### Barrier B: Input cleanup

After every required logit block is verified:

- delete remaining transient HLT shards when reports no longer need them;
- retain only immutable input hashes and rebuild instructions;
- reconcile the quota ledger;
- write `cleanup_deployable_inputs.json`.

### Wave 6: Report and retention

- write the selection report;
- retain compact selected checkpoints required for reproducibility and final
  claims;
- optionally remove non-selected ablation checkpoints only through an explicit
  reviewed retention policy;
- write final actual-byte accounting.

No cleanup job uses broad wildcard deletion. It reads exact paths from the
artifact manifest, resolves every path beneath the campaign root, verifies the
recorded hash and artifact role, and records the deletion result.

## 15. Failure and Restart Semantics

- A failed model job leaves persistent inputs and previously completed selected
  checkpoints untouched.
- Rank-local RAM is assumed lost after failure.
- A failed target consumer prevents automatic target cleanup.
- A failed scoring member does not invalidate already verified logits from
  other members, but the report remains incomplete.
- A cleanup job runs only after all required success receipts exist.
- `afterany` may be used for workspace cleanup inside an allocation, never for
  deleting shared scientific inputs.
- Requeue validates current storage reservations and active artifact hashes
  before starting.
- A run cannot resume from a compact selected checkpoint while claiming an
  exact optimizer continuation. It is a warm start and is labeled accordingly.

## 16. Provenance and Reporting

Every run report records:

- `storage_profile=streaming_30gb_v1`;
- persistent quota-contract hash;
- target mode and target codec hash;
- pseudo execution mode (`full_rank_cache`, `bounded_lru`, or
  `joint_differentiable`);
- RAM topology and measured peak resident bytes per rank;
- reconstructor and HLT source hashes;
- consumer-only pseudo schema hash;
- whether any pseudo representation was written persistently (must be false);
- compact checkpoint contract and hash;
- restart semantics;
- cleanup receipt hashes for inputs it depended on.

The final report contains projected peak, measured peak, final retained bytes,
bytes by artifact class, regenerated pseudo batches, cache hit rate, startup
time, steady-state time, and any run restarted after RAM loss.

Final-test attestation must state:

```text
offline inputs loaded: false
offline targets loaded: false
teacher logits loaded: false
pseudo cache loaded from persistent storage: false
fusion fitted on final test: false
```

## 17. Performance Expectations and Gates

The profile trades repeated frozen inference and weaker resume behavior for
bounded storage. It must still avoid unnecessary runtime regressions.

Expected impact:

- reconstructor steady-state training: within 10% of the accelerated cache path
  when shared transient targets are approved;
- frozen tagger job: one additional pseudo-generation startup pass, with
  steady-state epochs within 5% of the old cache consumer;
- joint F-tier variants: no additional pseudo-generation pass;
- scoring: one pseudo-generation pass per source family and split, not per
  tagger;
- complete campaign wall time: target at most 1.6x the cache-heavy accelerated
  path under equal available GPUs.

Runtime exceeding these gates is not solved by spilling pseudo tensors to
shared storage. The implementation must profile source generation, RAM copying,
tagger compute, and scoring bundles separately and optimize the slow bucket.

## 18. Required Tests

### 18.1 Storage-codec tests

- packed target masks and memberships round-trip exactly;
- byte-shuffled float32 arrays round-trip bitwise;
- compact integer fields reject overflow;
- shard hashes detect changed bytes, order, dtype, or shape;
- full target objects reconstructed from compact shards satisfy all existing
  accounting invariants;
- real offline target samples compile through the production path.

### 18.2 Pseudo streaming parity

- consumer-only pseudo batches produce exactly the same tagger logits as the
  full old pseudo object on a deterministic batch;
- removed fields are proven unread by every registered tagger variant;
- frozen full-rank cache and bounded-LRU regeneration produce identical arrays;
- D1, D2, and E7 preserve expected root and hierarchy provenance;
- F0 uses one differentiable shared forward and never a detached cache;
- no pseudo output path can resolve beneath persistent campaign storage.

### 18.3 Checkpoint tests

- compact selected checkpoints reconstruct every B/C/D/E/F model strictly;
- warm starts copy the same tensors as full selected checkpoints;
- prediction logits match before and after checkpoint compaction;
- compact checkpoints contain no optimizer, scaler, duplicate EMA, or online
  model state;
- a failed allocation cannot label a compact warm start as exact resume.

### 18.4 Quota and cleanup tests

- concurrent reservations cannot exceed the hard cap;
- dry-run performs no reservation and writes no artifact;
- stale reservations require job-state audit;
- cleanup refuses paths outside the campaign root;
- cleanup refuses hash or artifact-role mismatches;
- cleanup waits for every required consumer receipt;
- projected, measured-peak, and retained-byte reports reconcile exactly;
- a deliberately oversized campaign fails before Slurm model submission.

### 18.5 Distributed runtime tests

- each rank owns disjoint ranges with complete global union;
- actual DDP RAM preflight includes model, optimizer, buckets, and pseudo cache;
- a rank-local pseudo-generation failure aborts all ranks without deadlock;
- distributed validation reduces exact numerators and denominators;
- coverage hashes use canonical ordered range records;
- rank zero is the only persistent writer.

### 18.6 Scientific parity pilot

On a fixed deterministic subset, compare the existing cache-heavy path and the
30 GB path with identical selected weights:

- target tensors: bitwise equal;
- generated pseudo consumer tensors: bitwise equal;
- tagger logits: bitwise equal when deterministic kernels permit, otherwise
  within the existing BF16/DDP numerical tolerance;
- one-update losses and gradients: within the existing parity tolerance;
- model-validation selection metric: no material regression attributable to
  storage mode.

## 19. Implementation Steps

### Step 1: Add storage contracts and quota accounting

Implement storage profiles, the persistent quota ledger, atomic reservations,
artifact classes, projection reports, and campaign-wide byte audits. Add dry-run
coverage and hard rejection above the configured cap.

### Step 2: Implement the compact lossless target codec

Add packed masks/memberships, narrow validated integers, reversible byte
shuffle, shard hashes, the complete-object loader, and full-schema forensic
sampling. Preserve float32 target values.

### Step 3: Add target-mode preflight and RAM workspaces

Measure real target size, select `shared_transient_compact` or
`rank_local_build`, validate `/dev/shm` and cgroup capacity, and implement
bounded rank-local workspace reservations and cleanup traps.

### Step 4: Implement compact selected checkpoints

Separate ephemeral resume payloads from persistent selected payloads. Update
strict loaders, warm-start provenance, checkpoint hashing, stage handoff, and
completion cleanup.

### Step 5: Implement consumer-only pseudo batches

Define the minimal in-memory schema, adapt prediction packaging and tagger
loaders, remove unused persistent fields, and prove forward/gradient parity for
every tagger family.

### Step 6: Implement frozen reconstructor RAM sources

Generate model-train/model-validation pseudo views from HLT-only inputs into
rank-local full or bounded-LRU caches. Add source hashes, deterministic
regeneration, telemetry, and explicit handling for D1, D2, E7, and diagnostics.

### Step 7: Add DDP tagger execution

Extend global batch plans, failure consensus, validation reductions, rank-zero
writes, and Slurm topology to RAM-heavy E/F/G taggers. Run the parity and speed
promotion gate.

### Step 8: Implement bundled scoring and logit-only fusion inputs

Group selected taggers by reconstructor source, generate each source split once
per scoring allocation, persist logits only, and bind every block to checkpoint,
source, split, and identity hashes.

### Step 9: Implement storage-wave orchestration and audited cleanup

Add Wave 0 through Wave 6 dependencies, consumer receipts, privilege and input
cleanup barriers, exact-path deletion, cleanup receipts, partial-stage reuse
validation, and 30 GB Tigris submitters.

### Step 10: Complete parity, failure, and campaign acceptance

Run codec, checkpoint, DDP, quota-race, cleanup-safety, scientific-parity, and
runtime tests. Queue a small real-data storage smoke, then the complete pilot
only after it stays below projected RAM and persistent-byte gates.

## 20. Files Expected to Change

Primary modules:

- `teacher_logit_reco/adaptive_binary_pseudooffline/cache.py`
- `teacher_logit_reco/adaptive_binary_pseudooffline/production.py`
- `teacher_logit_reco/adaptive_binary_pseudooffline/prediction_cache.py`
- `teacher_logit_reco/adaptive_binary_pseudooffline/tagger.py`
- `teacher_logit_reco/adaptive_binary_pseudooffline/tagger_runtime.py`
- `teacher_logit_reco/adaptive_binary_pseudooffline/training.py`
- `teacher_logit_reco/adaptive_binary_pseudooffline/distributed_stream.py`
- `teacher_logit_reco/adaptive_binary_pseudooffline/orchestration.py`
- `teacher_logit_reco/adaptive_binary_pseudooffline/report.py`

New modules are expected for:

- storage quota/reservations;
- compact target codecs;
- rank-local RAM workspaces;
- streaming pseudo sources;
- cleanup receipts and lifecycle audits.

Scripts and Slurm workers are expected for:

- storage projection and actual-byte auditing;
- compact target generation;
- bundled source-family scoring;
- privilege/input cleanup barriers;
- `streaming_30gb_v1` pilot submission.

Tests must be added under the existing
`tests/test_adaptive_binary_pseudooffline_*` family.

## 21. Definition of Done

The storage-constrained campaign is complete only when:

1. no persistent pseudo-view cache is created;
2. target values and consumed pseudo tensors preserve the scientific contract;
3. the projected peak is at most 24 GB and final retained projection at most 20
   GB;
4. measured persistent usage never exceeds 30,000,000,000 bytes;
5. every concurrent writer uses an atomic quota reservation;
6. all RAM workspaces are topology-tested and retain 20% headroom;
7. selected checkpoints contain no duplicate training state;
8. tagger and scoring outputs are bound to exact reconstructor/HLT identities;
9. cleanup is dependency-controlled, hash-checked, exact-path, and receipted;
10. DDP failures terminate without deadlock or partial persistent artifacts;
11. cache-heavy and streaming parity tests pass;
12. full model-validation selection remains unchanged;
13. final-test workers attest HLT-only streaming and no persistent pseudo cache;
14. the complete pilot report includes projected, peak, and retained storage;
15. the campaign can be rebuilt from the raw dataset, manifest contract, compact
    selected checkpoints, and recorded configuration.

Until all fifteen conditions hold, the existing cache-heavy full campaign must
not be submitted under a 30 GB research-storage quota.
