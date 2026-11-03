# Selected-relations ParT: speed optimization implementation plan

Written: 2026-11-02. Implementation status: an isolated batched-tree candidate,
optional bounded validation cache, opt-in REGION compilation, numerical tests,
paired/sustained profiling and gated launcher now exist in
[part_selected_fast](../part_selected_fast/README.md). No cluster jobs have been
submitted by this implementation work. **Native/GH200 parity, speedup and resource
recommendations are still pending the new cluster gates.** Checkboxes below are
end-to-end evidence milestones, not claims based only on local CPU tests.

## Objective and scope

Build a faster execution path for all four selected-relation models on Tigris
GH200s, preserving their scientific definition and the reusable short Ranger
recipe. Reuse the existing dataset, environment where compatible, normalization,
and comparison controls. Avoid a large persistent dataset/tree cache.

The goal is to remove avoidable implementation overhead and measure how close
training can get to standard ParT / ParT-EV throughput. **Baseline-equivalent
speed, a particular speedup, and identical final accuracy are not promises.**
The selected models still perform additional learned directed-pair computation.

| Architecture | Bias | EV | Unchanged parameter count |
|---|---|---|---:|
| `selected_shared` | Shared | Off | 2,153,406 |
| `selected_shared_ev` | Shared | On | 2,218,942 |
| `selected_layerwise` | Layerwise | Off | 2,157,158 |
| `selected_layerwise_ev` | Layerwise | On | 2,222,694 |

Use these references as the authority:

- [Reusable screen protocol](../part_ev_screen/REUSABLE_SCREEN_PROTOCOL.md).
- [Selected representation and existing memory optimization](../part_selected_screen/README.md).
- [Selected model](../part_selected_screen/model.py),
  [feature encoders](../part_selected_screen/features.py), and
  [REGION construction](../part_selected_screen/region.py).
- [Tigris contract](protocol.py), [benchmark](engine.py), and
  [native input fingerprints](inputs.py).
- [Existing numerical parity policy](../part_selected_screen/memory.py).

Read-only reference campaign on RC:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_tigris/selected_gh200_20261023T073044Z
```

Its frozen source and artifacts, not whatever is currently on `master`, define
the reference execution. Verify them before implementation and benchmarking.
The existing walltime recovery changes resources only; it does not authorize
replacing that campaign's implementation or numerical admission evidence.

## 1. Non-negotiable boundaries

### Protect existing work

- Do not edit frozen campaign source, metadata, benchmarks, checkpoints, or
  normalization. Do not cancel, restart, or migrate currently running jobs.
- Develop in a separate package and use new diagnostic/campaign output paths.
  Retain the current selected implementation as a reference and fallback.
- Do not upgrade packages in an environment used by active jobs. Reuse the
  environment if it supports the chosen implementation; report any additional
  dependency or environment requirement before changing it.
- Do not duplicate the JetClass ROOT dataset or create a persistent 100M-jet
  pair/tree cache. No automatic cleanup of existing user data.
- This plan does not authorize full production reruns. Hand off a verified
  launcher and measured resource recommendation for the user to queue later.

### Preserve the experiment

Keep seed 101, isolated added-parameter initialization, one GPU, batch 512,
native BatchNorm, two original data-loader workers, the same file/event stream,
and continuous train/validation iterators. Preserve native permutation/trimming,
128-particle cap, masks, padding, raw track aliases, labels, and input precision.

Keep 200,000 attempted updates, the unchanged `ranger_short_v1` optimizer/LR
schedule, ten checks at 20,000-update intervals, 1.28M validation exposures per
check, highest-accuracy/first-tie checkpoint selection, and final 5M validation.
Preserve AMP-skipped-update accounting, Ranger/Lookahead state, GradScaler state,
and restart behavior. No independent 20M test is opened by this work.

Preserve FP16 autocast training with FP32 parameters, FP32 evaluation, TF32 off,
and cuDNN benchmarking off. Retain PT/TRACK/REGION definitions, uncertainty
floors, fitted normalization, dense directed pair processing, and its BatchNorm
population including the existing padded entries. Do not restore triangular
packing by assuming the selected learned embeddings are symmetric.

CPU tree threads are separate from the two data-loader workers. Requesting more
CPU cores for deterministic tree construction must not alter loader sharding,
interleaving, batch order, or RNG consumption.

## 2. What can be cached, and what cannot

| Quantity | Cache rule |
|---|---|
| Fixed raw particle measurements and fixed per-particle transforms | Reuse only with matching input bytes and preprocessing/normalization identity. |
| Fixed TRACK compatibility features | Can be reused with correct particle identities, masks, indexing, and feature policy. Dense whole-population storage is not the default. |
| PT fractions and ranks | Depend on the retained particle population; a full-jet cache is not generally valid after dropping particles. |
| REGION tree and derived descriptors | Depend on the retained particle population; cache only the matching view, with correct leaf remapping. |
| Learned family embeddings, TRACK node embeddings, pair stem, attention bias, EV messages | Never cache across optimizer updates. Their weights change, and their gradients are required. |

Native training trimming randomly permutes constituents and can remove real
particles. Removing only padding does not change the physical tree, but removing
a real particle can change clustering topology, ancestor features, PT fractions,
and ranks. Pruning a cached full-jet tree is not a substitute for reclustering.

The current short screen uses 102.4M training exposures: roughly one population's
worth, not ten complete dataset passes. Measure actual repetition and cache hit
rates; do not assume every jet repeats ten times. Validation is an attractive
first cache target because evaluation retains all valid particles and checkpoint
validation revisits the 5M population. Match exact views even in validation.

### RAM feasibility

The user's possible approximately 400 GB host-RAM allocation is an upper resource
option, not a requirement to reserve or fill that amount.

- 400 GB / 100M jets is only about 4 kB per jet, before other process memory.
- At 128 particles, the current packed representation in `region.pack` occupies
  21,968 bytes per jet in array payloads, or approximately 2.20 TB for 100M such
  jets. This is an illustrative full-width calculation, not a measured average.
- One dense float32 pair channel at width 128 over 100M jets is approximately
  6.55 TB. Caching every expanded relation tensor is not viable in 400 GB.
- Measure actual multiplicity, packed bytes, object/index overhead, peak RSS,
  and reuse distance before selecting a cache size. A compact topology-oriented
  representation may be much smaller, but topology alone does not contain all
  the continuous quantities needed for REGION.
- RAM is local to each allocated node/job and disappears at job end. Four jobs
  on different nodes do not share a single cache. Startup/rebuild cost must be
  included in runtime projections and restart tests.

## 3. Implementation layout

The layout below describes the planned responsibilities. The first implementation
uses `model.py` for the optional pure REGION compilation, `profiling.py` for scopes
and resource telemetry, and `scripts/submit_part_selected_fast.py` behind the Bash
wrapper. A separate `features.py` is deferred until a measured GPU bottleneck
justifies more fusion. See the README for the actual, tested CLI, not this table.
Old packages and frozen references remain unchanged.

| Proposed location | Responsibility |
|---|---|
| `part_selected_fast/model.py` | Opt-in execution adapter; same parameters, buffers, and state-dict keys as the reference selected model. |
| `part_selected_fast/trees.py` and `csrc/selected_tree_batch.cpp` | Batched CPU tree backend, direct packing, resource-bounded thread pool. |
| `part_selected_fast/cache.py` | Optional bounded, model-independent RAM cache and telemetry. |
| `part_selected_fast/features.py` | Candidate GPU fusion and pair-encoder execution optimizations. |
| `part_selected_fast/parity.py` | Primitive, model, optimizer, stream, and restart comparisons. |
| `part_selected_fast/protocol.py` | New execution contract referencing verified original science and donor artifacts. |
| `scripts/profile_part_selected_fast.py` | Read-only reference diagnostics and paired candidate benchmarks in a new output directory. |
| `scripts/run_part_selected_fast.py` | New campaign driver using established training/evaluation semantics. |
| `sbatch/submit_part_selected_fast_tigris.sh` and worker | Frozen-source dependency graph, explicit resource limits, dry-run and status. |
| `tests/test_part_selected_fast_*.py` | Numerical, cache, lifecycle, contract, and fake-Slurm regression coverage. |

Keep an explicit reference mode and individually selectable optimization flags
for diagnostics. Freeze the admitted combination before production. If a selected
optimization is unavailable or fails, stop or use a previously declared reference
fallback and record it; never silently benchmark one path and train another.

## 4. Execution phases

### Phase A — Establish a measured reference

- [ ] Verify frozen source/spec/input/normalization hashes and runtime identity.
- [ ] Profile all four selected variants, first on a small diagnostic batch and
  then the real batch-512 stream. Include variable multiplicity and full length.
- [ ] Separately time loader wait, input copies, native trimming, raw-token
  construction, PT, TRACK, device-to-host transfer, CPU clustering, validation/
  packing, host-to-device transfer, REGION ancestor/descriptors, pair stem/heads,
  transformer, backward, optimizer, validation, and checkpoint writing.
- [ ] Record CPU thread utilization, GPU utilization, CUDA allocated/reserved
  peaks, process/child RSS, pinned buffers, and transfer volume where measurable.
- [ ] Use scoped CPU timing and CUDA events/profiler traces with synchronization
  only at measurement boundaries. Do not sum overlapping CPU/GPU spans as if
  they were sequential. Repeat end-to-end timing without detailed profiling.
- [ ] Add short standard ParT / ParT-EV throughput probes on the same GH200 and
  runtime. These are disposable diagnostics, not retraining comparison controls.

Output: `profile.json` and a short `profile.md` ranking measured bottlenecks.
The CPU-tree hypothesis remains a hypothesis until these measurements exist.
Prioritize subsequent work by measured walltime, not parameter count.

### Phase B — Batched, parallel CPU trees and direct packing (first candidate)

- [ ] Extract a pure C++ per-jet computational core into the new backend without
  changing merge distances, float64 arithmetic, canonical leaf ordering, tie
  rules, E-scheme recombination, K=2/4/8 assignments, or final float32 casts.
- [ ] Accept a batch in one call. Parallelize independent jets, never the
  order-sensitive operations inside an individual tree. Preserve output order.
- [ ] Keep Python/pybind object creation out of parallel regions. Release the
  GIL around native computation; make per-jet storage thread-local and propagate
  exceptions deterministically. No nested oversubscribed thread pools.
- [ ] Return packed arrays directly rather than constructing hundreds of Python
  dictionaries and repeatedly converting tiny tensors/NumPy arrays per batch.
- [ ] Consolidate duplicate tree checks into one equivalent validation boundary.
  Preserve rejection of malformed/nonfinite trees; do not remove error checks
  merely to improve a benchmark.
- [ ] Test 1/2/4/8 and, if allocated resources permit, 16 tree threads. Account
  for the existing loader workers and training/control threads in CPU requests.
- [ ] Keep native trimming at its current location. Copy only required inputs,
  and consolidate transfers where beneficial. Do not advertise zero transfers
  while tree construction still runs on the CPU.

Acceptance: identical categorical/topological results and deterministic packing
at every thread count; numerical and model parity as defined in Section 5;
measured improvement including transfer and packing overhead.

### Phase C — Bounded RAM caching (conditional on reuse measurements)

- [ ] Begin with immutable validation trees/fixed descriptors. Measure training
  hits before enabling a training cache. Compare cache-disabled, cold, and warm
  end-to-end operation on the actual stream, not a repeated single batch alone.
- [ ] Key records by dataset/preprocessing/backend/normalization identity and
  exact retained particle content. Use stable file/entry plus particle IDs only
  if these can be transported without altering the numerical stream; otherwise
  use verified content-based identity and measure its hashing overhead.
- [ ] Initially permit exact-order view keys. Permutation-invariant keys and
  remapping are a later optimization requiring duplicate-particle/tie tests.
  Changing a retained real particle must produce a miss, not an approximate hit.
- [ ] Store only fixed information: topology/node data and/or fixed transforms,
  never tensors attached to autograd or learned outputs. Separate normalization-
  dependent data from raw trees in cache invalidation rules.
- [ ] Use packed arenas/arrays with explicit offsets and bounded index overhead,
  not millions of Python tree dictionaries. Compact integer widths must preserve
  sentinel values and pass round-trip tests. Do not downcast float features as
  an unreported memory optimization.
- [ ] Add a strict byte budget, deterministic safe eviction, immutable entries,
  hit/miss/build-time counters, and allocation-failure handling. A miss or eviction
  must produce the same computation as no cache. Never spill to home storage.
- [ ] Select the budget below the actual Slurm/cgroup memory allowance after
  measured loader, process, backend, and transfer-buffer headroom. Do not pin
  the whole cache; use bounded pinned staging buffers only if beneficial.
- [ ] Ensure an empty cache after restart changes only speed, not outputs, RNG,
  sample order, or optimizer state. Do not serialize the RAM cache in checkpoints.

Decision: enable only a measured net benefit including startup, lookups, misses,
and memory overhead. If the short training stream has poor reuse, ship the cache
disabled for training rather than forcing a large upfront precompute.

### Phase D — GPU feature and encoder optimization

- [ ] Profile and fuse pure tensor feature normalization/masking/assembly where
  supported by the existing runtime. Keep CPU tree building outside compiled
  graphs. Measure compilation cost, graph breaks, and dynamic-shape recompiles.
- [ ] Preserve the existing single packed EV-stem allocation. Do not reintroduce
  per-layer copies or activation recomputation to obtain a smaller memory figure.
- [ ] Explore a fused implementation of TRACK's first pair linear map to avoid
  materializing the full 81-channel concatenation. A secondary candidate is to
  distribute endpoint-only linear terms and compute their projections per node.
- [ ] Treat algebraic reassociation as numerically sensitive: autocast rounding
  of `right-left` and `left*right`, accumulation order, and backward gradients
  may differ from the original concatenation-plus-Linear. Preserve the original
  trainable tensors and test it separately; reject if parity does not pass.
- [ ] Reuse identical geometry only after verifying matching epsilon, masks,
  precision, and coordinate definitions. Similar-looking calculations are not
  automatically interchangeable.
- [ ] Optionally batch the layerwise final bias projections while retaining
  separate head parameters and BatchNorm buffers/populations. Benchmark before
  keeping this; do not mistake kernel count reduction for measured speedup.

Do not freeze encoders, replace nonlinearities, reduce channels, enable TF32,
change attention/dropout, or change the optimizer to improve these timings.

### Phase E — Pipelining (optional, higher-risk)

- [ ] First overlap independent work within a batch where dependencies allow.
  Then assess bounded CPU-tree preparation ahead of GPU consumption.
- [ ] Preserve exactly the original post-trimming views. Do not move random
  trimming into ordinary loader workers or pre-draw future random numbers from
  the model's RNG stream without a proven equivalent state-management design.
- [ ] Include dropout/trimmer RNG ordering, GradScaler skips, validation-boundary
  transitions, exception propagation, signal handling, and checkpoint/restart
  semantics in the design. Never train on stale trees for a prefetched batch.
- [ ] Queued work must either be discardable/replayable without changing RNG or
  have explicit restart-state handling. Cache/queue speed state must not become
  an undocumented dependency for reproducing the next optimizer update.
- [ ] Keep this phase disabled if exact stream/RNG preservation cannot be shown.

Pipelining is not required to ship the simpler batched-tree optimization. It must
earn its complexity through an additional measured end-to-end gain.

### Phase F — Combined admission, queue-ready integration, and handoff

- [ ] Compare each candidate independently, then the combined path. Keep only
  admitted features; record rejected or deferred ideas and why.
- [ ] Create a new execution/campaign contract with original science references,
  source/backend hashes, resource settings, cache/fusion flags, and parity evidence.
  Do not make the old strict protocol accept new code by disabling its checks.
- [ ] Preserve state-dict/parameter compatibility, but keep campaign identity
  strict. Existing selected checkpoints may be loaded read-only for diagnostics;
  do not silently resume an old production run under the new execution contract.
- [ ] Reuse existing data, normalization and verified comparison results through
  read-only references. No duplicate control training or full control inference
  is needed solely to benchmark implementation speed.
- [ ] Provide a dependency graph for a new campaign:
  `prepare -> parity/benchmarks -> gate -> train[4] -> lock -> evaluate[4] -> report`.
- [ ] Keep every pre-production job strictly below six hours (normally at most
  five). Split long diagnostics into bounded jobs if necessary; never convert
  incomplete measurements into a passing gate.
- [ ] Determine CPU/RAM and train/evaluation walltimes from measured results with
  the existing 30% timing margin. Respect current partition/account limits and
  leave adequate GPU and host memory headroom. More RAM can worsen schedulability;
  choose the smallest allocation that delivers the measured benefit.
- [ ] Freeze source at submission, provide `--dry-run`, read-only `status`, safe
  retries, task-specific logs, explicit job ledgers, and duplicate-job protection.
  Show the four task mappings and do not submit the reused controls.

The old resource-only recovery launcher is not the new implementation launcher.
Old benchmarks do not authorize new kernels, a new tree backend, or new caching.

## 5. Correctness and equivalence gates

Test reference versus candidate on the **same GH200/software runtime**. Historical
SPORC controls remain useful scientific screening references but cannot establish
same-hardware throughput or bitwise cross-platform parity.

### Required test matrix

- [ ] All four architectures; FP32 evaluation, FP32 training and FP16 training.
- [ ] Empty/single-particle cases, varying multiplicities through 128, padding,
  invalid tracks, repeated/equal particle values, tied merge distances, phi wrap,
  unequal-depth trees, and native random permutations/truncations.
- [ ] Exact tree topology, canonical ties, assignments, masks, labels, input
  identities, and discrete control state. Require unchanged per-tree arithmetic
  for the batched CPU path; investigate any continuous tree-value discrepancy
  rather than relaxing the gate automatically.
- [ ] Raw and normalized family features, logits, loss, gradients for every
  parameter-owning module, BatchNorm buffers, and all parameter counts/keys.
- [ ] Multiple optimizer updates including Lookahead synchronization, a boundary
  of the LR schedule, and controlled AMP-skip/error tests. Check optimizer slow
  weights/counters, scaler state, and model/trimmer RNG restoration.
- [ ] Native stream fingerprints at batches 0, 20, 1020, 2020 and 4020, plus
  post-trim comparisons on a continuous training segment. Diagnostic work must
  not consume production batches or mutate production state.
- [ ] Cache on/off, misses/hits/eviction/corrupt-key rejection, alternate tree
  thread counts, and restart with an empty cache. Test interrupted writes and
  restart around a validation/checkpoint boundary.
- [ ] Strict checkpoint import into the identical architecture; malformed or
  incompatible metadata must still fail closed.

Use the existing numerical policy as the initial frozen gate, not a tolerance
chosen after observing a failing candidate:

| Mode | Logit rtol / atol | Module-gradient relative L2 / absolute floor |
|---|---|---|
| FP32 | `1e-5` / `1e-6` | `2e-5` / `1e-6` |
| FP16 | `2e-3` / `2e-3` | `5e-3` / `2e-5` |

Use the existing gradient-norm bound implementation, including its absolute
floor for near-zero gradients. Preserve finite-value rejection. Define and
record any additional buffer/optimizer-state comparison thresholds before the
candidate trial; do not silently widen the existing ones. Retain per-module
errors so a small feature encoder cannot hide behind the larger backbone.

Passing local parity does not guarantee an identical stochastic training
trajectory or final accuracy. Add a matched finite training pilot that exercises
normalization, dropout, trimming and cache lifetimes. Follow with the complete
screen when queued by the user, reporting all four results and differences, not
only favorable outcomes. No model-performance equivalence claim from speed alone.

## 6. Performance evidence and acceptance

Use paired reference/candidate trials with identical batch sequences and initial
state, on the same GPU/runtime, with repeated measurements and alternating order
where practical. Retain the native cold-start/trimmer behavior in an end-to-end
trial. Separate isolated warm compute tests from the production stream.

The final sustained benchmark must cover at least the existing selected gate:
20 warmup steps, 5,000 measured steps in 1,000-step windows, four full-length
steps, and 131,072 validation events. Preserve the existing AMP skip (<1% in
measured steps, no skips in full-length checks), precision, and GPU memory
(<90% allocated peak) gates; additionally account for reserved/device memory
and measured host-memory headroom. Do not fill available RAM to the cgroup limit.

Record for each model and execution variant:

- Cold preparation, compilation and cache-fill costs, not just warm iteration time.
- End-to-end jets/second, step-time median and slow-window/tail behavior.
- CPU tree/packing and GPU forward/backward time, loader wait and validation rate.
- Cache hit rate/bytes, host peak RSS, CPU allocation, GPU peak allocated/reserved.
- Projected complete training and 5M evaluation times, including checkpoint and
  restart/cache-rebuild costs; queue delay is separate and not predictable here.
- `reference_selected_time / candidate_time` and
  `candidate_time / matching_standard_model_time`, measured on the GH200.

Profile/caching startup is not free: include it in an amortized estimate for this
200k-update screen. Do not project from a fully warm synthetic cache unless its
hit pattern matches the real stream. If feasible, do a bounded concurrent-job
check before choosing four-way production concurrency, because loader/filesystem
contention is absent from a single-job microbenchmark.

A candidate is ready to replace the execution path only if correctness gates
pass, repeated end-to-end measurements show a gain beyond measurement variation,
resource limits pass, and the entire projected job is faster after startup costs.
Report actual ratios; do not relabel a marginal or noisy result as a large gain.
Revert unhelpful optimizations individually. Near-baseline speed is a stretch
goal, not a reason to change the model or suppress unfavorable measurements.

## 7. Explicitly out of scope without a separate scientific decision

- Computing all relations on untrimmed jets, freezing a retained subset, or
  disabling native trimming to make a cache reusable.
- Freezing learned relation embeddings, pruning relation channels, approximating
  clustering, changing normalization, or caching stale learned tensors.
- Changing batch size, accumulation, precision policy, LR/optimizer, validation
  frequency, selection rule, dataset, or training budget.
- Treating dense directed pair BatchNorm as equivalent to triangular BatchNorm,
  removing padded entries from its statistics, or assuming pair symmetry.
- A wholesale GPU-tree rewrite before simpler measured optimizations are tried.
  It may become a later candidate, but tie decisions and float64 tree numerics
  would require their own full equivalence gate.
- Resuming currently running jobs under new code or automatically launching
  full reruns as part of a diagnostic.

## 8. Completion checklist and later execution order

- [ ] A: Frozen reference audited; bottleneck profile and RAM/reuse measurements saved.
- [ ] B: Batched parallel tree construction/direct packing tested and timed.
- [ ] C: Bounded cache tested; default policy justified by measured reuse (off is valid).
- [ ] D: GPU/encoder candidates tested; only passing, useful changes retained.
- [ ] E: Pipelining either proven and measured or explicitly deferred.
- [ ] F: All-four-model combined numerical and sustained performance gates pass.
- [x] Existing selected/Tigris/recovery and shared-engine regressions pass;
  CUDA-only checks are honestly marked pending when running local CPU tests.
- [x] New launcher passes fake-Slurm tests: correct graph/resources, immutable
  snapshot, failed-gate blocking, no control retraining, no active-job mutation,
  no duplicate submissions, and safe resume after a cache loss.
- [ ] Handoff contains implementation flags, per-model speed/memory table,
  scientific caveats, estimated walltimes, and tested dry-run/queue/status commands.
- [ ] Persistent overhead is listed explicitly: source/backend binaries, small
  reports/traces, normal checkpoints/predictions; no new persistent jet cache.

### Implementation handoff (2026-11-02)

- Implemented: batched independent CPU trees/direct packing; isolated model and
  engine bindings; all-four-model parity including features, gradients, native
  trimming, Lookahead, LR boundary and serialized restart; bounded RAM-only
  validation cache; optional pure REGION compilation; immutable new campaign
  contract; paired and sustained gates; dry-run/status/review and explicit
  gated training/evaluation submission. The default command stops at admission.
- Default candidate: four tree threads, cache off, compilation off. Optional
  features require a fresh campaign and the same gates; no optimization is
  described as admitted before those measurements exist.
- Deferred: training cache, TRACK linear-map reassociation, per-layer head fusion,
  and pipelining. Native training has limited guaranteed reuse, and reassociation
  or prefetching can change AMP/RNG behavior. Measure the simpler candidate first.
- Still remote-only: compiled native equivalence, real GH200 parity/performance,
  thread/cache/compile sweeps, device utilization analysis, concurrent-job
  contention, and measured final resource recommendations. Local oracle tests
  and fake scheduler tests do not satisfy those evidence milestones.
- Local verification: 172 passed, nine explicitly skipped (eight CUDA checks and
  one Linux native build), across the selected/fast/Tigris/recovery regression
  suite. Restart/checkpoint tests cover the new execution; fake Slurm tests do not
  run production jobs or assert anything about actual queue availability.

Execute A -> B first. Re-profile, then choose C/D according to the remaining
bottleneck. E is conditional. Finish with combined F regardless of which optional
optimizations are retained. A faster, simpler verified subset is preferable to
shipping every proposed mechanism without a measurable benefit.

## Supporting reference

[PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
describes operation fusion, data-loading overlap, and avoiding unnecessary
CPU/GPU synchronization. These motivate candidate techniques, not a claim that
every technique works on the pinned cluster environment. The repository's frozen
implementation and actual parity/throughput measurements remain authoritative.
