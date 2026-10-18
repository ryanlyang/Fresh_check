# Grouped-attention screen: implementation plan

Date: 2026-10-17  
Status: initial four-arm software implemented and locally tested, 2026-10-17. The requirements below are retained as the implementation contract; remote donor/runtime checks and real GPU admission are still required. No jobs were submitted. Executable instructions: [part_group_screen/README.md](../part_group_screen/README.md).

## 1. Deliverable and governing documents

Implement the complete **initial screen** in [GROUPED_ATTENTION_STUDY_PLAN.md](GROUPED_ATTENTION_STUDY_PLAN.md), using [REUSABLE_SCREEN_PROTOCOL.md](../part_ev_screen/REUSABLE_SCREEN_PROTOCOL.md) as the training and control-reuse contract.

The deliverable is an independently frozen experiment that can:

1. Train fixed-physics, fully learned, and physics-guided learned grouping from scratch, plus the recommended matched-size random-group control.
2. Reuse verified `PART_RANGER` results, optionally also `EV_RANGER`, without training those controls again or repeating their completed evaluation.
3. Run the inherited short training recipe and selected-checkpoint evaluation on the same official 5M validation population.
4. Produce matched accuracy/AUC/rejection comparisons, learning curves, grouping diagnostics, and measured parameter/memory/throughput costs.
5. Resume interrupted work without changing the experiment or touching the donor campaign.

The study's later extensions are covered by the roadmap in Section 12, but are **not extra models in the initial submission**. This distinction keeps the original comparison interpretable and avoids turning one screen into an uncontrolled sweep.

### Fixed scope

| Category | Initial implementation |
|---|---|
| New train/evaluate tasks | `PART_GROUP_FIXED_RANGER`, `PART_GROUP_LEARNED_RANGER`, `PART_GROUP_HYBRID_RANGER`, `PART_GROUP_RANDOM_RANGER` |
| Reused control | `PART_RANGER`; optional read-only `EV_RANGER` |
| Backbone | Native shared-bias ParT, unchanged inputs and readout; no EV or selected relation families in grouped arms |
| Recipe | Complete donor `ranger_short_v1`, seed 101 |
| Budget | 200,000 attempted updates; batch 512; no accumulation; one donor-compatible SPORC A100 |
| Checks | Ten checks, every 20,000 updates; 1,280,000 validation examples per check |
| Selection / final evaluation | Highest validation accuracy, first exact tie; selected checkpoint on all 5M validation jets |
| Runtime | Donor-matched software, GPU model/memory, native BatchNorm, FP16 training / FP32 evaluation |
| Pre-production jobs | Each requests less than six hours |

Use the source protocol's exact LR function, optimizer, skip handling, continuous data streams, and precision context. Do not create independently adjustable LR/epoch/batch flags for this screen. Changing these settings constitutes another experiment with matching controls.

Control reuse is **result reuse**, not weight reuse. Every grouped model starts from the matched seeded native initialization; no trained donor weights or probe/recovery state initialize production training.

## 2. What already exists, and what needs adaptation

The following observations come from the current repository. The donor's frozen source and metadata remain the authority for a real campaign.

| Existing code | Reuse | Adaptation needed |
|---|---|---|
| [Reference model](model.py) | Pinned Weaver `0.4.17`, native ParT configuration, unchanged backbone weights | New grouped wrapper; do not extend the original full-scale architecture registry just to run this screen |
| [Screen protocol](../part_ev_screen/protocol.py) | Recipe dictionaries and `learning_rate` | Its `TASKS`, task-directory checks, source manifest, and admission review assume seven specific tasks |
| [Screen model](../part_ev_screen/model.py) | Isolated initialization pattern | New architecture factory and grouping diagnostics; existing EV diagnostics are not group diagnostics |
| [Precision](../part_ev_screen/precision.py) | Outer plus Weaver-native autocast, observed activation audit | Preserve `model.mod` compatibility; audit added modules and geometric reductions explicitly |
| [Training streams](data.py), [screen data](../part_ev_screen/data.py) | Continuous seeded loaders, finite validation, fingerprints | Resolve routing metadata without altering predictive inputs, RNG, or stream order |
| [Optimizer](optimizer.py), [checkpoint state](checkpoint.py) | Ranger/Lookahead state, scaler/RNG/trimmer capture, atomic saves | Include grouped parameters and any extra state in the new campaign's checkpoint schema |
| [Screen driver](../scripts/run_part_ev_screen.py) | Step/validation/evaluation semantics, per-file predictions and selection | Remove seven-task/model-specific coupling through explicit interfaces, not monkey-patching globals |
| [Reporting](../part_ev_screen/reporting.py) | Metric conventions and identity checks | Replace fixed EV contrasts with grouped contrasts and external-control references |
| [SPORC submitter](../sbatch/submit_part_ev_screen_sporc.sh) | Frozen source, job ledger, duplicate-submission protection, reviewed GPU admission | Dynamic new-model task count; no donor training/evaluation entries |

Two implementation traps deserve explicit handling:

- The native trimmer permutes and truncates particles inside `torch.no_grad()`. It returns the aligned features, four-vectors, mask, and optional pair inputs, but **not the permutation indices**. Capturing its output is safe; computing the trainable grouping branch inside that hook would disable its gradients.
- The current screen's `initial_parity` expects exact initial output equality and treats non-`part` architectures as EV. That is unsuitable for grouping with a nonzero `gamma=1e-3`. Test grouped parity with the branch deliberately disabled, then restore its real initialization.

The locally inspected Weaver implementation matches the SHA-256 pinned in `reference_metadata()`. Still verify that identity in campaign preparation; an installed version number alone is insufficient.

## 3. Implemented code organization

Generic screen reuse is separate from grouped architecture code. Implemented paths:

```text
part_screen_common/
    contracts.py          # shared-field compatibility and immutable manifests
    controls.py           # read-only donor validation and result references
    engine.py             # reusable step/validation/checkpoint/evaluation helpers

part_group_screen/
    config.py             # frozen settings and four-new-task registry
    protocol.py           # grouped schema, source manifest, imported donor contract
    geometry.py           # stable four-vector summaries, angles, masks
    grouping.py           # fixed clustering, soft router, hybrid, random assignment
    branch.py             # common pooling, group transformer, particle readback
    model.py              # pinned native wrapper, insertion hooks, context cleanup
    metadata.py           # routing identity alignment without predictive inputs
    diagnostics.py        # isolated measurements on a fixed development sample
    reporting.py          # grouped results joined with validated donor controls
    README.md             # tested operations and recovery guide

scripts/export_part_screen_controls.py
scripts/run_part_group_screen.py
sbatch/submit_part_group_screen_sporc.sh
sbatch/run_part_group_screen_sporc.sh
tests/test_part_group_*.py
tests/test_part_screen_controls.py
tests/test_part_group_training.py      # real-model and engine lifecycle tests
```

The implementation directly reuses the actual original training step, validation, metrics, loader, precision, optimizer, and checkpoint helpers. `part_screen_common.engine.ScreenEngine` adds orchestration with explicit protocol/model/diagnostic callbacks. The original driver and its globals were not modified or monkey-patched; historical trainers do not use the new engine. Regression and real-model restart tests cover the original and new paths.

If shared code is extracted, extend new source manifests to include every imported helper and worker script. Existing active campaigns continue using their own frozen `source/`; never rewrite those snapshots. A new source hash is expected, whereas the unchanged numerical/data semantics must be demonstrably equivalent.

Suggested model-facing interfaces:

```text
assign(hidden[B,N,128], vectors[B,N,4], valid[B,N], metadata)
    -> Assignment(weights[B,N,K], active[B,K], optional_diagnostics)

group_branch(hidden, vectors, valid, metadata)
    -> residual_correction[B,N,128]

build_group_model(architecture, seed, config, trim=True, use_amp=False)
    -> native-compatible four-input model with model.mod
```

`K=8` initially. Valid assignment rows sum to one; padded rows and unavailable slots are zero. All arms use the same branch interface. Diagnostics should be opt-in and bounded, not full training-population assignment dumps.

## 4. Work package A: contracts and reusable controls

Implement this before production architecture work, so the rest of the plan targets a definite compatible experiment.

### A1. Import and freeze the donor contract

Default donor:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z
```

Read the donor's `campaign.json`, frozen source identities, preprocessing configuration, runtime/admission evidence, and requested task definitions. Import the **whole** selected recipe and shared contract, not a handful of matching numbers.

Separate these records:

- `experiment_spec.json`: immutable new architecture configs, new tasks, seed/recipe/data/runtime contract, source hashes, and donor identity.
- `controls.json`: atomically refreshed validation status and references, tied to the immutable expected donor identity; retain an audit history of refreshes.
- `comparison_lock.json`: final immutable set of verified control/new-model checkpoint and prediction hashes used by the completed report.

This separation lets a control progress from pending to complete without changing the new training campaign hash or invalidating its checkpoints. A different donor hash, selected checkpoint, or changed committed artifact is not an ordinary status refresh.

Compare normalized shared fields. Runtime records contain a campaign hash, so whole-dictionary equality across different campaigns would reject legitimate reuse. Do not ignore GPU/precision/software differences while removing bookkeeping fields.

**Dataset evidence caveat:** the current inventory records names, sizes, timestamps, and entry counts, not full ROOT payload SHA-256s. Do not describe its JSON hash as a content checksum. Preserve the donor identity evidence and verify staged archive/member manifests when available. A relocated or insufficiently evidenced dataset needs additional read-only content verification and review; record exactly what is verified. Run expensive inventory/hash work in bounded pre-production jobs, splitting it if necessary.

### A2. Validate only the requested controls

For `PART_RANGER`, and `EV_RANGER` if requested:

1. Verify task identity, seed, complete recipe, data/preprocessing and source/precision/runtime evidence.
2. Validate all ten checkpoint-check records, 200,000 attempted steps, completion and optimizer records, and `best_model_val.pt` SHA-256. Recompute highest-accuracy/first-tie selection from the check records.
3. Validate the selected-checkpoint lock and full-validation metrics/prediction sidecars. Check hashes, shapes, finiteness, labels, and exactly 5M unique file/entry identities with 500k per class.
4. Record `pending_training`, `pending_evaluation`, `verified`, or `incompatible` with specific reasons. Corruption is not pending completion.
5. Never require unrelated donor models to finish just to reuse a completed requested control, unless a necessary donor artifact is genuinely unavailable.

Do not use the current `part_ev_screen.protocol.load_campaign` as a generic historical reader: it compares a campaign to the current fixed seven-task source/schema. Build an explicit supported donor-schema reader that validates against the donor's recorded frozen source instead.

Read only the metadata and prediction formats necessary for comparison; do not deserialize trained donor weights into a new model. The donor remains a read-only path, including during status refresh and report generation.

**Acceptance:** malformed hashes, missing checks, wrong seed/recipe/runtime, wrong selected checkpoint, duplicated/missing events, and changed prediction contents all fail their appropriate checks. A pending control allows admitted new training but prevents a final comparative report. Refreshing status cannot mutate `experiment_spec.json` or donor files.

## 5. Work package B: native integration and shared branch

### B1. Preserve native ParT and insertion order

Construct the ordinary native wrapper first so all existing `mod.*` state-dict keys and initial tensors match `PART_RANGER`. Register the added branch once, outside the native blocks, under a separate prefix. Construct it in an isolated CPU RNG scope, using stable component-specific seeds.

Use the same initialization for common branch components across all four arms; learned and hybrid routers also start identically. Creating one router must not change downstream branch initialization or the subsequent training RNG stream.

Recommended integration:

1. A native-trimmer output hook captures the **post-trim** four-vectors, mask, and aligned routing metadata for the current forward.
2. A forward hook on `mod.blocks[3]` receives the output after particle block four. It computes the differentiable branch inside the native forward/autocast context and returns the corrected tensor before block five.
3. The wrapper clears captured tensors in `finally`, including failures. No stale batch context, retained autograd graph, or duplicate hook registration may survive a forward.
4. Keep `model.mod` visible so the existing precision context controls Weaver-native autocast. Preserve native embeddings, pair bias, dropout, class blocks, and classifier.

Do not put the trainable branch in the trimmer hook. Do not rerun the trimmer, separately truncate vectors, replace particle attention, or rewrite the whole native forward without a demonstrated need. Single-GPU execution is the initial contract; mutable forward context is not assumed safe under `nn.DataParallel`.

### B2. Implement common geometry and pooling

Implement the study's seven descriptors and assignment-weighted hidden-state mean/four-vector sum in one place. Use only actual post-trim valid particles. Keep the distinction between summed-four-vector directions and the hybrid router's weighted angular centers.

Freeze units and all floors/masks in `config.py` before screening. Include transverse-momentum/energy denominators, rapidity boundaries, mass-squared roundoff, empty occupancy, circular-mean degeneracy, narrow jets, and the `r_jet >= 0.05` hybrid radius. Use stable expressions whose backward pass is finite at tested degeneracies; forward-only clamps are not sufficient evidence.

Use FP32 for sensitive geometry/reductions under local autocast-disabled scopes, while retaining the donor backbone precision and added learned projections under the normal training precision context. Record this branch-specific policy explicitly. Do not silently convert the whole model to FP32 to make a test pass, or detach differentiable assignments/centers to avoid a bad gradient.

Define behavior for no valid particles as an explicit invalid-input failure before attention; low-multiplicity but nonempty jets remain supported. Mask inactive slots before softmax. If a numerical occupancy policy removes a soft slot, record removed mass; do not renormalize or change policy without a versioned definition.

### B3. Implement group processing and readback

- Token builder: hidden mean projected to 128 plus the common `7 -> 64 -> 128` GELU descriptor MLP, then LayerNorm.
- Group processor: one pre-LayerNorm attention block, width 128, eight heads, FFN `128 -> 256 -> 128`, zero added dropout.
- Readback: normalized particle queries and normalized group keys/values, ordinary projected cross-attention, no added particle FFN or group-pair bias.
- Output: `H + gamma * correction`, with one trainable scalar initialized to `1e-3`; zero correction on padded queries.

Freeze normalization epsilon, linear biases, initialization choices, and explicit attention settings. Keep native backbone dropout unchanged. Return the corrected states in the native sequence-first layout and expected dtype.

**Acceptance:** exact equality of unchanged initial weights and post-construction RNG state; gate-disabled same-platform evaluation parity at the existing fixed tolerance; no branch gradient computation under `no_grad`; finite gradients into every intended active branch component at `gamma=1e-3`; masking and low-multiplicity tests; context cleanup after success and exceptions. Initial active-branch logits are not required to equal baseline logits.

## 6. Work package C: four assignment mechanisms

### C1. Fixed angular grouping

Build a simple CPU reference implementation first: initialize one cluster per valid retained constituent, repeatedly merge the minimum rapidity–azimuth-distance pair with four-vector addition, and stop at `min(8,N)` clusters. Return a one-hot assignment to the original retained particles.

This is the specified fixed-count angular tree, not an interchangeable inclusive FastJet call. If an external clustering library is used later, prove its stopping/recombination behavior matches the reference on fixtures. Do not install a new library into an environment used by active jobs.

Define canonical tie handling independently of array order. Test exact duplicate/degenerate inputs separately; slot permutations are harmless, but changed physical partitions for distinguishable inputs must be explained and tested.

Then implement a batched/device-oriented equivalent if the reference implementation is too slow. Avoid a Python CPU transfer/synchronization per merge for production. Optimize against the oracle without changing tie rules or recombination. Benchmark grouping cost within an end-to-end training step.

**Acceptance:** assignments match hand-worked fixtures and the oracle; membership conservation; deterministic output; wrapped-angle handling; permutation-equivalent group contents; `N<=8` behavior; fixed versus optimized agreement within declared geometric tolerances.

### C2. Fully learned router

Use eight trainable initial slot vectors, shared particle key/value projections, three assignment rounds, and a shared GRU-cell update between rounds. Softmax is over **groups per particle**. Slot updates use occupancy-normalized particle-value means; there are two recurrent updates and three assignment calculations.

Finalize and freeze the exact slot normalization/query projection and GRU convention in code/configuration before comparative runs. Both learned arms must use this single router implementation. Only the final assignment matrix enters common pooling; do not add the GRU's final slot state as another downstream feature.

Mask unavailable slots for `N<K`, and exclude padded rows from pooling. Start with temperature one, no hard sampling, no straight-through estimator, no detached assignments, and no balance/diversity loss.

**Acceptance:** row sums and masks; finite forward/backward; useful nonzero gradients into slot vectors/projections/recurrent weights on nondegenerate fixtures; identical results on repeated evaluation; permutation behavior; finite behavior for concentrated or diffuse assignments. Diagnose collapse rather than quietly changing the objective.

### C3. Physics-guided learned router

Extend the same router with the study's angular penalty, fixed `lambda=1`:

- Round one is the fully learned router exactly.
- Before rounds two and three, compute assignment-pT-weighted rapidity/circular-azimuth centers from the previous round.
- Subtract `lambda * delta_R_squared / (2 * r_jet_squared)` from neural scores; `r_jet` is the retained jet's pT-weighted RMS radius with floor `0.05`.
- Guard empty/degenerate centers with the documented jet-axis fallback while preserving gradients in the nondegenerate regime.

With `lambda=0`, use the neural-only path so equivalence cannot be spoiled by an unused invalid center calculation. The nominal `lambda=1` path must retain differentiable center dependence. No additional trainable parameters, hard radius cutoff, or physics-based slot initialization in this arm.

**Acceptance:** `lambda=0` outputs and gradients match learned routing under identical weights; round-one equality; finite center gradients and periodic-angle tests; learned/hybrid parameter counts and common initial weights agree.

### C4. Random grouping with matched sizes

Reuse the fixed grouping's group-size multiset, then assign the same retained particles to those capacities with a deterministic label-independent random rule. The common branch and its initial weights must match fixed grouping. This is a new trained mechanism control, not a saved baseline.

**Resolve metadata explicitly before calling this arm complete.** The current trainer passes four predictive tensors and labels, and the native trimmer does not expose its permutation. Neither batch position nor post-trim array index is a stable original identity.

The alternatives considered were:

1. Carry stable within-event constituent keys through a metadata-only adapter. If stable event identity is available, use a class-neutral opaque event key, never class labels or class-encoded filename prefixes. Metadata must not enter the 17 input features, physical descriptors, or classifier.
2. Align keys to actual trimmer outputs using verified pre/post-trim feature/vector correspondence or a parity-tested metadata-preserving trimmer adapter. Do not call the trimmer twice or consume an extra random permutation. Duplicate constituents and ties need explicit handling.
3. If source identifiers cannot be carried without changing the stream, document and review a stateless, order-independent content-key alternative derived only from the unchanged input tensors. Test exact duplicates, key collisions, and permutation behavior before accepting it as an equivalent control definition; do not silently substitute it.

**Adopted resolution:** alternative 3 is implemented as `canonical_post_trim_input_bits_v1`, documented in the study and queue guide. It avoids changing native loader/trimmer behavior. Keys use the actual post-trim input features and four-vector bits; canonical full-input ordering breaks distinguishable hash ties, and exact duplicates are interchangeable. Seed 731019 is frozen separately from training. Local tests cover permutation, duplicate physical membership, masked features, matched capacities, and unchanged global RNG. The real-data fingerprint comparison remains a GPU admission gate.

Freeze the random partition seed separately from the training RNG. Construct memberships with a local/stateless hash-based ordering, not draws from Python/NumPy/Torch global RNG. For the same retained input set and keys, membership is unchanged across passes, workers, resume, and constituent permutations. Native trimming may change the retained set, so memberships are not required to remain identical when that set changes.

**Acceptance:** exact matching of fixed-arm occupancy multiset and branch parameter count; deterministic membership; global RNG unchanged; labels/filenames not consulted; identical predictive batch fingerprints with metadata enabled/disabled; correct post-trim alignment including duplicate fixtures.

If this contract cannot be satisfied, block the random arm rather than call arbitrary index-based grouping equivalent. The three primary architectures may proceed through development, but omitting the recommended random arm from production must be an explicit recorded scope decision.

## 7. Work package D: reusable training, selection, and restart

Generalize only the pieces coupled to the seven-task screen. Engine helpers should receive explicit recipe, task, model factory, artifact namespace, diagnostics callback, and stop/checkpoint policy; do not monkey-patch `TASKS` or global protocol objects.

Retain these inherited semantics:

- Set LR before each attempted update using `part_ev_screen.protocol.learning_rate`.
- Keep seeded training/checkpoint-validation iterators continuous, with original worker/fetch/prefetch behavior.
- Maintain attempted counters and AMP skip accounting; do not advance Lookahead via a fake successful update after an overflow.
- Preserve FP32 validation and complete-population AUC/rejection evaluation.
- Save full model, optimizer/Lookahead, scaler, RNG, native trimmer counters, attempted step, validation cursor, and best-check state atomically.
- Resume by reconstructing/replaying the same streams before restoring model RNG/trimmer state, following the existing trainer. Account for replay cost in logs rather than claiming it is free.
- Preserve exact-tie checkpoint selection and validate ten completed checks before final lock.

Add group diagnostics on a fixed development sample under a state-restoring context. Diagnostics must not advance production data iterators, affect native dropout/trimming RNG, change BatchNorm running statistics, or leave the model in the wrong training mode.

Use fresh production initialization after admission. Probe weights/scalers are never promoted to production. The donor's recovery approval is evidence about its own campaign only; any new numerical-admission issue needs separate review, not an automatic tolerance change.

**Acceptance:** differential tests of extracted helpers against the existing screen routines on identical small fixtures; LR boundary tests; AMP overflow fixture; exact best-check tie behavior; interruption/resume before and after validation; preservation of selected checkpoint on retries; equivalent uninterrupted/resumed state within documented deterministic limits; original screen regression suite remains passing.

## 8. Work package E: finite evaluation, reporting, and artifacts

Evaluate **only new selected checkpoints**. Use the inherited finite validation loader, full class ordering, last partial batches, FP32 outputs, and existing metric functions. Export predictions atomically per input file with checkpoint/data/source identity sidecars so terminal interrupted evaluations can reuse verified complete files.

The existing prediction NPZs contain logits and labels; event identities are represented by the sidecar's input-file record and contiguous entry interval. Reconstruct `(verified source file, entry index)` for joins. Do not assume explicit event-ID arrays exist, join by unrelated local filenames such as `file_003.npz`, or require absolute directory paths to match after a validated relocation.

Verify full coverage and label agreement before joining donor and grouped results. Stream per-file checks/comparisons where possible instead of loading every model's 5M predictions simultaneously.

### Required reports

- Summary table: model, grouping strategy, parameters, selected check, accuracy, delta versus `PART_RANGER` in percentage points, macro OVO AUC, separately labeled macro OVR AUC, and cross-entropy.
- Mechanism contrasts: fixed minus random; hybrid minus learned; each grouped arm minus ParT. Optional EV-reference contrasts explicitly compare different architectures, not grouping added to EV.
- Rejection tables: study-defined 30%/50% operating points and labeled Hqql 99% / Tbl 99.5%, including achieved efficiency and accepted QCD counts in JSON.
- Learning curves and matched-check differences; best-through-common-check summaries for incomplete progress views.
- Cost table: parameter counts, end-to-end training/inference throughput, grouping time, memory, AMP skips, and runtime, separating measured quantities from estimates.
- Diagnostic plots: occupancy, entropy, effective group count, angular widths, slot similarity, residual scale, correction norms, and a bounded set of example assignments.

Diagnostics are not new selection metrics. The primary accuracy-selected checkpoint remains the checkpoint for every reported class/metric. A single-seed screen has no across-seed error bar; repeated use of 5M validation is not independent test confirmation.

If a required control is pending, produce a clearly labeled progress artifact but no completed comparative report. Once it validates, report generation should not queue baseline inference. An optional EV control may be omitted from a labeled partial view; do not publish a promised EV comparison without its validated artifacts.

### New-campaign layout

```text
experiment_spec.json
source/ and source.sha256
launch.json
controls.json and control_validation_history/
benchmarks/<NEW_RUN_ID>.json
production_approval/
runs/<NEW_RUN_ID>/seed_101/
    optimizer.json
    last.pt
    best_model_val.pt
    validation/check_01.json ... check_10.json
    diagnostics/check_01.json ... check_10.json
    training_complete.json
    full_validation/file_*.npz and file_*.json
    full_validation/metrics.json
selection/locked_checkpoints.json
comparison_lock.json
reports/grouped_screen_report.json and grouped_screen_report.md
reports/learning_curves.pdf and grouping_diagnostics/
submitted_job_ids.tsv
logs/
```

Project disk requirements before launching. Four sets of raw FP32 logits for 5M events and ten classes alone require 800,000,000 bytes in total, before labels, sidecars, checkpoints, temporary files, and headroom. Do not duplicate donor predictions or create a 100M-event grouping cache by default. Never delete existing artifacts automatically to make a storage check pass.

**Acceptance:** exact metrics agreement with existing evaluator on shared fixtures; score ties and zero-QCD cases; shuffled file numbering with correct identity joining; missing/duplicate/mutated files rejected; partial-write/retry behavior; report marked complete only with all required verified results.

## 9. Work package F: admission and SPORC launch workflow

Create a separate submitter and worker, following existing freezing/ledger/signal conventions. All array sizes derive from `new_models`, not the number of report rows or control references.

```text
prepare + donor compatibility/status check
                 |
       new-model GPU benchmark array
                 |
       explicit admission review
                 |
       new-model training array
                 |
       selected-checkpoint lock
                 |
       new-model full-validation array
                 |
       report <--- verified donor result registry
```

Implemented modes on the **new** `submit_part_group_screen_sporc.sh`: `prepare`, `benchmark`, `review`, `train`, `evaluate`, `refresh-controls`, `report`, and read-only `status`, plus `--dry-run` and guarded `--resume`. The original seven-task launcher is unchanged.

Operational requirements:

1. Freeze committed source/configuration before queueing; verify worker source hashes at job start. Never execute new jobs from a mutable checkout.
2. Use donor-compatible SPORC hardware/software; read runtime details from verified artifacts. Check account/partition availability rather than assuming old permissions or silently switching to debug/Tigris.
3. Keep pre-production requests below six hours; use bounded or separate probes if a clustering reference implementation is slow. Benchmark actual batch 512, native trimming, worst-length 128, real data fetching, backward/optimizer, validation, and checkpoint I/O.
4. Test both backbone and added-module precision, finite gradients, observed AMP skips, and memory margin. Use the existing screen admission policy as the starting contract; tighten or separately review grouping-specific checks rather than weakening a failed threshold to obtain a pass.
5. Choose production and final-evaluation walltimes from sustained end-to-end measurements with the inherited planning margin. Benchmark failures do not trigger production submission.
6. Permit controls to be pending during new training. Do not attach stale donor job IDs as the sole dependency: completion is an artifact condition. A report that finds a pending control exits with a clear incomplete status and can be rerun after `refresh-controls`; do not occupy a GPU or silently poll forever.
7. Reject concurrent duplicate campaigns/tasks, active conflicting retries, and changes to frozen configuration. Lock writes and keep a ledger of exact submitted IDs; retry only terminal/incomplete stages and preserve verified completed artifacts.
8. Support read-only progress queries. No launcher mode cancels unrelated jobs, deletes data, modifies the donor environment, or automatically submits future-study variants.

**Acceptance:** shell syntax and mocked Slurm tests; dry-run performs no writes/submissions; exactly four (or explicitly chosen three) new training and evaluation tasks; no donor tasks; pending controls do not prevent train admission; missing review prevents production; duplicate/retry guards and dependency failures work; resource/time/source/env settings survive submission correctly.

## 10. Test matrix and merge gates

| Test group | Essential assertions |
|---|---|
| Contracts/controls | Correct frozen donor, full recipe/runtime comparison, pending versus incompatible, immutable training spec, complete prediction identity coverage |
| Geometry | Periodic azimuth, valid masks, safe low multiplicity, four-vector conservation, finite boundary gradients |
| Fixed/random grouping | CPU oracle parity, deterministic ties, matched sizes, permutation behavior, stable metadata, no label/RNG leakage |
| Learned/hybrid grouping | Correct softmax axis, three rounds/two updates, trainable assignments, finite centers, exact `lambda=0` equivalence |
| Shared branch/native adapter | Common initialization, state-key preservation, post-trim alignment, gate-disabled baseline parity, masks and hook cleanup |
| Training/precision | Actual observed dtypes, continuous fingerprints, LR boundaries, skip handling, unchanged backbone RNG/dropout, selection ties |
| Restart | Optimizer slow buffers, scaler/RNG/trimmer restoration, replayed cursor, interruptions around validation and atomic writes |
| Evaluation/reporting | Per-file recovery, full 5M/class coverage, correct joins/contrasts, metric equivalence, no invented finite rejection |
| Launcher | Frozen source, dynamic task count, no control submissions, guarded admission/resume, walltime policy, no destructive actions |
| Regression | Existing screen model/protocol/data/precision/training/reporting/launcher and relevant full-scale helper tests remain passing |

Run cheap synthetic CPU tests during development. Use GPU-only tests for actual FP16, worst-length memory, native trimming/dropout behavior, and sustained streaming cost. Lightweight tests may use a reduced synthetic budget through internal fixtures; production configuration must reject those fixtures as substitutes for the fixed recipe.

A test plan is not a test result. Record executed commands, environment, outcomes, and unresolved failures when implementation occurs. Do not report the screen ready based only on mocked tests.

## 11. Execution order and readiness checklist

Recommended implementation sequence, with a usable checkpoint after each stage:

1. **A — Reuse foundation:** donor contract reader, control validator/status registry, fixture tests, and isolated grouped task/config schema.
2. **B — Backbone integration:** common geometry/branch and native hook adapter; gate-off parity and unchanged initialization/RNG tests.
3. **C1 — Fixed groups:** reference oracle, then measured production implementation.
4. **C4 — Random control:** metadata feasibility spike, reviewed deterministic rule, fixed/random matched-branch tests.
5. **C2/C3 — Learned and hybrid:** shared router, finite gradients, collapse diagnostics, zero-locality equivalence.
6. **D — Shared engine:** conservative extraction/adapters, differential regression tests, checkpoint/resume and diagnostics isolation.
7. **E — Results:** new-only finite evaluation, verified control joins, metrics and grouped reporting.
8. **F — Operations:** frozen SPORC submission, dry-run/mock tests, resource projection, then bounded real GPU admission when authorized.
9. **Production handoff:** review the evidence, freeze the final comparison, and provide exact tested queue/status/recovery commands. Submit only when authorized.

The ordered model work follows the study's fixed-first design. No candidate is chosen or discarded merely because another has already produced an attractive score; implement the three primary strategies before comparative production screening.

Local software handoff:

- [x] Four initial architectures have versioned definitions, common initialization, and separate new-task registration.
- [x] Read-only donor validation, pending-result handling, and immutable result references are implemented and fixture-tested.
- [x] CPU/native-model tests, original regression tests, mocked scheduler/dependencies, and report/plot fixtures pass (commands and limits in the queue guide).
- [x] Checkpoint/restart, first-tie selection, per-file evaluation recovery, and matched reporting are implemented.
- [x] Parameter counts are measured locally; runtime/storage/memory gates and cost reporting are implemented.
- [x] Deployment, admission, status, and terminal-retry instructions are documented. No remote writes or submissions performed.

Actual production admission remains gated on:

- [x] Three primary architectures and the included random control have versioned definitions.
- [ ] Donor compatibility passes; required results are verified or clearly pending.
- [ ] Production recipe/runtime and initial backbone/data fingerprints match the donor contract.
- [ ] CPU/unit, differential, regression, and real GPU admission checks pass.
- [x] Resume, checkpoint selection, finite-population evaluation, and joined reporting work end to end on development fixtures. Actual 5M evaluation is a subsequent queued stage.
- [ ] Counts, costs, memory/storage estimates, and measured walltimes are recorded.
- [x] A dry-run performs no writes; mocked submission queues only new tasks and respects dependency/retry guards.
- [ ] Source/configuration is frozen and the operations guide is executable as written.

It is finished only after every included new model completes training and final validation, required controls validate, the comparison lock is frozen, and the full report is generated. A successful launcher or a single completed training is not completion of the study.

The local deliverable is therefore **ready to queue prepare/admission**, not already GPU-admitted or empirically complete. The user must deploy the committed implementation; prepare then freezes source/configuration and production requires reviewed, passing probes on the donor-compatible A100 runtime.

## 12. Remaining study ideas: implementation hooks, not automatic extra runs

This covers the rest of the study document without expanding the first production queue.

| Follow-up | Implementation approach | Required comparison / gate |
|---|---|---|
| Different `K` / insertion point | Versioned config, branch insertion adapter | Same recipe, comparable branch accounting; no post hoc silent default changes |
| Multi-scale `8 -> 4` | Nested fixed tree or second learned assignment `B`; compose `C=A B`, carry occupancy in hidden pooling | Single-scale branch with comparable capacity/compute; conservation tests at both scales |
| Adaptive group usage | Measured occupancy first; later explicit presence/sparsity mechanism | Frozen objective and empty-slot rules; show whether collapse differs from useful variable usage |
| Repeated grouping at depth | One branch interface with separately identified insertion instances | Quantify extra processing/parameters; distinguish from router refinement rounds |
| Group-pair physics bias | Separate group-pair encoder using group vectors | Do not reuse the native particle pair normalization blindly; isolate its added effect |
| Grouping plus EV | Compatible backbone factory and branch adapter | Matched ParT, EV, groups, and groups+EV factorial; reuse only contract-compatible completed cells |
| Capacity/compute control | Measure actual added budget, then design a new ordinary-ParT control | Do not reuse the EV-width-matched baseline as though it matches grouping automatically |
| `K=1`, descriptor removal, generic summary tokens | Explicit ablation configs/factories | Test one claimed mechanism at a time; common inputs/recipe and fair tuning budget |
| AdamW or additional seeds/full budget | Reuse the complete corresponding protocol, with compatible controls | New matched control where seed/budget/recipe differs; no comparisons disguised as architecture-only effects |

No unsupported novelty or guaranteed gain is an implementation acceptance criterion. The initial goal is a reproducible, fair answer about these grouping mechanisms and their costs. Literature claims remain governed by the study plan's references and require separate research when developed into a paper.
