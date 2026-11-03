# Faster selected-relation execution on Tigris

Implementation of [the optimization plan](../part_selected_tigris/OPTIMIZATION_PLAN.md).
This is an **isolated candidate execution**, not a change to the existing study.
Local CPU tests are not evidence of a GH200 speedup. Compilation against the
original native tree builder, CUDA parity, and measured speed/resource admission
run in the new campaign before training can start.

## What changes

- One native call builds a batch of REGION trees, with independent jets processed
  by a bounded OpenMP team. The per-tree merge arithmetic, canonical tie ordering,
  float64 calculations and float32 outputs follow the original compiled builder.
- Direct contiguous packing replaces the per-jet Python dictionaries, repeated
  validation/conversion, and many small field transfers. Categorical indices are
  stored as int32 on the CPU and expanded to the reference int64 on device.
- The existing GPU REGION arithmetic consumes those packed fields. PT, TRACK,
  learned encoders, bias heads, EV messages and the training loop remain unchanged.
- An optional, bounded **validation-only RAM cache** stores immutable packed tree
  bytes keyed by exact ordered particle content and campaign identity. It never
  caches learned features, spills to disk, or enters a model checkpoint. Misses
  rebuild the exact view; it is disabled by default.
- `--compile-region` optionally tries compilation of only the pure GPU REGION
  descriptor function. It is off by default and must pass the same numerical and
  speed gates. Compilation failure is an error, not an invisible fallback.

Native trimming still runs in its original location. We do not cache a full-jet
tree and prune it after removing real particles. We do not precompute the entire
100M population, freeze learned encoders, or assume pair symmetry. TRAINING tree
caching, TRACK algebraic reassociation, and ahead-of-model pipelining are deferred:
they need measured benefit and separate AMP/RNG equivalence evidence.

## Unchanged experiment

| Task | Architecture | Parameters |
|---|---|---:|
| 0 | Selected, shared bias | 2,153,406 |
| 1 | Selected, shared bias + EV | 2,218,942 |
| 2 | Selected, layerwise bias | 2,157,158 |
| 3 | Selected, layerwise bias + EV | 2,222,694 |

All four use seed 101, the existing `ranger_short_v1` recipe, one GH200, batch 512,
two original data-loader workers, 200,000 attempted updates, and ten checkpoint
checks. This is about 102.4M training exposures, **not ten full 100M-jet passes**.
FP16 training/FP32 evaluation, normalization, masks, native BatchNorm population,
Ranger/Lookahead, selection and restart rules are unchanged. Final evaluation is
the official **5M validation set, not the 20M test set**.

The existing shared/layerwise ParT and ParT-EV controls are reused read-only. The
existing selected GH200 campaign supplies the numerical implementation reference:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_tigris/selected_gh200_20261023T073044Z
```

Its source, environment, inputs and native backend must authenticate. New source
must still match its numerical modules; this implementation does not weaken that
check. Changing any legacy numerical source requires a separate reviewed study.
Old selected production checkpoints are not resumed under the new contract.
Reuse of SPORC A100 controls remains a cross-hardware scientific comparison; the
performance probes compare implementations on the **same GH200 in one job**.

## Queue diagnostics first

After committing/pushing these files locally and pulling them on **Tigris**, use
the existing reference environment. No environment clone or package installation
is needed. The wrapper reads `SELECTED_PYTHON` from the reference `launch.env`;
`--python /absolute/existing/env/bin/python` is an explicit override, subject to
the same frozen environment check.

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

FAST_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_fast/selected_fast_$(date -u +%Y%m%dT%H%M%SZ)"

bash sbatch/submit_part_selected_fast_tigris.sh \
  --mode benchmark --output "$FAST_ROOT" --concurrency 2 --dry-run

bash sbatch/submit_part_selected_fast_tigris.sh \
  --mode benchmark --output "$FAST_ROOT" --concurrency 2
```

This queues only:

```text
prepare (5h CPU)
  -> paired profile[0-3] (5h GPU each)
  -> sustained benchmark[0-3] (5h GPU each)
  -> gate (2h CPU)
```

All dependencies use `afterok` with invalid-dependency cancellation. The default
**does not queue training**. Every pre-production time limit is below six hours.
`--mode profile` stops after the shorter paired profiles. `--mode run` is an
explicit alternative that queues the complete gated graph, including training.
No commands above cancel or modify old jobs.

Defaults: 12 CPU cores/job, four tree threads, 128 GiB host RAM, zero cache, two
concurrent GPU tasks, 168h training and 12h evaluation limits. These are initial
resource ceilings, **not measured recommendations or runtime promises**. The
launcher checks the partition's actual maximum walltime. It also checks the
visible physical/cgroup/Slurm memory allowance inside a GPU job.

For another candidate, create a **new output directory**. Examples:

- `--tree-threads 8 --cpus 12`: benchmark a larger bounded tree team.
- `--cache-gib 8 --memory-gib 128`: test exact-view validation reuse.
- `--compile-region`: test the optional compiled GPU descriptor path.

The cache reserves at most half the requested host RAM; all process/loader RSS
plus a conservative allowance for the entire cache must still pass the headroom
gate, even if the diagnostic did not fill the arena. Each node/job has its own cache. A
warm repeated sample is labeled as a cache microbenchmark, not a claim that the
production stream achieves that hit rate. A small cache may provide no benefit
on the 5M validation stream; leaving it off is valid.

## Inspect, then train

```bash
bash "$FAST_ROOT/source/sbatch/submit_part_selected_fast_tigris.sh" \
  --mode review --output "$FAST_ROOT"

bash "$FAST_ROOT/source/sbatch/submit_part_selected_fast_tigris.sh" \
  --mode status --output "$FAST_ROOT"
```

Review and status do not submit jobs. Task-specific logs and the exact submitted
IDs are in `logs/` and `submitted_job_ids.tsv`. Profiles are in
`profiles/SELECTED_*.json`, `.md`, and one-step `.trace.json` files. Sustained
measurements are in `benchmarks/SELECTED_*.json`.

Once **all four** gates have passed, explicitly queue:

```bash
bash "$FAST_ROOT/source/sbatch/submit_part_selected_fast_tigris.sh" \
  --mode train --output "$FAST_ROOT" --resume --concurrency 2
```

This queues `gate -> train[4] -> lock -> evaluate[4] -> report`. The unchanged
comparison reporter writes `reports/selected_tigris_report.{json,md}` inside
**this new campaign** with the new execution schema/runtime identity.

Repeat nondefault resource/cache/compile flags when resuming; settings cannot
change inside an existing campaign. A failed parity/speed gate is not overridden
by `--resume`: inspect the evidence and create a fresh candidate. A terminated
training task can resume from its own compatible checkpoint with an empty RAM
cache. Resume refuses duplicate submission while any ledger job is active. If a
submission fails midway, the ledger retains already accepted jobs; it does not
cancel them automatically. Inspect those exact IDs before deciding what to do.

## Admission and evidence

1. Compile the new backend in the campaign, without changing Conda. Compare
   packed results exactly with the authenticated original compiled backend:
   widths 1/2/11/32/64/128, multiple thread counts, empty/single/duplicate jets,
   padding and malformed inputs. This CPU preparation must pass first.
2. For every architecture compare exact tree fields; normalized and learned
   family outputs; logits/loss; per-module gradients; BatchNorm buffers; FP32
   train/eval and FP16 train. Exercise native trimming, a six-update Lookahead
   cycle, LR steps 139999/140000, and a serialized optimizer/RNG restart with a
   cold cache. Frozen tolerances are in `parity.POLICY`; they are not fitted to
   observed errors.
3. Alternate reference/candidate order over three trials of 20 warmup + 200
   measured updates on identical streams. Save short profiler traces and a
   disposable standard-ParT or standard-ParT-EV timing probe on the same GPU.
   Those probes are not new scientific control runs. Paired ranges must show
   at least 1.05x conservative speedup; this is not a confidence interval.
4. Retain the existing sustained 5,000-update / five-window admission, full
   128-particle checks, 131,072 validation events, AMP-skip and precision gates,
   checkpoint timing and native fingerprints at 0/20/1020/2020/4020. Add measured
   host RSS and reserved-GPU headroom checks. Require a faster complete-job
   projection including candidate startup and retain the existing 30% walltime
   margin. Reference validation timing uses a short cold sample, so projections
   remain uncertain. Queue time and large multi-job contention are not predicted.

Parameter/state-dict identity and local numerical tests do **not** guarantee
identical final accuracy. The optional fusion path can fail strict parity; the
batched CPU builder can fail compilation on a particular compiler. Both failures
block production rather than silently changing execution or tolerances.

## Persistent storage and validation status

No new ROOT dataset, disk tree cache, cloned environment or cached learned
features. Persistent additions are a source snapshot, small copied input metadata,
native build files, reports/traces, benchmark checkpoints, and (only after the
user queues training/evaluation) ordinary checkpoints/predictions. Preparation
requires 6 GiB free as an initial allowance, not a reservation or guarantee against
other jobs filling storage. Optional Torch compilation may also create its normal
compiler cache; leave it off for the smallest/most predictable storage footprint.

Local tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_part_selected_fast_model.py tests/test_part_selected_fast_protocol.py tests/test_part_selected_fast_launcher.py tests/test_part_selected_fast_native.py -q
```

Verification on 2026-11-02: the combined selected/fast/Tigris/recovery suite passed
172 tests, with nine skips (one native Linux build and eight CUDA-only checks).
The native integration test explicitly skips without Linux C++/OpenMP and Ninja.
It is **not replaced by an oracle while claiming native coverage**. The Windows
CPU development environment cannot establish native/GH200 parity or throughput;
the queue above produces that evidence. Existing selected, Tigris and recovery
regression tests must also pass. No production jobs are submitted by local tests.
