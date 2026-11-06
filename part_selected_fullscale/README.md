# Seven-run Tigris campaign: benchmark stage

Implements the model and benchmark stage of the
[seven-run plan](../part_selected_tigris/FULLSCALE_SEVEN_RUN_CAMPAIGN_PLAN.md), using
the numerical definitions and optimized backend described in the
[selected-relations guide](../part_selected_tigris/SELECTED_RELATIONS_GUIDE.md).

**This launcher cannot submit production training.** The next decision is based
on the measured reports. Full-training/evaluation submission and the final
scientific-control audit remain a separate stage after benchmark review.

## Models

All seven use seed 101, batch 512 and the full Ranger recipe: 1,000,000 attempted
updates, with 50 accuracy-selected validation checks of 1.28M events. Diagnostic
updates use disposable models; none become production checkpoints.

| Index | Run ID | Parameters | Diagnostic |
|---:|---|---:|---|
| 0 | PART_MATCH_STATE_SHARED | 2,323,750 | Standard checks |
| 1 | PART_MATCH_STATE_LAYERWISE | 2,327,894 | Standard checks |
| 2 | SELECTED_PART_MATCH_STATE_LAYERWISE | 2,337,814 | Selected profile + sustained benchmark |
| 3 | STATE_EV_SHARED | 2,323,710 | Standard checks |
| 4 | EV_LAYERWISE | 2,212,774 | Standard checks |
| 5 | SELECTED_PART_SHARED | 2,153,406 | Selected profile + sustained benchmark |
| 6 | SELECTED_STATE_EV_LAYERWISE | 2,337,382 | Selected profile + sustained benchmark |

Counts have been checked by constructing the pinned models locally. All capacity
controls use shared bias and no EV/state. Their particle FFN widths are 599, 601,
and 601 respectively; class FFNs remain 512. Both State-EV models use all eight
blocks, rank 32, query chunks of 64 and shared-per-forward pair normalization.
Selected State-EV receives the enriched stem in both bias and state messages.

The baseline ParT and ordinary EV controls, and the independently queued
layerwise State-EV run, are not trained or changed here. `controls/registry.json`
records read-only locations and metadata hashes, **not an assertion that their
results have passed a full scientific comparability audit**.

## Before queueing

Commit/push the implementation and pull it on Tigris, or use a clean committed
worktree containing it. The launcher rejects a dirty checkout so its frozen
snapshot cannot accidentally omit local changes. It does not commit or push.
Tracked Python bytecode/cache entries are omitted from both the source snapshot
and its manifest. The completed snapshot is verified before the first job is
submitted, and again by every batch worker; real source inventory and hash checks
remain mandatory.

The existing read-only reference must still exist:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_tigris/selected_gh200_20261023T073044Z
```

It supplies the verified dataset inventory, ARM environment identity, selected
normalization, native stream fingerprints and original compiled tree backend.
No new dataset copy, environment installation or normalization fit is needed.
Use `--reference` for an explicitly relocated compatible reference, not a SPORC
campaign. The launcher obtains the existing interpreter from its `launch.env`;
`--python /absolute/existing/environment/bin/python` overrides that choice but
does not bypass the environment checks.

Allow approximately 12–13 GiB free at preparation, including a 4-GiB reserve.
This is a conservative seven-run output estimate, not a storage reservation.
The benchmarks themselves retain only small reports, compiled code, one source
snapshot, and one restart/weights probe per model. No persistent pair/tree cache,
full-population predictions or large profiler traces are produced at this stage.
Review recalculates the eventual output allowance from measured checkpoint sizes.

## Dry-run and queue on Tigris

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

SEVEN_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_fullscale/seven_$(date -u +%Y%m%dT%H%M%SZ)"

bash sbatch/submit_part_selected_fullscale_tigris.sh \
  --mode benchmark --output "$SEVEN_ROOT" --concurrency 1 --dry-run

# Run after inspecting the dry-run. This submits diagnostics, not training.
bash sbatch/submit_part_selected_fullscale_tigris.sh \
  --mode benchmark --output "$SEVEN_ROOT" --concurrency 1

squeue --me -o "%.18i %.32j %.12T %.12l %R"
```

Default graph:

```text
prepare (CPU, 5h)
  -> profile[2,5,6] (one GH200 each, 5h each)
  -> benchmark[0-6] (one GH200 each, 5h each)
  -> summary (CPU, 30min; also reports missing/failed benchmarks)
```

Each task requests 12 CPUs and 128 GiB host RAM, with four tree threads, two
loader workers, cache off and compilation off. Default concurrency is one GPU
job at a time to limit filesystem contention while comparing executions;
`--concurrency 2` or `3` is available. This is not multiple GPUs per model.

Every diagnostic is below six hours. Compute phases have a 4h50m internal
deadline, plus Slurm termination handling. An incomplete check fails and records
the reason; it never becomes successful admission. Production-duration requests
are deliberately absent. Existing jobs are never cancelled or changed.

`--mode profile` stops after the three paired profiles. A later
`--mode benchmark --output "$SEVEN_ROOT" --resume` can finish the graph after
the earlier jobs terminate. Completed, authenticated evidence is reused. Active
jobs prevent duplicate submission; successful submissions are fsynced to the
ledger even if a later submission fails. A failed saved measurement is preserved;
fixes to source/execution require a fresh campaign, not editing its snapshot.

If an older campaign failed immediately with `Frozen source file inventory
changed`, the initial launcher included Git-tracked Python cache files in its
manifest while the worker's verifier excluded them. Update to the corrected
launcher and use a **new output directory**, not `--resume` on that frozen
campaign. Keep the old logs/manifest unchanged; do not disable the verifier or
delete source files to make its check pass. This failure occurs before model
preparation and does not require changing the scientific recipe or walltime.

Native compilation runs in a child process with the same interpreter and source.
This prevents PyTorch's compiler helper from exposing setuptools-vendored package
metadata on the parent's import path. The parent still checks the complete
reference environment before preparation and again when freezing the spec, then
authenticates the compiled backend's source/ABI/binary hashes. No package versions
are ignored, installed, upgraded, or removed. This also fixes the older failure
listing new `installed_distributions.jaraco.*`, `autocommand`, etc. after a
successful compile; retry that failure with a fresh campaign after updating.

## What is measured

- CPU prepare authenticates inputs/environment/source, copies only small metadata,
  compiles the new native backend, checks exact trees/ties/padding/thread counts,
  and constructs all seven models to verify counts and active architecture.
- Three selected profiles run the actual corresponding reference and fast models
  on matched streams: three alternating-order trials, each 20 warmup + 200
  measured updates. FP32/FP16 features, logits, losses, gradients, BatchNorm,
  trimming/RNG, and serialized optimizer restart are checked. Selected State-EV
  additionally checks zero-state identity and nonzero state/family/head gradients.
  Only bounded one-step aggregate profiler scopes are saved. Scopes overlap and
  must not be summed as sequential time.
- Selected sustained probes measure 5,000 updates in five 1,000-update windows.
  Standard checks measure 1,000 updates; they are shorter startup/resource probes,
  not equally extensive performance evidence. Both use native-stream fingerprints,
  observed FP16 training/FP32 validation, bounded full-128 AMP calibration plus
  ten no-skip confirmations, and allocated/reserved GPU and process/worker memory.
- All seven time 131,072 validation events including macro AUC calculation,
  checkpoint writes/reloads and selected-weight writes. The full recipe's LR
  boundaries and Lookahead/RNG serialization are checked on disposable pilots.
- Projections include one million updates, 50 validations, at least 1,152 restart
  writes, up to 50 selected-weight writes, measured startup and a 30% allowance.
  Queue delays/restart replay are excluded. Small-sample AUC extrapolation is an
  approximation; a selected projection above 80% of the 480h ceiling is flagged
  for a longer bounded timing check before making a production decision.

The original four-model fast campaign's gates are untouched. Here a profile
without a measured 5% conservative gain may still complete its diagnostics, but
is flagged as not ready for resource review; it does not authorize production.
Failed numerical/input checks block subsequent measurements. Negative speed
results are retained rather than described as an optimization success.

## Review

```bash
bash sbatch/submit_part_selected_fullscale_tigris.sh \
  --mode review --output "$SEVEN_ROOT"

cat "$SEVEN_ROOT/submitted_job_ids.tsv"
```

`review` is read-only. The queued summary writes
`reports/benchmark_review.json`, with per-model throughput, memory, proposed
walltime, measured checkpoint sizes, missing/failed evidence, and storage reserve.
The four standard runs are assessed against 216h; the three selected runs against
`max(216h, margin-inclusive projection)` and a 480h planning ceiling. Actual
partition/account/QOS limits and maintenance must be checked again before any
production submission. The report always says `authorizes_training: false`.

Inspect `logs/sel7_*`, `failures/`, `profiles/<RUN_ID>.json`, and
`benchmarks/<RUN_ID>.json`. A successful Slurm job is not a scientific result.
Linux native/CUDA/GH200 performance cannot be established by local CPU tests;
these queued checks provide that evidence.
