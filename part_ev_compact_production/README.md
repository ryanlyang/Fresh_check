# Independent two-run compact State-EV production on Tigris

Queue-ready production orchestration for the **passed compact benchmarks**, not
a rerun or replacement of the existing seven-model campaign. No new preflight or
benchmark GPU jobs are needed. Nothing has been submitted by implementing this.

## Frozen experiment

| Index | Run | Parameters | Updates | Updates/check | Training walltime from the supplied benchmark |
|---|---|---:|---:|---:|---:|
| 0 | `COMPACT_STATE_LAYERWISE` | 2,262,950 | 1,000,000 | 20,000 | 150 h (6 d 6 h) |
| 1 | `PART_MATCH_COMPACT_COMPUTE` | 2,263,662 | 1,243,800 | 24,876 | 75 h (3 d 3 h) |

Both train fresh at seed 101, batch 512, with **50** validation checks of 1.28M
events each. One GH200, 12 CPUs and 128 GiB host RAM per GPU job; Tigris account
`reu-aisocial` inherited from the passed benchmark. Same original Ranger
(RAdam + Lookahead) recipe, FP16 training, FP32 validation, TF32 off, native
BatchNorm, 100M training pool, unchanged preprocessing and two data workers.
GradScaler starts at 32768, as in the passed benchmarks. No optimizer tuning,
gradient clipping, additional features, selected relations, or early stopping.

The compact model is **all eight** state-conditioned blocks, independent
layerwise attention-bias heads, rank 16, and a shared pair-only state projection.
It keeps ordinary EV in every block. The control is ordinary shared-bias ParT,
without EV/state: only particle FFNs widen to 570. All model construction and
training-step arithmetic come from the original authenticated benchmark source.
The new production files do not reimplement or change those models.

The control receives 24.38% more exposure (636,825,600 vs 512,000,000 jets).
Its LR milestones stretch proportionally: 70% flat, then the same 15 decays.
It does **not** get more checkpoint-selection opportunities. These are 50
reporting intervals, not 50 complete dataset passes. Selection uses highest
validation accuracy, keeping the first tie, not independently maximized AUC.

The launcher recomputes/authenticates the benchmark summary, then freezes the
measured schedules, FLOP curves, parameters, resource recommendations and source
hashes in `production.json`. The numbers above describe jobs 869415–869417;
they are not hardcoded substitutes for checking evidence. A different benchmark
requires reviewing its printed plan. Training recommendations must fit 20 days.
The benchmark's weights are disposable and are **never** loaded for production.

The FLOP match counts dense forward, backward and checkpoint recomputation.
It excludes non-dense operations, optimizer work, validation and data replay;
it is not a walltime/GPU-hour match. Training records every attempted update's
actual trimmed length, including AMP-skipped backward passes. The final report
shows both frozen estimates and realized cost-curve totals. No budget is extended
or shortened based on observed accuracy or later length distributions.

## Queued graph

```text
train 0 (150 h) -> afterok evaluate 0 (5 h) --+
                                           +-> afterok report (1 h, CPU)
train 1  (75 h) -> afterok evaluate 1 (5 h) --+
```

Exactly **five jobs**, no array throttle and no cross-training dependency.
Both training jobs can start whenever Slurm finds resources. Each evaluation
starts after its own training succeeds. Failure/timeout exits nonzero and blocks
dependent work; it is not mistaken for successful completion. Slurm's current
partition/QOS/account eligibility is checked with `sbatch --test-only` for every
resource shape before any job is submitted. This does not guarantee a start time.

Evaluation is the **full official 5M validation split**, using the accuracy-selected
checkpoint, FP32 logits, and restartable per-file prediction shards/checksums.
The report audits all 50 checks, optimizer identity, selected weights, prediction
coverage, labels and metrics before publishing accuracy, macro OVO/OVR AUC and
QCD rejection. This population overlaps checkpoint selection. **It is not an
independent 20M test.** Test data are not opened or queued. Historical baseline
results are not silently imported into this two-arm comparison.

## Source isolation and storage

At submission, the eight new implementation/worker files are frozen from the
committed repository into a new production directory. Workers use absolute paths,
verify their manifest and benchmark artifacts, and import scientific code only
from the passed benchmark's `source/`. The live checkout supplies neither models
nor training code to queued jobs. Later pulls cannot alter either snapshot.
The benchmark's launch environment and installed-package checks remain enforced,
including its fixed duplicate-metadata inventory convention.

Do not edit frozen snapshots, delete benchmark metadata/source, or change the
shared Python environment while these jobs are queued/running. Existing seven-run
jobs, other campaigns, data and environments are never modified or cancelled.

The read-only plan prints an evidence-based free-space requirement (roughly
7–8 GiB including a **4 GiB reserve**, depending on measured checkpoint size).
It conservatively budgets:

- one atomic latest full-state restart per model, including temporary-write space;
- at most 50 improving selected-weight snapshots per model plus the final best copy;
- two sets of 5M validation logits/labels, JSON, small logs, and the small source overlay.

There are no dataset copies, persistent pair caches or saved training predictions.
No cleanup/deletion happens automatically. Remaining storage is checked on resume;
checkpoints require the 4 GiB reserve. Other jobs can still consume shared space:
these checks do not reserve quota. Keep an eye on `df -h /home/ryreu/atlas`.

## Queue after committing and pushing the new files

Run on **Tigris**. First locate the successful benchmark by its known summary job:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

mapfile -t compact_logs < <(find \
  /home/ryreu/atlas/Fresh_check/checkpoints/part_ev_compact \
  -maxdepth 3 -type f -name 'evcompact_summary_869417_*.out' -print)

if [[ ${#compact_logs[@]} == 1 ]]; then
  COMPACT_BENCHMARK="$(dirname "$(dirname "${compact_logs[0]}")")"
  COMPACT_PRODUCTION_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_compact_production/full_$(date -u +%Y%m%dT%H%M%SZ)"

  bash sbatch/submit_part_ev_compact_production_tigris.sh \
    --mode plan \
    --benchmark "$COMPACT_BENCHMARK" \
    --output "$COMPACT_PRODUCTION_ROOT"
else
  unset COMPACT_BENCHMARK COMPACT_PRODUCTION_ROOT
  echo "Expected exactly one successful benchmark log; inspect before submitting."
fi
```

Check for **1,000,000 / 1,243,800 updates**, **150 / 75 hours**, and sufficient
free storage. Planning is read-only and never submits. Then, in the same shell:

```bash
bash sbatch/submit_part_ev_compact_production_tigris.sh \
  --mode submit \
  --benchmark "${COMPACT_BENCHMARK:?Run the plan block first}" \
  --output "${COMPACT_PRODUCTION_ROOT:?Run the plan block first}" \
  --reviewed

squeue --me -o "%.18i %.35j %.12T %.12l %R"
```

Save the printed production path. There can be only one production owner per
benchmark in this launcher's registry. An existing output cannot be overwritten.
Each accepted job ID is durably recorded in `submitted_job_ids.tsv`; an uncertain
`sbatch` response leaves an intent journal and blocks automatic resubmission.
Inspect the scheduler and journal if submission was interrupted—do not create
another output in an attempt to bypass the duplicate guard.

## Restart and monitoring

Every 1,000 attempted updates, and before/after each validation, `last.pt` is
atomically replaced with model + full Ranger state (including Lookahead slow
weights/counter) + GradScaler + Python/NumPy/Torch/CUDA RNG + trimmer counters +
training/validation stream cursors + committed best pointer + compute histogram.
The best weight checkpoint is for evaluation; it is **not** the resume source.

A Slurm warning or 15-minute-before-allocation-end deadline requests a stop.
At the next safe update boundary the worker saves and exits **75**, blocking
evaluation. If stopped during validation, the pre-validation checkpoint remains
authoritative; partially consumed validation is redone. Interruption during data
replay never overwrites the committed RNG/cursors. Abrupt process death/OOM can
lose work since the latest atomic checkpoint, but does not authorize evaluation.

On restart, deterministic loaders replay consumed batches (not gradient updates)
before restoring saved RNG and continuing. Late-run replay can take substantial
time and consumes walltime. This is restart support, **not** an automatic segmented
chain or a promise of bitwise GPU reproducibility. Completion requires the full
attempted-update and validation budgets; AMP skipped updates are reported.

Read-only validation history/progress:

```bash
bash sbatch/submit_part_ev_compact_production_tigris.sh \
  --mode status --output "$COMPACT_PRODUCTION_ROOT"
```

Training logs contain JSON progress every 100 updates. Durable outputs:

```text
runs/<run_id>/seed_101/last.pt
runs/<run_id>/seed_101/validation/check_01.json ... check_50.json
runs/<run_id>/seed_101/selected/check_*.pt
runs/<run_id>/seed_101/best_model_val.pt
runs/<run_id>/seed_101/training_complete.json
runs/<run_id>/seed_101/full_validation/metrics.json
reports/production_report.json
reports/production_report.md
```

After investigating a genuine failure, and **only once no jobs for this production
remain pending/running**, explicitly resume missing stages with:

```bash
bash sbatch/submit_part_ev_compact_production_tigris.sh \
  --mode resume --output "$COMPACT_PRODUCTION_ROOT" --reviewed
```

Optional `--indices 0` or `--indices 1` limits a resume. Completion artifacts are
audited before skipping training/evaluation; a final report is deferred if an
unscheduled arm is incomplete. No jobs are cancelled by this command. Existing
completed evaluations are reused; partial evaluations verify and reuse committed
file shards. Unresolved submission intents block retry until reviewed manually.

## Local verification

```text
.venv/Scripts/python.exe -B -m pytest tests/test_part_ev_compact_production.py tests/test_part_ev_compact_production_launcher.py tests/test_part_ev_compact_production_evaluation.py -q
```

Tests cover real model counts and step composition, shifted full schedules,
Ranger/RNG restart continuity, interrupted validation/replay, source/evidence
tamper rejection, storage guards, Slurm dependency/resource shapes, ambiguous
submission, duplicate protection, prediction resumption and metric auditing.
Mock Slurm tests submit no actual jobs. Production still requires the passed
GH200 evidence, matching RC environment and real Slurm resource validation.
