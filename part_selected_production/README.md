# Seven full-schedule production runs on Tigris

This is the production successor to the successful seven-model benchmark and
four-arm restart retry. It does **not** change the benchmark or its model code.
The original scientific snapshot, prepared inputs/backends, environment and
retry evidence remain required dependencies; do not delete them while running.

All seven runs are fresh, seed 101, one GH200 each, batch 512, the original full
Ranger learning-rate schedule, 1,000,000 attempted updates and 50 checks of
1.28M validation jets each. A check is 20,000 updates, not a dataset pass.
Training uses the full 100M training pool. Best accuracy (first tie) selects the
checkpoint for subsequent full-5M validation; no test data is opened.

| Index | Model | Parameters | Train limit | Evaluation limit |
|---:|---|---:|---:|---:|
| 0 | PART_MATCH_STATE_SHARED | 2,323,750 | 48h | 2h |
| 1 | PART_MATCH_STATE_LAYERWISE | 2,327,894 | 48h | 2h |
| 2 | SELECTED_PART_MATCH_STATE_LAYERWISE | 2,337,814 | 192h | 3h |
| 3 | STATE_EV_SHARED | 2,323,710 | 144h | 2h |
| 4 | EV_LAYERWISE | 2,212,774 | 72h | 2h |
| 5 | SELECTED_PART_SHARED | 2,153,406 | 192h | 3h |
| 6 | SELECTED_STATE_EV_LAYERWISE | 2,337,382 | 336h | 4h |

The three capacity controls (0,1,2) have **shared attention bias, no EV, no
state**, regardless of the target named in their IDs. Task 2 retains selected
relations. State tasks use optimized width-32 state messages in **all eight**
particle blocks, not just the last four. Model counts/configurations and the
recipe are those authenticated by the benchmarks, not rebuilt from a new checkout.

## Queue

First commit/push these changes locally, then pull them on **Tigris**. The submitter
requires committed, clean production source files. No new GPU benchmark or
preflight is submitted. CPU admission revalidates the existing evidence and
environment, and `sbatch --test-only` checks every requested resource shape
against the current scheduler before the first real submission.

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

SEVEN_BENCH="/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_fullscale/seven_buildfix_20261106T005700Z"
SEVEN_RETRY="${SEVEN_BENCH}/benchmark_retries/retry.xsunwcjm"
SEVEN_PROD="/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_production/seven_full_20261107"

bash sbatch/submit_part_selected_production_tigris.sh \
  --mode plan --benchmark "${SEVEN_BENCH}" --retry "${SEVEN_RETRY}" \
  --output "${SEVEN_PROD}" --concurrency 3

bash sbatch/submit_part_selected_production_tigris.sh \
  --mode submit --benchmark "${SEVEN_BENCH}" --retry "${SEVEN_RETRY}" \
  --output "${SEVEN_PROD}" --concurrency 3 --reviewed

squeue --me -o "%.18i %.30j %.12T %.12l %R"
```

`plan` is read-only. `submit` freezes only the small new orchestrator and creates
a separate production directory; it never edits the benchmark, retry, controls,
dataset or environment. A benchmark has one production owner to prevent an
accidental second campaign under a different output name.

There are seven individual training jobs (different walltimes), seven evaluation
jobs, and one 30-minute CPU report. Default concurrency 3 uses three `afterany`
training lanes: failure of one model does not prevent an unrelated model from
starting. Set `--concurrency 7` to leave all seven trainings independently eligible.
This controls training concurrency, not evaluation concurrency. Each evaluation
uses `afterok` on its own training, so it need not wait for the slowest model.
The report uses `afterok` on all newly queued evaluations and audits all seven.
Invalid dependencies are canceled; they do not masquerade as completed science.

These are long single allocations, **not** three-day segments. Maintenance and
account/QOS limits can delay or reject them, even if the benchmark passed. A
requested walltime is a limit, not an estimated start time or runtime guarantee.

## Progress, restart and partial submissions

```bash
bash sbatch/submit_part_selected_production_tigris.sh \
  --mode status --output "${SEVEN_PROD}"
```

Full restart `last.pt` is committed every 1,000 updates and around validation.
It contains the model, RAdam moments, Lookahead slow weights/counter, scaler,
Python/NumPy/CPU/CUDA RNG, trimmer counters, completed update and validation
stream cursors. Restart replays both streams before restoring RNG; it can be
time-consuming but does not replay optimizer updates or reset the schedule.

A Slurm warning/near-deadline stop saves a coherent restart and exits 75, so it
**does not release evaluation**. A stop inside validation leaves the previous
pre-validation checkpoint; the next run repeats that check. Candidate best
weights are written separately before their pointer is committed in `last.pt`.
Only all 1M updates plus all 50 checks can create a valid completion marker.

After all jobs belonging to this production have left the queue, explicitly
resume missing stages with the same output. Inspect errors before resuming a
genuine failure; no failure is silently treated as a planned handoff.

```bash
bash sbatch/submit_part_selected_production_tigris.sh \
  --mode resume --output "${SEVEN_PROD}" --concurrency 3 --reviewed
```

Optional `--indices 2,6` restricts a recovery to those tasks. A completed training
is not retrained if only evaluation is missing. Evaluation resumes verified
per-file prediction shards. Existing pending/running jobs block resubmission;
do not blindly cancel other campaigns. All accepted job IDs are durably recorded
in `submitted_job_ids.tsv`. An interrupted/ambiguous `sbatch` response leaves
`submission_intent.json` with `status: unknown`; inspect that exact request and
the scheduler before manually resolving it. The launcher will not guess whether
it was accepted. A failed `--test-only` submits **zero** jobs; fix the resource
issue and use `resume` for the already-frozen output.

Do not edit frozen source or `production.json`. Pulling the working repository
does not change queued workers, whose code and scientific dependencies are pinned.

## Storage and results

Admission uses the benchmark's measured conservative storage projection (about
12.1 GiB including a 4 GiB free-space reserve for this evidence). All seven
outputs are included, even when training concurrently. Free space is not reserved
against other work. No full-dataset cache, ROOT copy, environment clone, dense
relation cache or profiler trace is created. Storage checks fail without deleting
existing artifacts. Keep the original scientific snapshot/backends/evidence.

There is one rolling full restart per model plus atomic-write headroom, at most
50 small improvement-only weight snapshots per model, the finalized best model,
and seven full-5M logit outputs. This retention is already covered by the measured
51-weight-per-model estimate. There are not 50 optimizer checkpoints per model.

Final outputs: `reports/production_report.md` and `.json`, including seven-model
accuracy/AUC, internal matched contrasts, QCD rejection with passing counts, all
50 validation checks, selected checkpoints and training/validation timings.
Per-job session records include elapsed process time (including restart replay).

The external baseline ParT, ordinary EV and standard layerwise State-EV controls
are **not rerun**. Their existing metadata registry is preserved, but external
metric contrasts are explicitly pending a separate compatibility audit (recipe,
population, runtime and checkpoint selection). This avoids mixing SPORC/DDP,
short-screen or training-check metrics with the full-5M GH200 results. The seven
new trainings and their evaluations do not depend on those external controls.

These remain single-seed validation results, not independent test results or
multi-seed significance claims. The 5M population overlaps checkpoint selection.
