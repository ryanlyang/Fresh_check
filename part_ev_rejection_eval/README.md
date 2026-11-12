# Independent full-validation rejection: ParT versus layerwise State-EV

Evaluate the completed seed-101 full-schedule checkpoints, **without training**,
without waiting for the unused fourth State-EV training segment, and without
requiring the canceled members of the original seven-run campaign.

## What runs

- Two **independent** jobs on `debug`, each requesting one A100, 8 CPUs, 64 GiB
  host RAM, and **05:00:00**. No array throttle, no serial dependency between the
  two models, and no new GPU preflight/benchmark.
- A one-CPU, 2-GiB, 15-minute report job after both evaluations succeed.
- All **5M official validation jets**, 500k per class, batch size 512, FP32,
  the original finite validation loader and tensor preprocessing. No test data.
- The first highest-accuracy checkpoint from all 50 completed checks is audited
  for each run. Neither last weights nor a newly selected checkpoint are used.
- ParT remains native BatchNorm; State-EV retains the saved SyncBatchNorm model
  in evaluation mode. One GPU suffices for inference.
- Accuracy and QCD rejection at **30% and 50% signal efficiency** for all nine
  signal classes. Hqql/Tbl are explicitly included at these requested efficiencies,
  not their conventional 99%/99.5% working points. Exact log-odds thresholding and
  tie handling match the existing rejection calculation.
- No softmax or multi-class AUC computations: this is a rejection report, **not**
  a substitute for a complete final campaign report with AUC.

The original inference batching, precision and architecture are unchanged. Faster
turnaround comes from independent scheduling, removing obsolete dependencies,
avoiding unnecessary metrics, and resumable prediction files—not reduced data or
approximate metrics. Debug has only two A100s total; queue time is not guaranteed.
Five hours is a requested limit, not a guarantee of completion.

## Isolation and safety

Only new files implement this workflow. Pulling them does not change frozen source
used by existing jobs. The submitter freezes committed adapter files into a new
output directory, and each GPU worker imports its model, preprocessing, precision,
and audit code from that donor's **existing frozen source** in a separate process.
It re-execs the original donor Python with the original runtime settings and
checks the original source, environment, admission, GPU, selected weights, and
training records. These checks are not weakened or overwritten.

No campaign, source, checkpoint, old metrics, or old selection locks are edited.
No old job is canceled automatically. Submission and workers reject active donor
chains; submissions also reject active duplicate evaluations in the same output
parent directory. Use the default output parent for duplicate detection across
new submissions. Do not independently restart a donor chain while these jobs run.
New evaluation workers are protected by per-model file locks.

Predictions are saved atomically per ROOT file with hashes, checkpoint/file/entry
identity, observed FP32 evidence, and label hashes. A retry reuses verified complete
files and recomputes an uncommitted partial file. Changed committed outputs fail
closed. Reserve **2 GiB free**; the two uncompressed prediction sets themselves
are about 480 MB total (float32 logits and int64 labels), plus small metadata.
The final report verifies prediction hashes and cross-model label identities.

## Queue from sporcsubmit

First commit/push the new evaluator files locally, then on RC:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

export REJECTION_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_rejection_eval/rejection_$(date -u +%Y%m%dT%H%M%SZ)"

bash sbatch/submit_part_ev_rejection_eval.sh \
  --mode plan --output "$REJECTION_ROOT"
```

`plan` is read-only and checks the completed training metadata and selected weight
hashes. It does not open the ROOT data or require all seven old tasks to finish.
The stronger donor runtime audits execute in the GPU workers.

The old pending chain currently comprises:

| Job | Purpose |
|---|---|
| 21992687 | Unused fourth training segment |
| 21992688 | Old completion gate |
| 21992689 | Old evaluation |
| 21992690 | Old report |

After the plan succeeds, inspect these exact jobs. Cancel only pending members if
you want to replace that path with this evaluation. This does not delete results:

```bash
squeue -j 21992687,21992688,21992689,21992690 \
  -o "%.18i %.30j %.12T %R"

scancel --state=PENDING 21992690 21992689 21992688 21992687

bash sbatch/submit_part_ev_rejection_eval.sh \
  --mode submit --output "$REJECTION_ROOT"

squeue --me -o "%.18i %.30j %.12T %.12l %R"
```

The state-filtered cancellation leaves any job that has started running alone.
The launcher refuses submission if any old donor job is still active; inspect
that job rather than bypassing the guard. No baseline training needs rerunning.
If another historical baseline evaluation chain is active, it is also reported.

Defaults select the full-schedule SPORC control and State-EV campaigns discussed
in this study. `--part DIR` and `--state DIR` can explicitly override the donors;
the same seed, full-budget and population checks still apply. `--partition tier3`
is an alternative if debug access is unavailable. Account defaults to `reu-aisocial`.

## View results, partial progress, and retry

```bash
bash sbatch/submit_part_ev_rejection_eval.sh \
  --mode status --output "$REJECTION_ROOT"

cat "$REJECTION_ROOT/report.md"
```

`status` is read-only, counts saved file sidecars, and displays each completed
model immediately without waiting for the other. Once both finish it prints
rejection tables and `100 * (R_state / R_part - 1)` changes. Positive means better
rejection. Zero-QCD-acceptance rows use `ZB`, with no finite percentage claim.
The result paths are **this new evaluation directory**, not donor `full_validation`
directories; the earlier reader for donor metrics will still find those absent.

If a job times out or submission partially fails, first inspect logs and the new
`submitted_job_ids.tsv`. After all jobs from this evaluation are terminal:

```bash
bash sbatch/submit_part_ev_rejection_eval.sh \
  --mode resume --output "$REJECTION_ROOT"
```

This retains the frozen snapshot, skips models with completed results, and queues
only unfinished evaluations plus the report. No automatic cancel/retry loop or
silent environment compatibility override is used.

### Immediate startup failure in the first submitted version

The original worker located its Python driver relative to `BASH_SOURCE`. That is
incorrect under `sbatch`, which executes a spool-directory copy of the shell
script. The corrected worker locates the driver using the explicit `--output`
campaign argument. A relocated-worker regression test covers both GPU jobs and
the CPU report, including paths containing spaces.

For jobs `22080741` / `22080742`, inspect their stderr to confirm whether this was
the startup error. After committing/pushing/pulling the fix, use a **new output
directory** and the normal `plan` / `submit` commands above. Do not use `resume`
on the old output: it deliberately retains the old frozen worker. No failed
snapshot needs editing or deletion. No additional cancellation is needed if
the old jobs and their dependent report are all terminal; the duplicate guard
will refuse otherwise.

Interpretation: these are **single-seed validation results**, overlapping checkpoint
selection, not independent test results. The training runtimes differ (one GPU/native
BatchNorm versus two GPUs/SyncBatchNorm); the comparison does not isolate architecture.
