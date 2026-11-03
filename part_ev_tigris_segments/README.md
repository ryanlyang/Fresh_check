# Benchmark-dependent, three-day Tigris campaign

Queue the entire chain now, while the GH200 benchmark is pending:

```text
existing benchmark 723341
    -> CPU admission gate (30 min)
    -> segment 0 (72 h, 1 GH200)
    -> segment 1 (72 h, 1 GH200)
    -> segment 2 (72 h, 1 GH200)
    -> segment 3 (72 h, 1 GH200)
    -> full-completion gate (15 min)
    -> full-5M validation (5 h, 1 GH200)
    -> report (4 h 30 min, CPU)
```

All successors use `afterok`, with invalid dependencies cancelled. This does
not guarantee an immediate start after the predecessor; normal queue/resource
limits still apply. See [Slurm dependencies](https://slurm.schedmd.com/sbatch.html#OPT_dependency).

This trains **one fresh seed-101, width-32, all-eight-block, layerwise State-EV**
model (2,327,462 parameters). It retains batch 512, native single-GPU BatchNorm,
the full Ranger recipe, 1,000,000 updates and 50 validations at 20,000-update
intervals. The final evaluation covers official **5M validation**, not the
independent 20M test population. No additional seeds or control models are
queued here.

The original GH200 benchmark and all SPORC sources, approvals, checkpoints and
jobs remain untouched. The new production directory is deterministically
`checkpoints/part_ev_tigris_segments/benchmark_723341`. It does not reuse benchmark
weights or SPORC training checkpoints and does not copy the dataset.

## Queue now

Commit/push the new implementation locally, then run this block on **Tigris**.
The launcher locates the benchmark by its ledger, so it works before a pending job creates
its stdout log and after a completed job disappears from `squeue`.

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only

  bash sbatch/submit_part_ev_tigris_segments.sh \
    --mode submit \
    --benchmark-job 723341 \
    --segments 4 \
    --reviewed

  squeue --me -o "%.18i %.30j %.12T %.12l %R"
)
```

`--reviewed` explicitly permits **conditional** automatic training after all
admission checks pass; it does not bypass them. `--dry-run` prints requests
without creating directories or submitting jobs. No command automatically
cancels the SPORC chain or any other experiment.

The launcher verifies the benchmark job ID against its original ledger. If it
is still active, the admission job depends on that ID. If it has already left
the queue, accounting must show `COMPLETED|0:0`; admission still verifies the
actual benchmark result. Failed, missing or ambiguous evidence blocks training.
If the benchmark is outside the default checkpoint directory, add
`--benchmark /absolute/path/to/its/directory`; no stdout log is needed.

The CPU gate checks numerical/parity evidence, memory, precision, throughput,
unchanged sources/environment and the full-schedule projection. It stops if the
recommended remaining runtime exceeds the queued slots, or if the benchmark
needs more than the queued five-hour final evaluation. It never silently
reduces batch size, changes the recipe, loosens tolerances or stretches a job's
walltime. All phases use the benchmark's frozen ARM Python, account and Tigris
partition; GPU phases request one GPU, 8 CPUs and 128 GiB host memory.

## What happens at each handoff

At allocation end minus **one hour**, the trainer stops at a safe point, saves
and exits successfully only after verifying a committed checkpoint handoff.
Startup, the small first-slot restart check, data replay and validation all
count toward this deadline. Slurm's hard walltime remains 72 hours.

`last.pt` contains model and optimizer state, Lookahead slow weights/counter,
AMP scaler, update/validation cursors, RNG states and trimming counters. The
next segment continues from this file, **not** the accuracy-selected weights.
The learning-rate schedule and exposure counter never restart. The existing
deterministic loader replays consumed batches before restoring model RNG; this
uses I/O and allocation time but does not repeat optimizer updates.

Partial validation is not counted. Its pre-validation checkpoint is retained,
and that check is repeated on resume. Best candidates are versioned by check
(at most 50) so an interrupted validation commit cannot overwrite the best
model referenced by the latest committed restart. Only one `last.pt` is kept;
atomic replacement temporarily needs space for both old and new copies. Each
chain also stores one source snapshot, logs and small receipts. No dataset or
per-segment full checkpoint copies are made; no cleanup/deletion is performed.

A small disposable GPU check in the first slot tests save/restore equivalence
of weights, optimizer/Lookahead and scaler before production starts from a
fresh seed. Its updates do not count toward, or initialize, production.

Actual exceptions, external cancellation/termination, failed checks or a slot
that makes no committed progress return failure and block successors. A full
completion audit requires all 1M updates, all 50 validation checks and the first
highest-accuracy selected checkpoint before evaluation is allowed.

If training finishes early, later training slots audit completion and exit
without additional updates. They still need a brief GPU allocation; the final
evaluation can therefore wait for unused slots to clear. Four slots are a
bounded initial chain, not a promise of completion or an unbounded submit loop.
Replay cost and future filesystem contention can require another chain.

## Progress, recovery and continuation

```bash
EVTG_CAMPAIGN=/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_tigris_segments/benchmark_723341

bash sbatch/submit_part_ev_tigris_segments.sh \
  --mode review --campaign "${EVTG_CAMPAIGN}"
```

The launcher prints the new `Chain:` directory. Each chain contains
`submitted_job_ids.tsv`, `logs/`, `admitted.json` and `receipts/segment_*.json`.
Validation records remain under
`runs/EV_STATE_LAYERWISE_RANGER/seed_101/validation/check_*.json` in the production
directory. The chain source/config are immutable; ordinary later pulls do not
change queued workers.

If a job fails, inspect its log before taking action. Partial submissions retain
the IDs already queued in the ledger and never cancel them silently. Once all
jobs recorded in previous chains are terminal, continue the **same campaign**:

```bash
bash sbatch/submit_part_ev_tigris_segments.sh \
  --mode continue --campaign "${EVTG_CAMPAIGN}" --segments 4 --reviewed
```

This creates a new chain against the retained latest checkpoint. It also handles
an admission-job failure or interrupted completion-marker commit. Use `--segments`
to choose 1–8 slots; a single slot can audit an already-completed training run.
If the benchmark itself failed, fix/re-benchmark in a separate diagnostic first;
`continue` cannot bypass it. The launcher refuses active duplicate chains.

If training and its final handoff completed but evaluation/report failed, use
the original completed chain printed at submission:

```bash
bash sbatch/submit_part_ev_tigris_segments.sh --mode evaluate --chain "${EVTG_CHAIN}"
# Or, when evaluation already completed:
bash sbatch/submit_part_ev_tigris_segments.sh --mode report --chain "${EVTG_CHAIN}"
```

These paths recheck full completion. Evaluation reuses only hash-verified
per-file predictions; failed/incomplete files are recomputed. Reports are
`reports/tigris_state_report.json` and `.md`. They do not automatically compare
the one-GH200 run to the two-A100 run: distributed batch ordering/trimming and
platform arithmetic can differ, even with the same model, seed and recipe.

Local tests cover exact native Ranger/dropout/RNG resume, a stop mid-validation,
failure propagation, immutable benchmark evidence and fake-Slurm dependencies.
Actual GH200 operation still depends on the queued benchmark and first-slot
restart check. This implementation submits nothing until its launcher is run.
