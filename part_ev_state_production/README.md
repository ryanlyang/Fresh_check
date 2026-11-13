# State-bias and relation–state multiplication production

Two independent full-schedule, seed-101 campaigns on SPORC `tier3`.
Each uses its own already-passed benchmark's frozen model, preprocessing,
environment and numerical settings. Pair memory is not included or modified.

| Model | Parameters | Training allocation | Evaluation allocation |
|---|---:|---|---|
| State-conditioned attention bias | 2,329,510 | 2 A100s, 308 h (12 d 20 h) | 1 A100, 4 h |
| Relation–state multiplication | 2,343,846 | 2 A100s, 226 h (9 d 10 h) | 1 A100, 2 h |

These are the recommendations from the November 13 benchmarks, including
all 50 validations, checkpoint writes and a 30% margin. The launcher recomputes
the recommendations from each benchmark; it does not substitute a fixed 12-day
limit. Queue delay, restart data replay and future I/O contention are excluded.
Submission checks the live partition limit and runs `sbatch --test-only` for
every resource shape before submitting the first job. A failure stops submission;
it never silently reduces the walltime or training budget.

## Unchanged scientific setup

- One fresh training per model, seed 101. No benchmark weights or old best weights
  are used to initialize training.
- 1,000,000 update attempts, global batch 512 (256 per rank), two same-node A100s,
  DDP and SyncBatchNorm. Skipped AMP updates are counted and monitored, as in the
  benchmark/reference trainer; they do not cause extra data exposure.
- Full 100M training pool; original Ranger recipe and learning-rate schedule.
  Native trimming, state architecture, checkpointed chunking, precision, and
  parameter initialization come from the exact benchmark source snapshot.
- 50 accuracy-selection checks, one per 20,000 updates, each on 1.28M validation
  events. A check is not a full pass through the training dataset.
- First highest-accuracy checkpoint wins ties. Final FP32 evaluation covers all
  50 official validation files: 5M jets, 500k per class. AUC and QCD rejection
  are computed at that checkpoint, not maximized independently.
- No test-set evaluation. These remain single-seed validation results overlapping
  checkpoint selection, not independent test results or evidence of reproducibility.

Each campaign queues only `train -> afterok:evaluate -> afterok:report`.
The two training jobs have no dependency on one another and no concurrency cap.
Training is one long allocation, not a three-day segment chain. All jobs request
8 CPUs and 64 GiB host RAM; the final CPU report requests one hour.
There is no additional GPU preflight or benchmark job.

## Queue from sporcsubmit

First commit/push these production files locally, then pull on RC. Existing
benchmark snapshots, queued campaigns, environments and data are left untouched.
The fresh production launcher refuses uncommitted production source files.

Set paths and review both plans (read-only):

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

BIAS_BENCH="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_state_bias/state_bias_20261113T063807Z"
MULTIPLY_BENCH="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_state_multiply/state_multiply_20261113T065552Z"
BIAS_PROD="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_state_production/bias_full_seed101"
MULTIPLY_PROD="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_state_production/multiply_full_seed101"

bash sbatch/submit_part_ev_state_production.sh \
  --mode plan --model bias --benchmark "$BIAS_BENCH" --output "$BIAS_PROD"

bash sbatch/submit_part_ev_state_production.sh \
  --mode plan --model multiply --benchmark "$MULTIPLY_BENCH" --output "$MULTIPLY_PROD"
```

If both plans are correct, submit each independently:

```bash
bash sbatch/submit_part_ev_state_production.sh \
  --mode submit --model bias --benchmark "$BIAS_BENCH" \
  --output "$BIAS_PROD" --reviewed

bash sbatch/submit_part_ev_state_production.sh \
  --mode submit --model multiply --benchmark "$MULTIPLY_BENCH" \
  --output "$MULTIPLY_PROD" --reviewed

squeue --me -o "%.18i %.40j %.12T %.12l %R"
```

The original dedicated SPORC Python environment is read from each benchmark.
No environment installation, dependency update or compiled-extension rebuild is performed.
Keep the benchmark/reference directories and environments available: production
imports their authenticated frozen science, rather than making a second full
repository copy. Production orchestration gets its own small committed snapshot.
Later checkout pulls cannot alter already queued workers.

## Progress, recovery and outputs

Read-only status (no checkpoint or prediction payload loading):

```bash
bash sbatch/submit_part_ev_state_production.sh --mode status --output "$BIAS_PROD"
bash sbatch/submit_part_ev_state_production.sh --mode status --output "$MULTIPLY_PROD"
```

`logs/evstateprod_*_train_*.out` records training every 100 updates and all validation
checks. `runs/<run_id>/seed_101/validation/check_*.json` stores check accuracy/AUC.
The final outputs are `full_validation/metrics.json` under the run and
`reports/production_report.{json,md}` under the campaign. The report rechecks
selection, predictions and metrics, and includes the learning curve and rejection
counts. It does not automatically mix results from historical one-GPU controls.

`last.pt` contains the full model, optimizer/Lookahead, scaler, step/validation
cursors and every rank's RNG/trimmer state. It is atomically saved every 1,000
updates and on validation boundaries. Best weights are separate. Before walltime
expiry, a monotonic deadline and Slurm's 15-minute warning request a coordinated
save; interruption exits nonzero (normally 75, or torchrun's wrapper exit 1),
so `afterok` evaluation cannot run on incomplete training.

Only after the old chain has left the active queue, explicitly resume:

```bash
bash sbatch/submit_part_ev_state_production.sh \
  --mode resume --output "$BIAS_PROD" --reviewed
# Or the same command with "$MULTIPLY_PROD".
```

Resume audits existing artifacts and queues only missing stages. It refuses an
active/pending chain, a duplicate output for the same benchmark, changed frozen
evidence, or an unresolved submission attempt. If a partial submission or an
invalid dependency leaves jobs pending, inspect the recorded IDs and reconcile
them before retrying; the launcher never automatically cancels jobs.

Training continuation replays the deterministic train/validation streams before
restoring the saved per-rank random state. This preserves the existing stream
semantics but costs allocation time; no promise is made that a restart fits in
the original estimate. Validation interrupted mid-check restarts from its
pre-validation commit. Final 5M evaluation resumes by verified per-file prediction
chunks; it does not require training to restart. Genuine errors block successors.

## Persistent storage

No dataset duplication, pair-feature cache or persistent pair-state tensor cache.
Storage planning conservatively covers 50 possible improving weight candidates,
the final best copy, full/temporary restart writes, 5M prediction chunks and logs.
It includes a 4-GiB free reserve and totals roughly 6.5 GiB per fresh campaign.
The submission registry also counts remaining output budgets of other campaigns
created by this launcher, sharing that reserve rather than ignoring the other arm.
This is a headroom check, not a filesystem quota reservation against unrelated jobs.
It never deletes existing checkpoints, results or data.

## Local verification

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_part_ev_state_production_protocol.py tests/test_part_ev_state_production_engine.py tests/test_part_ev_state_production_launcher.py
```

Tests cover real benchmark admission logic, frozen imports, independent Slurm
dependencies, partial submission, storage/partition guards, two-process CPU
stop/resume (including partial validation), selection and evaluation chunk reuse.
Local CPU tests do not replace the existing remote A100 benchmark evidence.
