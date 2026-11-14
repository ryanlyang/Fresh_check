# State-bias, relation–state multiplication and pair-memory production

Three independent full-schedule, seed-101 production options on SPORC `tier3`.
Each uses its own already-passed benchmark's frozen model, preprocessing,
environment and numerical settings. Use `--model bias`, `--model multiply`,
or `--model memory`; each submission creates only that model's chain.

| Model | Parameters | Training allocation | Evaluation allocation |
|---|---:|---|---|
| State-conditioned attention bias | 2,329,510 | 2 A100s, 308 h (12 d 20 h) | 1 A100, 4 h |
| Relation–state multiplication | 2,343,846 | 2 A100s, 226 h (9 d 10 h) | 1 A100, 2 h |
| Evolving pair memory | 2,364,294 | 2 A100s, 372 h (15 d 12 h) | 1 A100, 4 h |

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
Training jobs have no dependency on the other models and no concurrency cap.
Training is one long allocation, not a three-day segment chain. All jobs request
8 CPUs and 64 GiB host RAM; the final CPU report requests one hour.
There is no additional GPU preflight or benchmark job.

## Queue pair memory only from sporcsubmit

Bias and multiplication are already queued. **Do not resubmit those chains.**
The memory addition changes only the production model registry; it reuses the
existing engine and the exact pair-memory science snapshot admitted by benchmark
22089090 / summary 22089091. No model, optimizer, precision, batch-size, or schedule
change and no new GPU preflight are needed. The original benchmark launcher stays
benchmark-only.

Commit/push the new production files locally, then run this in Bash on
`sporcsubmit`. It locates the benchmark by its unique summary log rather than
guessing the timestamp. The plan is read-only; the pull only updates the working
checkout, not the frozen sources used by queued jobs.

```bash
MEMORY_BENCH=""
MEMORY_PROD="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_state_production/memory_full_seed101"

mapfile -t MEMORY_LOGS < <(find \
  /home/ryreu/atlas/Fresh_check/checkpoints/part_ev_pair_memory \
  -maxdepth 3 -type f -name 'evmem_summary_22089091.out' -print)

if [[ ${#MEMORY_LOGS[@]} -eq 1 ]]; then
  MEMORY_BENCH="$(realpath -e -- "$(dirname -- "$(dirname -- "${MEMORY_LOGS[0]}")")")"
  (
    set -euo pipefail
    cd /home/ryreu/atlas/Fresh_check
    git pull --ff-only
    bash sbatch/submit_part_ev_state_production.sh \
      --mode plan --model memory \
      --benchmark "${MEMORY_BENCH}" --output "${MEMORY_PROD}"
  )
else
  echo "Expected exactly one pair-memory summary log; inspect paths before submitting."
fi
```

Review the printed model, seed, 1,000,000 updates / 50 checks, 372-hour training
and 4-hour evaluation allocations. If the plan passed and is correct, submit
**only memory**:

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  test -n "${MEMORY_BENCH:-}"
  test -n "${MEMORY_PROD:-}"
  test -f "${MEMORY_BENCH}/summary.json"

  bash sbatch/submit_part_ev_state_production.sh \
    --mode submit --model memory \
    --benchmark "${MEMORY_BENCH}" --output "${MEMORY_PROD}" \
    --reviewed

  squeue --me -o "%.18i %.40j %.12T %.12l %R"
)
```

Expect `evstateprod_memory_train -> evstateprod_memory_evaluate ->
evstateprod_memory_report`, with the latter two depending only on the preceding
memory job. No jobs in the existing bias/multiplication chains are changed.
372 hours is below the reported 480-hour `tier3` limit; the launcher still
rechecks current limits, account/QoS admission and aggregate storage at submission.

## Original two models: queue reference

First commit/push these production files locally, then pull on RC. Existing
benchmark snapshots, queued campaigns, environments and data are left untouched.
The fresh production launcher refuses uncommitted production source files.

The following is retained for fresh bias/multiplication campaigns, not as part of
the memory submission above. Do not rerun it for the already-queued outputs.
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
bash sbatch/submit_part_ev_state_production.sh --mode status --output "$MEMORY_PROD"
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
# Or the same command with "$MULTIPLY_PROD" or "$MEMORY_PROD".
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
Pair memory exists only within each forward pass. Its learned readers, writers
and gates (including their optimizer states) are checkpointed; the per-jet memory
activations are not saved or carried between batches.
Storage planning conservatively covers 50 possible improving weight candidates,
the final best copy, full/temporary restart writes, 5M prediction chunks and logs.
It includes a 4-GiB free reserve and totals roughly 6.5 GiB per fresh campaign.
The submission registry also counts remaining output budgets of other campaigns
created by this launcher, sharing that reserve rather than ignoring the other models.
For fresh pair memory the conservative requirement is approximately 6.44 GiB
including that reserve; additional remaining budgets from active campaigns are
counted separately. Having 6.44 GiB free alone need not be sufficient.
This is a headroom check, not a filesystem quota reservation against unrelated jobs.
It never deletes existing checkpoints, results or data.

## Local verification

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_part_ev_state_production_protocol.py tests/test_part_ev_state_production_engine.py tests/test_part_ev_state_production_memory.py tests/test_part_ev_state_production_launcher.py
```

Tests cover real benchmark admission logic, frozen imports, independent Slurm
dependencies, partial submission, storage/partition guards, two-process CPU
stop/resume (including partial validation), selection and evaluation chunk reuse.
Pair-memory coverage also exercises its real graph through the production
checkpoint/restore and rank-zero validation helpers, exact two-rank CPU continuation
through Lookahead updates, optimizer coverage of every memory head, and batch-local
memory lifetime. Fake-Slurm tests submit memory after both other chains and check
their source bytes, job IDs, independent dependencies and aggregate storage guards.
Local CPU tests do not replace the existing remote A100 benchmark evidence.
