# Three-day continuation chain for the approved two-A100 State-EV run

This is a scheduling adapter for `part_ev_layerwise_ddp`, not another model or
training recipe. It replaces the single nine-day request with **four sequential
72-hour allocations** by default. Each uses two A100s on the same tier3 node,
eight CPUs and 64 GiB host memory. Nothing is submitted or cancelled merely by
installing this code.

The existing campaign remains the source of truth:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_layerwise_ddp/layerwise_2g_20261029T162243Z
```

Unchanged: layerwise attention bias, optimized all-eight-block width-32 State-EV,
seed 101, 2,327,462 parameters, global batch 512 (256/rank), SyncBatchNorm, Ranger,
the original learning-rate schedule, one million training steps, 50 validation
checks, accuracy-based checkpoint selection and final full-5M validation. This
does not launch additional seeds or the shared-bias/capacity controls. The final
evaluation is validation, not the independent 20M test population.

## Handoffs and restart state

At allocation end minus **one hour**, the trainer requests a coordinated stop at
its existing safe checkpoints. Startup, data replay and validation all count
toward this deadline. The spare hour is for checkpoint/loader/process shutdown,
not another hour of training. Slurm still enforces a 72-hour hard limit.

The existing atomic `last.pt` contains model weights, optimizer moments,
Lookahead slow weights/counter, AMP scaler, training/validation cursors and each
rank's Python/NumPy/Torch/CUDA RNG and trimming state. The next job resumes from
this **latest state**, not the best-validation weights. Learning rate and total
exposure do not restart. A partly completed validation is repeated from its
pre-validation checkpoint; it is not counted as a completed check.

The unchanged data loader restores its position by replaying the earlier train
and validation streams before restoring model RNG. This consumes I/O and wall
time but does not perform additional optimizer updates. Replay time is recorded
per rank in each segment receipt. Four slots provide room beyond the previous
216-hour planning estimate, which excluded replay; completion is not guaranteed.
If replay alone consumes a slot without committed progress, the chain fails
rather than repeatedly submitting unproductive work.

A segment returns success only after either verified training completion or a
deliberate deadline stop with a new, hash-bound checkpoint handoff. Real errors,
external termination signals, missing state and invalid runtime/source evidence
remain failures. Successors use `afterok`, with invalid dependencies cancelled.

After the last slot, a short CPU gate audits full training completion before
allowing one-GPU full-5M evaluation, then the CPU report. If training finishes
early, unused training slots audit completion and exit without constructing or
training the model. They still need a brief two-GPU allocation, so queue gaps can
delay the final evaluation. There is no background self-submitter or automatic
cancellation of unrelated jobs.

## Queue the replacement

First commit/push this implementation locally, then on **sporcsubmit**:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

EVSEG_CAMPAIGN=/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_layerwise_ddp/layerwise_2g_20261029T162243Z

bash sbatch/submit_part_ev_ddp_segments.sh \
  --mode review --campaign "$EVSEG_CAMPAIGN"

squeue -j 21988787,21988788,21988789 \
  -o "%.18i %.30j %.12T %.12l %R"
```

Those IDs are the previously queued `evddp_train`, `evddp_evaluate` and
`evddp_report` jobs. **If they are still the pending old chain**, cancel only
those three. If training has started, do not cancel it just to run these commands;
inspect progress first and decide on a checkpoint-safe transition.

```bash
scancel 21988789 21988788 21988787
```

Wait for cancellation to finish, then:

```bash
bash sbatch/submit_part_ev_ddp_segments.sh \
  --mode submit --campaign "$EVSEG_CAMPAIGN" --segments 4 --reviewed

squeue --me -o "%.18i %.30j %.12T %.12l %R"
```

The launcher refuses to submit while any job recorded in the original campaign
or a previous segment chain remains active. It never runs `scancel` itself.
It reuses the passed DDP preflight and approval: **no new GPU preflight is
queued**. `--dry-run` prints the proposed layout without writing/submitting;
it is not a substitute for the evidence checks performed by real submission.

The new chain directory, job IDs and logs are printed at submission. Save that
path as `EVSEG_CHAIN`. Names are `evseg_segment` (four jobs), `evseg_gate`,
`evseg_evaluate`, and `evseg_report`. Each chain has its own immutable source
snapshot and config, so later repository pulls do not alter queued workers.
The original frozen campaign/preflight/source files are not rewritten.

Three-day requests may fit scheduling gaps that a nine-day request cannot.
They cannot bypass drained nodes, security maintenance or resource/QOS limits,
and successors are eligible after the prior job succeeds, not guaranteed to
start immediately.

## Read-only progress

From the current repository on sporcsubmit (not a frozen worker snapshot):

```bash
cd /home/ryreu/atlas/Fresh_check
python3 -B scripts/status_part_ev_ddp_segments.py
```

This stdlib-only viewer defaults to the campaign above and the seed-101 ParT /
ordinary-EV controls in `part_ev_state_fullscale/state_full_20261024T185049Z`.
It follows the **running segment** using `squeue`, falling back to the latest log
when the scheduler is unavailable. All validation checks come from the shared
run directory, not an individual segment. It prints every check's accuracy/AUC,
best metrics, best accuracy through the common completed check, total progress,
percentage toward the next check, and a conditional next-result ETA.

During restart data replay, it shows replay progress separately and withholds
the training ETA. Stale logs or unconfirmed running jobs also suppress live ETAs.
Validation itself has no batch-level progress log, so only its typical duration
is reported. Nothing is submitted or changed, and no model checkpoint, prediction
array or dataset is opened. This is a status display, not an integrity audit.
Use `--campaign PATH --controls PATH` to override the roots, `--no-controls` for
State-EV alone, or `--offline` to skip the scheduler query. No Conda activation is
needed. Historical controls use native single-GPU BatchNorm rather than the
State-EV run's two-GPU SyncBatchNorm; their contrasts are descriptive.

For the raw chain ledger and committed handoffs:

```bash
cat "$EVSEG_CHAIN/submitted_job_ids.tsv"

evseg_jobs="$(awk '{print $2}' "$EVSEG_CHAIN/submitted_job_ids.tsv" | paste -sd, -)"
sacct -j "$evseg_jobs" -X -P -o JobID,JobName,State,ExitCode,Elapsed,Timelimit

find "$EVSEG_CHAIN/receipts" -maxdepth 1 -type f -name 'segment_*.json' \
  -exec cat -- {} \;
```

A receipt reports before/after step and validation-check counts, checkpoint
hash, status (`paused`, `complete`, `already_complete`), and elapsed/replay
seconds for both ranks. It is written only after a valid handoff. The training
run's existing validation JSON files and logs continue to be updated normally.

## Recovery and additional slots

If a segment genuinely fails, inspect the error first. Do not force a dependency
past a failure or rerun a committed segment index. Once all jobs from the old
chain are terminal, submit a **new** chain against the same campaign. Its first
slot picks up the most recent valid `last.pt`. This also handles the unlikely
case where all four slots succeed but the final gate says the budget is not yet
finished. You can choose 1–8 slots with `--segments`; the per-slot walltime and
training recipe do not change. Nothing resets the run or deletes prior evidence.

If submission itself fails partway, the printed ledger retains the IDs already
queued. Inspect them before submitting again; the launcher does not silently
cancel the jobs it already submitted.

If training completed and only evaluation/report failed, reuse that completed
chain instead of submitting more training:

```bash
bash sbatch/submit_part_ev_ddp_segments.sh --mode evaluate --chain "$EVSEG_CHAIN"
# Or, if evaluation is already complete:
bash sbatch/submit_part_ev_ddp_segments.sh --mode report --chain "$EVSEG_CHAIN"
```

Both paths recheck completion and refuse active duplicates. Evaluation preserves
the existing resumable, hash-checked per-file predictions.

If training finished but the process died before committing its last handoff
receipt (or the chain was only partly submitted), use a fresh `submit` chain with
`--segments 1`. It audits the completed run and skips training, then allows
evaluation. It never restarts the completed model from scratch.

## Storage and verification

Segmentation does not copy the dataset or keep a full restart checkpoint for each
slot. One `last.pt` is atomically replaced in the original run; best-validation
weights, the existing bounded set of selected candidates (at most 50), and final
predictions retain the old policy. Each new chain adds one source snapshot,
small receipts and logs. No deletion/cleanup is performed here. Check free space
before submission: checkpoint replacement temporarily needs both old and new
files, and final validation outputs still need room.

Local tests cover exact two-process CPU/Gloo stop/resume equivalence, including
an interrupted validation and rank-local deadline, full optimizer/RNG state,
failure propagation, completion gating, and fake-Slurm dependency/resource
requests. They do not replace cluster execution or guarantee filesystem and
GPU runtime. The original passed two-A100 preflight remains required.
