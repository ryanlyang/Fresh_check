# Full-schedule all-eight State-EV and capacity-matched ParT

Replace only the **three last-four State-EV** runs in the existing full-schedule
SPORC campaign. Preserve its three baseline ParT runs and its one ordinary-EV run.
Create six fresh trainings in a **new**, separately frozen campaign:

| Model | Seeds | Parameters | Training source |
|---|---|---:|---|
| Baseline ParT | 101, 202, 303 | 2,143,486 | Existing full campaign |
| Ordinary ParT-EV | 101 | 2,209,022 | Existing full campaign |
| Optimized all-eight State-EV | 101, 202, 303 | 2,323,710 | New |
| Capacity-matched ParT | 101, 202, 303 | 2,323,750 | New |

## What changes, and what stays fixed

State-EV uses the previously implemented `state_fast` model: width 32 in **all
eight particle blocks**, query chunks of 64, shared per-forward pair
normalization, and activation checkpointing. This is the optimized all-eight
model, not the last-four or width-16 approximation. The GPU gate checks its
nonzero-state forward/backward equivalence as well as initialization parity.

The capacity control is native shared-bias ParT with particle FFN hidden widths
**512 -> 599 in all eight particle blocks**. Particle embeddings, attention,
shared pair-bias encoder, class-attention blocks and classifier are unchanged.
It has **no EdgeValue or state-conditioning branch**. All added parameters are
active; there are no dummy or frozen parameters used to inflate the count.
The match is approximate: the control has 40 extra parameters, **0.00172%** of
the State-EV total. Its class-block FFN widths remain 512.

Both new families start from scratch, not from screen or last-four checkpoints.
The original full Ranger recipe is reused without modification: global batch
512 on one GPU, 1,000,000 updates, 50 validation checks (one per 20,000 updates),
original RAdam/Lookahead settings and learning-rate schedule, and the same
100M training/5M validation data. A check is **not** a complete training-data
pass. Checkpoint selection remains highest validation accuracy, retaining the
first checkpoint in a tie. AUC is reported at that checkpoint.

This controls for total parameter count, not FLOPs or elapsed training time.
State-EV versus wide ParT tests whether the combined EV/state mechanism offers
more than widening baseline ParT. It does not isolate state conditioning from
ordinary EV at equal capacity; the ordinary-EV comparison is seed 101 only.
Report all three matched seeds, not the best seed.

## Queue safely on sporcsubmit

Commit/push these new files locally and pull them on RC first. No historical
implementation or existing campaign source is edited by this setup. The
launcher archives committed source, so later repository pulls do not change
its queued jobs. GPU validation still has to run on RC; local CPU tests do not
establish CUDA admission or walltime.

### 1. Read-only migration plan

Run on **sporcsubmit**, not Tigris. Do not paste Markdown fences into the shell.

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

CAP_PY=/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python
CONTROL_ROOT=/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_state_fullscale/state_full_20261024T185049Z
CAP_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_state_capacity/state_capacity_$(date -u +%Y%m%dT%H%M%SZ)"

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$CAP_PY" scripts/switch_part_ev_state_capacity.py --controls "$CONTROL_ROOT"
```

The plan derives job IDs from the original campaign ledger and verifies its
source, data/recipe/environment compatibility and approval evidence. For the
known submission it must preserve **21964940_0, _1, _2, _6** and propose replacing
only **21964940_3, _4, _5**, plus superseding evaluation **21964941** and report
**21964942**. It does not touch the shortened speed study or other projects.

### 2. Apply only that switch

After checking the plan, run:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$CAP_PY" scripts/switch_part_ev_state_capacity.py \
  --controls "$CONTROL_ROOT" --apply

squeue --me -o "%.20i %.32j %.2t %.12M %.12l %R"
```

The helper refuses if a last-four training has already started or completed,
if ownership/script identity differs, or if a kept control failed. Send that
output before proceeding; it never silently cancels running training.

For pending replacements it holds each target, verifies that Slurm exposes
an individual task with a distinct numeric job ID, then cancels that task.
If isolation fails it stops, possibly leaving that **target** held, and never
falls back to cancelling the parent training array. This precaution follows
Slurm's documented distinction between split array tasks and the parent array.
[Slurm array documentation](https://slurm.schedmd.com/job_array.html),
[scancel documentation](https://slurm.schedmd.com/scancel.html).

The old evaluation/report chain is replaced because its whole-array dependency
would otherwise include the cancelled last-four tasks. No old checkpoints,
logs, data or campaign files are deleted or rewritten. Cancellation changes
scheduler state; restoring those jobs would require an explicit resubmission.
If interrupted, rerun the read-only plan; already-cancelled targets are safe to
skip on a subsequent `--apply`.

### 3. Prepare and benchmark the replacements

```bash
bash sbatch/submit_part_ev_state_capacity.sh \
  --mode prepare --output "$CAP_ROOT" --controls "$CONTROL_ROOT"
```

This submits a CPU prepare job followed by two seed-101 GPU benchmarks, one
per new architecture. Each requests **5 hours**, below the six-hour limit for
checks. Defaults are the original tier3 / reu-aisocial / A100 / 8 CPU / 64G
resources. Explicit changes must match the frozen control launch contract;
the GPU runtime and seed-101 input fingerprints must also match its benchmarks.
Preparation verifies the data bytes and requires 20 GiB free space.

Wait for both benchmarks, then inspect:

```bash
bash sbatch/submit_part_ev_state_capacity.sh --mode review --output "$CAP_ROOT"
```

Training is blocked unless both pass initialization parity, precision,
full-length memory/AMP and sustained throughput checks. All-eight State-EV
also needs nonzero-state gradient/checkpoint-backward evidence. The wider ParT
is compared to a separately initialized width-599 native ParT, not expected
to produce the narrower model's logits.

### 4. Queue six new trainings and the combined evaluation chain

After reviewing the benchmark report:

```bash
bash sbatch/submit_part_ev_state_capacity.sh \
  --mode train --output "$CAP_ROOT" --reviewed --concurrency 3
```

The measured projection includes the full 1M steps, fifty validations and
checkpoint I/O; requested walltime adds a 30% planning margin. The common array
walltime uses the slower new architecture. You can set a larger
`--train-hours N` / `--evaluate-hours N`, but not undercut the recommendation.
The launcher checks partition MaxTime and never substitutes a shorter training
schedule to fit. Queue delays, contention and restart replay are not bounded
by this estimate. Default concurrency is 3, optionally up to 6.

The new array has:

| New task index | Training |
|---:|---|
| 0, 1, 2 | All-eight State-EV, seeds 101, 202, 303 |
| 3, 4, 5 | Wide ParT, seeds 101, 202, 303 |

The chain is:

```text
six new trainings + still-active kept control tasks (individually)
    -> CPU checkpoint lock/import
    -> ten full-5M validation evaluations
    -> combined report
```

Completed controls are audited for all 50 checks and checkpoint integrity and
do not need a scheduler dependency. Their selected checkpoints are copied
byte-for-byte into the new output namespace, retaining original metadata and
source-campaign identity. Control training is never restarted by the new
launcher. The checkpoint lock refuses incomplete or changed source evidence.

The ten evaluations are **validation**, not the independent 20M test set.
The 5M population overlaps checkpoint selection. Results include paired
three-seed accuracy/AUC contrasts, all 50 learning-curve checks, and classwise
QCD rejection. Ordinary-EV contrasts are explicitly seed-101-only.

```bash
cat "$CAP_ROOT/submitted_job_ids.tsv"
cat "$CAP_ROOT/reports/state_capacity_report.md"
```

Save `CAP_ROOT`; reuse that exact directory in later shells. The detailed report
is `reports/state_capacity_report.json`. Retry terminal phases with `--resume`;
active or partially submitted pipelines block duplicate submissions. Evaluation
retries reuse hash-verified per-file outputs. A failed benchmark is retained;
inspect it and use a new campaign for code changes. Do not use the old launcher's
whole-array resume command, which would bring back the replaced last-four runs.

## Local verification

The `tests/test_part_ev_state_capacity_*` tests cover model counts, original
backbone preservation, all-eight state placement, nonzero gradients, exact
Ranger checkpoint/resume at the decay boundary, frozen controls, capacity
contrasts, immutable imports, and fake-scheduler cancellation/dependency safety.
They do not contact Slurm. Run those tests and the existing fullscale/speed
regressions before pushing changes.
