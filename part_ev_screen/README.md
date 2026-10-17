# SPORC EdgeValue screen — queue guide

This is the seven-run, single-seed experiment in
[the agreed plan](../part_ev_fullscale/CONTEXT_GATING_SCREEN_PLAN.md).
It is separate from the existing full-scale campaigns. Nothing here changes or
submits their jobs. The implementation is locally tested; GPU fit, throughput,
and numerical checks still run on SPORC before production is authorized.

| Index | Run | Optimizer recipe |
|---:|---|---|
| 0 | ParT, standard shared pair bias | Ranger |
| 1 | ParT + existing EV | Ranger |
| 2 | EV + jet context | Ranger |
| 3 | EV + relation-gated particle values | Ranger |
| 4 | EV + both mechanisms | Ranger |
| 5 | ParT, standard shared pair bias | AdamW |
| 6 | ParT + existing EV | AdamW |

All use seed 101, batch 512, the official 100M training population, and 200,000
attempted updates (102.4M exposures). Validation occurs every 20,000 updates:
ten checks of 1.28M validation examples, reporting accuracy, CE, macro OVO AUC
and macro OVR AUC. Highest validation accuracy selects the checkpoint; ties keep
the earlier checkpoint. All seven selected checkpoints are then evaluated on the
same complete 5M validation population. **No 20M test files are needed or read.**

The context MLP scales the 64-channel relation representation using the masked
mean of current-layer particle states. The rank-four gate uses pair features to
modulate the native particle value channels. Both start at the identity; they
retain the standard shared attention bias and the existing EV initialization.
The gate is query-chunked with activation recomputation, avoiding a full
`batch × heads × particles × particles × value_channels` tensor.

## 1. Synchronize the code, then check SPORC

Commit and push the implementation from your local checkout first. The launcher
refuses uncommitted relevant code and freezes a committed source snapshot.
On **sporcsubmit** (not Tigris):

```bash
cd /home/ryreu/atlas/Fresh_check
git status --short
git pull --ff-only
hostname
uname -m
my-accounts
sinfo -p tier3,sporc -o '%12P %12t %24G %12l'
df -h /home/ryreu/atlas
```

Expected architecture: `x86_64`. Default requests are account `reu-aisocial`,
partition `tier3`, one `a100` GPU, eight CPUs, 64G host RAM, and two data workers.
Use the live account/partition output to confirm availability. If the accepted
production partition is `sporc`, explicitly pass `--partition sporc` at prepare;
there is no silent fallback to another partition or to debug. All runs must use
the same GPU model and memory capacity. Keep at least 8 GiB free after preparing
the environment; the launcher never deletes data or checkpoints.

## 2. Create the isolated environment

```bash
bash scripts/setup_part_ev_screen_sporc_env.sh
```

This clones the existing x86 `atlas_kd_sporc` environment into
`/home/ryreu/miniconda3/envs/part_ev_screen_sporc`, retains its CUDA-enabled PyTorch,
and installs pinned Weaver 0.4.17 plus the reference preprocessing dependencies
and matplotlib. The donor and Tigris environments are untouched. If the new path
already exists, it is not modified. An interrupted installation must be inspected
before any retry; an existing directory alone does not prove setup succeeded.
Prepare records installed distribution versions and relevant Weaver source hashes.

## 3. Queue preparation and the seven GPU probes

```bash
SCREEN_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/ev_context_gate_200k_$(date -u +%Y%m%dT%H%M%SZ)"

bash sbatch/submit_part_ev_screen_sporc.sh \
  --mode prepare --output "$SCREEN_ROOT" --partition tier3 --concurrency 2

printf 'Save this campaign path: %s\n' "$SCREEN_ROOT"
squeue --me -o '%.18i %.25j %.2t %.10M %.12l %R'
```

Add `--dry-run` to preview without creating files or submitting jobs. Data defaults
to `/home/ryreu/atlas/datasets/JetClass/Pythia/{train_100M,val_5M}`; specify the
JetClass parent with `--data-root` if it moved.

This queues only **prepare (2h) → benchmark array (seven tasks, 2h each)**.
Every pre-production allocation is under six hours. Each benchmark checks initial
same-platform reference parity, 20 warmup + 4,200 measured updates across multiple
ROOT fetch windows, full-128-particle forward/backward memory, finite losses and
AMP behavior, peak-LR execution, checkpoint-write time, and validation AUC cost.
An AdamW peak-LR probe checks execution, not mature optimizer-state behavior.
Benchmark weights are separate artifacts and are never used for training.

### Recovery from the first 40-GB A100 out-of-memory probes

The `context_gate_20261016T204528Z` benchmarks used the v1 screen driver. That
driver enabled outer FP16 autocast but built Weaver with `use_amp=False`;
Weaver's inner autocast context disabled it again. The transformer therefore ran
in FP32. Both ordinary ParT probes completed, but all five EV probes exhausted
the approximately 39.52-GiB GPU. Those probes do **not** establish the memory fit
or throughput of the intended mixed-precision recipe.

The v2 driver scopes both native and outer AMP to the training forward, retains
FP32 parameters/GradScaler, and explicitly uses FP32 for validation, diagnostics,
and initial parity. The first benchmark training and validation forwards audit
actual Linear/Conv1d output dtypes. Their evidence is saved in the benchmark JSON
and required before production. Batch 512, all seven architectures, initialization,
optimizers, and data/exposure budgets are unchanged. No gradient accumulation or
additional activation checkpointing was introduced by this fix.

After committing/pushing the fix locally, run on **sporcsubmit**:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

# NEW campaign: do not --resume the v1 snapshot or edit its frozen source.
SCREEN_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_$(date -u +%Y%m%dT%H%M%SZ)"
bash sbatch/submit_part_ev_screen_sporc.sh \
  --mode prepare --output "$SCREEN_ROOT" --partition tier3 --concurrency 2
printf 'Save this campaign path: %s\n' "$SCREEN_ROOT"
```

The existing dedicated environment and staged data are reused without modifying
them. All seven probes must be rerun, including the two controls, because their
timings used the old precision behavior. This queues only the two-hour prepare
and benchmark phases, not training. Keep the original campaign for its logs.
Actual GPU fit still needs these new measurements; CPU regression tests do not
prove a 512-jet batch fits. If it still fails, inspect the new traceback before
choosing targeted recomputation, a larger-memory GPU, or a common smaller batch.

All seven probes must finish successfully. A timeout or missing output is not a
pass. Defaults require <5% measured skipped updates, no skipped full-length probe
updates, and peak allocated memory below 90% of the GPU capacity. Native and
worst-length peaks are both recorded. Matching input fingerprints must agree
across all tasks. These probes do not establish eventual convergence or predict
queue waits and multi-job I/O contention.

## 4. Review the probes, then queue production and dependent evaluation

Restore `SCREEN_ROOT` to the printed campaign path if starting a new shell.

```bash
bash sbatch/submit_part_ev_screen_sporc.sh \
  --mode review --output "$SCREEN_ROOT"
```

The review prints throughput, memory, validation timing, skipped updates, and the
recommended training hours using the slowest measured complete fetch window,
validation/AUC and checkpoint costs, plus a 30% planning margin. This is not a
guarantee; restarts additionally replay the consumed data stream. If the probes
look healthy and the requested duration fits the partition:

```bash
bash sbatch/submit_part_ev_screen_sporc.sh \
  --mode train --output "$SCREEN_ROOT" --reviewed --concurrency 2
```

This queues the whole remaining graph:

```text
train[0..6] → full-validation evaluation[0..6] → report
```

Dependencies use `afterok` on the same SPORC scheduler. Training walltime comes
from the probes. `--train-hours N` may increase it, but cannot lower it below the
measured recommendation. Evaluation walltime is estimated separately with a
two-hour minimum; report gets two hours. If a measured training request exceeds
the site's limit, do not shorten it blindly: inspect timing/resources first.

There are **seven training runs**, not fourteen. The additional seven jobs only
evaluate their saved best checkpoints. No test-set evaluation is queued. A failed
array prevents downstream execution; no automatic cancellations or resubmissions
are performed by this launcher.

## Monitoring and safe recovery

### Short AMP recovery check after a completed probe

If the v2 benchmarks completed training/validation and passed memory, precision,
and initial parity checks, but failed only because a full-length stress update
was AMP-skipped, use the separate recovery diagnostic. Do not rerun the long
benchmark or edit its failed report. Commit/push these diagnostic scripts locally,
then pull them on **sporcsubmit** and run:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only
SCREEN_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z"

bash sbatch/submit_part_ev_screen_amp_recovery.sh \
  --campaign "$SCREEN_ROOT" --concurrency 2
```

Add `--dry-run` to inspect the requests without creating files/jobs. The launcher
queues seven **one-hour, one-GPU** tasks on the original partition/resources,
followed by a **10-minute CPU summary** using `afterany` so failures are included.
It refuses active original/recovery jobs for this campaign. A unique
`diagnostics/amp_recovery.XXXXXX` directory holds frozen diagnostic code, an input
manifest, per-step JSONL, and one JSON result per model. The launcher prints the
directory, job IDs, and summary log path. It does not submit training.

Each task restores its saved `benchmarks/RUN_ID_probe.pt`, including weights,
optimizer/Lookahead state, GradScaler, and RNGs. These checkpoints were saved
**after** the original three full-length stress attempts; they cannot reconstruct
which of those attempts skipped. The original benchmark source and environment
are reused without editing them. Checkpoint/report hashes and the recreated first
training-batch fingerprint must match.

The check uses the original synthetic stress input: batch 512, all 128 particle
slots active, padding filled with the first real particle, and trimming disabled.
It holds the original recipe's peak LR constant for 8 calibration attempts, then
requires 10 consecutive finite, non-skipped optimizer updates. This is a bounded
stress continuation, **not** a training resume or a performance measurement.
The scaler is restored, never manually reset or lowered. Calibration may back off
the scale but must end with a successful update. A confirmation skip, nonfinite
loss/state, missing optimizer step, or original memory-limit violation fails.
Every attempt logs the scale before/after, loss, unscaled gradient norm, actual
optimizer step count, and allocated/reserved peak memory. No weights are saved.

After completion, read the printed summary log, or rerun this read-only summary
with the **exact diagnostic directory printed by the launcher**:

```bash
RECOVERY="/absolute/path/printed/by/the/launcher/amp_recovery.XXXXXX"
/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python \
  "$RECOVERY/check_part_ev_screen_amp_recovery.py" --phase summary --output "$RECOVERY"
```

Passing means only that this short synthetic check recovered. It does not prove
long-run stability or authorize production: the original all-seven benchmark
gate remains unchanged. Review the numerical records before deciding how to
proceed. Missing/failed tasks are not silently treated as passing.

### Launch production after reviewed AMP recovery

After reviewing passing recovery records for **all seven** models, use the
recovery-aware submitter instead of the original `--mode train` command. Commit
and push the new approval scripts locally, then on **sporcsubmit**:

```bash
cd /home/ryreu/atlas/Fresh_check &&
git pull --ff-only &&
bash sbatch/submit_part_ev_screen_after_recovery.sh \
  --campaign /home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z \
  --recovery-job 21904258 \
  --reviewed --concurrency 2
```

`--recovery-job` identifies the exact diagnostic via its submission ledger;
`--recovery /absolute/diagnostic/directory` is an alternative. `--dry-run` previews
requests without approving evidence, creating files, or submitting jobs.

The submitter freezes a separate `production_approval/` wrapper. It validates the
original benchmark/checkpoint hashes, precision/parity/memory/input contracts,
every recovery attempt, the matching per-step logs, and all seven model identities.
The original failed benchmark JSONs, diagnostic results, `campaign.json`, and
frozen `source/` are **not edited**. Approval records the reviewed basis and exact
evidence hashes; each worker verifies them again and checks the actual GPU/runtime.
This is an explicit admission decision based on the original benchmarks plus the
short recovery check, not a claim that the original benchmarks passed.

Production uses the **unchanged frozen trainer**, with only its admission check
replaced for that invocation. All seven runs start from their original seed-101
initialization with fresh optimizers and the normal fresh GradScaler. **No probe
weights, optimizer state, or diagnostic scaler settings enter production.** Data,
batch 512, AMP, model architectures, schedules, and the 200,000-update budget stay
unchanged. Existing production files block a first launch rather than silently
resuming or overwriting them.

The submission graph is `train[7] -> full-validation[7] -> report`, using `afterok`
dependencies on SPORC and two concurrent GPU tasks per array by default. Training
and evaluation walltimes use the original measured projections plus 30%; the
report requests two hours. `--train-hours N` can increase, but cannot decrease,
the training estimate. The short recovery loop is **not** used for throughput
estimates. No new benchmark or 20M-test jobs are submitted. The measured margin
is not a guarantee under filesystem contention or restart replay.

The printed job IDs are also appended to the campaign's existing
`submitted_job_ids.tsv`. Monitor those IDs and the normal `logs/evscr_train_*`
files; training logs retain cumulative AMP skip counts and validation progress.

```bash
# Read-only approval/timing review, including while production is running:
bash sbatch/submit_part_ev_screen_after_recovery.sh \
  --campaign "$SCREEN_ROOT" --mode review

# Only after inspecting terminal failures and any stale dependent jobs:
bash sbatch/submit_part_ev_screen_after_recovery.sh \
  --campaign "$SCREEN_ROOT" --reviewed --resume
```

Use this same submitter for subsequent `--mode evaluate --resume` or
`--mode report --resume` recovery. Active jobs from the campaign or its AMP
diagnostics block duplicate submission. Resume uses only ordinary production
checkpoints; the approval never grants permission to load probe weights.

### Ordinary training and evaluation monitoring

```bash
cat "$SCREEN_ROOT/submitted_job_ids.tsv"
squeue --me -o '%.18i %.25j %.2t %.10M %.12l %R'
ls "$SCREEN_ROOT/logs"
```

Logs contain JSON training progress every 100 updates. Per-run outputs live under
`runs/RUN_ID/seed_101/`: `validation/check_*.json`, `diagnostics/check_*.json`,
`last.pt`, `best_model_val.pt`, and `training_complete.json`.

Native/context/gate diagnostics use a fixed separate validation batch, restoring
RNG/trimmer state afterward. Checkpoint selection is never based on those samples.

After inspecting a terminal failure, retry its phase with `--resume`:

```bash
# Incomplete GPU probes (not a change to model, environment or resource contract):
bash sbatch/submit_part_ev_screen_sporc.sh --mode benchmark --output "$SCREEN_ROOT" --resume

# Interrupted training; completed tasks are retained, others resume last.pt:
bash sbatch/submit_part_ev_screen_sporc.sh --mode train --output "$SCREEN_ROOT" --reviewed --resume

# Training all completed, but evaluation or reporting was interrupted:
bash sbatch/submit_part_ev_screen_sporc.sh --mode evaluate --output "$SCREEN_ROOT" --resume
```

Do not run all three recovery commands together. Any still-live job from this
campaign blocks a duplicate pipeline. Inspect stale dependent jobs before deciding
whether to cancel those specific IDs. Resume restores optimizer/Lookahead state,
AMP scaler, RNGs, trimmer counters, consumed train/validation cursors and LR
position. Deterministic data replay can be slow; do not confuse it with stalled
training. Validation checkpoints are atomic; interruption during validation
restarts that check from its saved pre-validation state.

Full-validation predictions are committed one ROOT file at a time as FP32 logits,
labels and a hashed file/entry manifest (about 1.7 GB total across seven models).
Interrupted files are recomputed; changed committed files fail closed. Raw data
and another campaign's results are never deleted. Relevant source/environment or
resource changes require a new campaign, not edits to its frozen `source/` tree.

## Results

- `reports/screen_report.md` and `.json`: all seven models, paired mechanism and
  optimizer-recipe contrasts, accuracy/CE/OVO/OVR AUC, class rejection/counts,
  parameter counts, resource and stability measurements.
- `reports/learning_curves.pdf` and `.png`: matched-check accuracy/AUC and accuracy
  contrasts.
- `selection/locked_checkpoints.json`: all seven validation-selected checkpoints,
  locked before common-population evaluation.

Rejection uses 30%/50% signal efficiency for the seven nonleptonic signals, 99%
for Hqql and 99.5% for Tbl. Zero-background cases retain counts and a null rejection
instead of invented finite values. The 5M validation population overlaps checkpoint
validation: it is a common comparison population, **not an independent test**.
No across-seed uncertainty is claimed from these single-seed runs.

## Local verification

```powershell
.venv\Scripts\python.exe -m pytest tests/test_part_ev_screen_model.py tests/test_part_ev_screen_protocol.py tests/test_part_ev_screen_data.py tests/test_part_ev_screen_training.py tests/test_part_ev_screen_reporting.py tests/test_part_ev_screen_launcher.py -q
```

Tests cover exact control initialization and RNG behavior; explicit pair-value
forward/backward references; padding/permutation semantics; schedule boundaries;
real-ROOT finite validation; replay/resume; selection/ties; all-seven evaluation
and reporting; and a mocked Slurm dependency graph. Passing these CPU tests does
not replace the remote GPU benchmark gate.
