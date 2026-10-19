# Layerwise attention-bias Ranger screen

Two new seed-101 runs on SPORC `tier3`, compared with the existing read-only
`PART_RANGER` and `EV_RANGER` controls. This is a separate campaign and does not
modify the grouped study, donor screen, environment, or their frozen sources.

| Bias | EV off | EV on |
|---|---|---|
| Shared | Reuse `PART_RANGER` (2,143,486 parameters) | Reuse `EV_RANGER` (2,209,022) |
| Layerwise | New `PART_LAYERWISE_RANGER` (2,147,238) | New `EV_LAYERWISE_RANGER` (2,212,774) |

## Architectural change

Keep the native shared four-feature pair encoder and its 64-channel stem. Give
each of eight particle-attention layers its own final 64-to-8 scalar-bias head
(native convolution, BatchNorm and GELU). Initialize all eight as independent
copies of the original head, including BN buffers. Thus initial outputs match
the shared model; head parameters can subsequently specialize by layer. Only
3,752 parameters are added to each counterpart. Class-attention blocks are unchanged.

EV already has independent value projections in each layer. Those projections,
their initialization, shared input stem and attention-weight/dropout reuse stay
unchanged. This is **layerwise versus shared attention bias**, not shared versus
layerwise EV projections, and does not add TRACK/PT/REGION or grouped attention.

## Unchanged reusable recipe

Follow [the reusable screen protocol](../part_ev_screen/REUSABLE_SCREEN_PROTOCOL.md):
official 100M training population, seed 101, Ranger short schedule, batch 512,
one A100 per task, two loader workers, 200,000 attempted updates, ten checks at
20,000-update intervals, 1.28M validation examples per check, highest-accuracy
checkpoint (first tie). Train from scratch, not from donor checkpoints.
Final evaluation uses the official 5M validation population, **not 20M test**.
FP16-autocast training, FP32 parameters/evaluation, native single-GPU BatchNorm,
disabled TF32, and the inherited loaders, scheduler and optimizer are unchanged.

Donor contracts check source, environment, runtime, data metadata and recipe.
The report verifies completed controls, checkpoint identities and prediction
coverage; it does not substitute missing controls with zeroes or partial results.

## Queue everything in one command

First commit and push these changes locally. On **sporcsubmit**:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only && bash sbatch/submit_part_layerwise_screen_sporc.sh
```

This freezes the committed source into a new directory and submits:

```text
train[0-1%2] -> checkpoint lock -> evaluate[0-1%2] -> combined report
```

There is **no separate prepare/preflight/benchmark job or manual review step**.
Each training task creates/validates shared campaign metadata under a lock,
then performs a small startup check: first real-batch identity, initial FP32
parity against the appropriate shared model, one FP16 training step and an FP32
forward. The probe runs in a **child Python process inside the same training
allocation**, with the same frozen script and interpreter. The child exits before
production starts, discarding its model, optimizer, scaler, loader, lazy imports,
module search-path changes and GPU context. The parent still strictly rechecks
the donor environment, then creates fresh seeded training state (or restores
`last.pt` on resume). No environment check is cached, disabled or relaxed.
The initial AMP step may be skipped while the fresh scaler adjusts; it is
recorded and is not represented as sustained numerical validation.

This direct-submit policy is recorded explicitly. Startup checks are **not** a
sustained throughput/memory/stability benchmark. GPU behavior has not been
validated locally; an in-job startup failure stops that task and blocks its
dependents. This bypass does not weaken other studies' preflight requirements.

Defaults: `reu-aisocial`, `tier3`, one A100/task, eight CPUs, 64G host RAM,
two concurrent tasks, **48-hour training and 4-hour evaluation limits**. Lock
and report request two hours each. These durations are planning allowances,
not measurements for the new models. `--train-hours 72` can request more time.
The two new models require about 3 GiB of free disk headroom for artifacts.
No existing jobs are cancelled, and no datasets or checkpoints are deleted.

Optional read-only preview:

```bash
bash sbatch/submit_part_layerwise_screen_sporc.sh --dry-run
```

Save the printed campaign directory as `LAYER_ROOT`. Read progress without
changing the campaign (also works while all tasks are pending):

```bash
LAYER_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_layerwise_screen/REPLACE_WITH_PRINTED_DIRECTORY"
bash sbatch/submit_part_layerwise_screen_sporc.sh --mode status --output "${LAYER_ROOT}"
squeue --me -o "%.18i %.30j %.2t %.10M %.12l %R"
cat "${LAYER_ROOT}/submitted_job_ids.tsv"
```

Logs are `logs/lw_train_<job>_<task>.out` / `.err`. Array index 0 is ParT
layerwise; index 1 is ParT-EV layerwise. A `--` in progress means no recorded
check, not necessarily a failure. Sources are frozen: later pulls cannot alter
pending tasks. Do not edit a campaign's `source/` directory.

## Results and retries

The combined report is `reports/layerwise_screen_report.md` and `.json`. It
contains accuracy, both macro OVO/OVR AUC definitions, all recorded classwise
rejections, learning histories, parameters and matched contrasts:

- EV gain with shared bias: `EV_RANGER - PART_RANGER`.
- EV gain with layerwise bias: `EV_LAYERWISE_RANGER - PART_LAYERWISE_RANGER`.
- Layerwise gain separately without and with EV.
- Interaction: the layerwise EV gain minus the shared EV gain.

These are single-seed screening-validation comparisons, not across-seed evidence
or independent test results. Repeated validation use is exploratory.

If donor full-validation results are still pending, the report exits incomplete
and records `reports/progress.json`; new training/evaluation are preserved.
After the donor completes, and all prior jobs in this campaign are terminal:

```bash
bash sbatch/submit_part_layerwise_screen_sporc.sh --mode report --output "${LAYER_ROOT}" --resume
```

After a timeout/failure, inspect the logs. To resume training and requeue its
dependent stages once prior campaign jobs are terminal:

```bash
bash sbatch/submit_part_layerwise_screen_sporc.sh --output "${LAYER_ROOT}" --resume
```

Use `--mode evaluate --resume` instead if training and selection are complete.
The launcher refuses live duplicates; it never cancels stale jobs automatically.
Existing valid checkpoints/prediction chunks are reused with identity checks.
Code changes require a new frozen campaign, not patching old snapshots.

### Startup-to-training environment-check failure

Jobs `21921608_0` and `_1` in `layerwise_20261019T083422Z` completed the startup
probe but failed the environment recheck before the production training loop.
The old launcher ran both stages in one process; the inventory matched before
the probe and differed afterward. The old exception did not name the differing
field, so the exact remote package/source discrepancy is not established.

The probe is now process-isolated as described above. `startup.json` records
`environment_changes_after_probe`, and any remaining environment failure lists
the differing fields and expected/actual values. Real environment changes still
block training. The model, optimizer, data, schedule and control contracts are
unchanged. After committing/pushing/pulling this fix, submit a **new campaign**
with the default command; do not `--resume` the old frozen source. Keep its logs.
