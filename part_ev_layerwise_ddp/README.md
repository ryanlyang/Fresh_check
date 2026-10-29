# Full layerwise all-eight State-EV on two A100s

This adds **one fresh seed-101 run**, not another three-model campaign. The model
is the benchmarked layerwise all-eight State-EV: width 32, query chunks 64,
2,327,462 parameters. Existing shared-State-EV, wider-ParT, baseline and ordinary
EV jobs and artifacts are never changed or cancelled.

Training uses two A100s **on the same node**, DDP, SyncBatchNorm, a global batch
of 512 (256 per GPU), and one training data worker per rank. The original full
Ranger recipe is retained: 1,000,000 update attempts, 512M jet exposures from the
official 100M pool, and 50 validation checks after each 20,000 steps. Each check
uses the existing 1.28M-event streaming validation recipe. Selection is by
accuracy, first tie retained. This is not 50 passes over the dataset.

The source of compute/memory evidence is the successful scaling diagnostic:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_gpu_scaling/scaling_20261028T024248Z
```

It measured 1,113 training jets/s and 17.92 GiB allocated peak per GPU. Its
128-hour projection covered training only. This launcher also measures
checkpoint and validation costs and adds a 30% planning margin; queue and
restart replay remain excluded. It never substitutes a shorter step budget to
fit a partition's time limit.

## Queue

Commit/push locally and pull on `sporcsubmit`. Do not upgrade the existing
`part_ev_screen_sporc` environment. First queue the new production-path check:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

bash sbatch/submit_part_ev_layerwise_ddp.sh --mode prepare \
  --scaling /home/ryreu/atlas/Fresh_check/checkpoints/part_ev_gpu_scaling/scaling_20261028T024248Z
```

Save the printed new campaign path. Preparation freezes source and reads the
old evidence; the queued preflight requests **04:30:00**, two A100s, eight CPUs
and 64G host RAM on `tier3`, account `reu-aisocial`. It checks seven resumed
updates across a Lookahead cycle, model/scaler/slow-weight recovery, and FP32
validation on 128k events. It does not repeat the 4,200-step scaling benchmark.
Its weights are disposable and never initialize production.

After the preflight succeeds:

```bash
bash sbatch/submit_part_ev_layerwise_ddp.sh --mode review --output /absolute/new/campaign

bash sbatch/submit_part_ev_layerwise_ddp.sh --mode train \
  --output /absolute/new/campaign --reviewed
```

This queues one two-A100 training job, then one-A100 full-5M FP32 validation,
then a CPU report. Dependencies are `afterok`. No independent 20M test is
queued. Use `--dry-run` to inspect requests without writes or submissions.
Use `--train-hours N` only to increase the measured recommendation, not reduce it.

## Recovery and validation

- `last.pt` is saved atomically every 1,000 steps, before/after validation, and
  on a coordinated stop when possible. It contains model, complete Ranger
  state (including Lookahead slow weights/counter), scaler, step/validation
  counters, and each rank's Python/NumPy/Torch/CUDA RNG and trimmer state.
- Resume reconstructs the same rank-sharded training streams and rank-zero
  validation stream, replays consumed batches, then restores random states.
  Replay can be slow. World size, workers, batch size, GPU model and NCCL
  settings cannot silently change on resume.
- Both ranks participate in small heartbeat collectives during rank-zero-only
  validation and replay. The unwrapped model runs FP32 evaluation; SyncBatchNorm
  does not synchronize batch statistics in evaluation mode.
- Selected candidates are versioned by check, so interruption during validation
  cannot overwrite the previous committed selection. `best_model_val.pt` is
  finalized after all 50 checks. Up to roughly 0.5 GiB of candidate weights may
  be retained; nothing is automatically deleted.
- Slurm's advance TERM warning is forwarded by torchrun to workers. If its
  shutdown grace period prevents a fresh save, the previous atomic checkpoint
  remains valid. Hard failures and timeouts are **not** automatically requeued.
- After confirming all jobs in this new chain are terminal, use the same train
  command with `--resume --reviewed`. It resumes only this campaign and replaces
  its failed downstream submission chain. `--mode evaluate --resume` retries
  only evaluation/report; committed per-file prediction chunks are hash-checked.
  Failed preflight evidence requires a fresh campaign, not edits to its snapshot.

## Output and interpretation

Logs: `logs/evddp_*`; job IDs: `submitted_job_ids.tsv`.
Run artifacts: `runs/EV_STATE_LAYERWISE_RANGER/seed_101/`.
Final reports: `reports/layerwise_ddp_report.json` and `.md`, including all 50
validation checks, final accuracy, macro AUC and QCD rejection/counts.

This is single-seed validation overlapping checkpoint selection, not independent
test evidence. DDP changes event sharding/order and native per-rank trimming;
SyncBatchNorm aggregates batch statistics. The global batch and optimizer recipe
are preserved, but historical one-GPU controls are **not runtime-matched**.
Consequently this report does not silently merge those controls or present their
differences as an isolated architecture-only contrast. This implementation does
not authorize additional control trainings or change their existing jobs.

Local CPU tests exercise distributed update/checkpoint/validation control flow.
Actual A100 checkpoint recovery and validation are admitted only by the new
preflight; performance scaling was measured by the existing diagnostic.
