# Full-schedule ParT versus last-four State-EV (SPORC tier3)

Seven **fresh** trainings, no reused screening checkpoints or Tigris results:

| Array index | Model | Seed | Parameters |
|---:|---|---:|---:|
| 0 | Baseline ParT, standard shared pairwise bias | 101 | 2,143,486 |
| 1 | Baseline ParT | 202 | 2,143,486 |
| 2 | Baseline ParT | 303 | 2,143,486 |
| 3 | Last-four State-EV, width 32 | 101 | 2,266,366 |
| 4 | Last-four State-EV, width 32 | 202 | 2,266,366 |
| 5 | Last-four State-EV, width 32 | 303 | 2,266,366 |
| 6 | Ordinary ParT-EV | 101 | 2,209,022 |

State-EV retains ordinary EV in all eight particle-attention blocks and adds
state-conditioned messages only in blocks 5–8, with state width 32, query chunks
of 64, checkpointed gradients, and shared differentiable pair normalization.
It reuses the tested `part_ev_state_speed` implementation without modifying it.

## Frozen full recipe

- Official 100M training population and 5M validation population; unchanged full
  particle features, 128-particle maximum and standard pairwise features.
- Batch 512, **1,000,000 attempted updates**, 50 validation checks every 20,000
  updates. This is 512M jet exposures (~5.12 dataset-pass equivalents), not 50
  passes over all 100M jets. AMP-skipped attempts consume budget as in the controls.
- Each check consumes 1.28M validation events from the continuous validation
  stream. Both streams remain alive between checks; matched model seeds use
  matched independently seeded input streams.
- Original Ranger recipe: Weaver RAdam (betas 0.95/0.999, epsilon 1e-5), Lookahead
  k=6 / alpha=0.5, no weight decay, no gradient clipping.
- LR 0.001 through zero-indexed step 699,999; 15 exponential drops beginning at
  step 700,000, every 20,000 steps, reaching 0.00001 for steps 980,000–999,999.
  This calls the existing full-scale LR implementation, not the compressed screen LR.
- FP16 AMP training with FP32 parameters/scaler, explicitly audited native Weaver
  autocast; FP32 validation. TF32 disabled. Single GPU, native BatchNorm, two data
  workers. Identical runtime contract for all seven runs.
- Select the highest validation-accuracy checkpoint; keep the first exact tie.
  AUC is recorded but does not select checkpoints. No early stopping.

References: [ParT paper, section 5.1](https://arxiv.org/html/2202.03772v3#S5.SS1),
[official launcher](https://github.com/jet-universe/particle_transformer/blob/main/train_JetClass.sh),
[local full-scale recipe](../part_ev_fullscale/protocol.py).

The three-seed main contrast is State-EV minus baseline ParT, paired by seed.
Ordinary EV is a **seed-101-only** comparison, not a three-seed estimate.

## Queue prepare and GPU preflights

First commit/push the new package, driver, launchers, and tests from your local
checkout. The launcher only snapshots committed source and refuses dirty relevant
files. It does not install packages or modify existing environments/campaigns.

On **sporcsubmit**, in Bash:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

FULL_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_state_fullscale/state_full_$(date -u +%Y%m%dT%H%M%SZ)"

bash sbatch/submit_part_ev_state_fullscale.sh \
  --mode prepare --output "${FULL_ROOT}" --partition tier3
```

Save that exact output directory. After reconnecting, set `FULL_ROOT` to the saved
path rather than generating another timestamp.

This queues CPU prepare -> **three representative GPU preflights** (indices
0,3,6, all seed 101), **not training**. Each requests **5 hours**, below six hours.
Prepare verifies all 1,050 ROOT headers, staging receipts and actual ROOT hashes.
Keep at least 16 GiB free beyond the staged data and other active campaigns.

Defaults: account `reu-aisocial`, partition `tier3`, one `gpu:a100:1`, 8 CPUs,
64G host memory, Python
`/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python`.
`--python`, `--data-root`, `--account`, `--partition`, `--gpu`, `--cpus` and
`--mem` are frozen at prepare. No silent rerouting to debug or smaller batch.

Preflights check strict initialization parity, nonzero State-EV gradient equivalence,
real-data sustained training, full-128-particle AMP recovery and memory headroom,
FP32 validation, matching seed-101 input fingerprints and timing projections.
All three must pass on the same GPU/runtime contract. Probes never initialize
production weights, optimizer or scaler. Model initialization for all seven tasks
is checked at prepare; local tests cover the three matched seeds.

```bash
bash sbatch/submit_part_ev_state_fullscale.sh \
  --mode review --output "${FULL_ROOT}"
```

Review is read-only, including while campaign jobs are active. It reports measured
full-schedule estimates per architecture (including all 50 validations), a 30%
training margin, and an evaluation recommendation of at least 5 hours. Runtime
projections are not guarantees and exclude queue time and restart data replay.

## Queue production after passing review

```bash
bash sbatch/submit_part_ev_state_fullscale.sh \
  --mode train --output "${FULL_ROOT}" --reviewed --concurrency 3

squeue --me -o "%.18i %.30j %.2t %.10M %.12l %R"
```

This submits **training[7] -> full-5M-validation[7] -> report** with `afterok`
dependencies. Both arrays default to at most three concurrent tasks; use
`--concurrency 7` if you want all seven eligible simultaneously. They each request
one A100. The training array uses the largest measured recommendation for its
walltime, not the 60-hour short-run limit. The earlier screen suggested ~162h for
State-EV, so ~216h (nine days) is a planning starting point, **not a hardcoded cap**.

Optional `--train-hours 216` and `--evaluate-hours 8` can increase allocations;
neither can undercut the measured recommendation. The launcher checks partition
MaxTime and refuses an over-limit request; it never silently shortens training.
Account/QOS restrictions can still cause Slurm to reject a request. If the site
limit cannot accommodate the measured time, stop and arrange a checkpointed
segmented plan rather than lowering the training budget. No automatic segmented
resubmission is implemented here.

After completion, results are under:

```text
reports/state_fullscale_report.md
reports/state_fullscale_report.json
runs/<RUN_ID>/seed_<SEED>/validation/check_01.json ... check_50.json
```

These are **validation**, not independent test results. The full 5M evaluation
overlaps checkpoint selection. All seven finalists are locked before evaluation.
The 20M test is deliberately not required at prepare or automatically queued;
keep the selected checkpoints for a later, separately frozen test campaign.

## Resume and safety

- Checkpoints contain model, complete Ranger/Lookahead optimizer state, AMP
  scaler, RNG, trimmer state and consumed train/validation batch cursors.
- A Slurm warning 180 seconds before time limit requests a checkpointed exit 75.
  Restart replays consumed input batches without optimizer steps and restores RNG.
  Replay near the end of 1M steps can be expensive; allow extra walltime if needed.
- Fifty checks and first-tie selection are validated before reporting. Missing or
  mismatched artifacts fail closed. Original Tigris/screen controls are not donors.
- The immutable git snapshot is used by pending and running workers; later pulls
  in the main checkout cannot change their source. Do not edit a campaign snapshot,
  frozen launch files, model/optimizer recipe or the active Python environment.
- A ledger and submission lock prevent duplicate pipelines. No jobs are cancelled
  automatically by this launcher. `--dry-run` creates no files or jobs.
- Only after all jobs in that campaign are terminal, use `--mode train --resume
  --reviewed` to resume incomplete trainings. Completed trainings are verified and
  skipped. This also queues a new dependent evaluation/report chain.
- `--mode evaluate --resume --evaluate-hours 8` retries full validation without
  retraining. Completed per-file logits are hash-verified and reused. A failed
  original dependency is not repaired by a new job; the new submission supplies
  new dependencies. Resolve any stale pending jobs explicitly before retrying.
- `--mode report --resume` retries only reporting. Preserve checkpoint/data files
  until all requested training and validation phases finish.
