# Faster state-conditioned EV: SPORC tier3

Three new seed-101 runs, with the same shortened Ranger recipe and reused
ParT, ordinary EV, and original state-EV controls. Existing campaigns and
checkpoints are read-only. No jobs are submitted by installing these files.

| Task | Run | State blocks | State width | Parameters |
|---:|---|---|---:|---:|
| 0 | EV_STATE_FAST_RANGER | All eight | 32 | 2,323,710 |
| 1 | EV_STATE_LATE4_RANGER | Blocks 5-8 | 32 | 2,266,366 |
| 2 | EV_STATE_LATE4_R16_RANGER | Blocks 5-8 | 16 | 2,237,694 |

All retain ordinary EV in every block. The full optimized version keeps the
original state-model parameter names, initialization, and mathematical function:

- Process 64 receiver particles per chunk instead of 16.
- Compute pair-feature normalization once per model forward and reuse it across
  blocks, **without detaching gradients**. Clear the cache on success and errors.
- Pass normalized tensors explicitly into checkpointed functions so backward
  recomputation never depends on a cleared mutable cache.

Checkpointing remains enabled. No compiler, precision, optimizer, input, batch,
or learning-rate changes are bundled into this study. Sharing normalization and
larger chunks trade memory for less repeated work; the cluster memory check must
pass. There is no automatic batch reduction or fallback to a different model.

The two late-block variants change the architecture and need new training to
measure accuracy retention. Even same-math optimized training need not follow a
bitwise-identical optimization trajectory. This study measures that too.

## Queue benchmarks first

Commit and push `part_ev_state_speed/`, `scripts/run_part_ev_state_speed.py`,
`sbatch/*part_ev_state_speed.sh`, and `tests/test_part_ev_state_speed_*.py`
locally before pulling on RC. The submitter snapshots committed source and
refuses dirty relevant source. It does not submit from an uncommitted checkout.

Run on **sporcsubmit**, in Bash (not PowerShell):

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

SPEED_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_state_speed/speed_$(date -u +%Y%m%dT%H%M%SZ)"

bash sbatch/submit_part_ev_state_speed.sh \
  --mode prepare --output "${SPEED_ROOT}" \
  --partition tier3 --concurrency 2
```

Save the printed campaign path. After reconnecting, set SPEED_ROOT to that
exact path; do not generate a fresh timestamp for review or training.

This submits CPU prepare -> three GPU benchmarks, **not training**. Prepare and
each benchmark request five hours, below the six-hour pre-production limit.
GPU tasks request one A100, eight CPUs, 64G host RAM, and use the unchanged
`/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python` environment.
There are no package installs, cancellations, or data deletions.

Defaults, overridable at prepare and frozen afterward:

- `--donor`: `/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z`
- `--state-donor`: `/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_message_study/messages_parityfix_20261019T053047Z`
- `--data-root`: `/home/ryreu/atlas/datasets/JetClass`
- Account `reu-aisocial`, partition `tier3`, A100 model and capacity matching the controls.

The original state run must have a valid training-complete record and selected
checkpoint. Its full-validation evaluation may still be pending: that does not
block benchmarking or training, but it must finish before the combined report.
Prepare verifies source/runtime contracts, dataset inventory and staging hashes,
and the original selected state checkpoint. Keep at least 12 GiB free.

`--dry-run` displays requests without files or submissions. Live campaign jobs
block duplicate submissions; changing the main checkout cannot alter a snapshot.

## What the benchmarks establish

1. The inherited strict initialization parity check against ordinary EV, with
   FP32 evaluation/training and FP16 training-mode forwards, fixed warmups,
   both execution orders, restored state/RNG, and unchanged tolerances.
2. Nonzero state-message forward and gradient equivalence against the unoptimized
   implementation, including padding, chunk boundaries, shared-normalization
   gradients and checkpoint recomputation. The late-block variants are compared
   with their own equivalent unoptimized architectures, not claimed identical to
   the original full model.
3. The full optimized model loads the original trained state checkpoint strictly
   and matches its FP32 logits on 64 real events (`rtol=1e-5`, `atol=1e-6`). This
   is a numerical regression check, not an accuracy assessment on 5M events.
4. The existing 4,200-update sustained throughput probe, worst-length memory and
   finite-update checks, AMP skip accounting, validation timing and input-stream
   fingerprints. All allocations must stay below 90% of the matched GPU capacity.
5. Direct FP32 inference timings for ParT, ordinary EV, original state EV, and
   the candidate **on the same GPU and same resident batch**. Batch size 512
   measures throughput; batch size 1 measures latency. Fixed warmups and three
   rotated-order repeats exclude dataset loading, transfers, AUC computation,
   and model creation from the timed region. All use fresh seeded weights for
   speed measurements only; the separate regression check uses trained weights.
   Trimming is disabled for this controlled comparison: all models see the same
   128 particle slots, with their real padding masks. Batch-one timing is for
   that fixed input shape, not a distribution-wide single-jet latency estimate.

No speedup is assumed, and no result is selected according to speed or accuracy.
The benchmark's `*_speed.json` files retain all per-model timings and repeats.
Validation's `inference_seconds` remains a different end-to-end measurement.

After all benchmarks finish:

```bash
bash sbatch/submit_part_ev_state_speed.sh --mode review --output "${SPEED_ROOT}"
```

The summary gives training jets/s, projected training hours, peak memory,
forward throughput, batch-one latency and time ratios against EV/original state.
For time ratios, smaller is faster (0.5 means half the time; 2 means twice).
A failed/incomplete numerical, memory or precision check blocks production.

## Queue the three matched training runs

After reviewing the benchmark summary:

```bash
bash sbatch/submit_part_ev_state_speed.sh \
  --mode train --output "${SPEED_ROOT}" --reviewed --concurrency 2
```

Dependencies are automatic: train[3] -> full-5M-validation evaluate[3] -> report.
Walltimes use the measured projections with 30% headroom; `--train-hours N` can
increase but cannot undercut that recommendation. All three start from scratch,
not from the donor checkpoint: batch 512, 200,000 attempted updates, ten checks
20,000 updates apart, the same full 100M training population, seed 101, unchanged
Ranger/decay/precision. No extra control training or 20M test evaluation occurs.

The final report compares all six models on the same full-5M validation population,
including deltas against ordinary EV and the original state model. Training-time
validation is not mixed with full-validation accuracy. These remain single-seed
screening results; they cannot establish independent test gains or reproducibility.

```bash
cat "${SPEED_ROOT}/submitted_job_ids.tsv"
squeue --me -o "%.18i %.35j %.2t %.10M %.12l %R"
cat "${SPEED_ROOT}/reports/state_speed_report.md"
```

All output stays under SPEED_ROOT. Per-check metrics are in
`runs/RUN_ID/seed_101/validation/check_*.json`. Reports retain all metrics,
provenance and inference timings in `reports/state_speed_report.json`.

## Retry terminal jobs

Keep failed evidence and frozen source. Do not edit old campaign snapshots.
An incomplete benchmark may be resumed after its array is terminal:

```bash
bash sbatch/submit_part_ev_state_speed.sh --mode benchmark --output "${SPEED_ROOT}" --resume
```

Committed passing training probes are reused; an interrupted inference timing
probe can rerun. A committed failed training probe needs inspection and a new
campaign after a fix. There is no automatic tolerance adjustment.

For interrupted production, use `--mode train --reviewed --resume` after all
previous campaign jobs are terminal. Completed trainings are verified/skipped;
incomplete runs restore optimizer, scaler, RNG, trimmer and data replay state.
Use `--mode evaluate --resume` or `--mode report --resume` for downstream retries.
If the original state's full-5M results were pending, rerun report once available.

## Local verification

```powershell
$stateSpeedTests = @((Get-ChildItem tests/test_part_ev_state_speed_*.py).FullName)
& .\.venv\Scripts\python.exe -m pytest @stateSpeedTests -q
```

CPU numerical/gradient, RNG, exact resume, protocol/approval and simulated Slurm
tests do not establish CUDA memory fit or a speedup. Those require tier3 benchmarks.
