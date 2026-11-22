# Triangle restart diagnostic on SPORC debug

This is a separate **diagnostic**, not a benchmark retry or a production launcher.
It investigates the small loss differences in benchmark `22138556` using that
benchmark's existing `probes/restart.pt` and its unchanged frozen scientific code.
No tolerance, model, training recipe, environment, existing checkpoint, benchmark
artifact, or queued job is changed. The live checkout is not used for the model.

## Resources and scope

- Partition `debug`, two matching same-node A100s, eight CPUs, 128 GiB host RAM.
- One-hour walltime; cooperative stop five minutes before the limit. No requeue.
- The launcher checks the live debug partition limit and runs `sbatch --test-only`.
  Debug access and scheduling availability are not guaranteed; no silent tier3 fallback.
- 42 synthetic updates per GPU (six sequences of seven), global batch 512,
  128 particles before native trimming, unchanged FP16 training and FP32 validation.
- No ROOT data loading, timing rerun, dataset copying, training completion marker,
  or production submission. It needs the saved probe; it will not silently start fresh.
- Output budget 1 GiB; submission requires 2 GiB free. Two rank-local diagnostic
  roundtrip checkpoints and bounded JSON evidence remain in the new output directory.

## What it tests

First, two low-instrumentation sequences run from the same checkpoint state,
restored from memory and disk. Their losses are compared with the **unchanged**
original loss tolerance. This result is recorded, not used to stop collection.

Then instrumented sequences compare an uninterrupted seven-update reference,
its repeat, a disk-start repeat, and a sequence saved/restored after update three.
Each sequence crosses the original step-20,000 validation boundary and a Lookahead
boundary. All serialized model weights/buffers, optimizer moments/slow weights/
counter, AMP scaler, rank-local RNG/trimmer state, and fixture progress are compared
exactly immediately after restoration. Every post-update state is inspected;
bounded reports include first differing tensor paths and largest absolute errors.
Losses, scales, skipped updates, learning rates, and Lookahead counters are recorded.
Rank agreement on update schedules is checked separately from local repeatability.

The synthetic inputs and rolling fixture cursor reproduce the failed probe, **not**
the production ROOT data stream. Model construction creates a new DDP reducer;
pre-checkpoint kernel/optimizer cache history from the failed process is unavailable.
Instrumented CPU snapshots add synchronization. Therefore failure to reproduce the
small loss mismatch does not prove the old benchmark was sound. No default GPU
determinism setting is changed. Results require review before any fix or benchmark retry.

`complete: true` means evidence was collected—even if exact comparisons differ.
`authorizes_training` and `authorizes_benchmark_retry` are always false.
Execution/I/O/deadline errors exit nonzero and retain partial rank event logs.

## Queue

Commit and push the new diagnostic files, then run this on **sporcsubmit**:

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only

  TRI_RESTART_CAMPAIGN="${PWD}/checkpoints/part_ev_triangles/triangles_profilefix_20261118T211519Z"
  TRI_RESTART_ROOT="${PWD}/checkpoints/part_ev_triangle_restart/restart_$(date -u +%Y%m%dT%H%M%SZ)"
  printf 'Diagnostic output: %s\n' "${TRI_RESTART_ROOT}"

  bash sbatch/submit_part_ev_triangles_restart.sh \
    --mode plan --campaign "${TRI_RESTART_CAMPAIGN}" --output "${TRI_RESTART_ROOT}"

  bash sbatch/submit_part_ev_triangles_restart.sh \
    --mode submit --campaign "${TRI_RESTART_CAMPAIGN}" --output "${TRI_RESTART_ROOT}" \
    --reviewed

  squeue --me -o '%.18i %.40j %.12T %.12l %R'
)
```

Only one debug diagnostic is submitted. Readout/support and Tigris jobs stay intact.
Submission freezes just the diagnostic source, binds donor/source/checkpoint hashes,
and records an intent before `sbatch`. If submission errors after creating an output,
inspect that output's intent, ledger, and Slurm before trying a new directory; an
ambiguous submission can already have created a job. Existing outputs are never reused.

## Review

Use the printed diagnostic output path (not the donor campaign path):

```bash
bash sbatch/submit_part_ev_triangles_restart.sh \
  --mode review --output "$TRI_RESTART_ROOT"
```

The example queue block uses a subshell; its variables do not persist afterward.
Set `TRI_RESTART_ROOT` to the printed path before reviewing in another shell.
Read `report.json`, `rank_0.json`, and `rank_1.json`; `events_rank_*.jsonl` preserves
partial progress. A restoration mismatch is not ordinary repeat noise. Exact
restoration with subsequent small variation needs comparison against the repeated
uninterrupted reference before choosing a precision-specific tolerance. Do not bypass
the original full benchmark or automatic-production admission gates.
