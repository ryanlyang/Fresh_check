# Tigris benchmarks for the three State-EV extensions

Independent, **benchmark-only** adapters for the three passed SPORC experiments.
Existing SPORC benchmarks, production queues, environments and data stay unchanged.
Nothing here submits production or cancels a job on either cluster.

| Model key | Experiment | Parameters |
| --- | --- | ---: |
| `bias` | State-conditioned attention bias | 2,329,510 |
| `multiply` | Relation-state multiplication | 2,343,846 |
| `memory` | Evolving pair memory | 2,364,294 |

All retain their original all-eight-block, layerwise State-EV architecture,
standard relations, initialization, Ranger recipe and seed 101. These are **not**
the compact models. The runtime projection covers 1,000,000 updates, 50 validation
checks, and checkpoint writes. Checks are not full dataset epochs.

## What changes on Tigris

- One GH200, global/local batch 512, native BatchNorm, two data-loader workers.
  SPORC uses two A100s, local batch 256 per rank, and SyncBatchNorm. This is the
  same intended global-batch recipe, **not a bitwise-identical training trajectory**:
  data sharding, trimming, numerical execution and BatchNorm implementation differ.
- Use the existing ARM CUDA environment at
  `/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python`.
  No installation or environment update is performed. Weaver model, data and
  optimizer sources must match the SPORC reference. ARM package versions are
  recorded independently and must stay unchanged after submission.
- Fresh benchmark weights only; no SPORC checkpoint transfer or resume.
- FP16 training, FP32 validation/inference, TF32 disabled, unchanged correctness
  tolerances and architecture. No silent batch-size or precision fallback on OOM.

The small launcher/adapter is frozen from committed Git files. Scientific code
is imported from each **authenticated, unchanged SPORC benchmark source snapshot**,
not the evolving live checkout. Pulling later commits therefore does not replace
either the queued SPORC science or this queued Tigris science. Keep those donor
directories and the shared ARM environment intact while jobs are pending/running.

## Jobs and checks

Each invocation creates an independent chain:

`CPU prepare -> afterok: GH200 benchmark -> afterany: CPU summary`

| Phase | Walltime | Resources |
| --- | --- | --- |
| Prepare | 2 hours | CPU only; 8 CPUs, 128 GiB host RAM |
| Benchmark | 5 hours 30 minutes | 1 GH200; 8 CPUs, 128 GiB host RAM |
| Summary | 15 minutes | CPU only; 2 CPUs, 8 GiB host RAM |

Three invocations mean **nine jobs, only three GPU jobs**. There are no
cross-model dependencies or array/concurrency caps. Slurm still controls when
resources become available. The benchmark has an internal five-hour deadline.

Each GPU job checks initialization parity, active forward/gradient correctness,
the model-specific delayed-learning or memory-history requirements, sustained
real-data training (20 warmup + 4,200 measured updates), fixed-128-particle
training at batch 512, a seven-update full-state restart, diagnostics, 128,000
validation events, FP32 inference throughput and batch-one latency. The restart
check includes optimizer, scaler, RNG and sequence-trimmer state.

Require finite/active model state, healthy AMP, and at least 10% **allocated**
device-memory headroom. Reserved memory is also reported. CPU-only parity cannot
admit a GPU benchmark. The full-schedule estimate uses the slowest measured
training window and adds validation/checkpoint costs plus 30%; it excludes queue
time, future filesystem contention and restart replay. The report flags whether
the estimate fits a 20-day training allocation; actual partition limits are
checked again when a future production launcher is added.

## Queue on Tigris

First commit/push these new files locally, then pull them on Tigris. Do not modify
existing frozen campaign directories. Run this setup and read-only plan block in
one Tigris shell; keep it open for the submission block below:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

EVEXT_REPO=/home/ryreu/atlas/Fresh_check
EVEXT_PYTHON=/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python
EVEXT_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

BIAS_BENCH="${EVEXT_REPO}/checkpoints/part_ev_state_bias/state_bias_20261113T063807Z"
MULTIPLY_BENCH="${EVEXT_REPO}/checkpoints/part_ev_state_multiply/state_multiply_20261113T065552Z"
mapfile -t EVEXT_MEMORY_LOGS < <(find \
  "${EVEXT_REPO}/checkpoints/part_ev_pair_memory" \
  -maxdepth 3 -type f -name 'evmem_summary_22089091.out' -print)

BIAS_GH="${EVEXT_REPO}/checkpoints/part_ev_extensions_tigris/bias_${EVEXT_STAMP}"
MULTIPLY_GH="${EVEXT_REPO}/checkpoints/part_ev_extensions_tigris/multiply_${EVEXT_STAMP}"
MEMORY_GH="${EVEXT_REPO}/checkpoints/part_ev_extensions_tigris/memory_${EVEXT_STAMP}"

if [[ "${#EVEXT_MEMORY_LOGS[@]}" -eq 1 ]]; then
  MEMORY_BENCH="$(dirname -- "$(dirname -- "${EVEXT_MEMORY_LOGS[0]}")")"
else
  unset MEMORY_BENCH
  echo "Expected exactly one pair-memory summary log; locate its campaign before submitting."
fi

(
  set -euo pipefail
  test -x "${EVEXT_PYTHON}"
  : "${MEMORY_BENCH:?Pair-memory benchmark directory not identified}"

  bash sbatch/submit_part_ev_extensions_tigris.sh \
    --mode plan --model bias --benchmark "${BIAS_BENCH}" \
    --output "${BIAS_GH}" --python "${EVEXT_PYTHON}"
  bash sbatch/submit_part_ev_extensions_tigris.sh \
    --mode plan --model multiply --benchmark "${MULTIPLY_BENCH}" \
    --output "${MULTIPLY_GH}" --python "${EVEXT_PYTHON}"
  bash sbatch/submit_part_ev_extensions_tigris.sh \
    --mode plan --model memory --benchmark "${MEMORY_BENCH}" \
    --output "${MEMORY_GH}" --python "${EVEXT_PYTHON}"
)
```

If all three plans pass, submit from the same shell:

```bash
(
  set -euo pipefail
  cd "${EVEXT_REPO}"
  bash sbatch/submit_part_ev_extensions_tigris.sh \
    --mode benchmark --model bias --benchmark "${BIAS_BENCH}" \
    --output "${BIAS_GH}" --python "${EVEXT_PYTHON}"
  bash sbatch/submit_part_ev_extensions_tigris.sh \
    --mode benchmark --model multiply --benchmark "${MULTIPLY_BENCH}" \
    --output "${MULTIPLY_GH}" --python "${EVEXT_PYTHON}"
  bash sbatch/submit_part_ev_extensions_tigris.sh \
    --mode benchmark --model memory --benchmark "${MEMORY_BENCH}" \
    --output "${MEMORY_GH}" --python "${EVEXT_PYTHON}"
  squeue --me -o "%.18i %.40j %.12T %.12l %R"
)
```

Submission validates the live partition, uses `sbatch --test-only` for all three
resource requests, then records job IDs and submission intent. It requires clean,
committed adapter files. `--mode plan` is read-only and does not guarantee that
later scheduler/storage checks will pass.

## Review and next step

```bash
cd /home/ryreu/atlas/Fresh_check
bash sbatch/submit_part_ev_extensions_tigris.sh --mode review --output "${BIAS_GH}"
bash sbatch/submit_part_ev_extensions_tigris.sh --mode review --output "${MULTIPLY_GH}"
bash sbatch/submit_part_ev_extensions_tigris.sh --mode review --output "${MEMORY_GH}"
```

If the shell variables are gone, use the exact printed benchmark directories.
Each contains `submitted_job_ids.tsv`, `logs/`, `campaign.json`, `benchmark.json`
and `summary.json`. Failed phases may leave a `*_error.json`. A summary after a
failed preparation reports missing evidence and exits nonzero; Slurm completion
alone never substitutes for a passed report.

After reviewing all passed reports, choose Tigris production walltimes and add
production submission separately. **`authorizes_training` stays false here.**
Do not cancel an already-running SPORC production job just because its Tigris
benchmark starts first. If production is later queued on both clusters, select
the first healthy production start per model and explicitly cancel its duplicate;
do not choose a replica based on which validation accuracy looks better.

## Storage and recovery

No dataset copies, persistent pair-feature cache, full-validation prediction
files, or test-data reads. Only the real-data benchmark opens train/validation
ROOT files; preparation checks their recorded metadata, not their full payloads.
The new namespace budgets 1 GiB per diagnostic plus one shared 4 GiB reserve. It
also counts outstanding registered SPORC State-EV production output budgets and
other pending Tigris-extension diagnostics before submission and execution.
These checks are **not a filesystem quota reservation** against unrelated jobs.

All outputs stay in `checkpoints/part_ev_extensions_tigris/<new-name>`.
Failed diagnostics retain their budget. Attempts cannot overwrite/reuse an output
directory or an already-started preparation/benchmark. For an intentional retry,
inspect the logs first and use a fresh name. If submission is interrupted, inspect
`submitted_job_ids.tsv`, `submission_intent.json` and Slurm before retrying:
some jobs may already exist. No automatic retries or cancellations are performed.

Local verification (CPU; actual GH200 checks run only on Tigris):

```powershell
.venv\Scripts\python.exe -B -m pytest -q tests/test_part_ev_extensions_tigris_protocol.py tests/test_part_ev_extensions_tigris_launcher.py tests/test_part_ev_extensions_tigris_runtime.py
```
