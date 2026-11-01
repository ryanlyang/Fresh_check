# One-GH200 timing check for layerwise State-EV

This is a **benchmark-only** Tigris adapter for the queued SPORC experiment:
layerwise attention bias, optimized State-EV in **all eight** particle blocks,
state width 32, seed 101, 2,327,462 parameters. It keeps global batch size 512
and the full Ranger recipe (1,000,000 updates and 50 checkpoint-validation
checks). No full training is queued by this launcher.

The SPORC segmented chain stays untouched and may continue to run. Tigris
uses its own source snapshot, environment fingerprint, logs and disposable
benchmark checkpoint. It never resumes or overwrites a SPORC checkpoint.

## Queue the diagnostic

First commit/push this implementation locally, then run on **Tigris**, not
`sporcsubmit`:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

test -x /home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python &&
bash sbatch/submit_part_ev_state_tigris_benchmark.sh --mode benchmark

squeue --me -o "%.18i %.30j %.2t %.10M %.12l %R"
```

This reuses the existing ARM environment from the earlier Tigris full-scale
ParT jobs. It does not install packages or change either cluster's environment.
If that Python is missing, stop and inspect the available environments; an
x86 SPORC environment cannot be substituted. Use `--python /absolute/path`
only for a compatible existing ARM CUDA environment.

The two submissions are:

| Job | Resources | Walltime | Purpose |
| --- | --- | --- | --- |
| `evgh_prepare` | CPU only, 8 CPUs, 128 GiB host RAM | 2 hours | Verify the reference, dataset metadata and ARM environment; freeze the diagnostic. |
| `evgh_benchmark` | 1 GPU, 8 CPUs, 128 GiB host RAM | 2 hours | After successful preparation, verify GH200/model operation and measure throughput, memory and validation. |

These are maximum allocations, not predictions of actual duration. The GPU
job checks for one visible GH200 with at least 80 GiB device memory. The
diagnostic does not automatically submit production training or cancel jobs.
`--dry-run` prints the proposed request without creating files or jobs.

Default read-only reference:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_layerwise_ddp/layerwise_2g_20261029T162243Z
```

Default new output is `checkpoints/part_ev_state_tigris/gh200_<UTC timestamp>`.
An optional `--output /absolute/new/path` must point to a new directory, separate
from the SPORC campaign. Dataset files are reused in place: no upload, copy,
deletion or new full-dataset rehash. Preparation checks the previously verified
inventory's file sizes/mtimes and pinned Weaver sources. Runtime/library
versions are recorded independently for ARM.

## Read the result

Copy the absolute directory printed after `Diagnostic:` into `EVGH_ROOT`:

```bash
EVGH_ROOT="/absolute/diagnostic/path/printed/by/the/launcher"
bash sbatch/submit_part_ev_state_tigris_benchmark.sh \
  --mode review --output "${EVGH_ROOT}"
```

Review is read-only. Logs are under `${EVGH_ROOT}/logs`; job IDs are in
`submitted_job_ids.tsv`. If submission only partly succeeds, inspect that
ledger before taking further action. A failed benchmark retains its evidence;
do not rerun submission with the same output directory.

The probe uses the existing all-eight model and numerical checks, 20 warm-up
updates, 4,200 measured real-data updates, a full-128-particle memory/stability
check, a checkpoint write, and 128,000 FP32 validation events. Projections use
the slowest measured training window, all 50 validations of 1.28M events each,
checkpoint writes, and cold data loading. Recommended walltime adds 30% margin.
Queue delays, restart replay and sustained shared-filesystem contention are
not covered by that margin as a guarantee.

## After the benchmark

Use the measured full-schedule estimate to choose a fresh Tigris production
run and, if needed, its segmented walltime. **That production launcher is a
separate next step; this change only prepares the diagnostic.** Keep the
SPORC chain queued meanwhile. A benchmark starting is not a reason to cancel
production on SPORC.

One GH200 uses native single-GPU BatchNorm; two A100s use SyncBatchNorm and
distributed data shards. Architecture, global batch, seed and training budget
are matched, but batch order/trimming and platform arithmetic can differ.
This is not a bitwise-equivalent cross-platform restart. If both production
campaigns are later queued, keep outputs independent and choose the result
source by completion/validation checks, not by whichever gives a nicer metric.
