# RPT SPORC inference-only campaign

This separate campaign runs the completed offline factorial models on SPORC
x86/A100 under the administrator-approved `debug` allocation. It does not train,
change model selection, overwrite the original campaign, or cancel the queued
Tigris inference job `207556` (or any other job). No production jobs were submitted
while preparing these files. Actual remote compilation, GPU parity, memory, and
timing validation have not yet been run.

## Launch

Use the existing x86 environment `/home/ryreu/miniconda3/envs/atlas_kd_sporc`.
Ninja has already been installed there. Do not reuse the ARM Miniforge prefix or
`atlas_kd_tigris`. Run from the current checkout on an x86 SPORC submit host:

```bash
bash sbatch/submit_relational_part_sporc_inference.sh \
  --source-campaign /absolute/path/to/completed/rpt_offline_factorial_CAMPAIGN \
  --dry-run

# Review the source and dependency graph, then deliberately submit:
bash sbatch/submit_relational_part_sporc_inference.sh \
  --source-campaign /absolute/path/to/completed/rpt_offline_factorial_CAMPAIGN
```

Alternatively, identify the exact campaign associated with the original job:

```bash
bash sbatch/submit_relational_part_sporc_inference.sh \
  --source-job 207556 \
  --search-root /home/ryreu/atlas/Fresh_check/checkpoints/relational_particle_transformer \
  --dry-run
```

Discovery requires exactly one ledger containing the two-field record
`infer 207556`. Ambiguous or absent matches fail; there is no latest-directory
fallback. The source must contain `selection/locked_models.json` and
`test/evaluation_plan.json`. The Python bootstrap performs the deeper source,
artifact, checkpoint, and test-plan checks.

The default output is a new sibling `rpt_offline_sporc_UTC_TIMESTAMP`; use
`--output /absolute/new/path` to name it. An existing output is rejected for a
fresh launch. Bootstrap freezes the driver, worker, and source into its own
`source/`, authenticates campaign inputs, and fixes file-to-task ownership before
submitting. All SPORC predictions and reports live in this new output, never in
the original Tigris prediction directories. Worker paths are absolute and safe
when Slurm executes the copied script from its spool directory.

## Frozen execution graph

Every edge is `afterok`; dependent jobs are killed by Slurm on an invalid
dependency. Default account is `reu-aisocial`, partition `debug`, GPU request
`gpu:a100:1`:

| Phase | Resources | Wall time | Array |
| --- | --- | --- | --- |
| build | 4 CPU, 16 GB | 1 hour | one CPU task |
| validate | 4 CPU, 64 GB, one A100 | 2 hours | one GPU task |
| infer | 8 CPU, 64 GB, one A100 | 23 hours 30 minutes | `0-39%2` |
| aggregate | 4 CPU, 64 GB | 12 hours | `0-11%2` |
| report | 2 CPU, 16 GB | 1 hour | one CPU task |

Inference owns the canonical 200 ROOT files across 40 tasks and retains 10,000-row
shards. The inference limit stays below the approved 24-hour debug maximum.
`--jobs` / `RPT_SPORC_JOBS` allows 1–200 tasks; `--concurrency` /
`RPT_SPORC_CONCURRENCY` allows 1–task-count concurrent GPU tasks.
`RPT_SPORC_AGGREGATE_CONCURRENCY` defaults to 2 (maximum 12).
`SBATCH_ACCOUNT`, `SBATCH_PARTITION`, `GPU_GRES`, `CONDA_BASE`, and `CONDA_ENV`
are explicit overrides recorded in `submission_config.json`; obtain any needed
allocation approval before changing scheduler settings. These settings and the
task count cannot change when resuming a campaign. If initially using a custom
Conda prefix/environment, set the same `CONDA_BASE` / `CONDA_ENV` on resume.

The build phase compiles the campaign-local x86 tree backend; it does not train
models. Validation must pass before inference: strict loading for all 12 new
factorial checkpoints, golden parity against saved Tigris 20M-test chunks from
all 12 old reference models, plus GPU memory and timing measurements. The timing
benchmark is diagnostic, not a scientific-performance or speed threshold gate.
Missing references or failed parity are blockers, not permission to skip checks.

The frozen parity policy compares the first 640 events from one original test
file per class (6,400 events total) with the saved Tigris logits for all twelve
historical checkpoints. These prefixes preserve the original batch size and
trimmer warm-up sequence. Each new checkpoint also receives CPU/GPU comparisons
on eight events with both fresh and warmed trimmer counters. Logits use FP32
with mixed precision disabled, absolute and relative tolerances of `5e-5`;
real-event tree topology/categories are exact and continuous tree values use
absolute tolerance `2e-6`. The policy is global, not adjusted by model or score.

`portability_validation.json` records the comparisons, GPU identity, peak
memory, and timing estimates. Inference requires the same GPU model/capacity as
the probe. The final report embeds the authenticated validation receipt and
summarizes all eight configurations, including accuracy, macro AUROC, rejection,
and matched architecture contrasts. Historical Tigris and new SPORC predictions
retain separate provenance. No accuracy or background-rejection threshold gates
are added.

Storage preflight budgets remaining predictions for **both** this campaign and
the still-queued Tigris campaign, plus 6 GiB headroom. Nothing is deleted
automatically. Keep the original checkpoints, normalization artifacts and
historical prediction files until aggregation and reporting have finished.

Both launcher and workers isolate Python user packages, disable bytecode writes,
use unbuffered output, prepend the active Conda `lib` directory, cap compilation
at `MAX_JOBS=4`, and set `OMP_NUM_THREADS=4`. Workers import only their frozen
source tree, never the inherited checkout.

## Inspect and resume

`logs/` contains phase/array stdout and stderr. `submitted_job_ids.txt` is appended
immediately after each successful submission, including on a partially submitted
graph. A failed `sbatch` stops further submissions; nothing is automatically
cancelled. Inspect those exact new-campaign job IDs and their Slurm states.

```bash
bash sbatch/submit_relational_part_sporc_inference.sh \
  --resume /absolute/path/to/rpt_offline_sporc_UTC_TIMESTAMP --dry-run
bash sbatch/submit_relational_part_sporc_inference.sh \
  --resume /absolute/path/to/rpt_offline_sporc_UTC_TIMESTAMP
```

Resume authenticates the frozen manifest/source and fixed configuration before
submission. It refuses any queued/running job belonging to this output, and
`flock` prevents concurrent submitters. It resubmits the dependency graph using
the same frozen driver and canonical task ownership; the driver checks completed
artifacts before reusing them. If cleanup is needed, handle only the new SPORC
campaign's IDs after inspection. Never cancel `207556` or other original Tigris
jobs as part of this workflow.

A bootstrap failure before `submission_config.json` is written has not queued
jobs. Preserve its directory for diagnosis and choose a new `--output` after
fixing the reported cause; do not treat a partially frozen bootstrap as a
resumable submitted campaign.
