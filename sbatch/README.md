# Fresh Same-HLT SLURM Runners

These scripts are for the research-compute Linux copy of this fresh project, for example:

```bash
export PROJECT_DIR=/home/ryreu/atlas/Fresh_check
export DATA_DIR=/home/ryreu/atlas/PracticeTagging/data/jetclass_part0
export CONDA_ENV=weaver
cd "$PROJECT_DIR"
```

Do not point `PROJECT_DIR` at the old project code. The only shared old path should be the JetClass data directory.

Before submitting real jobs, print the graph:

```bash
DRY_RUN=1 sbatch/submit_fresh_full_samehlt_reco7_vs_hlt5.sh
```

Full fresh run:

```bash
sbatch/submit_fresh_full_samehlt_reco7_vs_hlt5.sh
```

Smoke test only:

```bash
DRY_RUN=1 sbatch/submit_fresh_smoke_test.sh
sbatch/submit_fresh_smoke_test.sh
```

Useful overrides:

```bash
export OVERWRITE=1
export DEVICE=cuda
export BATCH_SIZE=128
export NUM_WORKERS=8
export CONDA_ENV=my_env_name
export PROJECT_DIR=/home/ryreu/atlas/Fresh_check
export OUTPUT_ROOT="$PROJECT_DIR/checkpoints"
```

The full submitter queues:

1. split manifest
2. fixed HLT cache
3. offline teacher
4. single HLT baseline
5. five HLT seed models
6. seven reco7 variants
7. HLT5 fusion
8. reco7+HLT fusion
9. leakage audits for reco7 and HLT5
10. final Markdown/JSON report

All downstream jobs use `afterok` dependencies.

## Full JetClass: original shared-bias ParT versus shared-bias ParT-EV

This is a separate experiment from the HLT workflows above. The entry point is
`submit_part_fullscale.sh`; it never uses an old campaign or a reconstructed
particle cache. The two architectures use the official 17-feature JetClass YAML
and raw particle four-vectors. Only ParT-EV adds a value-message projection to
each particle-attention block. Both retain the original shared pairwise bias.

The reference is the vendored `particle_transformer/train_JetClass.sh` and
`networks/example_ParticleTransformer.py`, with the older native-attention
implementation pinned to `weaver-core==0.4.17`. Its source checksum is verified.
This is a pinned implementation of the reference recipe, not a claim to reproduce
the original authors' exact software environment or published numerical results.
See the [ParT paper](https://arxiv.org/abs/2202.03772) and
[reference launcher](https://github.com/jet-universe/particle_transformer/blob/main/train_JetClass.sh).

The frozen recipe uses 100M training jets and the official 5M validation population;
50 reporting epochs of 20,000 minibatches; global batch 512; Ranger with the
reference defaults and zero weight decay; FP16 AMP; and the reference flat-then-
exponential learning rate. This is 1M minibatches / 512M event presentations
(approximately 5.12 dataset passes), **not five launcher epochs**. GradScaler
overflow skips are recorded separately. Every epoch evaluates 1.28M events from
a continuing validation stream, as specified by the reference launcher's budget.
The best validation accuracy selects the checkpoint, without early stopping.

Three matched random seeds give six training tasks. A dedicated environment is
cloned from the existing Tigris CUDA environment; its Weaver/data dependencies
are installed without modifying `atlas_kd_tigris`. Do not reuse the SPORC x86
environment. Commit and push the new setup files before running the commands
below; preparation freezes committed source, unaffected by subsequent work in
the main checkout.

On **Tigris**, after updating the repository:

```bash
cd /home/ryreu/atlas/Fresh_check
bash scripts/setup_part_fullscale_env.sh
```

After the uploader finishes all ten training archives and validation:

```bash
CAMPAIGN="/home/ryreu/atlas/Fresh_check/checkpoints/part_fullscale/part_ev_100m_v1"
bash sbatch/submit_part_fullscale.sh --mode prepare --output "${CAMPAIGN}"
```

Preparation checks all 1,050 filenames and ROOT header event counts. It does not
require test data. Once it completes, benchmark both architectures at each GPU
count before choosing production resources:

```bash
bash sbatch/submit_part_fullscale.sh --mode benchmark --output "${CAMPAIGN}" --gpus 1
bash sbatch/submit_part_fullscale.sh --mode benchmark --output "${CAMPAIGN}" --gpus 2
bash sbatch/submit_part_fullscale.sh --mode benchmark --output "${CAMPAIGN}" --gpus 4
```

Read `benchmarks/gpus_1.json`, `gpus_2.json`, and `gpus_4.json`: both model timings,
peak allocated GPU memory, AMP skips, and training-only duration estimates.
Estimates exclude checkpoint validation, restart I/O, and scheduler delays.
These jobs exercise real data loading, forward/backward, Ranger and (where
applicable) multi-node DDP. CPU unit tests do not replace this GPU check.

### Optional sustained walltime check (existing campaign stays unchanged)

The short 200-step benchmark can fit inside the first prefetched data slice.
To check whether the three-day training allocation has enough headroom without
changing an already queued training array, submit an independent diagnostic:

```bash
bash sbatch/submit_part_fullscale_timing.sh --campaign "${CAMPAIGN}"
```

This requests one GH200, 8 CPUs, 128G, and **two hours**, with a 100-minute soft
execution budget. It measures both architectures sequentially using the existing
campaign's frozen Python, model, optimizer, preprocessing, and loader. Targets
are 10,000 training minibatches and one full 1.28M-event validation pass per
architecture. Both data loaders stay alive together, as in production. Actual
full-state and best-model checkpoint writes are timed on the campaign filesystem;
the diagnostic's temporary checkpoint probes are then removed. There are no
model updates or writes to production checkpoints.

The submitter prints a unique `diagnostics/timing.*/report.md` and `report.json`
path, plus job/log paths. It freezes its own diagnostic scripts independently of
the campaign source. It never changes `submitted_job_ids.tsv`, `training_runtime.json`,
the original GPU benchmark JSONs, production walltimes, dependencies, or jobs.
Updating the main checkout to obtain this launcher does not alter queued jobs'
frozen source. Maintenance may still prevent the diagnostic from starting.

Projections include repeated ROOT fetches, validation, 1,100 full checkpoint
writes, and up to 50 best-checkpoint writes over the full run. The report shows
both whole-window and first-window-excluded projections and uses the larger for
planning. A 25% slowdown scenario illustrates headroom; it is **not** a confidence
interval or a bound on six concurrent jobs' filesystem contention. Queue waiting,
maintenance, and resume data replay are excluded. Too little training data, an
incomplete validation pass, or any probe error makes the estimate inconclusive.
`complete` records whether all requested targets finished; `usable` can be true
after a budget-limited run only if sufficient repeated-fetch timing and the full
validation pass were collected. Progress is saved during execution. A hard Slurm
kill may leave an unfinished report (and diagnostic-only temporary checkpoints);
never interpret that as a successful duration check. No automatic resource
changes are made from the results.

### Production training and testing

Multi-GPU execution uses one GH200/rank per node, with local batch 256 on two
GPUs or 128 on four. Unlike upstream's local-batch scaling, our adapter preserves
the single-GPU global batch and 1M-step budget. It retains the upstream DDP
SyncBatchNorm and native per-rank trimming policy; this is not promised to be
bitwise equivalent to a single-GPU run. All six runs must use the same GPU count
and recorded environment. The first training job freezes that choice.
Production refuses to start without a successful benchmark for that exact
campaign and GPU count. With four ranks, the native grouped-file worker
sharding divides 25 files/class/rank between two workers as 13 and 12;
therefore the event-presentation count is an approximate pass count, not a
claim of exactly uniform per-file sampling.

Example after reviewing the benchmarks (replace `2` if another count is chosen):

```bash
bash sbatch/submit_part_fullscale.sh --mode train --output "${CAMPAIGN}" --gpus 2
```

Training streams from ROOT files, with no persistent input or pair-feature
cache. Each seed keeps `last.pt` (complete optimizer, Lookahead, scaler, per-rank
RNG/trimmer state and consumed-batch cursors), `best_model_val.pt`, and small
validation logs. Atomic checkpoints are saved every 1,000 minibatches, around
validation, and on a handled walltime warning. Do not change GPU/worker counts
on resume. To recover after a terminal job, inspect logs and use the same command
with `--resume`; completed seeds are skipped.

The exact data-stream restart currently replays previously consumed batches
**without model or optimizer work**. That can still cost substantial ROOT I/O
late in a run. No completed training minibatches are deliberately re-optimized,
and the iterator is never reset at reporting-epoch boundaries (which would
otherwise repeatedly train on the beginning of the files). Hard termination
can lose work since the latest atomic checkpoint. No automatic retry loop is
submitted, and a failed training array does not release an `afterok` test job.

Testing remains a separate opt-in action, after the 200 official test ROOT files
are staged. It locks all six validation-selected checkpoints, then runs one GPU
per model and a dependent report. It can be submitted while training is still
running by supplying its array ID:

```bash
bash sbatch/submit_part_fullscale.sh --mode test --output "${CAMPAIGN}" \
  --test-data /home/ryreu/atlas/datasets/JetClass/Pythia/test_20M \
  --dependency TRAIN_ARRAY_ID
```

Replace `TRAIN_ARRAY_ID` with the actual numeric ID; omit the dependency when
all six trainings have already completed. Test jobs resume at ROOT-file
boundaries and preserve logits (approximately 4.8 GB total for six models) for
later diagnostics. Reports include accuracy, explicitly labeled macro OVO AUC
(the reference metric) and OVR AUC, and per-class QCD rejection at 30%/50%
signal efficiency, with 99% for Hqql and 99.5% for Tbl. Zero accepted QCD events
are represented by counts and a null finite rejection, never silently averaged
away. The report requires all six complete, same-population results.

Use `--help` for resource overrides and `--dry-run` to inspect requests without
creating files or submitting anything. These launchers do not delete datasets,
cancel other campaigns, or submit testing implicitly during training.

### Additional comparison: ParT (plain) versus ParT (plain) + EdgeValue

Use a **new campaign** with `--comparison plain` during preparation. The original
shared-bias campaign and its six running jobs are left unchanged. This comparison
uses `part_plain` and `part_plain_ev`, each with the same three seeds, train/validation
populations, optimizer, batch size, training budget, and checkpoint-selection rule
described above. Neither model adds a physics-derived pairwise bias to its attention
logits. The EdgeValue model still uses the standard pair features to construct its
value messages; "plain + EV" does not mean removing pair information from EV.

The comparison is frozen in `launch.env` and `campaign.json`. Only `prepare` accepts
`--comparison`; all subsequent phases infer it from that campaign. Do not point this
new comparison at the existing `part_ev_100m_20261013T044756Z` directory. Existing
campaigns without a comparison field retain their original standard/shared-bias
interpretation. Plain jobs have names beginning `parte_plain_`.

After committing/pushing the updated setup and pulling it on **Tigris**, this queues
the new preflight, a one-GPU benchmark, then the six one-GPU training runs. Training
waits for the benchmark's successful completion and verifies its passing result.
No testing is submitted by this block. It reuses the existing environment and
uploaded train/validation files; no new upload or dependency installation is needed.

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

PLAIN_CAMPAIGN="/home/ryreu/atlas/Fresh_check/checkpoints/part_fullscale/part_plain_ev_100m_$(date -u +%Y%m%dT%H%M%SZ)"
(
  set -euo pipefail
  bash sbatch/submit_part_fullscale.sh \
    --mode prepare --comparison plain --output "${PLAIN_CAMPAIGN}"
  PREPARE_JOB="$(awk -F '\t' '$1 == "prepare" {id=$2} END {print id}' \
    "${PLAIN_CAMPAIGN}/submitted_job_ids.tsv")"
  [[ "${PREPARE_JOB}" =~ ^[0-9]+$ ]]

  bash sbatch/submit_part_fullscale.sh \
    --mode benchmark --output "${PLAIN_CAMPAIGN}" --gpus 1 \
    --dependency "${PREPARE_JOB}"
  BENCHMARK_JOB="$(awk -F '\t' '$1 == "benchmark1gpu" {id=$2} END {print id}' \
    "${PLAIN_CAMPAIGN}/submitted_job_ids.tsv")"
  [[ "${BENCHMARK_JOB}" =~ ^[0-9]+$ ]]

  bash sbatch/submit_part_fullscale.sh \
    --mode train --output "${PLAIN_CAMPAIGN}" --gpus 1 --concurrency 6 \
    --time 4-00:00:00 --dependency "${BENCHMARK_JOB}"

  printf '\nSave this campaign path for results/testing: %s\n' "${PLAIN_CAMPAIGN}"
  cat "${PLAIN_CAMPAIGN}/submitted_job_ids.tsv"
)
```

Preparation requests two hours and the benchmark one hour, both below six hours.
The production jobs request four days. Concurrent new and existing training runs
can increase shared filesystem load; the two campaigns have independent outputs
and neither submitter changes the other's jobs. The one-GPU benchmark is a launch
check, not a guarantee that training will fit within its requested walltime.

Later, once the full test data have been staged, keep `PLAIN_CAMPAIGN` set to the
printed path (restore it explicitly in a new login session). To queue six test jobs
behind this campaign's training array, plus their dependent report:

```bash
(
  set -euo pipefail
  : "${PLAIN_CAMPAIGN:?Set this to the new plain campaign path}"
  TRAIN_JOB="$(awk -F '\t' '$1 == "train" {id=$2} END {print id}' \
    "${PLAIN_CAMPAIGN}/submitted_job_ids.tsv")"
  [[ "${TRAIN_JOB}" =~ ^[0-9]+$ ]]
  bash sbatch/submit_part_fullscale.sh --mode test --output "${PLAIN_CAMPAIGN}" \
    --test-data /home/ryreu/atlas/datasets/JetClass/Pythia/test_20M \
    --gpus 1 --concurrency 6 --time 1-00:00:00 --dependency "${TRAIN_JOB}"
)
```

If training is already complete and no longer known to the scheduler, omit the
dependency after confirming all six `training_complete.json` markers exist.
The test driver also checks completion and locks the validation-selected models.
Test data are not needed to prepare, benchmark, or train this campaign.

### Separate AdamW pilot: standard shared-bias ParT versus ParT-EV

The prepare-only flag `--training-recipe adamw` creates a **separate two-task
pilot**, not a continuation or optimizer change to any existing run. It compares
`part` and `part_ev` at the same matched seed, **101**. Both retain the standard
shared pairwise bias; `--comparison plain` is deliberately unsupported for this
pilot. Original-recipe standard and plain campaigns still have three seeds and
six tasks. The pilot is exploratory and does not replace the three-seed original-
recipe comparison.

The pilot keeps the full 1M-update budget, global batch 512, FP16 AMP, and validation
checkpoint selection. It uses AdamW with betas `(0.9, 0.999)`, epsilon `1e-8`, and
weight decay `1e-4` on weight matrices (including the 3-D EV projection tensors),
excluding biases, normalization parameters, the class token, and scalar/vector
parameters. The learning rate increases linearly over updates 1--50,000 to `1e-3`,
then follows cosine decay over the remaining 950,000 updates to `1e-5` at update
1,000,000. The global gradient norm is clipped at 1 **after** AMP unscaling.

Its optimizer/schedule profile is frozen into `campaign.json`, alongside the data
inventory and environment. `launch.env` records `PART_FULLSCALE_TRAINING_RECIPE`;
old launch environments without this field default to `original`. After prepare,
benchmark/training/testing infer the frozen profile, including when they are queued
before preparation completes. A resume cannot switch profiles. AdamW campaign names
default to `part_ev_adamw_*` and Slurm job names begin `parte_adamw_`.

On **Tigris**, after committing/pushing the code and pulling it into the main
checkout, this prepares the pilot and queues its one-GPU benchmark plus **two
one-GPU training jobs** behind successful preparation/benchmark dependencies. It
uses the existing `part_ev_fullscale` environment and the already staged training
and validation populations. No existing campaign is modified or cancelled.

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

ADAMW_CAMPAIGN="/home/ryreu/atlas/Fresh_check/checkpoints/part_fullscale/part_ev_adamw_100m_$(date -u +%Y%m%dT%H%M%SZ)"
(
  set -euo pipefail
  bash sbatch/submit_part_fullscale.sh --mode prepare \
    --comparison standard --training-recipe adamw --output "${ADAMW_CAMPAIGN}"
  PREPARE_JOB="$(awk -F '\t' '$1 == "prepare" {id=$2} END {print id}' \
    "${ADAMW_CAMPAIGN}/submitted_job_ids.tsv")"
  [[ "${PREPARE_JOB}" =~ ^[0-9]+$ ]]

  bash sbatch/submit_part_fullscale.sh --mode benchmark \
    --output "${ADAMW_CAMPAIGN}" --gpus 1 --dependency "${PREPARE_JOB}"
  BENCHMARK_JOB="$(awk -F '\t' '$1 == "benchmark1gpu" {id=$2} END {print id}' \
    "${ADAMW_CAMPAIGN}/submitted_job_ids.tsv")"
  [[ "${BENCHMARK_JOB}" =~ ^[0-9]+$ ]]

  bash sbatch/submit_part_fullscale.sh --mode train \
    --output "${ADAMW_CAMPAIGN}" --gpus 1 --concurrency 2 \
    --time 4-00:00:00 --dependency "${BENCHMARK_JOB}"

  printf '\nSave this pilot path for results/testing: %s\n' "${ADAMW_CAMPAIGN}"
  cat "${ADAMW_CAMPAIGN}/submitted_job_ids.tsv"
)
```

The preflight requests **two hours**, the benchmark **one hour**, and each training
job **four days**. The benchmark checks the pilot's actual optimizer/model profile;
it does not reuse a benchmark from an original-recipe campaign. Training validates
the benchmark's passing result before starting. The benchmark follows the actual
production learning-rate schedule for 20 computational warmup steps and 200 measured
steps. These 220 updates cover only the beginning of the 50,000-update learning-rate
warmup: this is a short throughput/numerical smoke test, not a convergence or
peak-learning-rate validation. It records gradient clipping and rejects more than
5% skipped AMP updates in the measured window. The launch chain submits no testing.
Use `--dry-run` on an existing campaign to display its resolved two-task resources
without submitting. A dry run against a missing campaign explicitly labels its
default original-recipe preview as unresolved.

After staging the 20M test data, restore `ADAMW_CAMPAIGN` to the printed path if in
a new login session. This queues **two test jobs**, followed by their report:

```bash
(
  set -euo pipefail
  : "${ADAMW_CAMPAIGN:?Set this to the new AdamW pilot campaign path}"
  TRAIN_JOB="$(awk -F '\t' '$1 == "train" {id=$2} END {print id}' \
    "${ADAMW_CAMPAIGN}/submitted_job_ids.tsv")"
  [[ "${TRAIN_JOB}" =~ ^[0-9]+$ ]]
  bash sbatch/submit_part_fullscale.sh --mode test --output "${ADAMW_CAMPAIGN}" \
    --test-data /home/ryreu/atlas/datasets/JetClass/Pythia/test_20M \
    --gpus 1 --concurrency 2 --time 1-00:00:00 --dependency "${TRAIN_JOB}"
)
```

As for the original campaigns, omit the training dependency only after confirming
both tasks completed if that job ID is no longer known to the scheduler. This is
one seed per architecture; any pilot difference is not a three-seed uncertainty
estimate. The original and plain comparisons keep their existing result reports.

### Near-parameter-matched control: wider ParT without EdgeValue

Use `--comparison parameter_matched` during preparation to add **only three new
training runs**: `part_matched` at seeds 101, 202, and 303. The existing ParT and
ParT-EV models are not rerun. This control retains standard shared pairwise
attention bias and has **no EdgeValue messages**. Only the eight particle-attention
blocks' feed-forward hidden widths increase from 512 to 544. The class-attention
feed-forward widths remain 512; the attention/embedding widths and all input
features stay unchanged.

The resulting model has **2,209,790** trainable parameters, versus **2,209,022** for
ParT-EV and **2,143,486** for baseline ParT. It exceeds ParT-EV by just **768
parameters (0.035%)**, so it is a near-parameter-matched rather than mathematically
identical-size control. All added parameters participate in the forward pass; no
unused padding parameters are added. Matching parameter counts does not imply
matching FLOPs, memory use, or runtime.

The control uses the **original** recipe: the same training/validation data,
three matched seeds, Ranger optimizer, learning-rate schedule, global batch 512,
FP16 AMP, full training budget, and validation checkpoint selection. Combining
`parameter_matched` with the AdamW pilot is rejected. The control has its own frozen
source, profile, outputs, and `parte_matched_*` Slurm jobs; existing standard, plain,
and AdamW campaigns are not modified.

After committing/pushing the setup and pulling it on **Tigris**, this queues the
preflight, a benchmark of the control architecture, and three one-GPU training jobs
with `afterok` dependencies. No test jobs are submitted yet:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

MATCHED_CAMPAIGN="/home/ryreu/atlas/Fresh_check/checkpoints/part_fullscale/part_matched_100m_$(date -u +%Y%m%dT%H%M%SZ)"
(
  set -euo pipefail
  bash sbatch/submit_part_fullscale.sh --mode prepare \
    --comparison parameter_matched --output "${MATCHED_CAMPAIGN}"
  PREPARE_JOB="$(awk -F '\t' '$1 == "prepare" {id=$2} END {print id}' \
    "${MATCHED_CAMPAIGN}/submitted_job_ids.tsv")"
  [[ "${PREPARE_JOB}" =~ ^[0-9]+$ ]]

  bash sbatch/submit_part_fullscale.sh --mode benchmark \
    --output "${MATCHED_CAMPAIGN}" --gpus 1 --dependency "${PREPARE_JOB}"
  BENCHMARK_JOB="$(awk -F '\t' '$1 == "benchmark1gpu" {id=$2} END {print id}' \
    "${MATCHED_CAMPAIGN}/submitted_job_ids.tsv")"
  [[ "${BENCHMARK_JOB}" =~ ^[0-9]+$ ]]

  bash sbatch/submit_part_fullscale.sh --mode train \
    --output "${MATCHED_CAMPAIGN}" --gpus 1 --concurrency 3 \
    --time 4-00:00:00 --dependency "${BENCHMARK_JOB}"

  printf '\nSave this control path for results/testing: %s\n' "${MATCHED_CAMPAIGN}"
  cat "${MATCHED_CAMPAIGN}/submitted_job_ids.tsv"
)
```

Preparation requests two hours, the benchmark one hour, and each training job four
days. Training requires the control campaign's own successful benchmark; the
existing baseline/EV benchmark is not reused. `--dry-run` on the prepared campaign
resolves the three-task array even before its CPU preflight finishes. A resume
cannot switch an existing standard/plain/AdamW campaign to this control.

Later, after staging the full test data, restore `MATCHED_CAMPAIGN` to the printed
path if needed. This queues **three test jobs**, followed by their control report:

```bash
(
  set -euo pipefail
  : "${MATCHED_CAMPAIGN:?Set this to the new parameter-matched control path}"
  TRAIN_JOB="$(awk -F '\t' '$1 == "train" {id=$2} END {print id}' \
    "${MATCHED_CAMPAIGN}/submitted_job_ids.tsv")"
  [[ "${TRAIN_JOB}" =~ ^[0-9]+$ ]]
  bash sbatch/submit_part_fullscale.sh --mode test --output "${MATCHED_CAMPAIGN}" \
    --test-data /home/ryreu/atlas/datasets/JetClass/Pythia/test_20M \
    --gpus 1 --concurrency 3 --time 1-00:00:00 --dependency "${TRAIN_JOB}"
)
```

Omit the dependency only after confirming all three trainings completed if their
job ID is no longer known to the scheduler. Compare the control's results with the
existing original-recipe ParT/ParT-EV results on the same test population and seeds;
the control campaign does not silently import or replace those external results.

## SPORC context/gating EdgeValue screen

The separate seven-run, seed-101, 200k-update screen is documented in
[`part_ev_screen/README.md`](../part_ev_screen/README.md). Its launcher is
`submit_part_ev_screen_sporc.sh`: preparation and seven two-hour GPU probes come
first; after review, it queues training, dependent full-5M-validation evaluations,
and reporting. It uses a dedicated x86 environment and does not alter the Tigris
full-scale campaigns or consume the 20M test set.
