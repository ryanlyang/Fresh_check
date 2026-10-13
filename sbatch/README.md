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
