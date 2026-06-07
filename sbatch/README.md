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
