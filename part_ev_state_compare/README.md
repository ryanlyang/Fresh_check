# Full-schedule, seed-101 State-EV comparison

Three new runs on SPORC `tier3`, with two existing full-schedule controls reused
read-only. This supersedes the proposed six-new-run capacity campaign; it does
not modify or resume that campaign, cancel jobs, or change the older controls.

## Models

| New training index | Model | Attention bias | State-EV | Parameters |
| --- | --- | --- | --- | ---: |
| 0 | `EV_STATE_LAYERWISE_RANGER` | Layerwise | Optimized, all eight blocks, width 32 | 2,327,462 |
| 1 | `EV_STATE_FAST_RANGER` | Shared | Optimized, all eight blocks, width 32 | 2,323,710 |
| 2 | `PART_STATE_MATCHED_RANGER` | **Shared** | **None** | 2,327,894 |

The capacity control is ordinary ParT with its eight particle feed-forward
hidden widths increased from 512 to **601**. Its particle embedding width,
attention, shared pair-bias encoder, class-attention blocks, and classifier are
unchanged. It has no ordinary EV or state branch. Width 601 is the closest
uniform integer FFN width to the layerwise State-EV parameter count: **432 more
parameters (0.0186%)**. Matching parameter count does not match FLOPs or latency.

Both State-EV models use the same existing optimized all-eight implementation:
query chunks of 64, shared pair normalization within a forward pass, and
checkpointed state-message chunks. Layerwise bias means eight independent final
bias heads on the same shared pair stem, initialized as copies of the shared
head. It does not mean eight independent pair encoders. State width and ordinary
EV behavior are otherwise the same in the two models.

Reused controls, both seed 101:

| Evaluation index | Model | Parameters | Original training task |
| --- | --- | ---: | --- |
| 3 | `PART_RANGER` | 2,143,486 | `21964940_0` |
| 4 | `EV_RANGER` | 2,209,022 | `21964940_6` |

The original control directory defaults to:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_state_fullscale/state_full_20261024T185049Z
```

Job IDs are read from its frozen submission ledger, not hard-coded into the
launcher. The other original seeds and any obsolete last-four runs are left
untouched. This campaign neither needs nor retries their old evaluation chain.

## Matched full schedule

All three new runs start from fresh seed-101 initialization, using the unchanged
full Ranger recipe (RAdam plus Lookahead), data preprocessing, input order,
global batch size 512, and learning-rate schedule of the original control
campaign. No donor training weights are used.

- Official 100M-jet training pool and 5M-jet validation pool.
- **1,000,000 optimizer-update attempts**, corresponding to 512M jet exposures.
- **50 validation checks**, each after 20,000 training steps; these are not
  50 full passes through the dataset.
- Same training-time validation population and accuracy-based checkpoint
  selection as the existing full-schedule controls.
- Final evaluation of all five selected checkpoints on the full 5M validation
  population, including accuracy, macro OVO/OVR AUC, and QCD rejection.
- No independent 20M test evaluation in this launcher. The full validation
  population overlaps checkpoint-selection data.

This is a single-seed comparison, not a three-seed reproducibility estimate.
Layerwise versus shared State-EV isolates the bias-sharing choice within the
state model. Layerwise State-EV versus the wider baseline compares two complete
models with approximately equal parameter counts; it changes both bias sharing
and message construction and therefore is not an isolated state-only contrast.

## Queue on SPORC

First commit and push this implementation locally, then pull it on
**sporcsubmit**. Relevant source must be committed and clean: the launcher runs
from a hashed snapshot of `HEAD`, not from a mutable working checkout.

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

COMPARE_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_state_compare/state_compare_$(date -u +%Y%m%dT%H%M%SZ)"

bash sbatch/submit_part_ev_state_compare.sh \
  --mode prepare \
  --output "${COMPARE_ROOT}"
```

Save the printed campaign path. Defaults are account `reu-aisocial`, partition
`tier3`, one A100 per GPU task, 8 CPUs, 64G host memory, and the unchanged
`/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python` environment.
Preparation retains the 20-GiB free-space guard; nothing is deleted.

This submits preparation followed by **three benchmarks**, not production
training. Preparation and each benchmark request **5 hours**, below the requested
six-hour pre-training limit. Benchmark concurrency is at most two.

The GPU gate checks native-length and full-128-particle training memory, AMP
recovery, validation throughput, initial baseline equivalence, and nonzero-state
forward/backward equivalence against the unoptimized implementation. It also
checks that the data stream, software, and GPU identity match the reused
controls. A local CPU test pass does not replace this GPU gate.

Once all three benchmarks pass:

```bash
bash sbatch/submit_part_ev_state_compare.sh \
  --mode review --output "${COMPARE_ROOT}"
```

Review the measured runtime estimates, then submit:

```bash
bash sbatch/submit_part_ev_state_compare.sh \
  --mode train \
  --output "${COMPARE_ROOT}" \
  --reviewed \
  --concurrency 3
```

This queues:

```text
three new trainings + existing ParT-101 and EV-101 completion
  -> lock five selected checkpoints
  -> five full-5M validation evaluations
  -> comparison report
```

Only the checkpoint lock waits for active original control jobs; new training
can start immediately. Completed controls must pass artifact/checkpoint audits
before their dependencies are omitted. A failed or unidentifiable control stops
submission rather than silently replacing it.

Training walltime comes from the new benchmarks with a 30% planning margin,
checked against the partition limit. It is not capped at the old short-run
walltime. Evaluation requests at least five hours. No source, checkpoint,
submission ledger, or Slurm job in the original campaign is changed.

## Results and retries

The job ledger is `${COMPARE_ROOT}/submitted_job_ids.tsv`; logs are in `logs/`.
Results are written to:

```text
reports/state_compare_report.md
reports/state_compare_report.json
```

Reports include all 50 validation checks, matched seed-101 contrasts, parameter
counts, QCD passing counts, and benchmark costs. They do not average different
architectures together or present a single seed as a multi-seed result.

Use `--resume` only after inspecting a terminal failed/interrupted phase. Active
duplicate campaigns are refused. Retries keep the same frozen source and
campaign contract. Training resumes from its own restart state; evaluations
reuse only hash-verified completed prediction chunks. Never edit the frozen
source to bypass a failed gate; a code change requires a fresh campaign.

For a non-mutating submission preview:

```bash
bash sbatch/submit_part_ev_state_compare.sh \
  --mode prepare --output /absolute/new/campaign/path --dry-run
```
