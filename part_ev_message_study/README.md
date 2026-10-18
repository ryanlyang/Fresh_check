# Five-run EdgeValue message study — SPORC tier3

Models and launchers are implemented and locally tested. **A100 fit, numerical
stability, and runtime still require cluster preflight.** No jobs have been
submitted. Existing campaigns and their frozen source are unchanged.

| Array index | New run | Change from ordinary EV |
|---:|---|---|
| 0 | EV_STATE_RANGER | Nonlinear messages conditioned on both particle states |
| 1 | EV_SEPARATE_RANGER | Independent, copy-initialized EV pair encoder |
| 2 | EV_STATE_SEPARATE_RANGER | Both changes |
| 3 | EV_ROUTING_RANGER | Separately corrected EV attention weights |
| 4 | EV_VARIANCE_RANGER | Weighted pair-feature variance alongside ordinary EV |

All five train **from scratch**, seed 101, with the donor's shortened Ranger
recipe: batch 512, 200,000 attempted updates, ten checks, compressed decay at
140k–196k, and the full 100M training population. Selected checkpoints are
evaluated on the common 5M validation population, **not the 20M test set**.
PART_RANGER and EV_RANGER are read-only reused controls, never new training tasks.

See the [architecture plan](../part_ev_screen/STATE_AND_SEPARATE_ENCODER_PLAN.md)
and [matched training protocol](../part_ev_screen/REUSABLE_SCREEN_PROTOCOL.md).
This implements the specific five-run study, not an arbitrary-model API.

## 1. Commit, push, then prepare on sporcsubmit

Commit and push the new package, driver, two Slurm scripts, tests and documentation
before pulling on RC. The launcher snapshots committed source and refuses dirty
relevant files. Use the **unchanged donor environment**, not atlas_kd_sporc;
no package installation is needed.

On **sporcsubmit**, in Bash:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

MESSAGE_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_message_study/messages_$(date -u +%Y%m%dT%H%M%SZ)"

bash sbatch/submit_part_ev_message_study.sh \
  --mode prepare --output "${MESSAGE_ROOT}" \
  --partition tier3 --concurrency 2
```

This queues only CPU prepare (five hours) followed by five GPU benchmarks
(five hours each, at most two concurrent). All pre-production allocations are
below six hours. Save the campaign path; after reconnecting, restore that exact
MESSAGE_ROOT rather than generating a new timestamp.

Defaults:

- Account reu-aisocial, partition tier3, one A100 per GPU task.
- Eight CPUs, 64G host memory, two workers, batch 512, no accumulation.
- Python: /home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python.
- Data: /home/ryreu/atlas/datasets/JetClass.
- Donor: /home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z.
- Minimum free storage: 12 GiB.

Use `--donor DIR` to name another compatible donor. Paths and launch resources
are frozen at prepare. The actual GPU model **and memory capacity** must match
the donor. Nothing silently switches to debug or changes batch/precision.
Add `--dry-run` to print requests without files or submissions.

Prepare verifies donor source, environment, recipe, admission and file inventory.
It reads all 1,050 ROOT files to compare SHA-256 with the upload receipts under
datasets/JetClass/receipts, so it can take longer than header-only prepare.
Keep those receipts.

The donor's inventory did not include payload hashes. This check combines
original staging hashes with its unchanged path/size/mtime/entry inventory;
it does not claim a retroactive training-time byte capture. Relocated/changed
inputs fail this version's strict reuse check and need a separate migration audit.

## 2. Review preflight, then authorize production

Once all five benchmarks finish:

```bash
bash sbatch/submit_part_ev_message_study.sh \
  --mode review --output "${MESSAGE_ROOT}"
```

Checks cover initial EV parity in evaluation and training, unchanged backbone
and dropout RNG, observed FP16 training/FP32 validation with FP32 parameters,
and matching donor input fingerprints across repeated fetch windows.

Timing uses 4,200 updates after 20 warmup updates. Full-128-particle testing uses
eight bounded AMP calibration attempts followed by ten finite, non-skipped
updates; the last calibration update must also succeed. Every skip is recorded.
Training/full-length/validation allocations must remain below 90% of donor GPU
memory. An unfinished five-hour benchmark is inconclusive, not approval.

These are new admission checks, not reuse of the old recovery authorization.
No probe model, optimizer or scaler state enters production. All five must pass.

After reviewing:

```bash
bash sbatch/submit_part_ev_message_study.sh \
  --mode train --output "${MESSAGE_ROOT}" \
  --reviewed --concurrency 2
```

This freezes approval against the benchmark evidence and queues:

```text
train [5] -> full-5M-validation evaluate [5] -> combined report
```

Both dependencies are `afterok`; no manual dependency editing is needed.
Training/evaluation walltimes use measured costs plus 30% headroom.
`--train-hours N` may increase, but not undercut, the recommendation.
Queue concurrency can change without changing the recipe, but additional
filesystem contention can make the timing estimate optimistic.

## 3. Monitor and resume

```bash
cat "${MESSAGE_ROOT}/submitted_job_ids.tsv"
squeue --me -o "%.18i %.35j %.2t %.10M %.12l %R"
```

Logs are under MESSAGE_ROOT/logs. Each run stores
runs/RUN_ID/seed_101/validation/check_01.json through check_10.json.
The original complete optimizer/scaler/RNG/trimmer checkpointing and deterministic
data replay are reused. Keep `last.pt` after interruptions.

Once previous campaign jobs are terminal:

```bash
bash sbatch/submit_part_ev_message_study.sh \
  --mode train --output "${MESSAGE_ROOT}" --reviewed --resume
```

Completed runs are verified and skipped; incomplete runs resume. Data replay
takes time and is not included in runtime projections. The launcher refuses
active duplicates and never cancels jobs.

Use `--mode benchmark --resume` for an interrupted check without a committed
result. A committed failed benchmark is retained: inspect it, fix the cause and
prepare a new campaign. Do not patch snapshots. Retry downstream work with
`--mode evaluate --resume` or `--mode report --resume`.

## 4. Controls and output

Missing donor training/evaluation is recorded as **pending**. New training may
proceed, but the final comparison remains blocked. No zeros or implicit control
retraining substitute for missing artifacts. Retry the report after they finish.

Reporting checks all ten validation records and first-best selection, optimizer/
runtime metadata, selected-checkpoint hashes, finalist locks, prediction
sidecars/hashes and complete coverage. It recomputes metrics from committed
logits and checks identical file/entry labels across all seven models.

```text
campaign.json                       # immutable new study
controls.json                       # references / pending statuses
data_contents.json                  # receipts and ROOT-byte hashes
production_approval.json             # reviewed new evidence
runs/.../best_model_val.pt
runs/.../full_validation/            # logits, identities, metrics
reports/message_study_report.json
reports/message_study_report.md
```

Reports include accuracy, both macro AUC definitions, loss, class rejections,
matched-check curves, paired correctness counts, state/separate interaction and
costs. All five models are shown against both controls, regardless of improvement.
Diagnostics cover branch sizes, fixed-attention endpoint sensitivity, encoder
separation, routing changes/entropies and weighted variance.

This remains a single-seed validation screen; the evaluation overlaps checkpoint
validation. It does not establish independent test performance or across-seed
significance. Capacity-matched controls and extra seeds follow promising results.

## Local checks

```powershell
$evMessageTests = @((Get-ChildItem tests/test_part_ev_message_*.py).FullName)
& .\.venv\Scripts\python.exe -m pytest @evMessageTests -q
```

Local tests cover explicit forward/gradient references, initialization and
dropout identity, separate BN storage, masks/permutations, CPU mixed-precision
dtype paths, exact Ranger resume, corrupt/incomplete controls, full-5M coverage
fixtures and simulated Slurm. They do not certify A100 memory or speed.
