# Offline RPT: four missing ablations, three seeds each

This is a **new follow-up campaign**, not a restart or replacement of the
completed offline study. It trains only 12 new checkpoints, evaluates them on
the same official 20M test events, and combines their results with the existing
12 checkpoints' full-test results. It does not modify the paper or old campaigns.

## Frozen model matrix

| Pair features | Bias sharing | EdgeValue | Run ID | Status |
|---|---|---|---|---|
| Standard four | Shared | No | OFF_RPT_BASE | Existing |
| Standard four | Layerwise | No | OFF_RPT_BASE_LAYERWISE | **New** |
| Standard four | Shared | Yes | OFF_RPT_BASE_SHARED_EDGEVALUE | **New** |
| Standard four | Layerwise | Yes | OFF_RPT_BASE_EDGEVALUE | Existing |
| Standard + selected | Shared | No | OFF_RPT_SELECTED_SHARED | **New** |
| Standard + selected | Layerwise | No | OFF_RPT_SELECTED_LAYERWISE | Existing |
| Standard + selected | Shared | Yes | OFF_RPT_SELECTED_SHARED_EDGEVALUE | **New** |
| Standard + selected | Layerwise | Yes | OFF_RPT_SELECTED_EDGEVALUE | Existing |

Every row has seeds **101, 202, 303**. Selected means **PT + TRACK + REGION**,
unchanged from the original offline transfer; no new family selection.

Shared+EV keeps the original shared pair encoder and original Weaver forward.
Its bias is reused by all particle-attention layers. Each layer/head has its own
value-message projection of the shared pair stem. The stem is packed/scattered
alongside the bias after Weaver trims/permutates particles, so relation messages
follow the same particle identities. Setting EV projections to zero must recover
the corresponding shared model's logits, input/parameter gradients and BN buffers.

Important limitation: historical shared and layerwise implementations differ in
trimming and pair-normalization packing, not just head-projection sharing. Standard
shared retains symmetric triangular pair packing; selected shared retains valid
directed pairs. Historical layerwise retains the original confirmation path and
dense projection-tail normalization. We preserve those definitions, rather than
silently changing the already evaluated models. EV-on/off within shared is a clean
addition; cross-path contrasts describe these complete implementations, not a
perfectly isolated test of bias sharing. Parameter counts also differ.

## Matched training and honest chronology

- Reuse the original split manifest: 1M training, 125k checkpoint-validation,
  125k reporting-validation jets, with identical identities, order and token bytes.
  The old 500k final-test cache is not rebuilt.
- The restored `JetClass_Pythia_train_100M_part1` contains the source files; this
  is **not** a change to 100M training jets.
- Reuse original offline relation/REGION normalizers. Do not refit on validation
  or test data. Rebuilt token hashes must match the original offline binding.
- Original `TrainingConfig`: BF16, microbatch 64, accumulation 2, AdamW,
  unchanged optimizer/schedule/checkpoint rule; max 40 epochs, min 12, patience 8.
  Original deterministic epoch sampler and collation are reused.
- No model-quality threshold cancels models or downstream evaluation. The
  original within-run early stopping remains part of the matched protocol.
- The four follow-up configurations are frozen **after inspection of earlier
  500k and 20M test results**. They are retrospective ablations, not an untouched
  confirmatory holdout experiment. Test metrics never select their checkpoints.

## Storage and dependencies

Read-only prerequisites (do not delete until reporting is complete):

- Original campaign `rpt_offline_transfer_20260809T195641Z_0737194d3c`, including all
  checkpoints, registries, split manifest, normalizers, offline cache binding,
  and compiled backend. Original large token/tree caches are **not** required.
- `rpt_offline_test20m_20260925T235646Z`: original plan, final report, and prediction
  NPZs with receipts. Existing predictions support event-paired comparisons.
- The original runtime directory recorded in that full-test plan's `source_root`.
- Extracted training ROOT files in `/home/ryreu/atlas/jetclass_part1` and extracted
  test ROOT files in `/home/ryreu/atlas/jetclass_test20m/test_20M`. Neither tar is needed.
- Original `atlas_kd_tigris` environment and Weaver source.

The bootstrap copies authenticated historical Python/C++ source and overlays only
the follow-up implementation into the new campaign's `source/`. Unrelated commits
in the main repository do not invalidate running jobs. Frozen source changes do.

One shared mmap token cache is approximately 9.15 GB (decimal), not 12 copies.
REGION trees are compact compressed shards, decompressed once per selected-model
process, with trees reconstructed lazily per batch. There are no persisted dense
pair matrices. New test logits are at most approximately 9.6 GB before compression
(plus small container/receipt overhead). Old full-test predictions are read in place.
Bootstrap requires 25 GiB free initially; after token caching, a measured conservative
tree bound plus remaining prediction budget and **6 GiB reserve** must fit. Your
reported 41 GiB may suffice, but this must be checked on the actual token population.
Other campaigns can consume that space later. Nothing is automatically deleted.

## Install the small code update

Use your usual commit/push/`git pull --ff-only` workflow if desired, including these
new files explicitly (do not add unrelated paper work). Alternatively, this
PowerShell block bundles only the runtime files, with one upload/MFA prompt:

```powershell
cd C:\Users\22rya\ComputerScience\CERN\Fresh_check
$bundle = Join-Path $env:TEMP ("rpt-offline-ablations-" + (Get-Date -Format yyyyMMddHHmmss) + ".tar.gz")
$files = @(
  "teacher_logit_reco/relational_part/offline_ablations.py",
  "teacher_logit_reco/relational_part/offline_ablation_data.py",
  "scripts/run_relational_part_offline_ablations.py",
  "scripts/evaluate_relational_part_offline_full_test.py",
  "sbatch/run_rpt_offline_ablations.sh",
  "sbatch/submit_relational_part_offline_ablations.sh"
)
tar -czf $bundle @files
if ($LASTEXITCODE -ne 0) { throw "Code bundle creation failed" }
scp $bundle "ryreu@tigris.rc.rit.edu:/home/ryreu/atlas/rpt-offline-ablations-code.tar.gz"
if ($LASTEXITCODE -ne 0) { throw "Code upload failed" }
ssh ryreu@tigris.rc.rit.edu
```

Then **inside the Tigris Bash shell** (not local PowerShell), unpack the six named
runtime files into the working repository. This updates only those code paths;
it does not overwrite either old campaign's frozen source or artifacts:

```bash
cd /home/ryreu/atlas/Fresh_check
tar -tzf /home/ryreu/atlas/rpt-offline-ablations-code.tar.gz
tar -xzf /home/ryreu/atlas/rpt-offline-ablations-code.tar.gz -C /home/ryreu/atlas/Fresh_check
bash sbatch/submit_relational_part_offline_ablations.sh --dry-run
bash sbatch/submit_relational_part_offline_ablations.sh
```

If you transferred through Git instead, skip the tar commands. Dry-run prints the
graph; it is not a remote data/environment validation. Save the printed campaign
root. The submission graph is:

```text
prepare -> cache[3, serial] -> storage check -> tree shards[126, max 8 CPU jobs]
 -> bind -> small real-Weaver GPU validation -> train[12, max 4 GPU jobs]
 -> lock all 12 -> full-test inference[20, max 8 GPU jobs]
 -> global metrics[12, max 3 CPU jobs] -> combined report
```

The 126 cheap CPU tasks cover 10k-jet shards (partial final validation shards);
they are not 126 training or test-inference jobs. The GPU validation runs FP32
zero-message logits/gradient/BN/trimming parity for shared standard and selected
models, then BF16 finite backward and strict state reload for every new model.
Local CPU stand-ins cannot attest to the installed real Weaver/CUDA interface.

Optional concurrency, set **before submitting**:

```bash
export RPT_ABLATION_TRAIN_CONCURRENCY=4
export RPT_ABLATION_TEST_CONCURRENCY=8
export RPT_ABLATION_TREE_CONCURRENCY=8
```

All phases have `afterok` dependencies. Runtime/data/storage failures stop dependent
work, not scientific underperformance. There is no submission in the implementation
step itself; only invoking the submitter above queues jobs.

## Monitor, resume, download

Set `ROOT` to the exact **new** campaign path printed by the submitter:

```bash
ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/relational_particle_transformer/rpt_offline_factorial_TIMESTAMP"
cat "${ROOT}/submitted_job_ids.txt"
squeue --me -o '%.18i %.22j %.2t %.10M %.45R'
ls -lh "${ROOT}/reports/"
```

Logs are under `${ROOT}/logs/`; epoch curves/registrations are under
`runs/<run_id>/seed_<seed>/`. Cache/tree shards and inference chunks commit receipts
atomically. Completed registrations/checkpoints are authenticated and skipped on
resume; unfinished training uses the original epoch-checkpoint resume. Restarting
a job never changes architecture, normalization or data selection.

After inspecting failures and cancelling **only stale jobs in this campaign's
job-ID ledger**, rerun:

```bash
bash sbatch/submit_relational_part_offline_ablations.sh --resume "${ROOT}"
```

The submitter refuses a duplicate graph while any previously submitted campaign job
is still live. Do not blanket-cancel other studies. Resume uses frozen campaign
source; a code correction requires an explicitly new campaign, not editing hashes.

Final files: `reports/offline_factorial_report.json` and `.md`. They contain all
eight configurations, three-seed accuracies, background rejection at 30/50/75/99/99.5%
signal efficiency and percent changes from base; the JSON retains per-seed metrics,
passing counts, conditional Wilson intervals and event-paired accuracy intervals.
Contrasts include EV at shared/layerwise bias, layerwise effects with/without EV,
their interaction, and selected-versus-standard features. Seed SD is not a CI;
conditional rejection intervals omit uncertainty in the empirical signal threshold.

Download the small `reports/`, `selection/`, and `campaign_spec.json` first. Preserve
all `best_model_val.pt` files and normalizers before any later cleanup.

## Local verification (2026-09-28)

The relational test suite passed **144 tests**, with **5 skips** (unavailable real
Weaver and POSIX-only integration on Windows). This includes 27 new follow-up
tests: model factories, both attention interfaces, active randomized trimming,
zero-message logits/gradients/BN parity, nonzero EV gradients, padding, strict
reload, cache/hash/collator equivalence, exact shard coverage, frozen-source
authentication, restart reuse, all-twelve locking even for bad metrics, global
old/new paired aggregation, and combined reports. Python syntax, Bash syntax,
submitter dry-run and whitespace checks passed. Existing Torch mask-type
deprecation warnings remain. Real-Weaver/CUDA validation and the measured remote
storage check are intentionally pending the queued stages; no remote jobs were
submitted during implementation.
