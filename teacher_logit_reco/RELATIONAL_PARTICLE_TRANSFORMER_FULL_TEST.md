# Supplemental offline RPT evaluation on the official JetClass 20M test set

This is **inference only**, using all twelve already-trained offline checkpoints:
`OFF_RPT_BASE`, `OFF_RPT_BASE_EDGEVALUE`, `OFF_RPT_SELECTED_LAYERWISE`, and
`OFF_RPT_SELECTED_EDGEVALUE`, each at seeds 101, 202, 303. It does not train the
missing shared-bias-plus-EdgeValue ablation or change the previous campaign.

## 1. Upload from Windows PowerShell

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\22rya\ComputerScience\CERN\Fresh_check\scripts\upload_relational_part_test20m.ps1"
```

Duo prompts require the user's interaction. The execution-policy exception is
limited to this PowerShell process; it does not change the machine's policy.
Review the local script before executing it. This copies
`C:\Users\22rya\Downloads\JetClass_Pythia_test_20M.tar` to
`/home/ryreu/atlas/jetclass_test20m/`, checks filesystem free space, and verifies
the upload's SHA-256 against the local archive. It refuses to overwrite an
existing archive or extracted directory. A failed upload can leave a partial
remote archive: inspect that exact file before deciding whether to replace it.
The source archive on Windows is never changed or removed.

Budget approximately **77 GB free** initially: 30.4 GB uploaded tar, 30.4 GB
extracted ROOT files, up to roughly 9.6 GB float32 logits plus filesystem and
metadata headroom. No full token, dense-pair, or REGION-tree cache is persisted.
The script checks `df`, not project/user quotas; check any separate quota too.
Do not silently redirect to an unapproved storage area if capacity is absent.

Upload the three new workflow files as well, from the repository in PowerShell
(or commit/push/pull these files through your usual workflow instead):

```powershell
Set-Location "C:\Users\22rya\ComputerScience\CERN\Fresh_check"
scp scripts/evaluate_relational_part_offline_full_test.py "ryreu@tigris.rc.rit.edu:/home/ryreu/atlas/Fresh_check/scripts/"
scp sbatch/submit_relational_part_offline_full_test.sh sbatch/run_rpt_offline_full_test.sh "ryreu@tigris.rc.rit.edu:/home/ryreu/atlas/Fresh_check/sbatch/"
```

These are new supplemental files, not modifications to the original model
source. The launcher copies the driver/worker into the evaluation directory so
later unrelated commits cannot change a running evaluation. Model and
preprocessing imports use the original offline campaign's pinned Git worktree.

## 2. Extract and submit on Tigris

After a successful checksum verification, log in:

```powershell
ssh ryreu@tigris.rc.rit.edu
```

Run the following **Bash** commands on RC:

```bash
cd /home/ryreu/atlas/Fresh_check
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python scripts/evaluate_relational_part_offline_full_test.py inspect-archive \
  /home/ryreu/atlas/jetclass_test20m/JetClass_Pythia_test_20M.tar
tar --keep-old-files -xf /home/ryreu/atlas/jetclass_test20m/JetClass_Pythia_test_20M.tar \
  -C /home/ryreu/atlas/jetclass_test20m

export RPT_FULL_TEST_JOBS=20
export RPT_FULL_TEST_CONCURRENCY=8
bash sbatch/submit_relational_part_offline_full_test.sh --dry-run
bash sbatch/submit_relational_part_offline_full_test.sh
```

Do not run extraction twice into an already populated directory. If extraction
was interrupted, inspect the directory and archive before resuming it; the
preparation job separately checks exact filename coverage and ROOT entry counts.
No archive deletion is performed automatically.

The default graph is:

1. One CPU preparation job: authenticate all checkpoint/normalizer/source
   lineage, test filenames, ROOT entry counts and bytes, and file-disjointness
   against **every original split**, including the old 500k final test.
2. **Twenty GPU array tasks, up to eight concurrent.** Each gets ten ROOT
   files (one million jets), runs all twelve checkpoints on each, and commits
   predictions every 10,000 jets. Input conversion and tree building are shared
   between models; each model retains its own learned encoders and attention.
3. Twelve CPU metric tasks, up to three concurrent. Each aggregates one
   checkpoint over all 20M events, never averaging per-file ROC operating points.
4. One CPU report job, depending on all metric tasks.

GPU tasks request one GH200, 8 CPUs, 64 GB RAM and a 48-hour limit. This limit
is a resource request, not a runtime estimate. Completed chunks are reusable
even if the containing GPU task times out. Initial concurrency is a scheduling
choice, not a performance filter; adjust it to available allocation/resources.
The work runs on allocated compute nodes, **not the login/submit node**.

No job is stopped or pruned because accuracy/rejection is disappointing.
Invalid checkpoints, overlap, missing/corrupted results, nonfinite outputs and
runtime failures remain errors. Downstream jobs with impossible dependencies
are cancelled so they do not linger indefinitely.

Unserialized Weaver trimming counters are restored to their freshly loaded
state at every 10,000-event chunk. This makes trimming behavior independent of
how many chunks the job completed before a retry. Trimmer flags, model weights,
and normalization statistics are unchanged; this supplemental execution policy
is explicit in the new evaluation plan. GPU roundoff can still depend on the
device/runtime; byte-identical cross-hardware logits are not claimed.

## 3. Monitor and resume

The submitter prints the supplemental output path and job IDs. Set `EVAL` to
that exact printed path; do not use the original offline campaign directory.

```bash
EVAL="/home/ryreu/atlas/Fresh_check/checkpoints/relational_particle_transformer/rpt_offline_test20m_TIMESTAMP"
cat "${EVAL}/submitted_job_ids.txt"
squeue --me -o "%.18i %.24j %.2t %.10M %R"
find "${EVAL}/predictions" -name complete.json -type f | wc -l
tail -n 10 "${EVAL}"/logs/rpt20m_run_*.out
```

There are 200 file completion records when inference is finished. Within each
file, valid chunk sidecars allow partial-file recovery. A committed chunk with
changed hashes is rejected, not silently reused. A data file without its final
sidecar is an interrupted write and is recomputed.

After failure, wait for/cancel **only this evaluation's** remaining stale jobs
(IDs are in its ledger), then:

```bash
bash sbatch/submit_relational_part_offline_full_test.sh --resume "${EVAL}"
```

This queues a new graph against the same frozen driver, assignments and
checkpoints. It also retries preparation if no plan was completed. Inference
skips authenticated completed chunks. The submitter refuses to duplicate a
still-live graph. Completed prediction files and prior campaign artifacts are
not deleted. Metrics are deterministically recomputed and immutable identical
outputs are accepted.

## 4. Results and statistical interpretation

Final outputs are `reports/full_test_report.md` and
`reports/full_test_report.json`. Download them from PowerShell:

```powershell
$evaluation = 'rpt_offline_test20m_TIMESTAMP' # exact name printed by submitter
$destination = "C:\Users\22rya\ComputerScience\CERN\a_download_checkpoints\$evaluation"
New-Item -ItemType Directory -Force -Path $destination | Out-Null
scp -r "ryreu@tigris.rc.rit.edu:/home/ryreu/atlas/Fresh_check/checkpoints/relational_particle_transformer/$evaluation/reports" "$destination/"
```

Report all four configurations and all three seeds. The report includes accuracy,
cross-entropy, class efficiencies, one-vs-rest AUC, confusion matrices, Brier
score, and the original 15-bin top-label ECE. Rejection targets are globally
frozen at 30%, 50%, 75%, 99%, and 99.5%; the last two are useful for leptonic
classes and the official paper's operating points. The discriminant, ceiling
order statistic and inclusive tie rule match the original campaign. Achieved
efficiency and QCD passing counts are always retained. Zero observed background
is flagged as empirically infinite (JSON null point estimate), not zero rejection
or proof of infinite population rejection. Three-seed means of rejection are
not finite if any seed saturates.

95% Wilson intervals for QCD false-positive rate/rejection **condition on the
empirical signal threshold**. They omit uncertainty in that threshold, so they
are not full ROC confidence intervals. Seed SD is not a confidence interval.

Paired accuracy intervals compare every model to its same-seed baseline, and
Selected EdgeValue additionally to Selected Layerwise. Sampling units remain
paired event identities, stratified by the ten fixed class counts. For scale,
the sufficient counts of paired losses/ties/wins are sampled via multinomial
draws: exactly the same bootstrap distribution, but not the same random draws
as the old event-index implementation. Seed 917301, 10,000 replicates, NumPy
PCG64, and linear 2.5%/97.5% quantiles are frozen in a **new supplemental v1
contract**, not represented as an unchanged old bootstrap artifact.

The models still have **1M training jets**, even though evaluation has 20M jets.
This is a supplemental test, not retraining, fresh model selection, proof of
scaling to 100M training jets, or establishment of architectural novelty.

## Validation status

Local tests cover input-conversion reuse and model isolation, a partial batch,
archive coverage/safety, split overlap under directory relocation, prediction
authentication, complete task allocation, original-metric equivalence, zero
background, and deterministic paired statistics. The twelve local checkpoint
hashes were checked against their original immutable lock. Real GPU execution
with the installed Tigris Weaver/backend must still be observed on RC; a local
CPU mock is not an authoritative Weaver equivalence test.
