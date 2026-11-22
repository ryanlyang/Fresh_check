# Recover the Tigris triangle profiling rejection

The Nov 22 diagnostic for `triangles_20261120T210558Z` showed every required
profiling stage, positive call counts, and positive CPU **or** GPU timing. The
old validator incorrectly required positive CPU time for every stage. Several
GPU regions had `cpu_us: 0.0` and positive `device_us`; CPU-only wrappers were
also present. The reported restart evidence passed on Tigris. This does **not**
resolve the separate SPORC restart diagnostic.

The corrected check requires an integer call count greater than zero, finite
nonnegative CPU and device times, and at least one positive time. Missing or
unexpected stages, zero time on both sides, and invalid timing still fail.
Model code, gradients, restart checks, numerical tolerances, measured timing,
training recipe and resource limits are unchanged.

## Recover saved measurements, without another GPU benchmark

Commit and push the fix locally first. On **Tigris**, run:

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only

  TRI_ORIGINAL="${PWD}/checkpoints/part_ev_rts_tigris/triangles_20261120T210558Z"
  TRI_RECOVERY="${PWD}/checkpoints/part_ev_rts_tigris/triangles_profilefix_$(date -u +%Y%m%dT%H%M%SZ)"
  printf 'Recovery output (keep this path): %s\n' "${TRI_RECOVERY}"

  bash sbatch/submit_part_ev_rts_tigris.sh \
    --mode plan-profile-recovery \
    --from-campaign "${TRI_ORIGINAL}" --output "${TRI_RECOVERY}"

  bash sbatch/submit_part_ev_rts_tigris.sh \
    --mode recover-profile \
    --from-campaign "${TRI_ORIGINAL}" --output "${TRI_RECOVERY}" \
    --reviewed
)
```

This submits **one CPU-only job, 30-minute walltime, 4 CPUs / 16 GiB RAM**.
It does not submit production. Inspection before submission also revalidates
the saved measurements; it opens metadata and frozen source, not datasets or
checkpoints, and does not run CUDA training.

The original pipeline must no longer have live jobs, and it must not already
have a production admission or an ambiguous submission. If a check stops the
command, inspect the stated cause rather than deleting journals or repeatedly
submitting under new names. An uncertain `sbatch` response is recorded in
`recovery_intent.json`; reconcile it with Slurm before any retry.

The recovery directory is a new frozen capsule in the same Tigris registry.
It inherits the original interpreter, account, donor and walltime ceilings.
It authenticates the original launch, source, campaign and measurements, then
copies `benchmark.json` **byte-for-byte**. That raw file keeps its original
`campaign_sha256`; it is checked against the original campaign, never relabeled
as a newly measured benchmark. The new campaign, summary and eventual
production plan record explicit recovery provenance and origin hashes.
The measured benchmark and production engine files must still be byte-identical.

No original JSON, frozen source, error log, queued job, environment, model or
SPORC artifact is changed. Old diagnostics and their conservative storage
reservations are not erased. A later `git pull` cannot alter this frozen capsule.

## Review the recovery

Use the exact recovery path printed above (the example assignment deliberately
requires you to paste that path, rather than selecting a potentially wrong run):

```bash
read -r -p 'Paste the recovery output path: ' TRI_RECOVERY
export TRI_RECOVERY
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  test -f "${TRI_RECOVERY:?}/summary.json"
  bash sbatch/submit_part_ev_rts_tigris.sh --mode status --output "${TRI_RECOVERY}"
  bash sbatch/submit_part_ev_rts_tigris.sh --mode review --output "${TRI_RECOVERY}"
)
```

Require `passed: true` and `complete: true`. A Slurm `COMPLETED` entry alone is
not production approval. Any other failed scientific or resource check still
blocks the recovery. Runtime recommendations come from the saved real timing
measurements, not the diagnostic stage timing. The usual 30% planning margin
does not guarantee completion under future contention or restart replay.

## Explicitly queue production after review

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  : "${TRI_RECOVERY:?Set this to the reviewed recovery output path}"
  bash sbatch/submit_part_ev_rts_tigris.sh \
    --mode dispatch --output "${TRI_RECOVERY}" --reviewed
  squeue --me -o '%.18i %.40j %.12T %.12l %R'
)
```

Dispatch rechecks the original and recovered evidence, current ARM environment,
aggregate persistent-storage budget, live partition limit and inherited
walltime ceilings. It queues the existing train → evaluate → report workflow:
one GH200 for training/evaluation, seed 101, 1,000,000 updates and 50 validation
checks. The same model/donor ownership guard prevents the original and recovered
outputs from both launching production. Repeating a fully successful dispatch
reports the existing jobs; a partial or uncertain dispatch requires inspection.

Storage can still block dispatch, as it did for the separate readout campaign.
This fix neither weakens that check nor deletes data. Readout and support do not
need a benchmark rerun for this triangle profiling issue.
