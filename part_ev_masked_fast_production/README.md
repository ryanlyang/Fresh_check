# Fast256 masked-predictive production on Tigris

Independent, candidate-only production for the execution-optimized masked
predictive objective on noncompact, all-eight-block, layerwise State-EV.
Existing source snapshots, campaigns, environments, datasets and queues are not
modified. No baseline/control is submitted and no old checkpoint is fine-tuned.

## Frozen training contract

- Fresh seed 101; one GH200, native BatchNorm, clean global batch 512.
- Same original model, masking objective, teacher taps, predictor, optimizer,
  learning-rate schedule, auxiliary weight schedule and EMA schedule.
- Auxiliary batch **256**: all 256 selected jets in one slot. The byte-identical
  measured implementation from `part_ev_masked_speed/step.py` is imported.
  Production additionally enables existing detached latent statistics every
  100 updates, without changing loss/gradient/RNG computations.
- Full **1,000,000 attempted updates / 50 validations**, one check per 20,000
  updates. Each training-time validation covers 1,280,000 events. Checks are not
  complete dataset epochs. AMP skips remain recorded, not hidden extra updates.
- Deployed student: 2,327,462 parameters; student + predictor: 2,626,598 trainable
  parameters; separate frozen EMA teacher: 1,926,412 parameters.
- Accuracy-selected, first-highest checkpoint; final FP32 evaluation of all 5M
  validation events. Student-only export: no predictor/teacher inference cost.
- One long allocation, **not three-day segments**. Complete latest state saved
  every 1,000 updates, at validation boundaries, and on cooperative shutdown.
  Restart includes Ranger/Lookahead, scaler, teacher, RNG, mask generators,
  trimming state and stream cursors. Resuming replays the consumed data stream;
  replay consumes allocation time. A genuine failure blocks dependent jobs.

The engine is reused in a private namespace with a new campaign/task identity
and the fast256 step. No historical module globals are patched. An interrupted
run resumes `last.pt`, not accuracy-selected weights. There is no automatic
cross-cluster race/cancellation or retry loop.

## Why a final preflight

The speed study measured a **2.54x auxiliary-active-step speedup** and a provisional
**255-hour allocation** for fast256, with about 34.82 GiB peak allocated device
memory. That was a short diagnostic, not production admission. Its failed fast32
check recorded only `Exact tensor changed: state`; the exact original tensor
cannot be inferred from that message.

This pipeline qualifies **only fast256**:

1. Collect a named FP32/AMP reproduction of the fast32 discrepancy. Fast32 stays
   excluded regardless of that diagnostic's outcome. A passing reproduction
   does not prove what caused the original failure; old evidence is preserved.
2. Require two full-step parity repetitions for fast256 in both FP32 and AMP,
   including variable lengths, nonzero State-EV branches, all unscaled gradients,
   exact masks/RNG/scaler/BN buffers and bounded optimizer/teacher/model state.
   Existing tolerances are not loosened.
3. Require seven-update repeat/disk/resume and schedule-boundary probes crossing
   Lookahead and validation, plus production-serializer roundtrips.
4. Require seven actual ROOT-stream auxiliary updates, checkpoint after three,
   fresh-session restoration, replay of three training and two validation batches,
   and matching subsequent inputs, masks, optimizer/teacher state and validation.
   These use explicit diagnostic schedule labels, not 100k trained updates.
5. Measure 4,200 sustained active updates, 200 clean-only updates, full-128-particle
   work, 128k-event FP32 validation, student-only inference, checkpoint/prediction
   I/O, and synthetic full-5M metric computation. No extra dataset cache is built.

The new walltime is recomputed from the slowest timing window, 879,999 active and
120,001 clean updates, **all 50 validations**, checkpoint writes and 30% margin.
It cannot be lower than the speed study's provisional recommendation (255h for
the current donor), or exceed the reviewed ceiling (default **480h**). Evaluation
gets at least two hours, increased if measurements require it. Live Slurm limits
are checked again at dispatch. If a required gate or ceiling fails, **no training
is submitted**; there is no silent fallback to a smaller batch or shorter recipe.

The unchanged objective does not guarantee identical final accuracy: AMP rounding
and training trajectories may differ. Final full-5M validation overlaps checkpoint
selection; it is not an independent test or a matched-control/multi-seed result.

## Pipeline and resources

`preflight -> afterok dispatch -> train -> afterok evaluate -> afterok report`

| Job | GPU allocation | CPUs / host RAM | Requested walltime |
| --- | --- | --- | --- |
| Preflight | 1 GH200 | 12 / 128 GiB | 5h30m |
| Dispatch | None | 2 / 4 GiB | 30m |
| Train | 1 GH200 | 12 / 128 GiB | Measured, at least 255h for current donor, at most 480h |
| Evaluate | 1 GH200 | 12 / 128 GiB | Measured, at least 2h |
| Report | None | 2 / 8 GiB | 1h |

Only preflight and dispatch are initially submitted. Dispatch creates the final
three jobs after successful qualification. Short CPU jobs have no advance signal
(avoiding the previous immediate-USR1 summary failure). GPU jobs get a 15-minute
shutdown warning, with an additional monotonic deadline when Slurm supplies its
end time.

The local headroom gate is approximately **8.47 GiB initially**, including a 4-GiB
shared reserve and the remaining candidate/probe/snapshot/output budget. Other
campaigns may consume free space: this is not a filesystem reservation. Nothing
is deleted automatically. Only this campaign's new files are written.

## Queue after committing and pushing

On **Tigris**, use the completed speed study below. `--reviewed` explicitly permits
automatic candidate production only after the new preflight passes:

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only

  SPEED_ROOT="$(pwd -P)/checkpoints/part_ev_masked_speed/speed_20261126T013751Z"
  FAST_ROOT="$(pwd -P)/checkpoints/part_ev_masked_fast_production/fast256_$(date -u +%Y%m%dT%H%M%SZ)"
  printf 'Speed study: %s\nNew production campaign: %s\n' "${SPEED_ROOT}" "${FAST_ROOT}"

  bash sbatch/submit_part_ev_masked_fast_production.sh \
    --mode plan --from-speed "${SPEED_ROOT}" --output "${FAST_ROOT}"

  bash sbatch/submit_part_ev_masked_fast_production.sh \
    --mode pipeline --from-speed "${SPEED_ROOT}" --output "${FAST_ROOT}" \
    --max-train-hours 480 --reviewed

  squeue --me -o '%.18i %.40j %.12T %.12l %R'
)
```

Save the printed `FAST_ROOT`. Do not repeat this block after jobs have been
recorded: it generates a new directory each time. A shared ownership record
prevents a second output from dispatching the same execution variant, but does
not make duplicate preflights useful.

For preflight-only submission, use `--mode preflight` instead of
`--mode pipeline --reviewed`; it queues a CPU summary after the GPU job, without
production.

## Review, explicit dispatch, and retry

Set `FAST_ROOT` to the actual path printed during submission, then:

```bash
bash /home/ryreu/atlas/Fresh_check/sbatch/submit_part_ev_masked_fast_production.sh \
  --mode review --output "${FAST_ROOT}"
```

For preflight-only campaigns, or a terminal dispatch failure after preflight
passed, explicit `--mode dispatch --output "${FAST_ROOT}" --reviewed` performs
the same admission checks. Do not manually race a still-pending dispatch job.
Completed dispatch is idempotent. Partial/unknown submission journals stop retries
until reconciled with Slurm; they are not permission to start another campaign.

After all relevant previous production jobs are terminal, explicit
`--mode resume --output "${FAST_ROOT}" --reviewed` authenticates the saved state
and submits only this candidate's required retry chain. Pending descendants may
need explicit cancellation first. No other campaign's jobs are cancelled.

Useful files:

- `preflight.json`, `summary.json`, or `preflight_failure.json` with named traces.
- `pipeline_submissions.json`, `production_submissions.json`, and retry journals.
- `production/runs/MASKED_PREDICTIVE_FAST256_STATE_EV_LAYERWISE/seed_101/validation/`.
- `production/reports/fast256_report.json` and `.md` after final evaluation.
