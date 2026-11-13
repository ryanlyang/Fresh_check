# State-conditioned attention bias: SPORC benchmark

Implements the independent [attention-bias plan](../part_ev_state_compare/STATE_CONDITIONED_ATTENTION_BIAS_PLAN.md).
The model is implemented; this launcher is **benchmark-only**, not production
admission. Existing models, environments, data, snapshots, and jobs are unchanged.

## Model and frozen scientific contract

- Noncompact standard-relations all-eight layerwise State-EV, state width 32.
- Eight zero-initialized bias-free `32 -> 8` heads add
  `tanh(B_l r_ij)` to the existing layerwise bias before the single native MHA.
- Native values, EV, and State-EV use that call's same post-dropout weights.
- Directed corrections, native trimming/masks, checkpointed query chunks of 64,
  graph-connected normalized-pair cache, FP32 correction projection/tanh.
- **2,329,510 parameters**: 2,048 more than the reference. No other planned idea
  or capacity reduction is bundled in.
- Target production recipe remains fresh seed 101, original full Ranger,
  1,000,000 update attempts, batch 512, and 50 validation checks. No production
  training or evaluation submission interface is included in this benchmark stage.

The benchmark runs on two same-node A100s with DDP/SyncBatchNorm, local batch
256, one data worker/rank, eight CPUs and 64 GiB host memory. It reuses the
reference's environment and audited train/val inventory, not its trained weights.
Prepare rechecks staging-receipt data hashes; no test data are opened. This can
take minutes and reads the full train/validation bytes, without making a copy.

## Queue from sporcsubmit

First commit/push the new implementation and pull it on RC. Do not edit an old
campaign's `source/` directory. New snapshots come from committed `HEAD`; later
pulls cannot change queued workers. Unrelated working-tree edits are not deleted.

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

BIAS_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_state_bias/state_bias_$(date -u +%Y%m%dT%H%M%SZ)"
BIAS_REFERENCE="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_layerwise_ddp/layerwise_2g_20261029T162243Z"

bash sbatch/submit_part_ev_state_bias.sh \
  --mode prepare --output "${BIAS_ROOT}" --reference "${BIAS_REFERENCE}" \
  --partition tier3 --dry-run

bash sbatch/submit_part_ev_state_bias.sh \
  --mode prepare --output "${BIAS_ROOT}" --reference "${BIAS_REFERENCE}" \
  --partition tier3

squeue --me -o "%.18i %.30j %.12T %.12l %R"
```

Default Python is `/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python`;
default account is `reu-aisocial`. Overrides `--python`, `--account`, and
`--partition` are frozen at preparation. No package installation occurs.
The launcher checks that the exact requested live partition is UP and accepts
the benchmark walltime. It never silently substitutes a different partition;
Slurm still validates account/QoS access and availability.

| Job | GPU request | Walltime | Dependency |
| --- | --- | --- | --- |
| `evbias_prepare` | None | 5 hours | None |
| `evbias_benchmark` | 2 A100s on one node | 5 hours 30 minutes | Successful prepare |
| `evbias_summary` | None | 15 minutes | Benchmark terminates |

All checks request less than six hours. The benchmark has a five-hour internal
budget plus scheduler shutdown allowance. Cancellation, timeout, missing results,
or a failed numerical/memory gate never authorizes training. A summary failure
can simply mean its upstream evidence is incomplete; inspect the prepare log.

## What the benchmark checks

- Fixed warmup/order-repeat identity checks in eval FP32, train FP32, and AMP;
  old initial weights and RNG streams must match. The new branch stays enabled.
- Active-state zero-bias gradients, active-bias chunk/checkpoint backward parity,
  and an active two-rank SyncBatchNorm/DDP global-batch reference check.
- 20 warmup plus 4,200 measured real-data updates, with repeated fetch windows;
  10 warmup plus 40 all-128-particle updates to stress memory with active heads.
- Parameters/scaler/Lookahead agreement across ranks, finite optimizer state,
  AMP skips, and at least 10% allocated GPU-memory headroom on both A100s.
- Full model/optimizer/slow-weight/scaler and per-rank RNG/trimmer checkpoint
  roundtrip followed by seven replayed updates across Lookahead synchronization.
  This checks stochastic model recovery on a retained batch; it does not claim
  to have timed replay of a production-scale data cursor.
- 128k-event rank-zero FP32 validation with the production validation batch 512,
  plus fixed-length FP32 inference throughput. Validation timing includes input
  and metric overhead; fixed inference excludes data loading. Neither is accuracy
  evidence for the eventual trained model.
- Fixed-sample per-layer/head bias magnitude, saturation, gradient norms,
  attention entropy/distribution shifts and state-message diagnostics. Evaluated
  without dropout; the comparison removes the bias at identical Q/K states.
  RNG, buffers, trimming counters and modes are restored; no data-cursor advance.

The new zero-gradient path can change gradient accumulation order even when
outputs agree exactly. A CPU diagnostic found float32 differences around `1e-7`
and float64 differences around `1e-16`. Gradient checks use the existing project
State-EV contract (`rtol=1e-4`, `atol=1e-5`), while forward and distributed checks
retain their separately recorded tolerances. No adaptive tolerance/warmup retries.

## Review and next decision

```bash
bash /home/ryreu/atlas/Fresh_check/sbatch/submit_part_ev_state_bias.sh \
  --mode review --output "${BIAS_ROOT}"
```

Review is read-only. Detailed evidence is in `benchmark.json`, with numerical
failures in `error_rank_*.json`, logs in `logs/`, and job IDs in
`submitted_job_ids.tsv`. The summary projects 1M updates, all 50 validations,
checkpoint writes and a 30% planning margin; queue time and long restart replay
are excluded. Twelve days is only a provisional production allowance.

This benchmark does not authorize reuse of the historical result as a fully
matched scientific control. Audit its source/runtime/recipe and segmented restart
history before making that comparison. It does not submit a replacement control.

Allow at least 6 GiB free for preparation/benchmark headroom. The only larger
probe artifact is a restart checkpoint; no pair tensors, dataset copy, or full-5M
prediction cache are created here. A future production storage check is separate.

For a code fix or failed probe, preserve evidence and prepare a new output.
The submitter refuses existing directories and has no implicit retry, cancellation,
environment-repair, or production mode.

## Local tests

```powershell
$biasTests = (Get-ChildItem tests/test_part_ev_state_bias_*.py).FullName
& .\.venv\Scripts\python.exe -B -m pytest @biasTests -q
```

CPU model, real-graph Gloo, checkpoint, protocol, and fake-Slurm tests are local
checks only. CUDA/AMP/SyncBatchNorm memory and runtime remain for the queued probe.
