# Selected pair relations on the reusable 100M-population Ranger screen

Four new seed-101 runs on SPORC tier3. Nothing is initialized from a donor;
four completed controls are reused as comparison artifacts only.

| New run | Attention bias | EV | Matching reused control | Parameters |
|---|---|---|---|---:|
| SELECTED_PART_SHARED | Shared | Off | PART_RANGER | 2,153,406 |
| SELECTED_EV_SHARED | Shared | On | EV_RANGER | 2,218,942 |
| SELECTED_PART_LAYERWISE | Layerwise | Off | PART_LAYERWISE_RANGER | 2,157,158 |
| SELECTED_EV_LAYERWISE | Layerwise | On | EV_LAYERWISE_RANGER | 2,222,694 |

Each adds 9,920 parameters to its standard-relation control. These are not
capacity-matched comparisons. All EV models retain independent per-layer
value projections; shared/layerwise refers to the scalar attention-bias heads.

## Locked recipe and populations

Follow [REUSABLE_SCREEN_PROTOCOL](../part_ev_screen/REUSABLE_SCREEN_PROTOCOL.md):
official 100M training population, seed 101, batch 512, one donor-matched A100,
two loader workers, native BatchNorm, FP16 autocast with FP32 master weights
and GradScaler, FP32 evaluation, TF32 disabled. Ranger/RAdam + Lookahead,
200,000 attempted updates, ten checks every 20,000 updates, original short
learning-rate schedule unchanged. This is 102.4M exposures, not five passes.

Checkpoint selection uses highest accuracy on the same continuous 1.28M/check
validation stream (first tie wins). Evaluate the selected checkpoint on all
5M validation jets. No independent 20M test is read. Predictions and class
order use the same evaluator and identity checks as the controls.

Default read-only donors:

```
/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z
/home/ryreu/atlas/Fresh_check/checkpoints/part_layerwise_screen/layerwise_20261019T190751Z
```

Preparation verifies both donor contracts, numerical source, data inventory,
environment, training completion, selected checkpoints and full-validation
prediction coverage/hashes. A missing or incompatible control blocks the new
campaign instead of silently retraining it or substituting another baseline.

## Added representation

Retain the four native ParT kinematic descriptors. Concatenate learned PT(8),
TRACK(12), REGION(12) embeddings, giving 36 pair-input channels and the same
64-wide pair stem. Reuse the historical feature definitions and MLP widths:

- PT: constituent momentum fractions/ranks and directional pair combinations;
  10→32→8 MLP.
- TRACK: raw impact parameters, uncertainties, significances, validity and
  pair compatibility; shared 7→32→16 endpoint MLP and 81→48→12 pair MLP.
- REGION: canonical beam-free angular clustering, exclusive K=2,4,8 groups,
  common ancestors, cluster fractions and geometry; 41→32→12 MLP.

MLPs use GELU and RMSNorm. The original particle embedding still receives
exactly the 17 unchanged native inputs. Four raw track aliases are appended
only for transport through the native sequence trimmer; they are removed
before particle embedding. In particular, clipped errors or tanh-transformed
impact parameters are NOT inverted to invent raw measurements.

The normalizer uses the first 50,000 examples of the seed-101 training stream,
with no model or validation input. Select up to 64 directed pairs per jet by
a separate fixed PCG64 stream; count applicable track nodes once each.
Use median/IQR scaling and [-8,8] clipping for the historically designated
continuous channels. Fit uncertainty floors from the first percentile of
valid positive errors in this same training subset, with minimum 1e-6.
Freeze and share these constants across all four runs. This is a new,
training-only fit, not reused statistics from the earlier 1M campaign.

Normalization collection uses a finite 98-batch prefix of that same two-worker
stream (50,176 events transported, last batch sliced to exactly 50,000).
Whole-batch quotas preserve native worker interleaving and allow natural worker
completion before shutdown. Production training/benchmark streams remain
infinite and unchanged. Collection errors remain fatal; none are suppressed.

Important differences from the standard controls / historical implementation:

- Directional features require directed pair processing. Pair BatchNorm sees
  the dense directed population, including native-style padded entries;
  standard controls use native symmetric triangular packing. Therefore the
  selected-minus-standard comparison includes this pair-encoder change, not
  only extra information. Mask added relation embeddings and EV messages.
- Relations are recomputed after the actual native permutation/truncation,
  using only retained constituents. No disabling trimming or stale tree cache.
- Unchanged backbone weights and the model/training RNG remain matched. Added
  parameters use an isolated seed; bias-head copies start identical. Equal
  selected/control logits are not expected or falsely claimed.
- REGION uses correct set-bit binary lifting for lowest common ancestors.
  The historical vectorized helper used `difference >= 2**level`, over-lifting
  some unequal-depth pairs. The new helper is tested against direct parent
  walks. Historical code/artifacts remain unchanged. This is an intended-feature
  follow-up, not bit-for-bit reproduction of that historical lookup defect.

The authenticated existing C++ tree backend is compiled once in the new
campaign. Trees and pairs are built on demand, with no persistent 100M-jet
tree/pair cache. Preparation/GPU benchmarking may be materially slower than
standard ParT; resource feasibility is measured before production starts.

## Memory-efficient execution

The selected models now use `selected_memory_v1`. Architecture, parameter
counts, feature definitions, normalization, native attention, batch 512,
precision, RNG, Ranger recipe and checkpoint selection are unchanged. The
completed shared/layerwise controls and their source are not modified.

- Pack the shared EV stem once in physical `[batch, query, key, channel]`
  order, while exposing the same logical axes to the original EV contraction.
  Each block now saves a view of that storage instead of copying it again.
  The batched matrix multiplication sees the same operand strides as before.
  For batch 512, 128 particles and 64 FP16 channels, one stem is 1 GiB;
  replacing eight separately saved copies with one removes 7 GiB of duplicate
  saved tensors. This is not a measured reduction in whole-model peak memory.
- Non-EV models no longer allocate an unused EV message tensor.
- Build and release raw relation families sequentially. In AMP, cast fixed
  track descriptors to the following Linear's existing autocast dtype before
  concatenation; its effective input is unchanged, without an 81-channel FP32
  temporary. Keep descriptor arithmetic and normalization in FP32 as before.

There is no activation checkpointing, CPU activation offload, reduced batch,
microbatching or repeated network computation during backward. Fewer copies
should help rather than impose a recomputation cost, but A100 throughput and
capacity still require measurement. Shared-storage gradient accumulation can
round differently in FP16; bit-identical trajectories or unchanged final
accuracy cannot be guaranteed.

Before timing, each GPU benchmark compares old/new execution on eight real
training jets at full constituent length: FP32 evaluation, FP32 training and
FP16 training, including gradients of every parameter-owning module. The
frozen tolerances are recorded in `initial_parity.memory_execution`. This
diagnostic restores weights, buffers, module modes, trimming and RNG, does not
step an optimizer and does not consume extra loader batches. Failure blocks
training. The existing full-batch 5,000-step throughput, precision, memory
headroom and runtime gates remain mandatory and unchanged.

## Queue everything

Commit/push locally first, then on **sporcsubmit**:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only
bash sbatch/submit_part_selected_screen_sporc.sh --dry-run
bash sbatch/submit_part_selected_screen_sporc.sh
```

The launcher freezes the committed source and submits the entire dependency
chain in one call:

```
prepare → benchmark[4] → gate → train[4] → lock → evaluate[4] → report
```

Preparation and each GPU benchmark request 5h (less than 6h). Gate/lock/report
request 2h. Training requests 96h, evaluation 12h, with two GPU tasks concurrent
by default (`--concurrency 4` permits all four). Walltimes are reservations,
not measured promises. The gate requires all four checks to pass, including
native-input stream fingerprints, FP16/FP32 precision, full-length memory,
finite training, bounded AMP skips and measured runtime projections with
30% margin fitting the queued training/evaluation walltimes. Failed checks
block downstream jobs. There is no separate manual preflight submission.

All production states are fresh; benchmark weights/optimizer/scaler are never
used for training. Later pulls cannot alter the frozen campaign. No installer,
deletion, control rerun or automatic cancellation is included.

## Inspect progress and final results

Use the campaign path printed by the submitter:

```bash
bash sbatch/submit_part_selected_screen_sporc.sh \
  --mode status --output /absolute/path/to/selected_campaign

cat /absolute/path/to/selected_campaign/reports/selected_screen_report.md
```

Status is read-only and prints every completed check, `--` for unfinished runs,
and best accuracy through each new/control pair's common check. Reports contain
all eight configurations, four matched selected-minus-standard comparisons,
EV and layerwise contrasts within selected models, AUC, paired accuracy counts,
and classwise QCD rejection at the same operating points as the controls.

After interruption, inspect logs and `submitted_job_ids.tsv`; use `--resume`
only when all campaign jobs have terminated. Default `--mode run --resume`
reuses valid prepared artifacts/passing benchmarks. `--mode train --resume`
starts at the benchmark gate; `--mode evaluate --resume` starts at checkpoint
locking; `--mode report --resume` only rebuilds the comparison. Completed
prediction files are verified before reuse. Changed source/configuration or a
failed recorded benchmark requires a fresh campaign, preserving old evidence.

Local tests use the CPU tree oracle. Actual CUDA capacity, compiled-backend ABI
and runtime/data agreement are checked by the queued jobs, not asserted locally.

### Recovery from benchmark array 21940378

Do **not** resume that campaign: it freezes the old allocation path. Preserve
its logs and benchmark reports. After committing/pushing locally, pull on
SPORC and submit a fresh campaign:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only
bash sbatch/submit_part_selected_screen_sporc.sh --train-hours 168 --dry-run
bash sbatch/submit_part_selected_screen_sporc.sh --train-hours 168
```

This seven-day reservation does not add training updates: the old non-EV
projections were already about 91 and 108 hours, exceeding the default 96-hour
reservation once the required 30% margin is included. The new measured gate
still decides whether the request is sufficient; no speedup is assumed.
Preparation and benchmarks retain their five-hour limits. Confirm the failed
chain's descendants have terminated before submitting replacements; inspect
its `submitted_job_ids.tsv` and cancel only that chain if anything remains.
The launcher does not cancel or overwrite any existing campaign.

### Recovery from preparation job 21939901

The C++ backend compiled and passed parity. Preparation then failed during
early shutdown of the infinite normalization DataLoader (`worker ... Aborted`).
The finite-prefix fix changes only normalization collection lifetime, not its
sampling or model/optimizer recipe. Commit/push/pull the fix and submit a fresh
campaign using the normal command above, **without `--resume`**: the failed
campaign has frozen the old implementation. Preserve it as diagnostic evidence.
Inspect its job ledger and cancel only any still-active jobs from that failed
chain before replacing it; the launcher normally cancels invalid dependencies.
