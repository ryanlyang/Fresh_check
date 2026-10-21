# Reusing the short-budget ParT / EdgeValue benchmark

Use this protocol to test a new architecture under the same conditions as the
seven-run SPORC EdgeValue screen, reusing its trained controls instead of training
identical baselines again. Train each new model from scratch; **reusing a control
means reusing its comparison results, not initializing the new model from it.**

This document records the existing recipe and specifies a reusable experiment
template. The generic control registry and launcher described below are a design
for a follow-up implementation, not commands already supported by the current
seven-task launcher. No active campaign needs to be changed.

The specific five-run Ranger follow-up now implements these reuse checks in
[part_ev_message_study](../part_ev_message_study/README.md), with a separate
tier3 launcher. The arbitrary-model/AdamW template below remains a design,
not a claim that the five-run launcher accepts arbitrary tasks.

The two-run [layerwise-bias Ranger screen](../part_layerwise_screen/README.md)
also reuses `PART_RANGER` and `EV_RANGER`. It keeps the shared pair stem and
existing per-layer EV projections, adding independent per-layer bias heads.
At the user's request, that study supports a one-command direct submission with
small checks inside training instead of separate preflight/review jobs. Its
metadata records that no sustained benchmark was performed; this is a
study-specific exception, not a change to other campaigns' admission policies.

The four-run [selected-relations screen](../part_selected_screen/README.md)
adds PT, TRACK, and REGION under shared/layerwise bias with EV off/on. It reuses
both shared controls above and the completed layerwise controls, with the same
seed-101 Ranger budget. Its launcher automatically queues preparation and GPU
benchmark gates before training; it does not inherit the layerwise study's
no-separate-preflight exception. See its README for raw-track transport,
training-only normalization, directed pair processing, and the corrected REGION
ancestor lookup relative to the legacy helper.

## 1. The exact reduced training budget

| Setting | Short-budget screen |
|---|---|
| Training population | Official JetClass Pythia 100M training jets, all files |
| Validation population | Official 5M validation jets |
| Seed | 101 for each model |
| Global batch | 512; one GPU, no gradient accumulation |
| Training budget | 200,000 attempted updates = 102.4M jet exposures |
| Reporting epochs / validation checks | 10, each after 20,000 updates |
| Checkpoint validation | 1,280,000 examples per check |
| Checkpoint selection | Highest validation accuracy; first checkpoint wins ties |
| Early stopping | None; finish the full budget |
| Final screening evaluation | Selected checkpoint on all 5M validation jets |

The original long recipe has 1,000,000 updates and 50 reporting epochs. This
screen divides its **exposure budget by five**, not its dataset size. A reporting
epoch is 10.24M exposures, not a complete pass through 100M distinct jets.
Streaming exposures do not guarantee that every training jet appears exactly once.

Keep the training and checkpoint-validation iterators continuous across checks;
do not reset them at each reporting epoch. An AMP-skipped update still consumes
a batch and advances the attempted-update count and learning-rate schedule.
Record skipped updates; do not silently add replacement steps.

### Ranger schedule (`ranger_short_v1`)

- Optimizer: pinned Weaver RAdam plus Lookahead, using the repository's
  state-complete `Ranger` wrapper.
- RAdam: learning rate `1e-3`, betas `(0.95, 0.999)`, epsilon `1e-5`, weight decay
  `0`. Lookahead: `k=6`, `alpha=0.5`. No warmup or gradient clipping.
- Keep `1e-3` for the first 140,000 attempted updates. Apply 15 multiplicative
  decays, each by `0.01**(1/15)`, at zero-based steps
  `140000, 144000, ..., 196000`. The final learning rate is `1e-5`.

The learning rate is assigned **before** the update. Thus the first decay applies
to attempted update 140,001. Reuse `part_ev_screen.protocol.learning_rate` rather
than substituting an epoch-based scheduler. Simply stopping the original long
schedule at 200,000 updates would miss its decay phase and is not this experiment.

### AdamW schedule (`adamw_short_v1`)

- AdamW: peak learning rate `1e-3`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight
  decay `1e-4`; no Lookahead.
- Linear warmup over 10,000 updates: update 1 uses `1e-7`, update 10,000 reaches
  `1e-3`. Cosine decay then reaches `1e-5` at update 200,000.
- Clip the global gradient norm at `1.0`, after AMP unscaling.
- Reuse `adamw_parameter_groups`: decay matrix/convolution weights and EV
  projections, but exclude biases, normalization parameters, scalar/vector
  parameters, and explicit exclusions such as the class token.

Use the whole matching recipe. Comparing Ranger to AdamW changes schedule,
regularization, and clipping as well as the optimizer; it is not an optimizer-only
ablation. The exact recipe dictionaries are stored in the donor `campaign.json`.

## 2. Which existing control to use

Donor campaign on RC:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z
```

| Control run ID | Architecture | Parameters | Recipe | Main use |
|---|---|---:|---|---|
| `PART_RANGER` | Standard shared-bias ParT | 2,143,486 | `ranger_short_v1` | Default baseline for a new architecture |
| `EV_RANGER` | Standard shared-bias ParT + EV | 2,209,022 | `ranger_short_v1` | Isolate improvements beyond existing EV |
| `PART_ADAMW` | Standard shared-bias ParT | 2,143,486 | `adamw_short_v1` | Baseline for an AdamW experiment |
| `EV_ADAMW` | Standard shared-bias ParT + EV | 2,209,022 | `adamw_short_v1` | Isolate improvements beyond EV with AdamW |

All four use seed 101. These are not the earlier small-data offline models,
Tigris full-budget models, or the plain/no-pair-bias ablations. The shared bias,
native backbone, and class-token readout remain in these controls; there are no
extra selected TRACK/PT/REGION features.

For a new EV extension with Ranger, report both `new - EV_RANGER` and
`new - PART_RANGER`. With AdamW, use the two AdamW controls instead. Report
accuracy differences as `100 * (new_accuracy - control_accuracy)` percentage
points. State whether AUC is macro OVO or macro OVR; do not mix them.

The submitted production graph was training `21904478`, evaluation `21904479`,
then report `21904480`. These IDs help locate logs, but do not establish that
artifacts are complete. At the time this protocol was requested, training was
still in progress. Do not publish unfinished controls as final results.

## 3. Conditions that make control reuse valid

Keep these fixed, except for the explicitly proposed architectural change:

- Dataset file identities and contents, labels/class order, and the unchanged
  `particle_transformer/data/JetClass/JetClass_full.yaml` preprocessing. Use
  the same 17 input features, 128-particle limit, wrap padding/mask, and native
  sequence trimming.
- Recipe, seed, initialization of unchanged backbone weights, batch size, and
  attempted-update budget. Initialize added modules without consuming the
  backbone or training RNG stream; the screen model contains an example.
- Data-stream seeds/order, two workers, `fetch_step=0.01`, and existing loader
  behavior. Matching only the model seed does not guarantee matching batches.
- Single-GPU native BatchNorm; the donor GPU model/memory capacity and software
  environment. The screen uses x86 SPORC A100s, not Tigris GH200s.
- FP16 autocast training with FP32 parameters and GradScaler; FP32 validation
  and evaluation; TF32 and cuDNN benchmarking disabled. Reuse the screen's
  precision context, which sets both outer and Weaver-native autocast.
- The same checkpoint-validation stream, selection/tie rule, full-validation
  event identities, evaluator, and metric definitions.

Use the donor's frozen metadata and source hashes as the authority, not whatever
code happens to be at the latest Git commit. A new campaign hash will naturally
differ: compare the shared experimental fields, not whole campaign hashes.

Changing the seed, budget, batch/accumulation, optimizer recipe, preprocessing,
or training precision requires a matched new control. Changing platform/runtime
also needs a reviewed comparability check or new controls. Moving an identical
dataset to another directory does not change the population if content and
file/entry identities are verified. Job IDs, queue concurrency, and walltime
alone are not model-training hyperparameters.

## 4. Create a read-only control registry

A small future exporter should create `controls.json` in a **new** experiment
directory. It should reference the donor artifacts, not rename them or overwrite
their campaign identity. For each requested control:

1. Validate the donor campaign hash, task/seed, full recipe, data inventory,
   preprocessing hash, frozen source, precision policy, and runtime.
2. Require `training_complete.json`, step 200,000, all ten validation checks,
   and the selected checkpoint's matching SHA-256. Confirm its selection
   agrees with highest accuracy / first tie across those checks.
3. Before treating full-validation results as reusable, verify the finalist lock,
   `full_validation/metrics.json`, checkpoint hash, all prediction-file sidecars
   and hashes, and complete file/entry coverage: 5M events, 500k per class.
4. Save the verified reference paths, hashes, recipe, selected check, and result
   status in the registry. Missing evaluation means `pending`, not zero or pass.

Useful donor artifacts, relative to its campaign directory:

```text
campaign.json
source/                                      # frozen implementation
production_approval/                         # reviewed admission evidence
selection/locked_checkpoints.json
runs/<RUN_ID>/seed_101/optimizer.json
runs/<RUN_ID>/seed_101/training_complete.json
runs/<RUN_ID>/seed_101/best_model_val.pt
runs/<RUN_ID>/seed_101/validation/check_01.json ... check_10.json
runs/<RUN_ID>/seed_101/full_validation/metrics.json
runs/<RUN_ID>/seed_101/full_validation/file_*.npz
runs/<RUN_ID>/seed_101/full_validation/file_*.json
reports/screen_report.json
```

Keep predictions and identity metadata for paired comparisons and recomputing
metrics. `last.pt` is a resume checkpoint, not the selected comparison model.
Neither benchmark/probe checkpoints nor AMP-recovery scaler state are training
initialization. The donor's reviewed recovery approval applies to that campaign;
new models must pass their own GPU/memory/precision/stability checks.

If a donor control is still running, register it as pending and allow new models
to train, but block the final comparison until the required artifacts validate.
Reusing a complete control requires neither retraining nor repeat inference when
the evaluation population and metric definitions are unchanged.

## 5. Reusable experiment template and implementation outline

Illustrative configuration for a **future** generic launcher (not an existing CLI):

```yaml
schema: reusable_part_screen_v1
name: my_new_ev_idea_seed101
donor_campaign: /home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z
donor_campaign_sha256: "<verified campaign.json content_hash>"
recipe_ref: ranger_short_v1
seed: 101
controls:
  - PART_RANGER
  - EV_RANGER
new_models:
  - run_id: MY_EV_IDEA_RANGER
    architecture: my_ev_idea
initialization: fresh_seeded
evaluation: official_validation_5M
```

Implementation sequence:

1. Build the control-registry exporter/validator above. Reuse the **complete**
   donor recipe, data, and runtime contracts; do not reconstruct them from a few
   YAML numbers. Freeze the new model configuration and source separately.
2. Add an extensible model/task registry in a separate module. Train only entries
   in `new_models`; controls are read-only references. **Do not append tasks to
   the active screen's `TASKS` or modify its frozen `source/`.** Its current
   trainer, locking logic, and report intentionally assume seven specific tasks.
3. Reuse the existing scheduler, optimizer, loader, precision, training/validation,
   and checkpoint semantics. Verify unchanged backbone initialization and input
   fingerprints before production; benchmark the new model on the donor runtime.
4. Queue `prepare -> new-model GPU checks -> review -> new-model training ->
   full-validation evaluation -> combined report`. Keep pre-production jobs
   under six hours. Reference controls are prerequisites for reporting, not new
   training-array entries. Choose training walltime from measurements.
5. Have the combined report join new results with verified donor results by
   event identity, and show matched recipe/seed contrasts. While training is
   incomplete, compare the same check numbers, not best-so-far at unequal budgets.

Acceptance tests should cover schedule boundaries (Ranger steps 139999/140000
and 195999/196000; AdamW steps 0/9999/199999), data fingerprints, precision,
resume state, checkpoint-selection ties, rejection of incompatible/incomplete
control artifacts, full population coverage, and a mock queue showing that reused
controls are never submitted for training. Extracting shared helpers must leave
the original seven-run behavior and active source snapshot unchanged.

## 6. Code to reuse and limits of the comparison

- [Screen protocol](protocol.py): short recipe dictionaries and exact LR function.
- [Optimizer implementation](../part_ev_fullscale/optimizer.py): Ranger state and
  AdamW parameter groups.
- [Screen model](model.py) and [reference backbone](../part_ev_fullscale/model.py):
  unchanged controls and isolated initialization for added mechanisms.
- [Precision context](precision.py), [data adapters](data.py), and
  [reference streams](../part_ev_fullscale/data.py): matching inputs and numerics.
- [Screen driver](../scripts/run_part_ev_screen.py) and [reporting](reporting.py):
  validation, checkpointing, finite evaluation, and existing contrasts.
- [Queue guide](README.md): existing campaign operations and reviewed recovery.

The full 5M evaluation overlaps the validation data used for checkpoint selection;
it is a common **screening validation population**, not an independent test.
The official 20M test is not used by this protocol. A reused single-seed control
supports exploratory matched comparisons, not across-seed uncertainty estimates.
Promising ideas can later receive matched additional seeds and independent test
evaluation. Repeated reuse of the validation benchmark does not make those later
architecture choices independently confirmed.
