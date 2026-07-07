# Deployable HLT Self-Dual-View 10-Class Plan

## Short Name

HLT-SDV, or deployable HLT self-dual-view fusion.

## Goal

Test whether the strong particle-level dual-view fusion signal from PD10-V2 can
be converted into a deployable HLT-only tagger.

PD10-V2 showed that a two-branch particle-level fusion model can be a very
strong privileged teacher:

```text
offline ParT teacher model-val accuracy:       0.838101
dual-view logit teacher model-val accuracy:    0.841944
particle dual-view teacher model-val accuracy: 0.844928
```

The V2 teacher is not deployable because it needs offline particles. This plan
asks whether the same architectural idea helps when both views are available
from HLT alone:

```text
inference:
  HLT particles
    -> deterministic second HLT degradation -> HLT2 particles
  HLT particles + HLT2 particles
    -> two-branch particle fusion model -> 10-class logits
```

This is not privileged distillation. It is a deployable self-dual-view HLT
tagger. The second view does not add physical information, but it may expose
which HLT features are stable under additional detector-like corruption.

## Reuse Existing PD10 Artifacts

Use the exact PD10 high-data root:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/privileged_distill_10class_5m_hlt0p4_run1
```

Reuse:

```text
split_manifest/split_manifest.json.gz
hlt_cache/
teachers/hlt_part_teacher_10class/best_model_val.pt
students/pd10_student_warm_start_ce_only/
final_report/pd10_report.json
```

Do not rebuild the original HLT cache. The only new cache is the derived HLT2
cache, generated from the already cached HLT particles.

Use the same splits:

```text
model_train: 5,000,000 jets
model_val:   1,000,000 jets
final_test:  1,000,000 jets
```

Keep the same 10-class task:

```text
QCD, Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl
```

## Core Hypothesis

HLT2 contains no new information beyond HLT:

```text
HLT2 = deterministic_corruption(HLT)
```

So this cannot recover information that was already lost before HLT. The
possible gain is instead from robustness and learned test-time augmentation:

```text
stable features:
  patterns that survive HLT -> HLT2 should be trusted more

fragile features:
  patterns that disappear or flip under mild extra degradation should be
  downweighted by the fusion head

learned uncertainty:
  disagreement between HLT and HLT2 branch embeddings can expose ambiguous jets

architecture/capacity:
  the two-branch fusion model is stronger than a single ParT, so we need
  capacity controls
```

The key scientific question is:

```text
Does HLT + HLT2 fusion beat an equally large HLT + HLT same-view fusion model?
```

If yes, the second degraded view is doing something useful. If no, any gain is
probably just more capacity or ensembling.

## Data Contract

### Parent View

The parent view is the existing PD10 HLT cache:

```text
source_view = HLT
source_cache = $PD10_ROOT/hlt_cache
```

This cache is already paired with the PD10 split manifest and labels.

### Derived View

The derived view is generated from the cached HLT particles:

```text
derived_view = HLT2
HLT2 = mild_second_layer_hlt_degradation(HLT)
```

The HLT2 generation must:

```text
preserve jet identity
preserve labels
preserve split membership
never read offline particles
be deterministic for a given jet identity, split, profile, and strength
record parent HLT cache hashes in metadata
```

At deployment, the same deterministic transform is applied to the observed HLT
jet before inference.

## HLT2 Degradation Profile

Add a new profile:

```text
hlt_second_degrade_mild_v1
```

This profile is intentionally milder than the original offline-to-HLT
degradation. It is a robustness probe, not a second full detector simulation.

Recommended observable target for the primary strength:

```text
additional constituent loss relative to HLT: 3% to 8%
mean HLT2 constituent count: slightly below HLT, not collapsed
HLT2-only ParT performance: worse than HLT, but not catastrophically worse
```

Use these initial strength points:

```text
s0p10: very mild
s0p20: primary/default
s0p35: stronger stress test
```

The implementation can reuse the existing HLT corruption primitives, but the
second-layer profile should have its own metadata and defaults. Suggested target
knobs at strength 1.0 before multiplying by `s`:

```text
extra_pt_threshold:      0.08
extra_merge_prob_scale:  0.15
extra_reassign_scale:    0.08
extra_smear_scale:       0.10
extra_eff_plateau_loss:  0.010 to 0.020
extra_tail_probability:  low, optional
```

Strength scaling:

```text
s = 0.0:
  HLT2 exactly equals HLT

s = 0.10, 0.20, 0.35:
  scale all second-layer corruption sources by s
```

Do not choose the final strength using `final_test`. Use model-val diagnostics
only.

## HLT2 Cache Layout

Write new caches under:

```text
$PD10_ROOT/hlt_self_dualview/hlt2_cache/hlt_second_degrade_mild_v1_s0p20/
```

with one cache block per split:

```text
model_train
model_val
final_test
```

Each cache metadata block must include:

```text
contract: hlt_self_dualview_hlt2_cache_v1
source_view: HLT
derived_view: HLT2
allowed_inputs: HLT_only
uses_offline_particles: false
student_deployment_inputs: HLT_plus_deterministic_HLT2
source_hlt_cache_dir
source_hlt_content_hash
source_hlt_jet_identity_hash
split_manifest_hash
hlt2_profile_name
hlt2_profile_version
hlt2_strength
hlt2_seed
deterministic_by_jet_identity: true
hlt2_content_hash
hlt2_particle_count_summary
```

The audit must verify:

```text
same labels as parent HLT cache
same jet identity hash as parent HLT cache
same number of jets as the split manifest
no offline inputs referenced
finite features and Lorentz vectors
HLT2 content hash differs from HLT for s > 0
HLT2 equals HLT exactly for s = 0
particle count and drop summaries are in the expected range
```

## Main Model: HLT + HLT2 Particle Dual-View Fusion

Use the same architecture pattern as the PD10-V2 particle dual-view teacher,
but both branches are deployable HLT-derived views.

```text
HLT  particles -> HLT branch  -> h_1
HLT2 particles -> HLT2 branch -> h_2

fusion_input = concat(
  h_1,
  h_2,
  abs(h_2 - h_1),
  h_1 * h_2
)

fusion_repr =
  LayerNorm(4D)
  Linear(4D, 512)
  GELU
  Dropout(0.05)
  Linear(512, 256)
  GELU
  LayerNorm(256)

logits = Linear(256, 10)
```

Initialize both branches from:

```text
$PD10_ROOT/teachers/hlt_part_teacher_10class/best_model_val.pt
```

This is the closest deployable analogue of the successful V2 particle dual-view
teacher. The only difference is that both branches are HLT-derived.

## Training Schedule

Use the same high-data split discipline as PD10:

```text
train: model_train only
select: model_val only
final report: final_test only after selection
```

Recommended schedule:

```text
stage 1:
  freeze both ParT branches
  train fusion MLP + classifier for 1 epoch
  head_warmup_lr = 1.0e-3

stage 2:
  unfreeze all parameters
  train 8 to 12 epochs
  branch_lr = 3.0e-5
  head_lr = 3.0e-4
  weight_decay = 1.0e-4
  dropout = 0.05
  batch_size = 128 if memory allows
  AMP enabled
  grad_clip_norm = 1.0
  early_stop_patience = 3
```

Select checkpoint by:

```text
primary: model_val cross entropy
tie-break: model_val accuracy
```

If two-branch memory becomes tight, use:

```text
batch_size = 64
gradient_accumulation_steps = 2
```

## Required Runs

### Existing Anchors

Do not rerun unless missing:

```text
hlt_part_teacher_10class
pd10_student_warm_start_ce_only
pd10_student_warm_start_dual_view_full_logits_t2_a0p3
pd10_student_warm_start_dual_view_full_logits_t2_a0p5
offline_part_teacher_10class
dual_view_logit_teacher_10class
particle_dual_view_teacher_10class
```

### New Cache Runs

```text
hlt2_cache_s0p00
hlt2_cache_s0p10
hlt2_cache_s0p20
hlt2_cache_s0p35
```

The `s0p00` cache is an identity-control cache and should be cheap. It is used
to prove that the HLT2 machinery can exactly reproduce HLT.

### New Model Runs

Primary run:

```text
sdv_hlt_hlt2_s0p20
```

Controls:

```text
sdv_hlt_hlt_same_view
sdv_hlt_hlt2_s0p10
sdv_hlt_hlt2_s0p35
sdv_hlt_hlt2_s1p00
hlt2_only_part_s0p20
tta_hlt_part_hlt_plus_hlt2_s0p20
```

Optional if the main run looks promising:

```text
sdv_hlt_hlt2_s0p20_shared_branch
sdv_hlt_hlt2_s0p20_frozen_branches
sdv_hlt_hlt2_s0p20_two_hlt2_samples
```

## What Each Run Answers

```text
sdv_hlt_hlt2_s0p20:
  Does deployable HLT self-dual-view fusion beat HLT ParT and warm CE-only?

sdv_hlt_hlt_same_view:
  Is any gain just two-branch capacity and fusion?

sdv_hlt_hlt2_s0p10 / s0p35:
  Is there a useful second-degradation strength?

hlt2_only_part_s0p20:
  Is the derived view individually useful, or only useful as a contrast view?

tta_hlt_part_hlt_plus_hlt2_s0p20:
  Does cheap logit averaging get the same benefit without training a fusion
  model?

shared_branch / frozen_branches:
  Is the gain from learned fusion over stable branch features, or from full
  two-branch fine-tuning capacity?
```

## Reporting

Create a final report under:

```text
$PD10_ROOT/hlt_self_dualview/final_report/
```

Required headline comparisons:

```text
best SDV final-test accuracy
delta vs HLT ParT
delta vs warm-start CE-only
delta vs V1 warm dual-view KD
delta vs same-view dual fusion
delta vs TTA averaging
```

Required metrics:

```text
accuracy
cross entropy
macro one-vs-rest AUC
macro FPR at signal efficiency 0.30 and 0.50
validation-threshold final-test FPR
ECE
per-class precision/recall/F1
confusion matrix
binary projection metrics for the same pairs as PD10
```

The report should explicitly answer:

```text
Did HLT+HLT2 beat HLT ParT?
Did HLT+HLT2 beat warm-start CE-only?
Did HLT+HLT2 beat HLT+HLT same-view?
Did HLT+HLT2 beat cheap TTA averaging?
Which HLT2 strength was best by model-val CE?
Did the final-test winner use final-test information for selection? No.
```

## Expected Outcomes And Interpretation

### Strong Positive

```text
HLT+HLT2 beats warm CE-only and same-view dual fusion.
```

Interpretation:

```text
The generated second HLT view is exposing useful stability/uncertainty
structure. This is a deployable version of the dual-view fusion idea.
```

Next:

```text
try two or three deterministic HLT2 samples
try shared-branch and lower-compute versions
try stochastic train-time HLT2 with deterministic evaluation samples
```

### Capacity-Only Positive

```text
HLT+HLT2 beats HLT ParT, but same-view dual fusion is just as good.
```

Interpretation:

```text
The two-branch/fusion architecture helps, but the degradation view is not the
reason. Use the cheaper or cleaner capacity control.
```

### Negative

```text
HLT+HLT2 does not beat warm CE-only.
```

Interpretation:

```text
The V2 teacher gain depended on privileged offline information, not just the
dual-view architecture. Continue with adapter/privileged-teacher approaches.
```

### TTA Match

```text
TTA averaging matches trained fusion.
```

Interpretation:

```text
The effect is mostly augmentation/ensembling. Prefer the cheaper approach unless
fusion has better FPR or calibration.
```

## Scientific Caveats

HLT2 is a deterministic function of HLT, so it cannot add missing detector
information. Any gain must come from one of:

```text
regularization
learned robustness
uncertainty from view disagreement
capacity/fusion effects
test-time augmentation effects
```

That is why the same-view and TTA controls are mandatory.

Also, because inference creates HLT2 on the fly, the export/deployment metadata
must be explicit:

```text
requires_offline_inputs: false
requires_teacher_features: false
requires_deterministic_hlt2_transform: true
hlt2_profile_name: hlt_second_degrade_mild_v1
hlt2_strength: <selected strength>
```

## Implementation Steps

### Step 1: Add Plan Constants And Layout

Add a small `hlt_self_dualview` module or package with:

```text
experiment root under $PD10_ROOT/hlt_self_dualview/
variant names
HLT2 cache paths
report paths
default strength list: 0.00, 0.10, 0.20, 0.35, 1.00
```

Do not duplicate PD10 split-building logic. Point to the existing PD10 split
manifest and HLT cache.

### Step 2: Implement HLT2 Cache Builder And Audit

Create a script that reads the parent HLT cache and writes HLT2 cache blocks:

```text
scripts/build_pd10_hlt2_cache.py
scripts/audit_pd10_hlt_self_dualview_inputs.py
```

The builder must never read offline particles. The audit must verify identity,
labels, hashes, finite arrays, and degradation statistics.

### Step 3: Implement Dual-HLT Dataset And Collate

Add a dataset/loader that yields:

```text
hlt_inputs
hlt2_inputs
labels
jet_identity
```

For `sdv_hlt_hlt_same_view`, both branches should receive the parent HLT cache.
For `sdv_hlt_hlt2_*`, branch 2 receives the selected HLT2 cache.

### Step 4: Implement Deployable Dual-HLT Fusion Model

Reuse the PD10-V2 particle dual-view architecture pattern:

```text
two ParT embedding branches
concat / abs-diff / product fusion
512 -> 256 fusion MLP
10-class classifier
```

Initialize both branches from the HLT ParT teacher checkpoint.

### Step 5: Implement Training, Prediction Cache, And Metrics

Add scripts for:

```text
scripts/train_pd10_hlt_self_dualview.py
scripts/evaluate_pd10_hlt_self_dualview.py
```

Final-test prediction must only need:

```text
HLT cache
HLT2 cache or deterministic HLT2 transform
selected SDV checkpoint
```

It must not need offline caches, teacher logits, or teacher representations.

### Step 6: Implement TTA And HLT2-Only Controls

Add cheap controls:

```text
HLT2-only ParT training/eval
HLT ParT logit average over HLT and HLT2
```

These controls prevent over-interpreting a dual-fusion gain.

### Step 7: Add Report Aggregation

Create:

```text
scripts/write_pd10_hlt_self_dualview_report.py
```

The report should import existing PD10 anchors and compare all new variants
against HLT ParT, warm CE-only, V1 dual-view KD, same-view dual fusion, and TTA.

### Step 8: Add Slurm Submitter

Create:

```text
sbatch/submit_pd10_hlt_self_dualview.sh
```

Dependency graph:

```text
existing PD10 split + HLT cache
  -> HLT2 cache/audit
  -> same-view, HLT2 strengths, HLT2-only, TTA
  -> final report
```

The submitter should support:

```text
SMOKE=1 with small max_train/max_val/max_test
SKIP_EXISTING=1
OVERWRITE=0/1
PD10_ROOT override
CONDA_ENV override
```

### Step 9: Smoke Test

Run a small smoke test using the existing PD10 root:

```text
20k train / 5k val / 10k test
```

Validate:

```text
HLT2 cache audit passes
s0p00 equals HLT exactly
all model variants train for at least one epoch
final report contains all headline comparisons
final-test path is HLT-only plus deterministic HLT2
```

### Step 10: High-Data Run

Queue the full 5M/1M/1M run with primary variants:

```text
sdv_hlt_hlt_same_view
sdv_hlt_hlt2_s0p10
sdv_hlt_hlt2_s0p20
sdv_hlt_hlt2_s0p35
sdv_hlt_hlt2_s1p00
hlt2_only_part_s0p20
tta_hlt_part_hlt_plus_hlt2_s0p20
```

Only expand to optional shared-branch or multi-HLT2 variants if the first run
shows a real gain over same-view dual fusion.
