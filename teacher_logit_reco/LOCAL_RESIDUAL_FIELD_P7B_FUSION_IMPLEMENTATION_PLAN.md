# A0/P7b Multi-Level Fusion With Independent-Seed HLT Control

## Status And Objective

This is the implementation plan for the next local residual-field curriculum
experiment. It starts from the completed first-stage pilot, preserves the
existing `A0` and `P7b` checkpoints, and adds one genuinely independent HLT
Particle Transformer.

The experiment compares:

```text
method complementarity:  A0 + P7b
ordinary seed diversity: A0 + independent HLT ParT seed
```

Both pairs receive the same fusion search space, fitting data, selection rules,
metrics, and deployment audit. The search includes calibrated late fusion,
learned logit fusion, frozen-representation fusion, and event-dependent gates.

## Motivation And Existing Evidence

The completed pilot gave these exploratory final-test results on 150,000
aligned jets:

```text
A0 accuracy:                    74.9460%
P7b accuracy:                   75.3280%
G0 = mean_logits(A0, P7b):      75.8273%
```

At 50% signal efficiency, `Hgg` versus `QCD` background rejection changed from
approximately `55.35` for A0 to `57.92` for P7b. Other class projections were
mixed. This suggests useful complementarity, but a normal two-seed neural
ensemble can also create complementary errors. The seed control is therefore
mandatory.

The campaign must answer:

1. Does an independently trained HLT ParT reproduce A0-level performance?
2. Does `A0 + P7b` beat `A0 + independent HLT seed` under matched fusion?
3. Does representation-level or gated fusion beat the best late fusion?
4. Are gains broad across classes and operating points?
5. Are they worth the additional inference cost?

## Naming And Collision Avoidance

The existing LPRF ID `A1` already means a schedule-matched HLT fine-tune that
is warm-started from a baseline checkpoint. It is not an independent seed.

Use this new identity instead:

```text
run_id:              A0_seed1
display_alias:       A1_independent
canonical_config_id: A_recipe-A0-from-scratch_seed-20522
```

The user-facing phrase “A1 model” in this campaign means `A1_independent`, but
all paths, reports, and CLIs must use `A0_seed1`. The existing `A1` branch and
artifacts must remain untouched.

## Members And Groups

| Member | Meaning | Initialization | Runtime inputs |
|---|---|---|---|
| `A0` | Existing clean HLT ParT | From scratch, seed `20421` | HLT particles only |
| `A0_seed1` | New independent HLT control | From scratch, seed `20522` | HLT particles only |
| `P7b` | Existing curriculum student | Selected-consumer initialization during training | HLT particles only |

Primary groups:

```text
F_method = A0 + P7b
F_seed   = A0 + A0_seed1
```

Member order is meaningful. A0 is always member A; P7b or A0_seed1 is member
B. Single-member rows for all three models must appear in every report. The
existing G0 result is the reproducibility reference for
`F_method/L0_mean_logits`.

## Deployment Contract

Every primary member and fusion must declare:

```text
runtime_inputs = HLT_only
uses_true_fields = false
uses_offline_particles = false
uses_teacher_logits_at_runtime = false
deployable = true
```

P7b may record oracle consumers and teacher logits in training provenance, but
prediction, representation extraction, and fused inference may not load an
oracle checkpoint, true residual field, offline particle collection, or
teacher-logit cache.

A deployed representation fuser may contain only frozen member checkpoint
hashes, its feature-extraction contract, stack-train normalization, fusion-head
parameters, and stack-train calibration. It must load successfully when all
oracle artifacts are unavailable.

## Data And Leakage Contract

Use the exact split manifest and HLT-cache hashes from the completed pilot:

```text
model_train  -> train A0_seed1
model_val    -> select the A0_seed1 checkpoint only
stack_train  -> fit fusion parameters and normalization
stack_val    -> select fusion family and hyperparameters
final_test   -> evaluate only after selected_fusion.json is frozen
```

Forbidden operations:

- Fitting temperatures, weights, normalizers, heads, or gates on stack-val.
  Stack-val may define a frozen deployment operating threshold after the model
  is fitted; that threshold is not a trainable model parameter and must not be
  optimized against final-test.
- Loading final-test from a candidate-fitting or selection process.
- Selecting a method because its final-test result is attractive.
- Combining different content hashes or jet-identity orderings.
- Giving the method and seed groups different candidate grids.

All blocks for a split must agree on jet count, labels, identity sequence,
source-manifest hash, HLT-content hash, HLT profile, and degradation strength.
Cross-split identities must be disjoint.

### Current final-test status

The existing final-test has already been inspected for A0, P7b, and G0. It is
valid for an explicitly exploratory comparison of predeclared new methods, but
it is no longer a pristine unseen test for a publication-level claim. Write:

```text
test_status = exploratory_previously_partially_opened
```

If unused source jets exist, create an immutable `confirmation_test` before
viewing new fusion outputs. A publication-level conclusion should use that
split once. Lack of a new split does not block the engineering pilot, but must
limit the claim in the report.

## A0_seed1 Training Contract

`A0_seed1` is an independent rerun of the exact A0 recipe. Only these semantic
changes are permitted:

```text
run_id and canonical_config_id
output paths
training seed: 20421 -> 20522
dataloader seeds derived from the new training seed
```

Architecture, model size, HLT-only field source, splits, jet caps, batch size,
gradient accumulation, epoch budget, early stopping, optimizer, schedule,
learning rates, weight decay, dropout, AMP policy, checkpoint-selection metric,
class order, manifest, HLT profile, and cache hashes must match A0.

It must use:

```text
baseline_checkpoint = null
require_baseline_warm_start = false
teacher_logits = null
target fields unavailable to the model
```

Do not route this job through the current `A1)` shell branch because it calls
`require_baseline_checkpoint`.

Before submission, write `a0_seed_recipe_audit.json`, a field-by-field diff
between A0's stored config and A0_seed1. Fail on every non-allowlisted change.
Also record checkpoint hashes, source commits and dirty-status hashes, training
module content hashes, data hashes, seeds, and deterministic-backend settings.

If training code materially changed after A0, run A0_seed1 from the original
source commit or retrain a matched `A0_seed0_recheck`. Never silently compare
different training recipes.

## Frozen Member And Feature Caches

All members run in eval mode with every parameter frozen, dropout disabled, and
no stochastic field corruption. Cache, for every member and split:

```text
logits             float32 [N, 10]
probabilities      float32 [N, 10]
jet_embedding      float16 or float32 [N, D]
labels             int64 [N]
jet_file_indices   int32 [N]
jet_entries        int64 [N]
```

Keep logits/probabilities in the existing LPRF prediction-block format. Store
representations separately so existing consumers remain compatible.

The canonical representation is the final pooled ParT class embedding
immediately before the classifier head:

```text
representation_name = pre_classifier_cls_embedding
```

For P7b this comes from its deployable augmented student after its HLT-derived
residual-field predictor runs. For A0 and A0_seed1 it is the corresponding
HLT-only ParT embedding.

The Weaver wrapper currently returns logits only. Add a narrow adapter that
captures classifier-head input without modifying upstream Weaver. It must:

- Locate the head structurally rather than guess a tensor.
- Assert a rank-2 `[batch, embedding_dim]` result in a one-batch preflight.
- Assert logits with and without capture agree within tolerance.
- Remove hooks after every forward pass.
- Fail closed for an unsupported Weaver layout.
- Never use logits as a mislabeled embedding fallback.

If member dimensions differ, learn stack-train-only projections to a common
dimension before difference/product features. Representation metadata must
bind the file to member checkpoint, prediction-content, jet-identity, labels,
manifest, and HLT-content hashes plus the extraction contract and runtime flags.

## Predeclared Fusion Search Space

Adding methods after stack-val inspection requires a new campaign ID.

### Late and calibrated methods

| ID | Method | Definition |
|---|---|---|
| `L0_mean_logits` | Uniform logit mean | `(z_A + z_B) / 2` |
| `L1_mean_probs` | Uniform probability mean | `(softmax(z_A) + softmax(z_B)) / 2` |
| `L2_temp_mean_logits` | Calibrated logit mean | Mean of per-member temperature-scaled logits |
| `L3_scalar_simplex_logits` | Scalar convex mixture | `w z_A + (1-w) z_B` |
| `L4_classwise_simplex_logits` | Per-class convex mixture | `w_c z_A,c + (1-w_c) z_B,c` |
| `L5_linear_stacker` | Multiclass linear stacker | Centered logits and/or probabilities |

Fit temperatures and mixture parameters on stack-train by cross-entropy. Pick
regularization and feature mode on stack-val from these fixed grids:

```text
temperature bounds: [0.25, 5.0]
scalar weight grid:  0.00, 0.01, ..., 1.00
classwise L2:        0, 1e-4, 1e-3, 1e-2, 1e-1
stacker features:    logits, probabilities, logits+probabilities
stacker C:           1e-4, 1e-3, 1e-2, 1e-1, 1, 10
```

L0 on F_method must reproduce existing G0 stack metrics within tolerance.

### Frozen-representation methods

Let `h_A`, `h_B` be normalized embeddings and `z_A`, `z_B` be logits. Define:

```text
x = [h_A, h_B, abs(h_A - h_B), h_A * h_B, z_A, z_B, z_B - z_A]
```

| ID | Method | Output |
|---|---|---|
| `R0_linear_embeddings` | Regularized linear interaction head | Ten logits |
| `R1_mlp_embeddings_logits` | LayerNorm plus one hidden GELU layer | Ten logits |
| `R2_scalar_event_gate` | Event-dependent scalar gate | `g(x) z_A + (1-g(x)) z_B` |
| `R3_classwise_event_gate` | Event-dependent class gate | `g_c(x) z_A,c + (1-g_c(x)) z_B,c` |
| `R4_A0_anchored_residual` | Small correction anchored to A0 | `z_A + delta_z(x)` |

Initial limits:

```text
hidden widths:             64, 128
dropout:                   0.0, 0.1
weight decay:              1e-5, 1e-4
maximum hidden layers:     1
maximum fusion parameters: 1,000,000
```

R2/R3 record gate distributions and entropy. R4 records correction norms and
the fraction of A0 predictions changed. Backbones stay frozen; any backbone
gradient or optimizer state invalidates the run.

### Deferred methods

Particle-token cross-attention, backbone unfreezing, end-to-end dual-backbone
training, and weight interpolation are not in the first pass. Token fusion is
unlocked only if representation fusion beats late fusion by at least `0.1`
percentage points on stack-val or materially improves the rejection objective
without harming accuracy. Weight interpolation is a poor initial fit because
A0 and P7b have different deployable computation graphs and input projections.

## Fit, Screening, And Selection

Fit every candidate for both groups independently with identical code and
grids. Normalization, calibration, weights, and learned parameters come from
stack-train. Early stopping and hyperparameter choice may use stack-val. No
candidate-fitting function may accept a final-test path.

Representation candidates use screening seed `5101`. Form the union of the
best two representation families from each group, then rerun every family in
that union for both groups with seeds `5102` and `5103`. This keeps the stability
budget symmetric even when the initial group rankings differ. Ranking uses mean
stack-val performance over the three heads and records the variance. The final
artifact averages the three cheap head outputs unless the selector chooses a
deterministic linear method.

### Predeclared champions

Freeze two choices before final evaluation.

`accuracy_champion`:

1. Highest mean stack-val ten-class accuracy.
2. Differences below `0.05` percentage points are ties.
3. Break ties with lower cross-entropy.
4. Then prefer fewer trainable fusion parameters.

`rejection_champion`:

1. Stack-val accuracy may be at most `0.10` percentage points below the best
   raw member in that group.
2. Minimize mean log FPR at 50% signal efficiency over `QCD` versus `Hgg`,
   `Hbb`, `Tbqq`, `Wqq`, and `Zqq`.
3. Rank with Jeffreys-smoothed FPR `(false_positives + 0.5)/(N_QCD + 1)` so
   zero-count projections remain finite.
4. Break ties with lower cross-entropy, then lower complexity.

If both rules choose the same artifact, evaluate it once. Otherwise both
predeclared champions may be evaluated and neither may be hidden after the
final result is known.

Write `selected_fusion.json` atomically before final data is opened. It stores:

```text
campaign ID and timestamp
selection_source = stack_val
champion role
group, member IDs, and checkpoint hashes
method ID and complete hyperparameters
fit artifact and candidate-registry hashes
selection metrics and tie-break trace
final-test status
```

### Matched-control views

The report must provide three comparisons:

1. **Family matched:** F_method versus F_seed for each identical family.
2. **Best achievable:** each group's champion under the identical search budget.
3. **Recipe replay:** refit the F_method winner's frozen method/hyperparameters
   on F_seed stack-train data without another hyperparameter search.

Recipe replay is the strictest check of whether P7b contributes more useful
diversity than an ordinary second HLT seed.

## Metrics

### Multiclass metrics

Report accuracy, cross-entropy, macro one-vs-rest AUC, macro and per-class
accuracy, confusion matrix, expected calibration error, and Brier score.

### QCD background rejection

Use the repository's established one-vs-one score:

```text
score(signal vs QCD) = logit_signal - logit_QCD
```

Report 30% and 50% signal-efficiency points for all nine non-QCD classes, with
headline rows for Hgg, Hbb, Tbqq, Wqq, and Zqq. Every row includes target and
realized signal efficiency, threshold, QCD false-positive count and support,
FPR, rejection, and a finite-sample interval.

Keep two threshold conventions distinct:

- `within_split_matched_efficiency`: derive threshold on the evaluated split;
  this is an ROC diagnostic.
- `stack_val_frozen_threshold`: freeze threshold on stack-val and apply it to
  final data; this is a deployable operating point.

Never show rejection without the false-positive count. Replace an infinite
point estimate with a sample-size-defined lower bound in human-facing reports.

### Complementarity and uncertainty

For each pair report disagreement rate, both-correct/A-only/B-only/both-wrong
counts, error-overlap Jaccard index, per-class disagreement, logit/probability
correlations, gain on A0-error events, and loss on A0-correct events.

Use deterministic paired resampling:

- At least 1,000 paired bootstrap replicates for accuracy and CE deltas.
- Stratified paired bootstrap intervals for FPR/rejection comparisons.
- Bootstrap seed and sampled-index hash in provenance.
- Exact or Wilson binomial FPR intervals when passing counts are small.

### Runtime metrics

Two models do not imply equal cost: P7b includes a deployable residual-field
predictor. Measure member and head parameters, peak GPU memory, median/p95 batch
latency, jets per second, and artifact sizes. Report improvement per added
millisecond relative to A0.

## Artifact Layout

Never write into the completed curriculum pilot directory:

```text
checkpoints/local_particle_residual_field_fusion/
  p7b_seed_control_<campaign_id>/
    campaign_manifest.json
    source_artifact_audit.json
    a0_seed_recipe_audit.json
    members/{A0,A0_seed1,P7b}.json
    taggers/A0_seed1/
    predictions/{A0,A0_seed1,P7b}/
    representations/{A0,A0_seed1,P7b}/
    candidates/F_method/<candidate_id>/
    candidates/F_seed/<candidate_id>/
    selection/candidate_registry.json
    selection/selected_fusion.json
    final_evaluation/
    final_report/
```

Final-report outputs:

```text
summary.md
run_report.json
provenance_audit.json
member_metrics.csv
fusion_candidate_stack_val.csv
selected_fusion_metrics.csv
paired_group_comparison.csv
binary_rejection.csv
complementarity.csv
runtime_metrics.csv
bootstrap_intervals.csv
```

Candidate and final artifacts are immutable by default. Reruns require a new
campaign ID or an explicit debug-only overwrite flag.

## Implementation Steps

### Step 1: Contracts and candidate registry

Add `local_particle_residual_field/fusion_campaign.py` with validated
`FusionCampaignConfig`, `FusionMemberSpec`, `FusionGroupSpec`,
`FusionCandidateSpec`, and `SelectedFusionArtifact`. Lock the members, groups,
candidate registry, split policy, deployment flags, and names. Reject legacy
`A1` as the independent control.

### Step 2: Independent HLT seed recipe

Add `A0_seed1` as a from-scratch `hlt_only` recipe. Prefer a new recipe branch
over weakening A1's warm-start guard. Implement the A0 config-diff audit and a
narrow Slurm wrapper for this control.

### Step 3: Source-artifact preflight

Add a read-only command that resolves and hashes the A0 checkpoint/report, P7b
deployable checkpoint/report, P7b's `selected_consumer.json`, split manifest,
HLT-cache manifest, and reusable predictions. Prove P7b loads without oracle
runtime resources. Write `source_artifact_audit.json` and block GPU jobs on
failure.

### Step 4: Prediction caching

Reuse `LocalResidualFieldPredictionConfig`. Cache A0_seed1 predictions and
reuse A0/P7b only after hash validation. Add binary-projection metrics to
prediction metadata. Development commands request stack-train and stack-val
only.

### Step 5: Representation extraction

Prefer these new files:

```text
teacher_logit_reco/local_particle_residual_field/fusion_features.py
scripts/cache_local_residual_field_fusion_features.py
```

Reuse hash-attested representation-cache patterns from
`constrained_coarse_to_fine/evaluation.py` and `posthoc.py`, with the stricter
LPRF deployment metadata above.

### Step 6: Shared metrics

Factor multiclass, calibration, complementarity, and binary-projection metrics
into a shared LPRF helper. Implement within-split and frozen-threshold rejection
with counts and intervals. Reproduce current A0/P7b final accuracy and the
manually verified QCD projections before trusting fusion results.

### Step 7: Late fusion

Extend the LPRF fuser or add a campaign-specific implementation for L0-L5.
Reuse tested pieces from `jetclass_fresh/fusion.py` and
`jetclass_fresh/independent_fusion.py` only where split behavior matches.

Do not call `fit_group_stacker` unchanged during development: it eagerly loads
and evaluates final-test. Refactor fitting, selection, and evaluation to accept
explicit split lists.

### Step 8: Representation and gating heads

Prefer:

```text
teacher_logit_reco/local_particle_residual_field/fusion_models.py
teacher_logit_reco/local_particle_residual_field/fusion_train.py
```

Implement R0-R4 over cached frozen features. Checkpoints store member/cache
hashes, normalization, candidate spec, curves, selected epoch, parameter count,
and deployment flags.

### Step 9: Candidate runner and selector

Add:

```text
scripts/run_local_residual_field_fusion_campaign.py
scripts/select_local_residual_field_fusion.py
```

The runner fits one candidate/group pair and writes stack-only metrics. The
selector requires identical search coverage for both groups, applies both
selection rules, writes the atomic selection artifact, and records tie breaks.
Fail if any candidate report contains final-test metrics before selection.

### Step 10: Locked final evaluation

Add a separate command accepting only `selected_fusion.json`. It must not
accept a method ID or hyperparameter override. Verify selection, registry,
member, cache, fit-artifact, and deployment hashes before evaluation. Write a
new immutable result directory and label current final-test exploratory unless
a pristine confirmation manifest is supplied.

### Step 11: Reporting

Prefer:

```text
teacher_logit_reco/local_particle_residual_field/fusion_campaign_report.py
scripts/write_local_residual_field_fusion_campaign_report.py
```

Lead with the best raw member, G0 reproduction, best late F_method, best
representation F_method, best F_seed, method-recipe replay on F_seed, paired
uncertainty, rejection, and runtime cost.

Every fusion row needs nonempty `run_id`, `group_id`, and `candidate_id`; do not
repeat the previous `best tagger: None` presentation defect.

### Step 12: Slurm orchestration

Add:

```text
sbatch/submit_lprf_p7b_fusion_campaign.sh
sbatch/submit_lprf_p7b_fusion_campaign_tigris.sh
```

Dependency graph:

```text
source preflight
  -> A0_seed1 training
  -> A0_seed1 stack prediction/representation caches

source preflight
  -> verify/reuse A0 and P7b stack caches

all stack caches
  -> L0-L5 and R0-R4 for both groups
  -> representation stability reruns
  -> selector
  -> selected final evaluation
  -> bootstrap and runtime jobs
  -> final report
```

Supported stages:

```text
preflight
train_seed_control
cache_stack
fit_candidates
select
evaluate_final
report
full_campaign
```

`full_campaign` expresses every dependency with `afterok` and cannot queue
final evaluation without the selector.

Every Tigris job must export `PYTHONNOUSERSITE=1`, use account
`reu-aisocial`, source `sbatch/common.sh`, and activate through `fresh_setup`.
Do not source the Python `conda` entry point as a shell script. Print all
resolved paths, job IDs, and dependencies. Support `PRINT_ONLY=1` and safe
resume based on immutable completed artifacts.

## Required Tests

Create a focused series such as
`tests/test_local_residual_field_fusion_campaign_stepN.py`.

### Configuration and seed control

- Reject legacy A1 as the independent seed ID.
- A0_seed1 is HLT-only, from scratch, and has no teacher inputs.
- Recipe audit permits only identity, seed, and output changes.
- Recipe audit rejects architecture, schedule, data, and selection changes.

### Alignment and deployment

- All member labels and identities align; split overlap fails.
- Checkpoint/cache hash mismatches fail.
- Oracle, true-field, or offline runtime metadata fails.
- P7b deployable loading works with oracle directories unavailable.

### Representations

- Captured embeddings have `[batch, D]` shape and leave logits unchanged.
- Unsupported classifier layouts fail instead of returning a fallback.
- Representation hashes bind to predictions and checkpoints.
- Fusion training gives no backbone gradients or optimizer state.

### Fusion and leakage

- Uniform logit/probability arithmetic matches direct calculation.
- Scalar/classwise weights are convex.
- Temperature, normalization, and weights use stack-train only.
- Hyperparameter selection uses stack-val only.
- Candidate fitting cannot open final-test.
- Both groups have identical candidate registries.
- Gates are finite, bounded, and correctly shaped.
- A zero R4 correction exactly reproduces A0 logits.

### Selection, metrics, and reports

- Both selectors reproduce deterministic toy cases.
- Zero-background ranking remains finite.
- Final evaluation requires a valid selection artifact and rejects overrides.
- Binary scores equal `logit_signal - logit_QCD`.
- FPR, counts, thresholds, and rejection match hand calculations.
- Stack-val thresholds are unchanged on final application.
- Paired bootstrap preserves alignment and is deterministic.
- Final rows are HLT-only/deployable and have stable IDs.
- Reports show method and seed controls and retain class regressions.

### Slurm

- Tigris account is exactly `reu-aisocial`.
- `PYTHONNOUSERSITE=1` is exported.
- Final evaluation depends on selection.
- Failed preflight/training prevents downstream jobs.
- Print-only displays the complete graph without submission.

## Acceptance And Interpretation Gates

Engineering completion requires passing provenance/alignment audits, a valid
A0_seed1 recipe audit, reproduction of raw metrics and G0, oracle-independent
selected artifacts, no final-test data in candidate reports, and an immutable
complete final report.

Interpret results as follows:

1. If F_method does not beat F_seed, G0 is consistent with ordinary ensembling.
2. If F_method beats F_seed across matched late methods, P7b likely adds
   specialized diversity.
3. If representation/gated fusion beats the best late method and its seed
   control, earlier fusion is justified.
4. If representation gain is below `0.1` percentage points and rejection does
   not improve, retain simpler late fusion.
5. A gain at only one rejection point is operating-point-specific, not general.
6. Final claims include uncertainty and latency, not accuracy alone.

The strongest positive outcome is:

```text
best(A0 + P7b, identical search budget)
  > best(A0 + A0_seed1, identical search budget)
  > A0
```

## Recommended Milestones

```text
Milestone 1:
  A0_seed1 audit/training/predictions, binary metrics,
  L0-L5 matched late-fusion comparison

Milestone 2:
  representation caches, R0-R4 heads, matched controls, selector

Milestone 3:
  locked final evaluation, bootstrap/runtime jobs, final report,
  optional pristine confirmation split
```

Milestone 1 is valuable on its own and should finish before token-level or
end-to-end fusion work begins.
