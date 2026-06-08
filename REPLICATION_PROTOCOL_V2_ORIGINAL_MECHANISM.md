# JetClass Same-HLT Reconstructor-Stack Replication Protocol V2

## Purpose

This document is a stricter replication protocol for the JetClass same-HLT reconstructor-stack result. The first fresh-start implementation was useful as an independence check, but it did not reproduce the original mechanism closely enough. It implemented a simplified reconstructor and a different dual-view tagger. Its low stacked-logistic-regression result is therefore not a decisive falsification of the original result.

V2 has a narrower target: reproduce the original mechanism as closely as possible while keeping the cleaner split discipline from the fresh-start check.

The goal is not to copy checkpoint names, old job IDs, or exact implementation quirks. The goal is to recreate the same *algorithmic structure*:

- one fixed HLT corruption profile shared by every model;
- an m2-style operation-aware HLT-to-offline reconstructor;
- a soft corrected view built from the reconstructor outputs;
- a dual-view tagger that sees both the original HLT view and the reconstructed/corrected view;
- seven reconstructor-behavior variants trained on the same data and same fixed HLT;
- a stacked logistic regression meta-classifier trained only on a held-out stacking split;
- strict audits showing no label leakage, no offline-test leakage, no split overlap, and no per-model HLT drift.

The expected outcome is not necessarily exactly the old `0.802` test accuracy, because this protocol uses a stricter five-way split and a 7-model set rather than the old full 12-model set. The expected outcome is that the seven m2-hybrid reconstructor-diverse models plus HLT should be clearly stronger than the HLT-only seed ensemble under the same split and fixed HLT.

If V2 still only reaches HLT-ensemble performance, then the original result needs a deeper audit. If V2 recovers a large gain, then the first fresh-start failure was likely due to an incomplete reproduction of the original mechanism.

## Non-Negotiable Experimental Constraints

### Fixed HLT Must Be Shared Across All Models

Generate the HLT-like view exactly once per split and cache it. Every model must consume the same cached HLT arrays. Do not regenerate HLT inside individual training jobs. Do not vary HLT parameters across variants. Do not vary HLT random seeds across variants.

Use the fixed HLT profile already specified in `jetclass_fixed_hlt.py`:

- `hlt_pt_threshold = 1.30`
- `merge_prob_scale = 1.35`
- `reassign_scale = 1.00`
- `smear_scale = 1.00`
- `eff_plateau_barrel = 0.99`
- `eff_plateau_endcap = 0.97`
- `eff_turnon_pt = 1.40`
- `eff_width_pt = 0.20`

The HLT cache should record, for every split:

- exact parameter values;
- HLT seed used for that split;
- content hash of the resulting HLT tokens and masks;
- source split manifest hash;
- number of jets;
- class counts;
- mean offline constituents and mean HLT constituents;
- decomposition of constituent loss into threshold, merge, and efficiency components if available.

All seven reconstructor variants and the HLT baseline must have identical HLT cache hashes for the same split.

### Use the Stricter Five-Way Split

Keep the stricter fresh-start split. Do not collapse it back to the original val/test-only stack protocol.

Use these split roles:

- `model_train = 500000` jets
- `model_val = 150000` jets
- `stack_train = 250000` jets
- `stack_val = 50000` jets
- `final_test = 500000` jets

These splits have different purposes and must not be interchanged:

- `model_train` is used to train offline teacher, HLT baseline, Stage-A reconstructors, and Stage2 dual-view taggers.
- `model_val` is used for model checkpoint selection and early stopping.
- `stack_train` is used to train the logistic-regression stacker.
- `stack_val` is used to select stacker hyperparameters and choose fusion method.
- `final_test` is locked until the final report and audits.

The final test labels must not influence any model selection, early stopping, fusion-weight selection, C selection, feature engineering, threshold selection, or debugging decision. If final-test performance is inspected during development, the run should be marked as exploratory and not as the locked final result.

### Offline Information Is Allowed Only In Training Targets And Evaluation

For real deployment, only HLT-like inputs are available. In this experiment, offline constituents and labels are available for supervised training and evaluation. The rule is:

- Offline constituents may be used as Stage-A reconstruction targets on `model_train` and `model_val`.
- Offline labels may be used to train taggers on `model_train`, select checkpoints on `model_val`, train stacker on `stack_train`, select stacker C/method on `stack_val`, and evaluate on `final_test`.
- Offline constituents must not be used inside the inference path for `stack_train`, `stack_val`, or `final_test` predictions.
- Offline teacher logits must not be included as stacker features unless a clearly separate teacher-distillation experiment is being run. For this V2 replication, do not include teacher logits in the fusion features.
- Final-test offline labels must only be read during final metric computation and final audits.

During inference, every base model should take only:

- cached HLT tokens;
- cached HLT mask;
- model weights/checkpoint;
- fixed preprocessing constants learned or specified before final-test evaluation.

If any prediction path reads offline constituents for `stack_train`, `stack_val`, or `final_test`, that is leakage.

## Original Mechanism To Reproduce

The first fresh implementation diverged too much from the original architecture. For V2, implement the original mechanism, not a simplified substitute.

The original mechanism has three tightly coupled pieces:

1. An operation-aware hybrid reconstructor.
2. A soft corrected-view builder.
3. A dual-view cross-attention tagger.

These should be treated as one system. Reproducing only the high-level idea of “HLT to offline reconstructor” is not enough.

## Part 1: Operation-Aware m2-Hybrid Reconstructor

### High-Level Behavior

The reconstructor receives an HLT-like constituent set and predicts a soft corrected representation. It does not produce a single fixed number of hard particles. It produces a candidate set with soft weights and several operation branches.

The branches are:

1. **Edit / unsmear branch**

   This branch corrects existing HLT constituents. It predicts token-level kinematic deltas for each HLT constituent. It is responsible for correcting pT, eta, phi, and energy without creating new particles.

2. **Split / unmerge branch**

   This branch treats an HLT constituent as a possible merged product and proposes split children. The original m2-hybrid style allows the parent to be redistributed into children. It includes parent-level split probability, child existence probabilities, child kinematic deltas, and parent uplift terms.

3. **Generator branch**

   This branch predicts missing constituents from a jet-level latent context. It should use learned generator queries and attention to the encoded HLT tokens. It should not be a simple unconditional MLP from the global pooled vector. The generator produces candidate missing tokens and soft existence weights.

4. **Budget heads**

   The reconstructor also predicts jet-level budget quantities, including total count and added count. These are used to calibrate soft candidate weights and avoid arbitrary over-generation.

### Required Architecture Details

The reconstructor should use a transformer-style token encoder, not only an MLP over token features.

Required components:

- input projection from per-token HLT features into an embedding dimension;
- relative-position bias or relative-position-aware token attention using eta/phi/dR relationships;
- multiple token encoder layers;
- layer normalization after token encoding;
- token-level edit heads;
- token-level split heads;
- attention-based jet-level pooling;
- attention-based generator decoder using learned generator queries;
- budget head from pooled jet context;
- output dictionary exposing all fields needed by the corrected-view builder and losses.

Recommended dimensions for faithful reproduction:

- token embedding dimension around `128`;
- number of attention heads around `8`;
- number of token encoder layers around `6`;
- feed-forward dimension around `512`;
- dropout around `0.1`;
- maximum generated tokens default around `56`, with variant-specific changes;
- maximum split children per parent around `2`.

Do not replace this with a single small MLP over raw tokens plus average pooling. That was a major divergence in the first fresh implementation.

### Input Representation

The reconstructor should receive both processed HLT features and raw HLT kinematics.

The original mechanism effectively separates:

- features used for token encoding;
- raw constituent four-vector-like quantities used to construct corrected candidate tokens.

Use raw HLT constituent fields at least:

- pT;
- eta;
- phi;
- energy.

Use feature preprocessing consistent across all models. If using canonical preprocessing, record the exact transform. Do not compute preprocessing constants from stack or test splits.

For token encoding, include enough features to reproduce the original behavior. The exact feature vector may include normalized kinematics and particle ID/track information, but the key requirement is that the same feature representation is used for every variant and every split.

### Edit / Unsmear Branch

For each HLT token, the edit branch should predict bounded deltas:

- log-pT delta;
- eta delta;
- phi delta;
- log-energy delta.

These deltas should be bounded by smooth nonlinearities such as tanh so that Stage-A does not explode numerically. The old mechanism used different scales for kinematic deltas. The implementation does not need to exactly duplicate constants, but should keep the same qualitative behavior:

- pT and energy deltas should be multiplicative in log space;
- eta and phi deltas should be additive and bounded;
- phi must be wrapped to the standard angular range;
- energy should be floored to be physically compatible with pT and eta.

The edit branch should also predict a token-existence or token-weight signal. The edited token weight is a soft confidence for retaining/correcting the original HLT constituent.

### Split / Unmerge Branch

The split branch is essential. Do not omit it. Do not reduce it to a generic generated-token branch.

For each HLT token, predict:

- parent split probability;
- child existence probabilities;
- child kinematic deltas;
- parent pT/energy uplift terms;
- split child tokens;
- split child soft weights;
- split-parent added support used later by corrected-view construction.

The parent split probability should reduce the keep/edit branch weight when splitting is high. The model should be able to choose between “keep/edit this parent” and “split this parent into children.”

The parent uplift terms matter. They let split products recover under-measured parent pT and energy before distributing to children. This is one of the differences between a generic soft generator and the original operation-aware setup.

Split children should be local to the parent but not hard-constrained to exactly conserve the original parent. The original m2-hybrid philosophy allowed split products to recover missing pT/energy rather than being forced to sum exactly to the HLT parent.

### Generator Branch

The generator branch should use learned generator queries and cross-attention to the HLT token encoding.

Required behavior:

- learn a bank of generator query embeddings;
- attend from generator queries to encoded HLT tokens;
- decode generator tokens from the attended generator states;
- predict generator soft existence weights;
- use a jet-level budget prediction to calibrate the total generator weight.

Generator token kinematics should be physically safe:

- pT positive;
- eta bounded;
- phi wrapped;
- energy positive and at least compatible with pT and eta.

The generator should be able to predict missing constituents while remaining tied to the HLT jet context. It should not use offline targets at inference.

### Budget Calibration

The reconstructor should not hard-code a fixed number of generated or split candidates. It should produce a large candidate set with soft weights, then calibrate those weights with budget heads.

Budget behavior should include:

- predicted total count;
- predicted added count;
- allocation between split-added support and generator-added support;
- scaling of token weights so the soft total count is calibrated;
- sparse penalties to discourage gratuitous candidate mass;
- optional separate sparse penalties for split and generator branches.

The important story is:

- The model proposes edit, split, and generate candidates.
- It predicts soft weights for those candidates.
- Count/budget losses and sparsity losses discourage arbitrary over-creation.
- The corrected view uses soft support, not a hard top-k oracle.

### Output Dictionary

The reconstructor should expose a structured output object or dictionary with at least these conceptual fields:

- candidate tokens for all edit/split/generated candidates;
- candidate weights;
- edited token candidates;
- edited token weights;
- split child candidates;
- split child weights;
- generated candidates;
- generated weights;
- per-parent split-added support;
- generator-to-parent assignment weights;
- budget total prediction;
- budget added prediction;
- auxiliary budget/efficiency signal if needed;
- action probabilities or compatibility fields if needed by the training/evaluation code.

The exact names can differ, but the downstream corrected-view builder must have the same information.

## Part 2: Soft Corrected View Builder

The corrected view is not simply “all predicted candidate particles.” In the original mechanism, the dual-view tagger primarily receives a parent-token-aligned corrected view with added support features.

This is an important distinction. The first fresh implementation built a corrected ParticleTransformer input from all candidates. That is not equivalent.

The original corrected view behaves roughly as follows:

1. Take edited token candidates corresponding to original HLT parent tokens.
2. Use their soft token weights to define which parent tokens remain active.
3. Compute normal per-token features from the edited parent tokens.
4. Scale features by token weight if configured.
5. Add extra channels describing reconstruction confidence and added support.
6. Include split-added support per parent.
7. Include generator-added support assigned back to parent tokens.
8. Include a smooth budget/efficiency share signal.

The corrected-view token feature dimension in the original m2-hybrid path was effectively:

- base 7 computed token features;
- plus token weight;
- plus parent-added support;
- plus efficiency/budget share.

So the corrected view has about 10 features per parent token. It is not a full 14-feature raw ParticleTransformer input over all candidates.

The V2 fresh implementation should reproduce this logic closely.

### Corrected View Details

For each parent token:

- compute corrected kinematics from the edit branch;
- compute token weight;
- compute parent-added support from split children;
- compute generator-added support by assigning generator weights back to parents using generator-to-parent attention;
- combine split-added and generator-added support into a parent-added scalar;
- compute an efficiency/budget share scalar;
- concatenate these support values with the corrected token features.

The corrected-view mask should be based on edited token weights exceeding a small floor. If all tokens are masked out for a jet, force at least one token valid to avoid empty-sequence failures.

The corrected view should be deterministic at evaluation time.

Do not use offline constituents when building corrected views for stack or final-test predictions.

## Part 3: Dual-View Tagger

The original dual-view tagger is not just two independent taggers averaged together. It has a specific fusion structure.

Reproduce a model with these behaviors:

- HLT branch consumes the original HLT feature sequence.
- Reconstructed branch consumes the soft corrected-view feature sequence.
- Each branch has its own input projection and transformer encoder.
- Each branch gets pooled with attention pooling.
- Include cross-attention from pooled HLT context into corrected-view tokens.
- Include cross-attention from pooled corrected-view context into HLT tokens.
- Concatenate pooled HLT, pooled corrected, HLT-to-corrected cross context, and corrected-to-HLT cross context.
- Use layer normalization and an MLP classifier head.

This cross-attention fusion is part of the original mechanism. A two-branch ParticleTransformer with only CLS concatenation is not the same mechanism.

Recommended architecture:

- embedding dimension around `128`;
- number of heads around `8`;
- number of layers around `6` per branch;
- feed-forward dimension around `512`;
- dropout around `0.1`;
- attention pooling query shared or separate;
- 4-context concatenation into an MLP head.

The dual-view tagger should be trained on labels using HLT inputs and reconstructed-view inputs. It must not access offline constituents during inference.

## Part 4: Training Stages

The training schedule should be close to the original, while still fitting the fresh-run compute budget.

### Stage 1: Offline Teacher And HLT Baselines

Train an offline teacher on offline constituents. This is mainly a reference upper bound. For this V2 run, do not use teacher logits as stacker features.

Train an HLT baseline on cached fixed-HLT constituents. Also train five HLT seed-control models if the HLT5 control is part of the same report.

The HLT5 control is important because it answers whether stacked logistic regression alone explains the gain. It should remain in the report.

### Stage A: Reconstructor Pretraining

For each of the seven variants, train the operation-aware reconstructor to map fixed-HLT inputs to offline constituent targets.

Use only `model_train` for training and `model_val` for checkpoint selection.

Stage-A losses should include:

- set matching loss between predicted candidates and offline targets;
- budget/count calibration loss;
- pT response loss;
- energy response loss;
- mass response loss;
- local consistency loss for split/generated candidates;
- sparsity penalty;
- optional split/generator sparse penalties;
- optional false-positive mass or anti-overlap penalty depending on variant.

The matching loss should be faithful to the original m2-hybrid behavior. Hungarian matching is preferred if feasible. If using Chamfer for speed, it must be noted as an ablation and not treated as the primary reproduction.

The loss must be numerically safe:

- sanitize NaN and Inf candidate tokens before matching;
- replace invalid costs with a large finite cost;
- avoid empty-match NaN means;
- track a non-finite penalty or diagnostic;
- clamp pT/energy/eta in physically safe ways.

Do not silently drop failed jets without recording the count.

### Stage B: Dual-View Tagger Training

Train the dual-view tagger using the frozen Stage-A reconstructor output.

Inputs:

- original fixed-HLT view;
- soft corrected view from the frozen reconstructor.

Targets:

- class labels.

Train on `model_train`, select checkpoint on `model_val`.

The classifier checkpoint saved for fusion should be the best model-val dual-view classifier checkpoint.

### Stage C: Optional Joint Fine-Tuning

The original training script had StageC joint fine-tuning capability. For this V2 replication, decide explicitly whether to include it.

If the goal is to reproduce the old `stage2` fusion exactly, then use the Stage2 pre-joint checkpoint for fusion and keep StageC optional.

If StageC is used, report both:

- Stage2 pre-joint checkpoint performance;
- StageC joint checkpoint performance.

Do not accidentally fuse a different checkpoint type than intended. Every fusion model spec must say whether it loads Stage2 or Joint.

For direct comparison to the old same-HLT 12-model result, fusing Stage2 dual-view checkpoints is the primary target.

## Part 5: Seven Model Variants

Keep seven models for this V2 run. They should be the same conceptual subset used in the same-HLT audit:

1. `m2_base`
2. `m2_consstrong`
3. `m2_budgetlite`
4. `m2_genlow`
5. `m2_genhigh`
6. `m2_topk60ish`
7. `m2_antioverlap`

Every variant must use:

- same file split;
- same HLT cache;
- same fixed HLT profile;
- same base architecture;
- same training seed unless the variant definition explicitly changes only the variant behavior;
- same model_train/model_val split roles;
- same stack_train/stack_val/final_test split roles.

The only differences should be reconstruction/loss/capacity knobs.

### Variant 1: `m2_base`

Purpose: baseline m2-hybrid operation-aware reconstructor.

Use default generated-token capacity around 56. Use default budget, sparsity, local, pT, mass, and energy losses. This is the reference model.

### Variant 2: `m2_consstrong`

Purpose: stronger reconstruction consistency and budget calibration.

Increase consistency pressure relative to base. This can include:

- larger budget/count loss;
- stronger pT response loss;
- stronger mass/energy response losses;
- slightly stronger local consistency;
- possibly slightly stronger dual-view consistency if the training framework includes it.

This model should learn a more globally calibrated corrected view.

### Variant 3: `m2_budgetlite`

Purpose: reduce over-constraint from count/budget matching.

Lower budget/count pressure. Keep enough sparsity to avoid unlimited candidate creation. Increase or preserve pT response pressure so the model can prioritize useful energy-scale corrections over exact token-count recovery.

This model tests whether relaxed budget matching produces a complementary classification view.

### Variant 4: `m2_genlow`

Purpose: reduce generation capacity and make generated additions sparse.

Lower maximum generated tokens, for example around 40. Increase generation sparsity modestly. Keep split/edit branches unchanged. This variant should emphasize correcting/splitting existing HLT structure rather than creating many new candidates.

### Variant 5: `m2_genhigh`

Purpose: increase generation capacity and allow more missing-token support.

Increase maximum generated tokens, for example around 72. Relax generation sparsity. Keep budget calibration active so the model cannot simply flood the corrected view. This variant should provide a complementary high-recall missing-constituent view.

### Variant 6: `m2_topk60ish`

Purpose: intermediate generated-token capacity.

Use maximum generated tokens around 60. Use moderate budget and sparsity. This variant sits between genlow and genhigh and was useful in the original family.

### Variant 7: `m2_antioverlap`

Purpose: discourage redundant added support.

Increase local consistency and anti-overlap pressure. Use an anti-overlap term so split/generated candidates do not collapse into many near-identical support points. This variant should encourage cleaner local structure.

## Part 6: Fusion Protocol

The fusion protocol should be stricter than the old original script.

For every base model, collect frozen predictions on:

- `stack_train`;
- `stack_val`;
- `final_test`.

For each split, every model prediction block must include:

- logits;
- probabilities;
- labels;
- jet identity list or identity hash;
- model name;
- checkpoint path;
- HLT cache content hash;
- source split hash;
- allowed-input declaration.

Before fitting fusion, verify all prediction blocks in a split have identical labels and identical jet identities.

### Fusion Methods To Report

Report at least:

- best single HLT baseline;
- best single dual-view reconstructor model;
- HLT5 seed-control stacked result if available;
- uniform probability average over HLT plus seven dual-view models;
- weighted probability average selected on `stack_val`;
- weighted logit average selected on `stack_val`;
- stacked logistic regression trained on `stack_train` and selected on `stack_val`;
- final selected method evaluated once on `final_test`.

### Stacked Logistic Regression

The stacker input should be concatenated base-model outputs. Use one of:

- logits only;
- probabilities only;
- logits plus probabilities.

The main result should use logits plus probabilities if this matches the original audit setup. Standardize features before logistic regression.

Train logistic regression only on `stack_train`. Use `stack_val` to select C and fusion method. Do not train the final stacker on `final_test`.

Recommended C grid:

- 0.03;
- 0.1;
- 0.3;
- 1.0;
- 3.0;
- 10.0.

For final reporting, either:

- train on `stack_train` and select on `stack_val`, then evaluate `final_test`; or
- after selecting C and feature mode on `stack_val`, refit on `stack_train + stack_val` and evaluate `final_test`, but this must be clearly reported as refit-after-selection.

The cleaner default is train on `stack_train`, select on `stack_val`, evaluate `final_test`.

## Part 7: Required Leakage Audits

The V2 run must include the audits that passed in the original investigation, adapted to the stricter split.

### Source Audit

Verify fusion sources contain only allowed model checkpoints:

- HLT baseline checkpoint;
- dual-view tagger checkpoint;
- Stage-A reconstructor checkpoint.

Flag any source path or metadata containing:

- teacher logits;
- offline teacher checkpoint used as prediction feature;
- target scores;
- oracle;
- offline constituent arrays in the prediction path.

Offline reconstructor checkpoints are allowed because the reconstructor was trained using offline targets, but inference must use only HLT inputs.

### Same-HLT Compatibility Audit

For each split and each model, verify the same HLT content hash. Every base model must consume the same cached HLT view.

### Split Audit

Verify no jet identity appears in more than one split. If using file-level separation, verify no file overlap. If using jet-level hash separation, verify no `(file, entry)` overlap.

### Prediction Alignment Audit

Before fitting any fusion model, verify all prediction arrays for a given split have:

- same label order;
- same jet identity order;
- same number of rows;
- finite logits and probabilities.

### Permuted-Label Control

Shuffle `stack_train` labels, train the stacker, evaluate `final_test`. Accuracy should collapse to near random. For ten classes, near random is about 0.10.

### Row-Shuffle Control

Shuffle the rows of all features relative to labels, train the stacker, evaluate final test. Accuracy should collapse to near random.

### Input-Zero Control

Evaluate the stacked system or at least each base model on a small sample where the HLT inputs are zeroed or destroyed. Accuracy should collapse to near random. This verifies that the model predictions actually depend on HLT input and are not cached label artifacts.

### HLT5 Control

Train five HLT-only models with different training seeds on the same HLT cache. Stack them using the same stacker protocol. This should be the baseline for “ensembling alone.”

The reconstructor-diverse stack must beat HLT5 by a meaningful margin to support the paper story.

### Holdout Stack Check

Optionally split `stack_train` into two halves. Train the stacker on one half, evaluate the other half and final test. This is useful to detect stacker overfit.

## Part 8: Metrics To Report

Report at least:

- accuracy;
- macro one-vs-rest AUC;
- cross entropy;
- per-class accuracy or confusion matrix;
- best single model metrics;
- HLT baseline metrics;
- HLT5 stacked metrics;
- uniform/weighted/stacked fusion metrics;
- final-test metrics with confidence intervals if possible.

Also report reconstruction diagnostics, even if classification is the primary target:

- mean HLT constituent count;
- mean offline constituent count;
- mean corrected-view active count;
- pT response to offline;
- pT response resolution;
- jet-axis delta-R to offline;
- fraction of jets improved vs HLT for pT response and axis;
- average split support;
- average generation support;
- average token weight.

These diagnostics help tell whether the reconstructor is physically meaningful or mainly producing a useful learned representation.

## Part 9: What Would Count As A Successful V2 Replication

A successful V2 replication does not require exact old numbers. It should show:

1. HLT baseline around the expected HLT-only scale.
2. HLT5 seed ensemble gives only a modest improvement over single HLT.
3. Individual reconstructor models may be only modestly better than HLT or even comparable.
4. The seven-model reconstructor-diverse stack plus HLT is substantially better than HLT5.
5. Leakage audits pass.
6. Permuted-label and row-shuffle controls collapse to random.
7. Zero-HLT control collapses to random.
8. Same-HLT hash compatibility passes.

If V2 gets only HLT5-level performance again, then inspect:

- whether the operation-aware reconstructor was actually implemented;
- whether the corrected view matches the original parent-token-aligned support view;
- whether the dual-view tagger includes cross-attention fusion;
- whether Stage-A and StageB training ran long enough;
- whether the base models learned diverse logits;
- whether the stacker feature matrix contains meaningful diversity.

## Part 10: Common Failure Modes To Avoid

### Failure Mode: Simplified Reconstructor

Do not implement a simple MLP reconstructor and call it m2-hybrid. That was likely the main reason the first fresh attempt failed.

### Failure Mode: Candidate View Instead Of Corrected Parent View

Do not feed all edit/split/generated candidates directly into a generic ParticleTransformer and assume it matches the original corrected view. The original corrected view is parent-token aligned and includes soft support features.

### Failure Mode: Missing Cross-Attention Dual-View Fusion

Do not replace the original cross-attention dual-view tagger with two independent encoders plus concatenation unless this is explicitly reported as an ablation.

### Failure Mode: Too Short Training

Twenty epochs may be insufficient. The original mechanism used much longer Stage-A and StageB schedules. If compute is limited, run a smaller debug subset first, but do not treat a short run as a definitive failure.

### Failure Mode: Hidden HLT Drift

Do not let each variant regenerate HLT on the fly. Cache HLT once.

### Failure Mode: Stack/Test Leakage

Do not fit the stacker on final test. Do not tune C on final test. Do not inspect final test repeatedly during implementation and report the best run as locked.

### Failure Mode: Teacher Feature Leakage

Do not include offline teacher logits or offline teacher predictions as features in the stacker.

### Failure Mode: Missing Per-Split Identity Checks

Do not rely only on array lengths. Check jet identities.

## Part 11: Implementation Plan

Implement this in stages. Do not try to change everything in one prompt.

### Step 1: Verify Data Splits And HLT Cache

Confirm the existing five-way split manifest has the intended sizes:

- `model_train = 500000`
- `model_val = 150000`
- `stack_train = 250000`
- `stack_val = 50000`
- `final_test = 500000`

Confirm cached fixed-HLT views exist for all five splits and include the fixed HLT parameter values and content hashes.

Deliverables:

- split audit report;
- HLT cache audit report;
- class counts per split;
- constituent-count summary per split.

### Step 2: Replace The Fresh Reconstructor With The Original-Mechanism m2-Hybrid Reconstructor

Implement the operation-aware reconstructor described above.

Required deliverables:

- transformer-style token encoder with relative position awareness;
- edit branch;
- split branch with parent uplift and child existence;
- attention-based generator branch;
- budget heads;
- output dictionary with candidate tokens/weights and corrected-view support fields;
- numerical-safety checks.

Do not yet train all seven variants. First run a forward-pass and loss smoke test.

### Step 3: Implement Original-Mechanism Losses

Implement Stage-A losses close to the original:

- set matching loss, preferably Hungarian;
- budget/count loss;
- pT response loss;
- mass response loss;
- energy response loss;
- locality loss;
- sparse loss;
- split/gen sparse diagnostics;
- anti-overlap option;
- nonfinite diagnostics.

Deliverables:

- unit tests with synthetic finite inputs;
- test where NaN/Inf candidates are sanitized or rejected before SciPy matching;
- diagnostic printout for one real batch.

### Step 4: Implement The Soft Corrected-View Builder

Implement parent-token-aligned corrected-view construction.

Deliverables:

- corrected-view feature tensor;
- corrected-view mask;
- support channels for token weight, parent-added support, and budget/efficiency share;
- test that output uses only HLT and reconstructor output;
- test that zero-HLT changes corrected-view features.

### Step 5: Replace The Fresh Dual-View Tagger With Cross-Attention Fusion

Implement a dual-view tagger with:

- HLT branch encoder;
- corrected-view branch encoder;
- attention pooling for both branches;
- cross-attention both directions;
- fused MLP classifier head.

Deliverables:

- forward-pass test;
- training over a tiny subset;
- checkpoint save/load test.

### Step 6: Train One Variant End-To-End

Train `m2_base` only.

Run:

- Stage-A reconstructor training on `model_train`;
- Stage-A selection on `model_val`;
- StageB dual-view tagger training on `model_train`;
- StageB selection on `model_val`;
- frozen prediction collection on `stack_train`, `stack_val`, and `final_test`.

Deliverables:

- Stage-A loss curves;
- StageB accuracy curves;
- final-test single-model metrics;
- HLT baseline comparison.

### Step 7: Train The Seven Variants

Train:

- `m2_base`
- `m2_consstrong`
- `m2_budgetlite`
- `m2_genlow`
- `m2_genhigh`
- `m2_topk60ish`
- `m2_antioverlap`

Keep HLT cache fixed and unchanged.

Deliverables:

- per-variant training report;
- per-variant HLT hash check;
- per-variant checkpoint paths;
- per-variant single-model metrics.

### Step 8: Run HLT5 Seed Control

Train five HLT-only seed models using the same fixed HLT cache and same split. Stack them with the same stacker protocol.

Deliverables:

- best individual HLT final-test metrics;
- HLT5 uniform average;
- HLT5 weighted average;
- HLT5 stacked logistic regression;
- comparison to single HLT baseline.

### Step 9: Collect Reco7 Plus HLT Predictions

For HLT baseline plus seven dual-view models, collect frozen logits/probabilities on:

- `stack_train`;
- `stack_val`;
- `final_test`.

Deliverables:

- prediction NPZs;
- prediction metadata JSONs;
- identity alignment audit;
- HLT hash compatibility audit.

### Step 10: Run Fusion

Train the stacked logistic regression on `stack_train`. Select C and method on `stack_val`. Evaluate final selected method on `final_test`.

Deliverables:

- single-model table;
- fusion-method table;
- final selected result;
- comparison to HLT5.

### Step 11: Run Leakage Audits

Run:

- source audit;
- same-HLT compatibility audit;
- split identity audit;
- prediction alignment audit;
- permuted-label control;
- row-shuffle control;
- zero-HLT input control;
- HLT5 control comparison;
- optional holdout stack check.

Deliverables:

- audit report JSON;
- audit summary text;
- explicit pass/fail for every audit.

### Step 12: Write Final Report

The final report should include:

- exact split sizes;
- HLT parameters;
- HLT hashes;
- model architecture summary;
- variant definitions;
- training budgets;
- HLT baseline result;
- HLT5 result;
- individual reco7 results;
- fusion results;
- leakage audits;
- reconstruction diagnostics;
- limitations and next checks.

## Final Reminder

The point of V2 is not to create a nicer simplified implementation. The point is to reproduce the specific mechanism that produced the surprising result:

- operation-aware m2-hybrid reconstructor;
- parent-token-aligned soft corrected view;
- cross-attention dual-view tagger;
- same fixed HLT;
- reconstructor-diverse stack.

If any of those are missing, the run should be labeled as an ablation, not as a replication.
