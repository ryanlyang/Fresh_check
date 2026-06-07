# JetClass Same-HLT Reconstructor Ensemble Replication Protocol

This document describes the clean-room replication target for the JetClass HLT-to-offline reconstruction/tagging experiment. It is intentionally a specification, not implementation code. The goal is to let a fresh Codex session recreate the experiment from scratch using only the local JetClass data, the fixed HLT generator, and the Particle Transformer codebase.

The core question is simple:

Can a diverse set of reconstruction-specialized HLT-only models, all trained and evaluated on the same HLT corruption, give a substantially better stacked classifier than an ensemble of ordinary HLT-only taggers trained with different random seeds?

The replication should avoid using the old project implementation. It may use the same experiment design and fixed-HLT generator. It should not try to reproduce exact old checkpoint names or exact old numeric values. The important outcome is whether the 7-model reconstructor-diverse stack is clearly stronger than the HLT-only seed ensemble under the same split, same HLT generation, and leakage audits.

## Available Local Resources

The fresh-start working folder on the independent Windows machine is:

`C:\Users\22rya\ComputerScience\CERN\Fresh_check`

The Particle Transformer reference implementation is in:

`C:\Users\22rya\ComputerScience\CERN\Fresh_check\particle_transformer`

The fixed HLT generator already written for this fresh-start check is:

`C:\Users\22rya\ComputerScience\CERN\Fresh_check\jetclass_fixed_hlt.py`

The fixed HLT generator should be treated as frozen experiment infrastructure. Do not silently modify its corruption logic while developing models. The point of the check is that every model sees the exact same generated HLT view.

The JetClass data on the research compute is expected at:

`/home/ryreu/atlas/PracticeTagging/data/jetclass_part0`

On the Windows local machine, the research-compute data folder will not exist. The training and analysis scripts should therefore accept a `--data_dir` argument and default to the research-compute path only in research-compute runners.

The usual research-compute working directory is:

`/home/ryreu/atlas/PracticeTagging`

## Particle Transformer Code To Reuse

The fresh implementation should use the Particle Transformer package in this Windows fresh-check folder as the tagger backbone instead of copying the previous project tagger code.

Important files to inspect:

`C:\Users\22rya\ComputerScience\CERN\Fresh_check\particle_transformer\README.md`

`C:\Users\22rya\ComputerScience\CERN\Fresh_check\particle_transformer\dataloader.py`

`C:\Users\22rya\ComputerScience\CERN\Fresh_check\particle_transformer\networks\example_ParticleTransformer.py`

`C:\Users\22rya\ComputerScience\CERN\Fresh_check\particle_transformer\data\JetClass\JetClass_full.yaml`

The Particle Transformer expects per-particle inputs organized into four logical groups:

`pf_points`: typically per-particle angular offsets such as `part_deta` and `part_dphi`.

`pf_features`: scalar per-particle features, including log-pT, log-energy, relative log-pT, relative log-energy, charge, particle-ID indicators, tracking quantities, and angular offsets.

`pf_vectors`: four-vectors, typically `part_px`, `part_py`, `part_pz`, and `part_energy`.

`pf_mask`: constituent-validity mask.

The reference JetClass YAML lists the full feature convention. Use it as the canonical guide for how to format tensors for Particle Transformer. The new code may build these tensors manually rather than relying on the original weaver training command, because this experiment needs HLT views, reconstructed views, dual-view training, and stacked fusion.

The clean-room implementation does not need to be bitwise identical to the old implementation. It should use Particle Transformer consistently for all taggers so the comparison is internally fair.

## JetClass Classes And File Mapping

The final classification is a 10-class JetClass problem.

Use this label order:

1. `QCD`
2. `Hbb`
3. `Hcc`
4. `Hgg`
5. `H4q`
6. `Hqql`
7. `Zqq`
8. `Wqq`
9. `Tbqq`
10. `Tbl`

The ROOT file prefixes should be mapped to those labels as follows:

`ZJetsToNuNu` maps to `QCD`.

`HToBB` maps to `Hbb`.

`HToCC` maps to `Hcc`.

`HToGG` maps to `Hgg`.

`HToWW4Q` maps to `H4q`.

`HToWW2Q1L` maps to `Hqql`.

`ZToQQ` maps to `Zqq`.

`WToQQ` maps to `Wqq`.

`TTBar` maps to `Tbqq`.

`TTBarLep` maps to `Tbl`.

Each class should have multiple ROOT files in the data directory. For the clean replication, use a stricter five-way split rather than reusing the same validation set for both model selection and stacker training.

The preferred split is by deterministic jet identity, not by random array position after loading. A jet identity should include at least the source ROOT file path and entry index. If enough files are available, reserve completely separate files for the locked final test. If file counts make that awkward, jet-level hash partitioning is acceptable, but the final metadata must prove that no jet identity appears in more than one partition.

Use these partitions:

`model_train`: `500,000` jets total. This is the only partition used to fit neural-network weights for HLT taggers, reconstructors, offline teacher, and dual-view taggers.

`model_val`: `150,000` jets total. This is used for neural-network early stopping, checkpoint selection, and model-level hyperparameter decisions. It must not be used to train the stacked logistic regression.

`stack_train`: `250,000` jets total. This is used to fit the stacked logistic regression and any stack-level calibration or fusion weights. Neural-network weights must already be frozen before this partition is used.

`stack_val`: `50,000` jets total. This is used to choose stack-level regularization, compare stack variants, select simple-fusion weights if a separate validation decision is needed, and decide whether the stacker is behaving sensibly. It must not be used to train neural-network weights.

`final_test`: `500,000` jets total. This is locked until the end. It is used exactly once for the final reported numbers after the model family, stacker, and audits are fixed.

Use deterministic class-wise sampling so the class balance is known. If the goal is a balanced 10-class benchmark, sample equal counts per class in every partition. Save the exact file list and jet-entry IDs for each partition.

If the fresh implementation first needs a smaller debug run, keep the same five-way split roles and reduce only the number of jets sampled from each partition. Do not collapse `stack_train` and `stack_val` back into `model_val`.

Use a max constituent count of `128`.

## Loading JetClass Data

The reference `particle_transformer/dataloader.py` shows how to read JetClass ROOT files with uproot. Use that file as the guide for branch names and feature conventions.

For this experiment, the raw constituent representation needed by the HLT generator should include at least these columns per constituent:

`pt`, `eta`, `phi`, `energy`, `charge`, five particle-ID flags, and four tracking-related attributes.

The fixed HLT generator expects token tensors with shape `[n_jets, max_constits, 14]` and masks with shape `[n_jets, max_constits]`.

The token column convention is:

Column 0: constituent `pt`.

Column 1: constituent `eta`.

Column 2: constituent `phi`.

Column 3: constituent `energy`.

Column 4: constituent charge.

Columns 5 through 9: particle-ID flags.

Columns 10 through 13: track-related attributes.

The labels should come from the filename or ROOT label branches according to the mapping above. For this replication, filename-based class assignment is acceptable and usually simpler, because each JetClass file contains a single process type. If label branches are used, verify that the label branch and filename mapping agree.

Sampling should be deterministic. Recommended seeds:

Base experiment seed: `52`.

`model_train` jet sampling seed: `153`.

`model_val` jet sampling seed: `254`.

`stack_train` jet sampling seed: `356`.

`stack_val` jet sampling seed: `457`.

`final_test` jet sampling seed: `558`.

`model_train` HLT corruption seed: `1053`.

`model_val` HLT corruption seed: `1054`.

`stack_train` HLT corruption seed: `1055`.

`stack_val` HLT corruption seed: `1056`.

`final_test` HLT corruption seed: `1057`.

All models in the 7-model reconstructor experiment must use the same partition jet IDs and the same cached HLT views for every partition.

## Strict Fairness And Leakage Rules

For this replication, imagine the deployed system receives only HLT constituents as input. Any validation or test-time use of offline constituents, offline jet kinematics, offline teacher scores, or offline-derived features is leakage unless it is explicitly part of evaluating the offline teacher reference.

Allowed uses of offline information:

Offline constituents may be used as supervised targets when training the reconstructor on `model_train`.

Offline constituents may be used for reconstruction validation and checkpoint selection on `model_val`, because that is still model-development data.

Offline constituents may be used to train and evaluate an offline-only teacher reference. That teacher is an upper-reference model, not an input to the HLT stack.

Class labels may be used for classifier training, validation, stacker training, stacker validation, and final test scoring according to the partition roles.

The fixed HLT generator necessarily starts from offline constituents in this synthetic study to create HLT-like inputs. After a partition's HLT view is generated and cached, every HLT model, reconstructor, dual-view tagger, and stacker must operate from the HLT view only. Treat the cached HLT view as the deployed input.

Forbidden uses of offline information:

Do not compute HLT or reconstructed Particle Transformer features using offline jet `pt`, offline jet `eta`, offline jet `phi`, offline jet energy, offline mass, or offline axis.

Do not pass offline constituents to the HLT baseline, dual-view tagger, reconstructor inference path, fusion model, or stacked logistic regression.

Do not include offline teacher logits, offline teacher probabilities, offline teacher embeddings, oracle reconstruction losses, target scores, or offline-vs-HLT residual labels as fusion features.

Do not train neural-network weights on `stack_train`, `stack_val`, or `final_test`. These partitions are for frozen-model evaluation and stack-level analysis only.

Do not tune the reconstructor, tagger, HLT generator, model list, or neural-network checkpoint using `stack_train`, `stack_val`, or `final_test` performance.

Do not choose final reported numbers by looking across multiple test evaluations. The `final_test` partition is locked until the model family, stacker choice, and leakage audits are fixed.

The safe mental model is: offline data can teach the reconstructor during training, but once the model is frozen, every reported HLT-side result must be obtainable from HLT constituents alone.

## Fixed HLT Generation

Use the fixed HLT generator in:

`C:\Users\22rya\ComputerScience\CERN\Fresh_check\jetclass_fixed_hlt.py`

The intended public function is `build_fixed_hlt_view`.

Inputs are offline constituent tokens and a mask. Outputs are HLT-like constituent tokens, HLT mask, and diagnostics.

The default HLT profile is the fixed m2-style profile:

`hlt_pt_threshold = 1.30`

`merge_prob_scale = 1.35`

`reassign_scale = 1.00`

`smear_scale = 1.00`

`eff_plateau_barrel = 0.99`

`eff_plateau_endcap = 0.97`

`eff_turnon_pt = 1.40`

`eff_width_pt = 0.20`

Every model in the 7-model experiment must use this exact HLT generation. Do not create per-model HLT variants. The only differences between the seven models should be the reconstruction objective or reconstruction dynamics.

The HLT view should be generated once per split and then reused for all models. This prevents accidental drift from per-model random HLT generation.

Save enough metadata to prove the HLT view was shared:

HLT parameters.

HLT seed per split.

Constituent-count summaries for offline and HLT views.

Drop decomposition diagnostics from the HLT generator.

A content hash or checksum for the generated HLT arrays or the source file list plus seeds.

## Building Particle Transformer Inputs Without Leakage

This point is critical.

For any view being classified, compute Particle Transformer inputs from that view only.

For the offline teacher, compute jet axis, jet energy, relative pT features, and relative energy features from offline constituents.

For the HLT baseline, compute all Particle Transformer inputs from HLT constituents only.

For reconstructed views, compute all Particle Transformer inputs from reconstructed constituents only.

Do not use offline jet `pt`, offline jet `eta`, offline jet `phi`, offline jet energy, or offline-derived relative features when evaluating an HLT or reconstructed model.

For each view, reconstruct the jet four-vector by summing constituent four-vectors. From that summed vector, derive the view-specific jet pT, eta, phi, and energy. Then compute per-constituent relative features such as angular offsets and log relative pT/energy with respect to that same view.

This is the main anti-leakage rule for the tagger input pipeline.

## Baseline Models To Train

There are two baseline/control families.

### Single HLT Baseline

Train one Particle Transformer classifier on the HLT view only.

Input: HLT constituents.

Target: 10-class JetClass label.

Loss: cross entropy.

Model validation: `model_val` HLT view for early stopping and checkpoint selection.

Stack and final evaluation: after the model is frozen, evaluate it on `stack_train`, `stack_val`, and `final_test` for fusion analysis.

This model is also included as one input to the main 7+HLT stacked fusion.

### Five-Seed HLT Ensemble Control

Train five independent HLT-only Particle Transformer classifiers.

All five must use the same file split, same sampled jets, and same fixed HLT view.

Only the training seed should differ.

Suggested seeds: `101`, `202`, `303`, `404`, `505`.

After training, evaluate the five frozen HLT models on `stack_train`, `stack_val`, and `final_test`, then use the same stacked logistic regression procedure used for the reconstructor-diverse models.

This control asks whether stacked logistic regression plus multiple neural networks is enough to explain the gain. If the 5-HLT stack is much weaker than the 7-model reconstructor-diverse stack, then the improvement is not just generic ensembling.

## Reconstructor-Diverse Seven-Model Experiment

Train seven reconstruction-specialized models. All use the same fixed HLT input views and the same five partition definitions: `model_train`, `model_val`, `stack_train`, `stack_val`, and `final_test`.

Each model has two conceptual parts:

1. A reconstructor that maps HLT constituents toward an offline-like representation.
2. A dual-view classifier that classifies jets using HLT information plus the reconstructed/corrected view.

The seven models should be intentionally different but still part of one coherent reconstructor family. Their differences should be reconstruction behavior and reconstruction loss priorities, not HLT-generation differences.

The seven models are:

1. `m2_base`
2. `m2_consstrong`
3. `m2_budgetlite`
4. `m2_genlow`
5. `m2_genhigh`
6. `m2_topk60ish`
7. `m2_antioverlap`

The recommended interpretation is:

`m2_base` is the default HLT-to-offline reconstruction objective.

`m2_consstrong` emphasizes classifier consistency and global reconstruction consistency more strongly.

`m2_budgetlite` relaxes count/budget matching and lets the model focus more on useful corrected structure.

`m2_genlow` restricts generated-token capacity so the model is forced to be conservative about added constituents.

`m2_genhigh` increases generated-token capacity and relaxes sparsity so the model can recover more missing soft structure.

`m2_topk60ish` is a middle-capacity generation model, between low and high generation capacity.

`m2_antioverlap` emphasizes locality/anti-overlap so generated or split candidates do not collapse into redundant nearby particles.

These are not arbitrary models. They probe a clean family of hypotheses: reconstruction diversity in generated content, budget handling, consistency strength, and candidate locality creates complementary HLT-derived views.

## Reconstructor Design

The reconstructor should be operation-aware but not oracle-based.

Input: HLT token sequence and HLT mask.

Target during Stage A training: offline token sequence and offline mask.

Allowed information during inference: HLT tokens only.

A practical reconstructor should produce a soft corrected particle view with candidate tokens and candidate weights. The exact architecture can differ from the old project, but it should preserve these reconstruction behaviors:

Edit or unsmear existing HLT tokens by predicting residual corrections in kinematics.

Split likely merged HLT tokens into child candidates.

Generate missing constituents from a jet/global latent state.

Predict soft weights or existence probabilities so the number of active generated/split candidates is learned rather than fixed by hand.

Use budget/count heads to calibrate the expected total corrected multiplicity and added multiplicity.

The generator capacity is a maximum number of candidate tokens, not a forced count. The model should decide how many generated candidates matter by assigning weights.

The reconstructed view can be represented as weighted candidate particles. When building Particle Transformer inputs, either keep weights as an extra scalar feature or fold weights into candidate pT/energy consistently. Do not use offline target information to decide which candidates survive at validation or test time.

## Stage A Reconstructor Training

Stage A trains the reconstructor from HLT to offline.

Use reconstruction losses that compare the predicted corrected view to offline constituents.

Recommended loss components:

Set-matching loss between predicted particles and offline particles. Hungarian matching is preferred if feasible. Chamfer-style matching is acceptable if it is implemented carefully and validated.

Global pT response loss, comparing reconstructed jet pT to offline jet pT.

Global energy response loss, comparing reconstructed jet energy to offline jet energy.

Mass response loss, comparing reconstructed jet mass to offline jet mass.

Soft count/budget loss, comparing predicted active count and added count to target offline-minus-HLT count statistics.

Sparsity loss for generated candidates, so the generator does not fill every available slot with noise.

Locality loss for generated or split candidates, encouraging them to stay near plausible HLT parents or high-density regions.

Optional axis loss, comparing reconstructed jet axis to offline jet axis, but keep it consistent across models unless it is explicitly part of a named variant.

Suggested base weights, as an approximate starting point:

Set matching: `1.0`.

Budget/count: `0.70`.

Sparsity: `0.010`.

Locality: `0.08`.

pT ratio: `0.12`.

Mass ratio: `0.02`.

Energy ratio: `0.02`.

Physics constraint term: `0.00` unless explicitly added.

Target added-particle scale: `0.90`.

Maximum generated candidate tokens for the base model: `56`.

The model variants should modify these knobs cleanly:

`m2_base`: use the base weights above.

`m2_consstrong`: increase global/consistency emphasis. Use stronger budget or consistency weights and allow light joint finetuning.

`m2_budgetlite`: reduce budget/count pressure and modestly increase pT response pressure.

`m2_genlow`: reduce max generated candidates, for example around `40`, and keep sparsity moderately strong.

`m2_genhigh`: increase max generated candidates, for example around `72`, and reduce sparsity pressure.

`m2_topk60ish`: use intermediate generation capacity, around `60`, with moderate budget and sparsity.

`m2_antioverlap`: increase locality or anti-overlap pressure and reduce the local radius so extra candidates are less redundant.

## Dual-View Tagger

The dual-view tagger should classify each jet using two views:

The original fixed HLT view.

The reconstructed/corrected view produced from the HLT view.

Use Particle Transformer encoders for these views. The simplest robust design is two Particle Transformer branches whose pooled outputs are concatenated and passed through a classifier head. A lighter alternative is one Particle Transformer over a combined sequence with view-type embeddings, but that is more likely to introduce implementation ambiguity.

The classifier target is the same 10-class JetClass label.

The dual-view classifier must never receive offline particles at validation or test time.

## Training Schedule

A clean staged schedule is recommended.

Stage 0: train an offline Particle Transformer teacher for reference only.

The offline teacher is not required for the main stack. It gives an upper reference and checks that the Particle Transformer implementation is reasonable. It must not be used as an input to the stack.

Stage 1: train the HLT baseline Particle Transformer.

This is the primary HLT-only baseline and the baseline model included in the 7+HLT stack.

Stage 2: train Stage A reconstructors for the seven variants.

Each variant maps HLT constituents to offline-like constituents using reconstruction losses only.

Stage 3: train dual-view taggers with reconstructors frozen or mostly frozen.

This is the main checkpoint to use for same-HLT stacked fusion. In the old experiment this was called the Stage2 or PreJoint output.

Stage 4: optional joint finetuning.

Joint finetuning can be useful, but for the clean replication of the same-HLT stack, use the Stage2 dual-view checkpoints unless a separate experiment explicitly compares Stage2 versus joint.

## Fusion / Stacked Logistic Regression

After all neural-network models are trained and frozen, evaluate each model on `stack_train`, `stack_val`, and `final_test`.

Save `stack_train` logits/probabilities/labels, `stack_val` logits/probabilities/labels, and `final_test` logits/probabilities/labels. Do not save or use offline teacher outputs as stack features.

Models included in the main stack:

`hlt_baseline`

`m2_base`

`m2_consstrong`

`m2_budgetlite`

`m2_genlow`

`m2_genhigh`

`m2_topk60ish`

`m2_antioverlap`

Compute simple ensemble baselines:

Uniform probability average.

`stack_val`-selected weighted probability average.

`stack_val`-selected weighted logit average.

Then train the main stacked model:

Use multinomial logistic regression.

Input features are concatenated logits and probabilities from all included frozen models.

Fit the logistic regression on `stack_train` only.

Choose regularization and any stack-level feature choices using `stack_val` only.

Use candidate inverse-regularization values such as `0.03`, `0.1`, `0.3`, `1.0`, `3.0`, and `10.0`.

Optimize `stack_val` accuracy.

After the stacker choice is fixed, evaluate once on `final_test`. Do not use `final_test` labels to choose regularization, feature mode, model subset, fusion method, calibration, or pruning path.

The final stacker should normally remain the model fit on `stack_train` with choices selected on `stack_val`. If a later paper-style run refits on `stack_train + stack_val`, report that as a separate variant and keep the locked `final_test` rule unchanged.

The stacked logistic regression is a meta-classifier. It learns class-specific linear corrections from the pattern of model logits/probabilities. It is more powerful than a weighted average, but it is still a simple supervised classifier trained only on frozen-model predictions from the stack-training partition.

## Leakage And Sanity Audits

The replication should include audits before interpreting results.

File split audit:

Verify no ROOT file appears in more than one split.

Jet identity audit:

Hash raw jets or stable `(file, entry)` identities from `model_train`, `model_val`, `stack_train`, `stack_val`, and `final_test`. Verify no overlap across any pair of partitions.

Partition-role audit:

Verify neural-network checkpoints were trained only using `model_train` and selected only using `model_val`. Verify the stacker was fit only using `stack_train`, selected only using `stack_val`, and reported on `final_test` only after all choices were fixed.

HLT sharing audit:

Verify all seven reconstructor models and all HLT baselines use the same HLT arrays or the same source file list, jet indices, HLT parameters, and HLT seeds.

Offline leakage audit:

Verify `model_val`, `stack_train`, `stack_val`, and `final_test` tagger input builders for HLT and reconstructed views do not use offline jet-level kinematics when computing Particle Transformer relative features. For every HLT-side result, the only physics inputs should be HLT constituents or reconstructor outputs produced from HLT constituents.

Fusion source audit:

Verify the stack inputs contain only model logits/probabilities and labels. Do not include offline teacher logits, offline teacher probabilities, fused teacher targets, target scores, oracle labels beyond class labels, or reconstruction losses as stack features.

Permutation audit:

Shuffle `stack_train` labels before fitting the stacked logistic regression. `stack_val` and `final_test` accuracy should collapse near chance level.

Holdout-stack audit:

Split `stack_train` predictions into two halves. Fit the stacked logistic regression on one half and evaluate on the other half, `stack_val`, and `final_test`. These metrics should be reasonably consistent.

Block shuffle audit:

Randomly permute the rows of one model's prediction block relative to the labels and other models. The stacked performance should degrade.

HLT-only ensemble control:

Compare the 7+HLT reconstructor-diverse stack to the 5-HLT-seed stack. If the 5-HLT-seed stack gives only a modest gain but the 7+HLT stack gives a much larger gain, the improvement is not just generic ensembling.

Blind-result discipline:

Do not tune the implementation to match the old numeric result. First produce the clean result, then compare qualitatively.

## Expected Interpretation

A successful replication does not require exact old metrics. It should show this qualitative structure:

The offline teacher is strongest or near strongest as an upper reference.

The single HLT baseline is weaker than offline.

The 5-HLT-seed stacked ensemble improves only modestly over the single HLT baseline.

The 7+HLT reconstructor-diverse stack improves substantially more than the 5-HLT-seed stack.

The leakage audits pass.

If that pattern holds, the result supports the claim that diverse HLT-to-offline reconstruction objectives create complementary HLT-derived representations that a simple stacker trained on a separate stack-training partition can exploit.

## Suggested Output Layout

Use a new output root under the fresh-start folder or research-compute working directory.

Suggested directory names:

`checkpoints/jetclass_fresh_hlt_baselines`

`checkpoints/jetclass_fresh_reco7`

`checkpoints/jetclass_fresh_fusion/reco7_plus_hlt`

`checkpoints/jetclass_fresh_fusion/hlt5_seed_control`

`checkpoints/jetclass_fresh_audits/reco7_plus_hlt`

Every run directory should save:

Configuration file.

Resolved data split file list.

Resolved sampled jet indices or sample seed metadata.

HLT parameter metadata.

Training curves.

Best `model_val` checkpoint.

Final `model_val`, `stack_val`, and `final_test` metrics, with clear partition labels.

For fusion runs, save:

Model list.

`stack_train`, `stack_val`, and `final_test` logits/probabilities.

Fusion report.

Stacker coefficients.

Audit report.

## Multi-Step Implementation Plan

Use this plan as a sequence of prompts. Each step should be independently reviewable before moving to the next.

### Step 1: Inspect Particle Transformer And Define Interfaces

Read the Particle Transformer README, example network, dataloader, and JetClass YAML.

Decide the exact Python interfaces for datasets, Particle Transformer input builders, tagger models, and training loops.

Write no training experiment yet. Only define the intended data objects and tensor conventions.

### Step 2: Implement JetClass Data Loading And Split Management

Write a loader that finds JetClass ROOT files, maps filenames to the 10 labels, creates deterministic five-way partitions, samples the requested number of jets for `model_train`, `model_val`, `stack_train`, `stack_val`, and `final_test`, and returns offline token tensors, masks, labels, and stable jet identities.

Save split metadata and verify no jet-identity overlap. If file-level separation is used for some partitions, also verify no file overlap where that guarantee is intended.

### Step 3: Integrate The Fixed HLT Generator

Use `jetclass_fixed_hlt.py` to generate HLT views for all five partitions.

Cache or save generated HLT metadata so all models reuse the same view.

Verify constituent-count and drop diagnostics look stable.

### Step 4: Build Particle Transformer Inputs From Any View

Implement a view-to-Particle-Transformer input builder.

It must compute jet axis, jet pT, jet energy, relative pT, relative energy, delta-eta, delta-phi, and delta-R from the provided view only.

Test it on offline, HLT, and dummy reconstructed views.

### Step 5: Train The Single HLT Baseline

Train one HLT-only Particle Transformer classifier.

Evaluate `model_val` only during development. Defer locked `final_test` evaluation until the final fusion/audit stage.

This verifies that the loader, HLT generator, and Particle Transformer training loop work.

### Step 6: Train The Offline Teacher Reference

Train an offline-only Particle Transformer classifier.

Use it only as an upper reference. Do not include it in fusion features.

### Step 7: Implement The Reconstructor Family

Implement the operation-aware HLT-to-offline reconstructor with edit/unsmear, split, generate, soft candidate weights, and budget/count calibration.

Start with `m2_base` only.

Train Stage A and inspect reconstruction losses and simple response diagnostics.

### Step 8: Implement The Dual-View Tagger

Connect the frozen or mostly frozen reconstructor output to a dual-view Particle Transformer classifier.

Train the Stage2 dual-view classifier for `m2_base`.

Evaluate against the HLT baseline on `model_val` during development. Defer stack partitions and `final_test` until the frozen-model fusion stage.

### Step 9: Add The Seven Reconstructor Variants

Add `m2_consstrong`, `m2_budgetlite`, `m2_genlow`, `m2_genhigh`, `m2_topk60ish`, and `m2_antioverlap`.

Confirm each variant changes only reconstruction/loss behavior, not HLT generation or data split.

Train all seven Stage2 models.

### Step 10: Implement Fusion And Stacked Logistic Regression

Evaluate the seven Stage2 models plus the HLT baseline on `stack_train`, `stack_val`, and `final_test`.

Save logits and probabilities for all three stack partitions.

Implement uniform average, weighted probability average, weighted logit average, and stacked logistic regression.

Fit the stacker only on `stack_train`, choose stack-level settings only on `stack_val`, and evaluate `final_test` only once.

### Step 11: Implement The Five-Seed HLT Control

Train five HLT-only Particle Transformer models with different training seeds and identical data/HLT views.

Run the same stacked logistic regression procedure on those five models.

Compare the 5-HLT stack to the 7+HLT stack.

### Step 12: Implement Leakage Audits

Add file split, jet hash, HLT sharing, fusion source, permutation-label, holdout-stack, and block-shuffle audits.

Do not interpret the result until these pass.

### Step 13: Write The Final Fresh-Start Report

Summarize single HLT, offline teacher, HLT5 seed stack, 7+HLT stack, simple fusion baselines, and audit outcomes.

State whether the 7+HLT stack is substantially better than the HLT-only seed stack.

Avoid overclaiming exact recovery of offline performance unless the clean-room result and audits support it.


### Step 14: Write Research-Compute Sbatch Scripts

After the Python implementation is complete and unit-tested on small local/debug samples, write sbatch scripts that run the full fresh-start experiment on the research compute. This step should create the complete orchestration needed for the final comparison: the single HLT baseline, the five-seed HLT control, the seven same-HLT reconstructor-diverse models, the fusion jobs, and the leakage audits.

The sbatch scripts should be written for the research-compute Linux environment, not for the Windows local folder. The Windows folder is the clean development origin. The research-compute project directory should be a clean copied/synced version of that fresh folder, for example a path such as `/home/ryreu/atlas/Fresh_check` or another explicitly chosen fresh directory. Do not point the fresh sbatch scripts at the old project code directory except for the shared JetClass data path.

The research-compute data path should remain:

`/home/ryreu/atlas/PracticeTagging/data/jetclass_part0`

#### General Sbatch Formatting Requirements

Every sbatch script should have a clear job name, output log path, error log path, partition, time limit, memory request, and environment setup.

Every script should start by printing the job ID, hostname, current date, working directory, command-line arguments, important environment variables, and resolved output directory.

Every script should use strict shell behavior: fail on unset variables and failed commands.

Every script should create a log directory before launching Python. Recommended log directory:

`offline_reconstructor_logs`

or, for a cleaner fresh run:

`fresh_check_logs`

Use log names that include the job name and SLURM job ID, such as:

`fresh_reco7_m2_base_%j.out`

`fresh_reco7_m2_base_%j.err`

Every script should activate the intended conda environment before running Python. The exact environment name is machine-specific, so the script should make this configurable. For example, use an environment variable such as `CONDA_ENV`, defaulting to the known research environment if one exists.

Every training script should write a run configuration JSON into its checkpoint directory. That configuration must include data split seeds, HLT seeds, HLT parameters, partition sizes, model variant name, training seed, and git commit or source snapshot hash if available.

Every script should print the exact Python command before executing it.

Every script should exit with a nonzero status if the expected checkpoint, report, or scores file is missing at the end.

Do not use the debug partition for full training. Debug is appropriate only for smoke tests, tiny data checks, or short audit runs.

#### Recommended Resource Requests

These are starting recommendations. Adjust after observing actual runtime and memory usage on the target cluster.

HLT-only tagger training, one seed:

Partition: `tier3` or the normal GPU training partition.

Time: `2-00:00:00`.

Memory: `64G` to `96G`.

GPU: one GPU if the cluster requires explicit GPU requests.

Offline teacher reference:

Partition: normal GPU training partition.

Time: `2-00:00:00`.

Memory: `64G` to `96G`.

GPU: one GPU.

One reconstructor-diverse dual-view model:

Partition: normal GPU training partition.

Time: `3-00:00:00` as the first safe setting.

Memory: `128G` as the first safe setting.

GPU: one GPU.

If Stage A reconstruction loading/caching is memory-heavy, increase memory before changing the experiment. If jobs are comfortably below memory limits, reduce later.

Seven-model fusion and stacked logistic regression:

Partition: normal CPU or GPU partition. GPU is useful only if inference is repeated; if it consumes saved logits only, CPU is enough.

Time: `1-00:00:00`.

Memory: `96G` to `160G`, depending on whether logits are streamed or fully materialized.

Use single-threaded or controlled-threaded sklearn settings to avoid memory blowups.

Five-HLT-seed fusion:

Partition: normal CPU or GPU partition.

Time: `12:00:00` to `1-00:00:00`.

Memory: `64G` to `128G`.

Leakage audits:

Partition: debug is acceptable for lightweight audits.

Time: `12:00:00` to `1-00:00:00`.

Memory: `64G` to `128G`.

Use smaller input-control samples first, then increase if needed.

#### Individual Scripts To Write

Write a small set of reusable scripts instead of many nearly identical one-off scripts.

1. HLT baseline training runner.

Suggested name:

`run_train_fresh_hlt_baseline.sh`

Purpose:

Train one HLT-only Particle Transformer on `model_train`, select checkpoint on `model_val`, and save frozen-model logits for later partitions only when explicitly requested.

Required arguments or environment variables:

Project directory.

Data directory.

Output root.

Train seed.

Partition sizes.

HLT cache path or HLT generation settings.

Expected outputs:

HLT baseline checkpoint.

Run configuration JSON.

Training curves.

`model_val` metrics.

2. HLT seed training runner.

Suggested name:

`run_train_fresh_hlt_seed.sh`

Purpose:

Train one HLT-only model for the five-seed ensemble control.

Required argument:

Seed identifier, such as `101`, `202`, `303`, `404`, or `505`.

Expected outputs:

One checkpoint per seed.

One run configuration JSON per seed.

One metrics file per seed.

3. HLT5 submitter.

Suggested name:

`submit_fresh_hlt5_seed_control.sh`

Purpose:

Queue five `run_train_fresh_hlt_seed.sh` jobs, one per training seed, then queue the HLT5 fusion job with an `afterok` dependency on all five seed jobs.

Expected behavior:

Print each submitted job ID.

Print the final dependency string.

Print the expected HLT5 fusion output directory.

Queue no final-test reporting job until all five seeds finish successfully.

4. Offline teacher runner.

Suggested name:

`run_train_fresh_offline_teacher.sh`

Purpose:

Train the offline-only Particle Transformer reference. This model is for upper-reference reporting only. It must not be included in fusion features.

Expected outputs:

Offline teacher checkpoint.

Offline teacher `model_val` metrics.

Optional locked final metrics only during final report generation.

5. Reconstructor-diverse model runner.

Suggested name:

`run_train_fresh_reco7_variant.sh`

Purpose:

Train one of the seven HLT-to-offline reconstructor-diverse dual-view models.

Required variant argument:

`m2_base`

`m2_consstrong`

`m2_budgetlite`

`m2_genlow`

`m2_genhigh`

`m2_topk60ish`

`m2_antioverlap`

This script should translate the variant name into reconstruction-loss settings, generation capacity, locality settings, and consistency settings. It must not alter the HLT generation profile or the data split.

Expected outputs:

Stage A reconstructor checkpoint.

Stage2 dual-view classifier checkpoint.

Run configuration JSON.

Training curves.

`model_val` metrics.

Reconstruction diagnostics on `model_val`.

6. Reco7 submitter.

Suggested name:

`submit_fresh_samehlt_reco7.sh`

Purpose:

Queue the seven variant jobs with the same HLT cache, same split metadata, and same base experiment seed. Queue the 7+HLT fusion job with an `afterok` dependency on all seven variant jobs and the HLT baseline job if the HLT baseline is not already complete.

Expected behavior:

Submit all seven jobs.

Print all job IDs with their variant names.

Submit one dependent fusion job.

Submit one dependent leakage audit job after the fusion job.

Never queue models with different HLT corruption settings.

7. Reco7 plus HLT fusion runner.

Suggested name:

`run_fuse_fresh_samehlt7_plus_hlt.sh`

Purpose:

Load the frozen HLT baseline and the seven frozen Stage2 dual-view models. Evaluate all of them on `stack_train`, `stack_val`, and `final_test`. Fit stacked logistic regression on `stack_train`, choose stack settings on `stack_val`, and report `final_test` exactly once.

Expected outputs:

Fusion report JSON.

Saved logits/probabilities NPZ for `stack_train`, `stack_val`, and `final_test`.

Uniform average metrics.

Weighted probability average metrics.

Weighted logit average metrics.

Stacked logistic regression metrics.

Stacker coefficients.

Model-source audit metadata proving no offline teacher outputs were used.

8. HLT5 fusion runner.

Suggested name:

`run_fuse_fresh_hlt5_seed_control.sh`

Purpose:

Run the same fusion logic as the 7+HLT stack, but only on the five independently seeded HLT-only models.

Expected outputs:

HLT5 fusion report JSON.

HLT5 saved logits/probabilities NPZ.

HLT5 stacked logistic regression metrics.

This control must use the same `stack_train`, `stack_val`, and `final_test` partitions as the 7+HLT fusion.

9. Leakage audit runner.

Suggested name:

`run_audit_fresh_samehlt7_plus_hlt.sh`

Purpose:

Run all leakage and sanity checks after the 7+HLT fusion report exists.

Required checks:

File and jet-identity split audit.

Partition-role audit.

HLT sharing audit.

Offline-input audit.

Fusion-source audit.

Permuted stack-label audit.

Holdout-stack audit.

Block-shuffle audit.

Comparison against HLT5 seed control.

Expected outputs:

Audit report JSON.

Human-readable audit summary.

Pass/fail flags for each audit.

10. Master submitter.

Suggested name:

`submit_fresh_full_samehlt_reco7_vs_hlt5.sh`

Purpose:

Queue the entire clean test. This is the script the user should run when ready for the final fresh-start replication.

It should queue:

The single HLT baseline if not already complete.

The five HLT seed-control jobs.

The seven same-HLT reconstructor-diverse jobs.

The HLT5 fusion job dependent on the five HLT seed jobs.

The 7+HLT fusion job dependent on the HLT baseline and seven reconstructor-diverse jobs.

The leakage audit job dependent on the 7+HLT fusion and, if available, the HLT5 fusion.

The final report job dependent on both fusion jobs and the audit job.

The master submitter should print a final summary containing every job ID, dependency chain, output directory, and log directory.

#### Dependency Structure

Use `afterok` dependencies so downstream jobs only run if upstream jobs complete successfully.

The intended dependency graph is:

HLT seed jobs feed into HLT5 fusion.

HLT baseline plus seven reco7 jobs feed into 7+HLT fusion.

HLT5 fusion and 7+HLT fusion feed into leakage audit and final comparison.

Do not make the seven reco7 jobs depend on each other. They should run independently and in parallel if resources allow.

Do not make the HLT seed jobs depend on each other. They should run independently and in parallel if resources allow.

If the cluster is busy or has job-count limits, add an optional submitter mode that queues a smaller number of concurrent jobs or uses job arrays.

#### Sbatch Script Safety Checks

Before submitting full jobs, each sbatch script should support a dry-run or print-only mode that prints the Python command without submitting or executing it.

Every submitter should check whether the expected output directory already exists. If it exists, either refuse to overwrite or require an explicit `OVERWRITE=1` environment variable.

Every training script should check that the data directory exists on the research compute.

Every fusion script should check that all required checkpoints exist before starting expensive inference.

Every audit script should check that required fusion reports and NPZ files exist.

Every final report script should check that the audit report passed before presenting final results as credible.

#### Smoke-Test Scripts

Before launching the full run, write one smoke-test submitter:

Suggested name:

`submit_fresh_smoke_test.sh`

Purpose:

Run the HLT baseline and one reconstructor variant on tiny partition sizes, then run fusion and audits on that tiny output.

Suggested smoke-test sizes:

`model_train`: `10,000` jets.

`model_val`: `3,000` jets.

`stack_train`: `5,000` jets.

`stack_val`: `2,000` jets.

`final_test`: `10,000` jets.

Suggested smoke-test resources:

Partition: debug if available.

Time: `4:00:00` to `12:00:00`.

Memory: `32G` to `64G`.

The smoke test is only for pipeline correctness. Do not interpret physics performance from it.

#### What Counts As A Successful Sbatch-Orchestrated Run

The final research-compute run is successful only if all of these exist:

Single HLT baseline checkpoint and metrics.

Five HLT seed checkpoints and HLT5 fusion report.

Seven same-HLT reconstructor-diverse checkpoints and 7+HLT fusion report.

Saved stack inputs for both HLT5 and 7+HLT fusion.

Leakage audit report with pass/fail status for every required check.

Final comparison report showing HLT baseline, HLT5 stack, 7+HLT stack, simple ensemble baselines, offline teacher reference, and audit status.

The final report should make the central comparison explicit: if the 7+HLT stack is substantially stronger than the HLT5 seed stack while all leakage checks pass, the result supports the reconstruction-diversity hypothesis rather than a generic ensemble-only explanation.
