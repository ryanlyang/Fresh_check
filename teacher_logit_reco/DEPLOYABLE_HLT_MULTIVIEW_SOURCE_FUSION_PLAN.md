# Deployable HLT Multiview Source And Fusion Plan

## Short Name

HLT-MV, or deployable HLT multiview source/fusion.

## Goal

Redo the HLT-only self-dual-view study in a cleaner way on the PDV3 HLT-v2
realistic dataset, using the existing caches and splits, and separating three
questions:

```text
1. Are deterministic HLT2 views useful as standalone source models?

2. Does particle-level HLT + HLT2 fusion beat logit fusion of separately
   trained source models?

3. Does the fusion model need pretrained single-view branches, or can it learn
   the useful cross-view representation from scratch?
```

This is still a deployable HLT-only strategy. At inference time the model sees
only the observed HLT jet, deterministically derives one or more HLT2 views
from that HLT jet, and then predicts with HLT-only-derived inputs.

## Dataset And Fixed Inputs

Use the PDV3 high-data HLT-v2 realistic root:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/privileged_distill_v3_av10_adapter_fixed_hlt_v2_realistic_s1p0_highdata_20260705_190747
```

Use these existing artifacts:

```text
inputs/split_manifest/split_manifest.json.gz
inputs/hlt_cache/
```

Use the same split sizes as the high-data PDV3 run:

```text
model_train: 5,000,000 jets
model_val:   1,000,000 jets
final_test:  1,000,000 jets
```

The original HLT cache must not be rebuilt. HLT2 caches should also be reused
if they already exist. If an HLT2 cache is missing, build only that missing
cache from the PDV3 HLT cache, never from offline particles.

The HLT2 strength grid for this plan is:

```text
s0p10
s0p20
s0p35
s1p00
```

Do not include `s0p00` in the model grid. `s0p00` is an identity-cache audit
tool, not a scientifically useful separate model.

## Output Layout

Write all outputs under a new root so this redo does not overwrite the earlier
HLT-SDV attempts:

```text
$PDV3_ROOT/hlt_multiview_source_fusion/
```

Suggested layout:

```text
source_models/
  hlt_part_seed8801/
  hlt2_part_s0p10_seed8811/
  hlt2_part_s0p20_seed8821/
  hlt2_part_s0p35_seed8831/
  hlt2_part_s1p00_seed8841/

hlt_random_seed_controls/
  hlt_part_seed9101/
  hlt_part_seed9102/
  hlt_part_seed9103/
  hlt_part_seed9104/

logit_fusions/
  source_5view/
  hlt_random_4seed/
  pretrained_dualview_4model/
  scratch_dualview_4model/

particle_dualview_pretrained/
  sdv_hlt_hlt2_s0p10/
  sdv_hlt_hlt2_s0p20/
  sdv_hlt_hlt2_s0p35/
  sdv_hlt_hlt2_s1p00/

particle_dualview_scratch/
  sdv_hlt_hlt2_s0p10/
  sdv_hlt_hlt2_s0p20/
  sdv_hlt_hlt2_s0p35/
  sdv_hlt_hlt2_s1p00/

controls/
  sdv_hlt_hlt_same_view/
  tta_hlt_part_hlt_plus_hlt2_s0p10/
  tta_hlt_part_hlt_plus_hlt2_s0p20/
  tta_hlt_part_hlt_plus_hlt2_s0p35/
  tta_hlt_part_hlt_plus_hlt2_s1p00/

triview/
  tri_hlt_hlt2_s0p35_s1p00/

final_report/
```

## Run Family A: Scratch Single-View Source Models

Train five single-view ParT source models from scratch:

```text
hlt_part_seed8801
hlt2_part_s0p10_seed8811
hlt2_part_s0p20_seed8821
hlt2_part_s0p35_seed8831
hlt2_part_s1p00_seed8841
```

The HLT model uses `inputs/hlt_cache`. Each HLT2 model uses the matching
second-degradation cache. All five train on `model_train`, select by
`model_val` cross entropy, and evaluate final-test only after model-val
selection.

These source models answer:

```text
How much information survives in each deterministic HLT2 view by itself?
```

They also provide clean branch initializers for the pretrained particle
dual-view models.

## Run Family B: Five-Source Logit Fusion

After the five source models finish, evaluate a fixed logit fusion over:

```text
hlt_part_seed8801
hlt2_part_s0p10_seed8811
hlt2_part_s0p20_seed8821
hlt2_part_s0p35_seed8831
hlt2_part_s1p00_seed8841
```

Start with uniform logit averaging:

```text
logits_fused = mean(logits_i)
```

If implementation time is small, also fit model-val nonnegative weights with a
simple constrained optimizer and apply the selected weights once to final-test.
Final-test must not choose the weights.

This run answers:

```text
Does a simple ensemble of all deterministic HLT-derived source views already
capture the available gain?
```

## Run Family C: Random-Seed HLT Ensemble Control

Train four independent HLT-only ParT models from scratch:

```text
hlt_part_seed9101
hlt_part_seed9102
hlt_part_seed9103
hlt_part_seed9104
```

Then evaluate logit fusion over those four HLT models:

```text
hlt_random_4seed
```

This is an important control. If the five-source HLT/HLT2 ensemble only matches
the random HLT seed ensemble, then the gain is probably generic ensembling, not
useful deterministic HLT2 diversity.

## Run Family D: Pretrained Particle Dual-View Models

Train four two-branch particle fusion models:

```text
sdv_hlt_hlt2_s0p10
sdv_hlt_hlt2_s0p20
sdv_hlt_hlt2_s0p35
sdv_hlt_hlt2_s1p00
```

For each strength:

```text
branch A: HLT particles, initialized from hlt_part_seed8801
branch B: HLT2 particles, initialized from matching hlt2_part source model
fusion head: random initialization
```

Use the established particle dual-view fusion form:

```text
h_1 = HLT branch embedding
h_2 = HLT2 branch embedding

fusion_input = concat(h_1, h_2, abs(h_2 - h_1), h_1 * h_2)

fusion_repr =
  LayerNorm
  Linear -> GELU -> Dropout
  Linear -> GELU -> LayerNorm

logits = Linear(fusion_repr, 10)
```

Use a staged schedule:

```text
stage 1:
  freeze both branches
  train fusion head for 1 epoch

stage 2:
  unfreeze both branches
  fine-tune with small branch LR and larger fusion/head LR
```

These runs answer:

```text
Does learned particle-level fusion beat logit fusion when each branch starts
from a strong view-specific source model?
```

## Run Family E: Scratch Particle Dual-View Models

Train the same four HLT + HLT2 particle fusion models from scratch:

```text
sdv_hlt_hlt2_s0p10_scratch
sdv_hlt_hlt2_s0p20_scratch
sdv_hlt_hlt2_s0p35_scratch
sdv_hlt_hlt2_s1p00_scratch
```

Do not initialize either branch from source checkpoints. Do not use a head-only
warmup stage that freezes nonexistent pretrained features. Train the full
two-branch model jointly from the beginning.

These runs answer:

```text
Is pretraining the single-view branches necessary, or can the dual-view model
learn useful cross-view features directly?
```

## Run Family F: Logit Fusion Of Dual-View Models

After the four pretrained particle dual-view models finish, evaluate:

```text
pretrained_dualview_4model_logit_fusion
```

After the four scratch particle dual-view models finish, evaluate:

```text
scratch_dualview_4model_logit_fusion
```

Use uniform logit averaging first. Optional model-val-selected nonnegative
weights are allowed if the same rule is used for all fusion families.

These runs answer:

```text
Are different HLT2 strengths complementary even after learned particle fusion?
```

This also gives a fair comparison to:

```text
source_5view logit fusion
hlt_random_4seed logit fusion
```

## Run Family G: Core Controls

Keep these controls:

```text
sdv_hlt_hlt_same_view
tta_hlt_part_hlt_plus_hlt2_s0p10
tta_hlt_part_hlt_plus_hlt2_s0p20
tta_hlt_part_hlt_plus_hlt2_s0p35
tta_hlt_part_hlt_plus_hlt2_s1p00
```

`sdv_hlt_hlt_same_view` checks whether two branches and a fusion head help even
without any second view.

The TTA controls use the single HLT model on HLT and HLT2 views, then average
logits. They check whether the benefit is just cheap test-time augmentation
rather than learned dual-view fusion.

## Run Family H: Tri-View Test

Run one tri-view model:

```text
tri_hlt_hlt2_s0p35_s1p00
```

Use three pretrained branches:

```text
HLT branch:       hlt_part_seed8801
HLT2 s0p35 branch: hlt2_part_s0p35_seed8831
HLT2 s1p00 branch: hlt2_part_s1p00_seed8841
```

This is the first multiview-capacity test beyond pairwise dual-view. It should
not block the main dual-view report, but it should be included in the final
summary if it completes.

Hold off on four-view until the tri-view result is known. Four-view is likely
worth trying if tri-view beats the best two-view model or if the logit-fusion
results show clear complementarity across HLT2 strengths.

## Selection And Reporting Rules

Every trained model must obey:

```text
train on model_train only
select checkpoint by model_val cross entropy
tie-break by model_val accuracy
evaluate final_test only after selection
```

Every report must include:

```text
model_val accuracy and cross entropy
final_test accuracy and cross entropy
per-class precision/recall/AUC
binary projection metrics
confusion matrix
ECE / calibration metrics where already supported
```

The headline table should compare:

```text
base HLT source model
best HLT2-only source model
HLT random 4-seed logit fusion
HLT/HLT2 source 5-view logit fusion
best pretrained particle dual-view
pretrained particle dual-view 4-model logit fusion
best scratch particle dual-view
scratch particle dual-view 4-model logit fusion
same-view particle dual-view control
best TTA control
tri-view model
```

## Expected Interpretation

The most interesting positive signal would be:

```text
best particle dual-view > source logit fusion > random HLT seed fusion > base HLT
```

That would suggest that deterministic HLT2 views provide useful, deployable
view diversity, and that particle-level learned fusion uses that diversity
better than a pure logit ensemble.

A weaker but still useful result would be:

```text
source logit fusion > random HLT seed fusion > base HLT
```

That would say HLT2 views are useful, but the current particle fusion model is
not yet the right way to use them.

A negative result would be:

```text
random HLT seed fusion >= all HLT2 methods
```

That would imply the apparent gain is mostly ordinary model ensembling, not the
deterministic degradation strategy.

## Implementation Steps

1. Add a new HLT-MV config layer and output layout helpers for the PDV3 root.

2. Add source-model training wrappers for HLT and HLT2 caches, with scratch
   training defaults and no AMP by default.

3. Add source-model prediction caching for `model_val` and `final_test`.

4. Add logit-fusion reporting for arbitrary source model lists:
   `source_5view`, `hlt_random_4seed`, pretrained dual-view four-model fusion,
   and scratch dual-view four-model fusion.

5. Reuse the existing HLT-SDV particle fusion model for pretrained branch
   dual-view runs, but wire branch A/B initialization to the matching source
   checkpoints.

6. Reuse the same HLT-SDV model for scratch dual-view runs with branch
   initialization disabled and head warmup disabled.

7. Add or adapt tri-view training so it consumes the canonical HLT source,
   HLT2 s0p35 source, and HLT2 s1p00 source checkpoints.

8. Add Slurm submitters that reuse existing HLT and HLT2 caches, never rebuild
   caches unless explicitly requested, and create one final report dependency
   graph.

9. Add smoke/dry-run tests for the dependency graph, output paths, no-offline
   input contract, and no accidental cache rebuilds.

10. Queue the PDV3-only high-data run after dry-run output confirms the graph:
    sources first, then logit source fusion and particle dual-view models, then
    dual-view logit fusions, controls, tri-view, and final report.
