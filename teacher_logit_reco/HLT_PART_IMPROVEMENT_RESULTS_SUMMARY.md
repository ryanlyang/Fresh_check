# HLT ParT Improvement Results So Far

Last updated: 2026-07-06

This note summarizes the HLT ParT improvement experiments we have run so far, what has and has not worked, and exactly what the promising setups change about ParT. The goal is to keep the story straight: several methods sound similar in conversation, but they modify different points in the ParT pathway.

The main working domain for the strongest results so far is:

- JetClass 10-class tagging.
- Fixed HLT degradation strength `0.6`.
- HLT-only inference.
- Canonical ParT-style particle inputs.
- Metrics centered on final-test 10-class accuracy, plus binary projections such as QCD-vs-Hgg FPR@50 where available.

For older binary studies, the main setup was QCD-vs-Hgg at HLT degradation `0.6`, with FPR at 50 percent signal efficiency as the primary metric.

## Short Version

The clearest wins so far have come from adding a small learned adapter around a real HLT ParT, rather than replacing ParT.

The best current story is:

1. Train or load a strong HLT ParT baseline.
2. Keep the ParT backbone as the final reasoner.
3. Add a small per-particle residual signal before the ParT transformer blocks.
4. Fine-tune the combined model.

The strongest old high-data 10-class result was:

| Run | Final-test accuracy |
|---|---:|
| HLT ParT baseline | `0.745424` |
| Best individual AV10 model, `av10_context_mlp_control` | `0.752787` |
| Best AV10 fusion | `0.752929` |

That is about `+0.0075` absolute accuracy over HLT ParT, roughly `3%` relative error reduction.

The strongest newer 500k ablation pilot result was:

| Run | Final-test accuracy |
|---|---:|
| HLT ParT baseline recheck | `0.733270` |
| Best individual, `av10_shuffled_feature_adapter` | `0.739700` |
| `av10_lc_mlp_delta_features` | `0.739634` |
| `av10_feature_mlp_adapter` | `0.739422` |
| Best ablation fusion | `0.741916` |

That result is exciting, but also diagnostic: the shuffled-feature adapter doing very well means part of the gain may be from adapter capacity, regularization, optimization dynamics, or residual embedding flexibility, not purely from semantically meaningful feature structure.

## What Worked Best

### 1. Architecture-view residual adapters into ParT embeddings

This is the most important family so far.

It does not replace ParT. It starts from the real HLT ParT pathway, then adds a learned residual signal into the per-particle embedding before the ParT transformer blocks.

Conceptually:

```text
HLT particle features F_i
  -> canonical ParT embedding
  -> h_i_base

architecture or feature adapter view
  -> delta_h_i
  -> optional gate_i

h_i_final = h_i_base + gate_i * delta_h_i

h_i_final
  -> normal ParT transformer blocks
  -> normal ParT classifier
```

What changes about ParT:

- The final model is still a ParT-style transformer over particles.
- The particle sequence, mask, and main ParT attention backbone are still used.
- The adapter changes the hidden particle embedding `h_i` before global ParT attention.
- It does not directly add new raw PF columns to the dataset.
- It does not replace the ParT classifier with PN, PFN, or PCNN.
- The useful branch gives ParT an extra residual hint per particle, and ParT decides how to use it.

Why this seems to work:

- ParT remains the strongest final reasoner.
- The adapter gives ParT a different local or feature-wise transformation of each particle.
- The adapter is small enough that it can help without turning the experiment into a completely different model.
- Zero-initialized residual projections let the model begin at the baseline ParT behavior, then learn only useful deviations.

Important variants:

| Variant | What it feeds into ParT |
|---|---|
| `av10_context_mlp_control` / `av10_feature_mlp_adapter` | Per-particle canonical features pass through an MLP/context adapter to predict `delta_h`. |
| `av10_pfn_context_to_part` | A PFN-style per-particle/global context branch predicts `delta_h`. |
| `av10_pcnn_context_to_part` | A PCNN-style particle context branch predicts `delta_h`. |
| `av10_pn_context_to_part` | A ParticleNet/PN-style local graph branch predicts `delta_h`. |
| `av10_all_views_to_part` | Multiple architecture views are combined before predicting `delta_h`. |
| `av10_part_only_adapter` | Uses ParT's own embedding to predict `delta_h`, with no raw-feature view. |
| `av10_shuffled_feature_adapter` | Uses deliberately shuffled feature semantics as a control. |

The old high-data result was especially interesting because the plain feature MLP control was the best individual model, not the more elaborate architecture views.

That means the current evidence says:

- Residual context into ParT embeddings is valuable.
- The best gain may not require a full PN/PFN/PCNN side model.
- Some of the benefit may be a learned pre-transformer adapter or optimization improvement.
- We need controls to separate "new architecture information" from "extra residual adapter capacity."

### 2. Feature MLP adapter

This is the surprisingly strong simple version.

Setup:

```text
canonical HLT features F_i
  -> small MLP or shallow context block
  -> delta_h_i

ParT embedding h_i_base
  -> h_i_base + gate_i * delta_h_i
  -> ParT transformer
  -> classifier
```

What it changes about ParT:

- It does not alter the raw particle features.
- It does not add another full classifier.
- It adds a learned per-particle hidden correction after the ParT embedding.
- ParT then attends globally over corrected particle embeddings.

Best old high-data result:

| Model | Final-test accuracy |
|---|---:|
| HLT ParT baseline | `0.745424` |
| `av10_context_mlp_control` | `0.752787` |

Current 500k ablation result:

| Model | Final-test accuracy |
|---|---:|
| HLT ParT baseline recheck | `0.733270` |
| `av10_feature_mlp_adapter` | `0.739422` |

This setup is probably the cleanest current candidate for "simple ParT improvement." It is not exotic. It is a learned residual adapter into ParT's per-particle representation.

### 3. LC MLP Delta Feature-Input Adapter

This is related to local compression, but it changes a different part of ParT.

The feature MLP adapter modifies the hidden embedding `h`. The LC MLP Delta adapter modifies the input feature rows `F`.

Setup:

```text
canonical HLT features F_i
  -> MLP delta adapter
  -> bounded delta_F_i

F_i_adapted = F_i + delta_F_i

F_i_adapted
  -> normal ParT embedding
  -> normal ParT transformer
  -> classifier
```

What it changes about ParT:

- It changes the feature vector ParT embeds.
- It does not change particle order, particle mask, or the ParT attention backbone.
- It keeps points and Lorentz-vector side inputs unchanged.
- Because points/vectors remain unchanged, feature deltas are bounded and diagnostics matter.

Current 500k ablation result:

| Model | Final-test accuracy |
|---|---:|
| HLT ParT baseline recheck | `0.733270` |
| `av10_lc_mlp_delta_features` | `0.739634` |

This is one of the strongest current pilot results. It says that "learned correction to the canonical PF feature row before ParT" has legs. It is also closer to the old reconstruction/intelligent-input-correction intuition than the embedding adapter is.

The important distinction:

```text
Feature MLP adapter:
  F -> ParTEmbed(F) = h_base
  F -> MLP -> delta_h
  h_final = h_base + delta_h

LC MLP Delta:
  F -> MLP -> delta_F
  F_final = F + delta_F
  h = ParTEmbed(F_final)
```

### 4. Score fusion

Score fusion is not a new ParT architecture. It combines model outputs after training.

Setup:

```text
model_1 logits
model_2 logits
...
  -> scalar-weighted logit/probability fusion or stacking
  -> final prediction
```

What it changes about ParT:

- Nothing inside ParT.
- It only combines scores after separate models have made predictions.

Old high-data AV10 fusion:

| Fusion | Final-test accuracy |
|---|---:|
| `av10_with_controls / scalar_weighted_logit_mean` | `0.752929` |
| `av10_architecture_view_core / scalar_weighted_logit_mean` | `0.752869` |

Current 500k ablation fusion:

| Fusion | Final-test accuracy |
|---|---:|
| Best ablation fusion | `0.741916` |

Fusion helps, but the more important scientific result is still the individual adapter improvements. Fusion is useful for performance, but it is a less clean explanation.

### 5. Offline transfer of the feature adapter

We also tested whether the same idea helps an offline ParT, not only an HLT-degraded ParT.

Current 500k offline transfer result:

| Model | Final-test accuracy |
|---|---:|
| Offline ParT baseline | `0.803374` |
| Offline feature MLP adapter | `0.806670` |
| Offline PCNN context adapter | `0.805558` |

This is a big clue. If the adapter helps offline data too, then the mechanism may be broader than "repair HLT degradation." It may be a generally useful ParT adapter: a small learned per-particle representation correction before global attention.

That is good news, but it changes the story. It means the result may be about improving ParT itself, not only exploiting HLT-specific detector degradation.

## Main Results Tables

### Old clean 500k 10-class AV10 run

Run root:

```text
architecture_view_part_10class_hlt0p6_20260701_113409
```

Data:

| Split | Size |
|---|---:|
| train | `500k` |
| model_val | `150k` |
| stack_val | `150k` |
| final_test | `500k` |

Setup:

- 10-class JetClass.
- HLT degradation strength `0.6`.
- HLT-only final inference.
- Warm-started from HLT ParT baseline.

Final-test accuracy:

| Variant | Final-test accuracy |
|---|---:|
| HLT ParT baseline recheck | `0.730100` |
| `av10_all_views_to_part` | `0.738496` |
| `av10_pn_context_to_part` | `0.738476` |
| `av10_pcnn_context_to_part` | `0.738350` |
| `av10_context_mlp_control` | `0.738254` |
| `av10_pfn_context_to_part` | `0.738154` |

Best gain over baseline:

```text
0.738496 - 0.730100 = +0.008396 accuracy
```

That is about `3.1%` relative error reduction.

Binary projection FPR@50 on final test:

| Variant | QCD-vs-Hbb | QCD-vs-Tbqq | QCD-vs-Hgg |
|---|---:|---:|---:|
| HLT ParT baseline | `0.00036` | `0.00108` | `0.02136` |
| `av10_all_views_to_part` | `0.00032` | `0.00090` | `0.02026` |
| `av10_pn_context_to_part` | `0.00034` | `0.00090` | `0.02026` |
| `av10_pcnn_context_to_part` | `0.00034` | `0.00086` | `0.02006` |
| `av10_context_mlp_control` | `0.00032` | `0.00090` | `0.02026` |
| `av10_pfn_context_to_part` | `0.00036` | `0.00090` | `0.02022` |

### Old high-data 10-class AV10 run

Run root:

```text
architecture_view_10class_hlt0p6_highdata_20260701_202441
```

Data:

| Split | Size |
|---|---:|
| train | `3M` |
| model_val | `1M` |
| stack_val | `1M` |
| final_test | `1M` |

Notes:

- The old fusion cache used `1M` stack_train, not `3M`.
- This run used HLT degradation strength `0.6`.

Final-test accuracy:

| Variant | Final-test accuracy |
|---|---:|
| HLT ParT baseline recheck | `0.745424` |
| `av10_context_mlp_control` | `0.752787` |
| `av10_pcnn_context_to_part` | `0.752686` |
| `av10_pfn_context_to_part` | `0.752681` |
| `av10_all_views_to_part` | `0.743791` |
| `av10_random_view_control` | `0.743820` |
| `av10_pn_context_to_part` | `0.743544` |

Best individual gain:

```text
0.752787 - 0.745424 = +0.007363 accuracy
```

Best fusion:

| Fusion | Final-test accuracy |
|---|---:|
| `av10_with_controls / scalar_weighted_logit_mean` | `0.752929` |
| `av10_architecture_view_core / scalar_weighted_logit_mean` | `0.752869` |

Binary projection FPR@50 on final test:

| Variant | QCD-vs-Hbb | QCD-vs-Tbqq | QCD-vs-Hgg |
|---|---:|---:|---:|
| HLT ParT baseline | `0.00025` | `0.00044` | `0.01944` |
| `av10_context_mlp_control` | `0.00017` | `0.00041` | `0.01809` |
| `av10_pcnn_context_to_part` | `0.00018` | `0.00043` | `0.01802` |
| `av10_pfn_context_to_part` | `0.00016` | `0.00043` | `0.01809` |
| `av10_all_views_to_part` | `0.00023` | `0.00048` | `0.01946` |
| `av10_random_view_control` | `0.00025` | `0.00047` | `0.01948` |
| `av10_pn_context_to_part` | `0.00023` | `0.00047` | `0.01927` |

Fusion binary projection FPR@50:

| Fusion | QCD-vs-Hbb | QCD-vs-Tbqq | QCD-vs-Hgg |
|---|---:|---:|---:|
| `av10_all_available / scalar_weighted_logit_mean` | `0.00019` | `0.00044` | `0.01821` |
| `av10_architecture_view_core / scalar_weighted_logit_mean` | `0.00018` | `0.00041` | `0.01831` |
| `av10_with_controls / scalar_weighted_logit_mean` | `0.00018` | `0.00041` | `0.01840` |

Interesting detail:

- Best 10-class accuracy came from `av10_context_mlp_control`.
- Best QCD-vs-Hgg FPR@50 among individual variants came from `av10_pcnn_context_to_part` at `0.01802`.
- The all-views model underperformed at high data, even though it looked good in the 500k run.

### Old binary QCD-vs-Hgg architecture-view run

Run root:

```text
architecture_view_part_qcd_hgg_binary_hlt0p6_500k_pilot_20260701_014844
```

Data:

| Split | Size |
|---|---:|
| train | about `500k` |
| validation | about `150k` |
| final_test | about `500k` |

Setup:

- Binary QCD-vs-Hgg.
- HLT degradation strength `0.6`.
- Primary metric: FPR@50, lower is better.

Final-test results:

| Variant | Final FPR@50 | Final accuracy |
|---|---:|---:|
| `av_baseline_recheck` | `0.020624` | `0.882968` |
| `av_pfn_only` | `0.019956` | `0.885472` |
| `av_random_view_control` | `0.019988` | `0.885484` |
| `av_all_views` | `0.020012` | `0.885480` |
| `av_context_mlp_control` | `0.020232` | `0.885204` |
| `av_pn_only` | `0.020488` | not listed here |
| `av_pcnn_only` | `0.020536` | not listed here |

Best binary gain:

```text
0.020624 -> 0.019956
```

That is about a `3.2%` relative FPR@50 reduction.

### Current 500k AV10 ablation pilot

Run root:

```text
architecture_view_10class_ablation_hlt0p6_pilot_20260703_225629
```

Data:

| Split | Size |
|---|---:|
| train | `500k` |
| model_val | `150k` |
| stack_val | `150k` |
| final_test | `500k` |

Final-test accuracy:

| Variant | Final-test accuracy |
|---|---:|
| `av10_hlt_baseline_recheck` | `0.733270` |
| `av10_shuffled_feature_adapter` | `0.739700` |
| `av10_lc_mlp_delta_features` | `0.739634` |
| `av10_feature_mlp_adapter` | `0.739422` |
| `av10_part_only_adapter` | `0.739368` |
| `av10_feature_mlp_adapter_wide` | `0.739162` |
| `av10_pcnn_context_repeat` | `0.738956` |
| `av10_pfn_context_repeat` | `0.738878` |
| `av10_extra_part_block` | `0.737886` |
| `av10_frozen_part_feature_adapter` | `0.737584` |
| `av10_larger_part` | `0.721196` |

Best fusion:

| Fusion | Final-test accuracy |
|---|---:|
| `av10_all_ablation / scalar_weighted_logit_mean` | `0.741916` |

Interpretation:

- Most adapter variants beat the HLT ParT baseline.
- `av10_lc_mlp_delta_features` is strong, supporting input-feature residual correction.
- `av10_feature_mlp_adapter` is strong, supporting embedding-space residual correction.
- `av10_part_only_adapter` is also strong, which suggests that the residual adapter pathway itself is powerful even without raw feature context.
- `av10_shuffled_feature_adapter` being the best individual pilot result is a warning: some improvement may be due to capacity, optimization, or regularization rather than semantic particle feature reasoning.
- `av10_frozen_part_feature_adapter` still beats baseline, which helps argue the gain is not only "we fine-tuned ParT for longer."
- `av10_larger_part` did not work in this pilot, but one under-tuned larger-ParT run is not a final verdict on capacity scaling.

### Current high-data AV10 ablation status

Run root:

```text
architecture_view_10class_ablation_hlt0p6_highdata_20260703_225629
```

The high-data ablation campaign is still incomplete in the downloaded diagnostics summarized here. The confirmed HLT baseline recheck result is:

| Variant | model_val accuracy | stack_val accuracy | final-test accuracy |
|---|---:|---:|---:|
| `av10_hlt_baseline_recheck` | `0.750513` | `0.750295` | `0.750823` |

This is not directly comparable to the older high-data baseline `0.745424` because the research-compute checkpoint folder was deleted and the cache/splits were rebuilt. Future high-data ablation comparisons should use this newer baseline as their local reference.

## What Has Not Worked As Clearly

### Local graph particle transformer

The local graph idea was:

```text
HLT particles
  -> local kNN / EdgeConv / point-attention neighborhood module
  -> enriched particles
  -> real ParT-style global attention
  -> classifier
```

Motivation:

- HLT degradation is local.
- Dropped, merged, smeared, or reassigned particles should be easier to reason about with local neighborhoods.

Outcome:

- The implementation became much more faithful after review: real ParT backbone, local adapters, protocol checks, and fusion.
- Individual local graph models did not become the clear breakthrough.
- Score fusion with local graph could help slightly, but the gains were small compared with the AV10 adapter results.

Current judgment:

- Good idea, but not the leading branch right now.
- It may still be useful as a diversity source in fusion or as a residual expert, but it is not yet the best single model.

### HLT4 heterogeneous score fusion

The HLT4 idea was:

```text
Train separate HLT models:
  ParT
  PN
  PFN
  PCNN

Fuse their scores.
```

Old high-data standalone final-test accuracies:

| Model | Final-test accuracy |
|---|---:|
| HLT ParT | `0.745834` |
| HLT PCNN | `0.692595` |
| HLT PFN | `0.682692` |
| HLT PN | `0.661307` |

Best old HLT4 fusion:

| Fusion | Final-test accuracy |
|---|---:|
| HLT4 `logits_probs` fusion | `0.748611` |

Outcome:

- Fusion improves over standalone HLT ParT.
- It does not reach the old AV10 best result around `0.7529`.

Current judgment:

- Useful baseline and advisor-facing control.
- Not the strongest improvement mechanism by itself.

### All architecture views together

The all-views model looked strong in the 500k pilot:

```text
av10_all_views_to_part final_test = 0.738496
baseline = 0.730100
```

But in the old high-data run:

```text
av10_all_views_to_part final_test = 0.743791
baseline = 0.745424
```

So the all-views model did not scale cleanly in that run. Combining many views can add noise, optimization difficulty, or redundant signals. The single-view or simple feature adapter variants are currently cleaner.

### Larger vanilla ParT

The newer ablation included a larger vanilla ParT-style control:

```text
av10_larger_part final_test = 0.721196
baseline = 0.733270
```

This did not work in the pilot. However, this should not be overinterpreted as "larger ParT can never help." A bigger ParT may need different learning rate, schedule, warmup, regularization, or more data. The immediate result only says that the current larger-ParT control did not explain the adapter gains.

### Reconstruction-style approaches

The reconstruction intuition was:

```text
HLT particles or jet summaries
  -> reconstruct offline-like information
  -> use that improved representation for tagging
```

Outcome so far:

- Reconstruction and residual-target ideas can help in some lower-data or binary settings.
- The gains have felt less robust at high data.
- A strong ParT can often learn direct tagging cues from HLT data without needing explicit reconstruction.

Current judgment:

- The useful part may not be reconstructing offline particles literally.
- A better version may be "learn a latent correction or context useful for tagging," which is exactly what the AV10 adapters are doing more directly.

## Setup Details To Avoid Confusion

### Are the AV10 models trained from scratch?

No, not in the main AV10 setup.

The AV10 adapter models generally start from a pretrained HLT ParT baseline checkpoint. Then they add an adapter and fine-tune. The baseline recheck is used to verify the starting HLT ParT under the same HLT cache and split setup.

The important distinction:

```text
From scratch:
  randomly initialized ParT -> train everything

AV10:
  trained HLT ParT baseline -> load weights
  add zero-init adapter
  train/fine-tune adapter + usually ParT at smaller LR
```

Because of this, we still need clean controls for:

- pure HLT ParT fine-tune with no adapter,
- adapter-only with frozen ParT,
- parameter-matched larger or extra-block ParT,
- shuffled/broken semantic controls.

Some of these controls now exist in the ablation campaign, but the high-data set of controls is still not complete in the downloaded results.

### What is an embedding in this context?

An embedding is a learned vector representation for each particle. ParT first maps raw/canonical particle feature rows into hidden vectors. Those hidden vectors are what the transformer blocks attend over.

For a particle `i`:

```text
F_i = canonical feature row
h_i = ParT embedding vector
```

The embedding adapter modifies `h_i`, not `F_i`.

### What does "learned residual context signal into the ParT pathway" mean?

It means the model learns a correction vector and adds it to the ParT particle embedding:

```text
h_i_final = h_i_base + delta_h_i
```

or with a gate:

```text
h_i_final = h_i_base + gate_i * delta_h_i
```

Then ParT continues normally.

This is not score fusion. It is internal feature/embedding fusion before ParT makes its prediction.

### What is the difference between AV10 and LC MLP Delta?

They modify different objects.

AV10 feature/architecture adapters:

```text
F_i -> ParT embedding -> h_i_base
adapter -> delta_h_i
h_i_final = h_i_base + delta_h_i
```

LC MLP Delta:

```text
F_i -> adapter -> delta_F_i
F_i_final = F_i + delta_F_i
F_i_final -> ParT embedding -> h_i
```

So:

- AV10 changes the hidden representation after embedding.
- LC MLP Delta changes the feature row before embedding.

### What is the difference between architecture-view and context MLP control?

Architecture-view variants try to use different inductive biases:

- PFN-style view: per-particle MLP plus global pooling.
- PCNN-style view: local/ordered particle pattern extraction.
- PN-style view: local graph/neighborhood reasoning.

The context MLP control does not use PN/PFN/PCNN. It just learns a residual context from canonical features. It was intended as a control, but it became one of the strongest models.

That means the current best result may be less about "PFN reasoning transferred into ParT" and more about "a compact learned pre-transformer adapter improves ParT."

### Does AV10 add extra particle fields?

Not in the main embedding-adapter setup.

It does not literally append PN/PFN/PCNN outputs as extra columns in the 19-ish input feature vector. Instead, it projects those views into a residual hidden vector `delta_h` and adds that vector to ParT's particle embedding.

The LC MLP Delta variant is closer to changing fields, because it modifies the canonical feature row before ParT embedding.

## Interpretation So Far

The most plausible current explanation is:

1. HLT ParT is already very strong.
2. Replacing ParT is hard.
3. But ParT may benefit from a small learned residual adapter before global attention.
4. That adapter can reshape each particle representation in a way the baseline embedding does not.
5. The gain does not require a full extra architecture view, because feature MLP and part-only adapters also work.
6. The gain is not purely ordinary score fusion, because individual internal-adapter models beat baseline.
7. The gain is not only HLT-specific, because offline feature adapter also improved offline ParT in the pilot.

The awkward but important finding:

```text
av10_shuffled_feature_adapter was the best current 500k individual ablation.
```

That means we should be cautious about claiming the adapter is using physically meaningful feature semantics. The safer claim is:

```text
Small residual adapters into or before ParT particle representations improve ParT performance in these runs.
```

Then the next question is whether the useful effect is:

- semantic feature processing,
- extra local/nonlinear per-particle compute,
- better optimization from zero-init residual pathways,
- regularization,
- implicit ensembling inside the network,
- or some combination.

## What To Trust Most Right Now

Most trusted positive signals:

1. Old high-data `av10_context_mlp_control` beat HLT ParT by about `+0.0074` accuracy.
2. Old high-data PCNN/PFN context variants were essentially tied with the feature MLP improvement.
3. Current pilot `av10_lc_mlp_delta_features` and `av10_feature_mlp_adapter` both beat baseline.
4. Offline feature adapter beat offline ParT in the current pilot.
5. Frozen-ParT adapter improved over baseline in the current pilot, so not all gain requires fine-tuning the whole ParT.

Less trusted or more ambiguous signals:

1. The shuffled-feature adapter winning the pilot complicates the physical interpretation.
2. The old all-views model did not hold up at high data.
3. The bigger-ParT control was poor, but may be under-tuned.
4. Local graph methods have not shown a clear high-data breakthrough.
5. Reconstruction-style improvements have not yet looked as robust as direct adapter improvements.

## Next Controls That Matter

The most important follow-ups are:

1. Complete the high-data ablation campaign under the rebuilt cache/split setup.
2. Run a pure HLT ParT fine-tune-only control with no adapter.
3. Repeat the best feature adapter and shuffled adapter with multiple seeds.
4. Check whether shuffled-feature performance persists at high data.
5. Compare parameter-matched and schedule-matched controls more carefully.
6. Re-tune the larger-ParT control before making a final capacity claim.
7. Report binary projections such as QCD-vs-Hbb, QCD-vs-Tbqq, and QCD-vs-Hgg for the current ablation campaign.
8. For offline transfer, rerun offline baseline and offline feature adapter at higher data to test if the adapter improves ParT generally.

## Bottom Line

The strongest branch right now is not reconstruction, not standalone PN/PFN/PCNN, and not local graph replacement. It is:

```text
real HLT ParT
  + small residual per-particle adapter
  + ParT remains the final global reasoner
```

The most useful concrete implementations are:

```text
Feature MLP adapter:
  canonical features -> delta_h -> add to ParT embedding

LC MLP Delta:
  canonical features -> delta_F -> modify ParT input features

Architecture context adapters:
  PFN/PCNN/PN view -> delta_h -> add to ParT embedding
```

The headline result to remember:

```text
Old high-data 10-class:
  HLT ParT baseline: 0.745424
  best individual AV10: 0.752787
  best AV10 fusion: 0.752929
```

The current scientific caution:

```text
The adapter idea works, but we still need to prove exactly why it works.
```

