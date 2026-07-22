# Five Models and Fusion

## Slide: Five Models and Fusion

**Main message**

After training five reco7 variants, fusion is a cheap logit-level stage that learns how to combine their predictions.

**Bullets**

- Train/score the first five reco7 variants from `v2_original_mechanism_step7/reco7`.
- Cache each model's logits on held-out stack splits.
- Fit the fusion model on `stack_train`.
- Select fusion settings on `stack_val`.
- Evaluate once on `final_test`.
- The fusion model is lightweight because it operates on cached logits, not raw particles.

**Speaker phrasing**

The source models do the expensive work. Fusion is deliberately small: it only sees each model's class-logit vector and learns a calibrated combination. This lets us test whether the reco7 variants make complementary mistakes.

**Current graphic membership**

The graphic currently shows the first five variants from:

```text
v2_original_mechanism_step7/reco7/
```

Those five are:

- `m2_antioverlap`
- `m2_base`
- `m2_budgetlite`
- `m2_topk60ish`
- `m2_genlow`

## Suggested Graphic

Use:

`teacher_logit_reco/presentation_assets/five_models_fusion.jpg`

Regenerate with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\make_five_models_fusion_jpg.ps1
```
