# Losses / Training Objective

These notes are for the slide that explains the simplified m2/reco7 reconstructor objective.

## Slide: Losses / Training Objective

**Main message**

The reconstructor is trained to turn the pseudo-HLT jet into an offline-like particle set using three reconstruction signals.

**Bullets**

- The model receives the pseudo-HLT jet as input.
- It predicts a corrected particle view.
- The corrected view is compared against the matched offline jet.
- The slide-level objective uses three terms:
  - **Set matching loss:** match the unordered predicted particle set to the offline particle set.
  - **Generation loss:** encourage the model to generate missing constituents, not only edit existing HLT particles.
  - **Jet pT loss:** keep the total reconstructed jet pT aligned with the offline jet pT.

**Equation**

```text
L = w_set L_set + w_gen L_gen + w_pT L_jet-pT
```

**Speaker phrasing**

The important thing is that the target is not particle-by-particle index matching. The offline and HLT views do not have the same observed constituent list, so the reconstruction loss has to treat particles as an unordered set. Then the generation term pushes the model to recover missing constituents, while the jet pT term keeps the whole reconstructed jet physically calibrated.

## Suggested Graphic

Use the generated JPG:

`teacher_logit_reco/presentation_assets/m2_losses_training_objective.jpg`

Suggested caption:

**The reconstructor is trained with an unordered set-level loss, a generation signal for missing constituents, and a jet-level pT calibration loss.**

Regenerate with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\make_m2_losses_training_objective_jpg.ps1
```
