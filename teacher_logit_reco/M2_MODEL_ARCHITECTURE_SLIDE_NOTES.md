# m2_base Reconstructor Architecture

## Slide: Model Architecture

**Main message**

The m2_base reconstructor encodes the pseudo-HLT particles once, then uses explicit branches plus a budget head to build three reconstructed particle views.

**Bullets**

- Input is the pseudo-HLT constituent set.
- A token encoder builds local particle representations.
- An attention pool builds a global jet summary.
- Token context and global context form a shared latent representation.
- Three candidate mechanisms produce candidate particles:
  - **Edit branch:** adjusts existing HLT particles.
  - **Split branch:** lets one parent particle become multiple child candidates.
  - **Generate branch:** creates new soft candidates for missing constituents.
- A **budget head** predicts counts/weights that control how much reconstructed content to keep.
- The branch outputs and budget head form a budgeted candidate bank.
- The model outputs **three reconstructed views**, not just one corrected view.

**Speaker phrasing**

The important design choice is that the model has explicit ways to explain the offline jet: it can edit observed HLT particles, split merged-looking particles, or generate particles that disappeared from the HLT view. The budget head then controls how many candidates survive and how strongly they contribute. The output is a small family of reconstructed views rather than a single deterministic correction.

## Suggested Graphic

Use:

`teacher_logit_reco/presentation_assets/model_architecture_simple.jpg`

Regenerate with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\make_model_architecture_jpg.ps1
```
