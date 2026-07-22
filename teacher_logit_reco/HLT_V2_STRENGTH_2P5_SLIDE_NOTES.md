# Pseudo-HLT View: HLT v2 at Strength 2.5

These notes are a first slide section for explaining the pseudo-HLT view used in the reco7 / m2-style setup.

## Slide 1: Why We Build a Pseudo-HLT View

**Main message**

We start from JetClass offline particle jets and create a paired, lower-quality pseudo-HLT view of the same event.

**Bullets**

- JetClass gives us an offline-like constituent representation of each jet.
- We create a pseudo-HLT version by applying a deterministic degradation pipeline to the same jet.
- The offline jet and pseudo-HLT jet remain matched one-to-one.
- The label is unchanged; only the input view changes.
- During training, we can use both offline and pseudo-HLT views.
- At inference, the deployed model only gets the pseudo-HLT view.

**Speaker phrasing**

The key point is that this is not a new dataset with different events. It is the same jet seen through two resolutions. The offline view is privileged information available during training, while the HLT-like view is the runtime constraint.

## Slide 2: HLT v2 Degradation at Strength 2.5

**Main message**

HLT v2 strength 2.5 is an amplified stress-test version of the pseudo-HLT degradation.

**Bullets**

- The profile name in code is `fixed_hlt_v2_realistic`.
- The strength convention is:
  - `0.0`: exact offline identity
  - `1.0`: mild realistic target point
  - `2.5`: amplified stress test
- The v2 pipeline changes the constituent set, not the class label.
- It can remove soft particles, merge nearby particles, drop particles probabilistically, smear kinematics, and locally reassign dense particles.
- This creates a harder HLT-like view where the model must learn under degraded particle information.

**Speaker phrasing**

Strength 2.5 should be described as intentionally more aggressive than the nominal v2 target. It is useful because it makes the offline-to-HLT gap large enough that reconstruction, correction, or privileged-training methods have a real problem to solve.

## Slide 3: How Constituents Get Dropped

**Main message**

The pseudo-HLT view observes fewer constituents because some particles are removed by thresholding and efficiency loss.

**Bullets**

- **Thresholding:** very soft particles are removed before the HLT-like view is formed.
- **Efficiency loss:** even particles above threshold can be dropped probabilistically.
- Efficiency depends on pT turn-on, eta region, local particle density, and jet-level quality.
- The event and label are unchanged; the HLT-like view simply has fewer observed constituents.
- This is the first and most important source of information loss.

**Effective parameters at strength 2.5**

| Parameter | Value |
| --- | ---: |
| pT threshold | `0.50 GeV` |
| Merge radius | `0.00625` |
| Merge probability | `0.625` |
| Barrel efficiency plateau | `0.9975` |
| Endcap efficiency plateau | `0.9900` |
| Barrel pT turn-on / width | `0.50 / 0.15 GeV` |
| Endcap pT turn-on / width | `0.875 / 0.20 GeV` |
| Density loss scale | `0.0375` |
| Jet quality sigma | `0.0625` |
| Smear scale | `0.50` |
| Reassignment scale | `0.375` |

**Speaker phrasing**

For this slide, we are only focusing on particle loss. The HLT-like view is not just a noisier copy of the offline jet; some constituents disappear entirely. That matters because a reconstructor cannot simply correct coordinates. It also has to reason about missing information.

## Slide 4: How to Position This in the Full Setup

**Main message**

The pseudo-HLT view defines the inference-time limitation; the offline view defines the privileged training signal.

**Bullets**

- Input at runtime: pseudo-HLT jet only.
- Extra training signal: matched offline jet.
- Target behavior: improve HLT-time tagging without requiring offline inputs at inference.
- Reconstruction target: produce an offline-like corrected view from HLT.
- Tagging target: classify using only deployable HLT-derived information.
- Evaluation should compare against a strong HLT-only Particle Transformer baseline.

**Speaker phrasing**

This is the core research tension: offline data cannot magically add information at inference, but it may help train a model that extracts more of the information still present in the HLT view.

## Suggested Graphic

Use the simplified JPG graphic:

`teacher_logit_reco/presentation_assets/hlt_v2_strength_2p5_simple.jpg`

Suggested caption:

**Pseudo-HLT v2 at strength 2.5 drops constituents through low-pT thresholding and probabilistic efficiency loss, leaving the same jet with fewer observed particles.**

Regenerate the JPG with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\make_hlt_v2_strength_jpg.ps1
```
