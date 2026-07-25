# Prediction-Anchored Bridge: Shuffle and Radius Follow-up

Status: deferred research follow-up; this note does not change the locked pilot
protocol.

Evidence source: `full_pilot_20260723_202206`, B3 job `15694`, evaluated on
the sealed `model_val_select` split and aggregated over seeds 101, 202, and
303.

## Observation

The physical-45 clean consumer gained `0.0246444` accuracy from its matched
`f0` endpoint to the 10% bridge endpoint. Its negative-control gains over the
same `f0` endpoint were:

| Condition | Mean accuracy gain | Fraction of real bridge gain |
|---|---:|---:|
| Correct 10% bridge | 0.0246444 | 100.0% |
| Event-shuffled delta | 0.0034089 | 13.8% |
| Particle-shuffled delta within each jet | 0.0182400 | 74.0% |
| Radius-group-permuted delta | 0.0224000 | 90.9% |
| Same-norm random delta | -0.0472178 | harmful |
| Sign-reversed delta | -0.0260889 | harmful |

The robust consumer showed the same ordering: real bridge `0.0182622`,
event-shuffled `0.0028267`, particle-shuffled `0.0128844`, and
radius-group-permuted `0.0168800`.

## Current interpretation

This is not evidence that arbitrary extra fields help. Random and
sign-reversed fields hurt substantially, while shuffling fields between jets
destroys most of the useful gain. It is evidence that much of the usable
within-jet signal is insensitive to exact particle correspondence and exact
radius-block identity.

Plausible explanations to distinguish later:

1. The bridge primarily supplies jet-level or regional summary corrections.
2. The three current radius blocks are strongly redundant.
3. The consumer pools the fields enough that exact particle placement matters
   less than their within-jet distribution.
4. The shuffle controls preserve mask, ordering, or marginal structure that is
   more informative than intended.
5. A control-generation or data-lineage bug exists. The low event-shuffle
   gain and harmful random/sign controls make gross label leakage less likely,
   but they do not replace a direct audit.

## Audit before interpreting the result physically

- Verify wrong-event maps never preserve the original event and are created
  independently of labels.
- Verify particle permutations move only valid particles, are non-identity
  when at least two valid particles exist, and do not move padding.
- Verify radius permutation moves the intended 15-channel blocks and does not
  accidentally leave one dominant block fixed.
- Report identity/permutation rates and per-class control gains.
- Repeat the controls on at least one independently trained seed and, when
  available, on both the robust and clean consumers.

## Deferred field/radius experiments

Do not block the current 10% pilot on these. Revisit after the first
deployable robust-versus-clean reconstruction comparison.

1. Add wider target scales, initially `0.20` and `0.40`, while retaining
   `0.02`, `0.05`, and `0.10`.
2. Compare the current radius-separated 45-vector with a lower-dimensional
   regional/scale-pooled target whose statistics are deliberately easier to
   predict.
3. Compare radius-aware heads with shared or fused radius heads.
4. Measure per-radius predictability and downstream gain rather than assuming
   every scale deserves equal output capacity.
5. Expand the field only if wider scales add out-of-sample consumer gain and
   the HLT-only predictor can recover a meaningful fraction of that gain.

## Robust/clean continuation policy to test

Use the robust physical-45 consumer for the breadth of the main pilot, but
retain two matched clean-consumer comparisons:

1. Train the canonical A3 HLG primary objective against both consumers
   (`D10_A3_hlg_primary` and `D10_TALT_A3`).
2. Train the simple particle-only primary objective against both consumers
   (`D10_A0_c0_delta` and `D10_TALT_A0`).

These two pairs separate teacher choice from architecture choice while
keeping the extra clean-consumer cost small. The selection protocol now treats
performance-rule failures as explicit quality warnings rather than execution
stops. The clean comparisons remain diagnostics, and neither consumer is
silently relabeled as having passed a rule that it failed.

Evaluating the already teacher-independent `D10_L0_bridge_only` checkpoint
through the clean consumer remains a useful optional follow-up, but it is not a
separate training row in the pilot.

## Revisit trigger

Return to this note after the paired A0 and A3 robust/clean results exist,
or earlier if a direct control audit finds an implementation problem. If the
deployable field predictor recovers little of the real bridge gain while the
shuffle behavior repeats, prioritize a regional/pooled target before simply
adding more residual channels.
