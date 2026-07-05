# HLT V2: Realistic Mild-Degradation Protocol

## Short Version

The current fixed-HLT degradation profile is too harsh for the next round of
experiments. Even with `hlt_degradation_strength=0.2`, the generated HLT view
still drops roughly 16% of constituents and produces a 10-class ParT gap of
about seven accuracy points:

```text
offline ParT final-test accuracy: ~0.810
HLT ParT final-test accuracy:     ~0.737
gap:                              ~7.3 points
```

That is useful as a stress test, but it is probably not the best representation
of the realistic HLT regime we want to study. This plan introduces a new HLT
profile:

```text
fixed_hlt_v2_realistic
```

with a different semantic contract:

```text
HLT v2 strength = 0.0
  exactly offline

HLT v2 strength = 1.0
  the realistic mild-HLT target profile

HLT v2 strength > 1.0
  stress-test degradation, not the default scientific target
```

The important change is that `strength=1.0` should no longer mean "full harsh
degradation." It should mean "the realistic HLT working point we actually want
to compare against."

## Motivation

The old profile came from a fixed same-HLT corruption used for a more severe
replication/stress-test setting. It has a useful implementation, but its
strength knob is misleading for our current purpose:

```text
old strength = 1.0
  original harsh fixed-HLT profile

old strength = 0.2
  20% of some parameter values, but not 20% of the performance degradation
```

Several pieces of the old implementation remain active even at low strength:

```text
density efficiency loss
fixed endcap turn-on offset
tail probability
small smearing floors
efficiency clipping below 1.0
merge-radius floor
```

As a result, `old strength=0.2` is still materially lossy:

```text
drop_total_fraction: ~0.166
drop_eff_fraction:   ~0.154
mean offline constits: ~39.3
mean HLT constits:     ~32.8
```

For the next PDV3 / AV10 / privileged-distillation experiments, we want a more
realistic baseline question:

```text
Can adapters and privileged distillation improve a strong HLT ParT when HLT is
only moderately worse than offline?
```

The desired regime is:

```text
offline ParT accuracy: ~81%
HLT v2 ParT accuracy:  ~78-79%
gap:                   ~2-3 accuracy points
```

This is a more believable and more useful gap. It gives the student room to
improve without turning the problem into "recover from a heavily damaged event."

## Scientific Contract

HLT v2 must satisfy four contracts.

### 1. Identity at Strength Zero

At `strength=0.0`, HLT v2 must be exactly offline:

```text
tokens_hlt == tokens_offline
mask_hlt   == mask_offline
```

No stochastic keep/drop, no pT smearing, no merge radius floor, no tail
probability, no sorting differences beyond the existing offline ordering. This
gives us a hard sanity check:

```text
HLT v2 strength 0.0 ParT ~= offline ParT
```

If that check fails, the HLT v2 implementation is wrong.

### 2. Realistic Target at Strength One

At `strength=1.0`, HLT v2 should be the default realistic profile. The target is
not exact detector simulation; it is a controlled JetClass HLT proxy whose
observable severity is mild:

```text
drop_total_fraction target:        ~0.05 to 0.08
mean HLT constituents target:      ~36 to 37 when offline mean is ~39
HLT-vs-offline 10-class ParT gap:  ~2 to 3 accuracy points
```

The exact numbers should be selected using model-val only. Final-test labels are
not used to choose the profile.

### 3. All Degradation Sources Scale

Every corruption source must scale with strength:

```text
threshold drop
local merging
efficiency plateau loss
turn-on loss
density-dependent efficiency loss
jet-quality efficiency variation
pT / eta / phi smearing
tail probability
local reassignment
```

No hidden degradation floor should remain when `strength=0.0`.

### 4. Versioned Metadata

Every cache must record:

```text
hlt_profile_name: fixed_hlt_v2_realistic
hlt_profile_version: v1
hlt_degradation_strength: <float>
hlt_params: {...}
hlt_content_hash: <hash>
split_manifest_hash: <hash>
```

Old `fixed_hlt` and new `fixed_hlt_v2_realistic` caches must never silently pass
as one another.

## Proposed HLT V2 Profile

HLT v2 should use a new parameter object rather than overloading the old
`FixedHLTParams` too much. The old dataclass does not expose all the knobs we
need to scale cleanly.

Suggested new fields:

```python
@dataclass(frozen=True)
class FixedHLTV2Params:
    profile_name: str = "fixed_hlt_v2_realistic"
    profile_version: str = "v1"

    # Low-pT threshold.
    hlt_pt_threshold: float = 0.20

    # Local merge behavior.
    merge_radius: float = 0.0025
    merge_probability: float = 0.25

    # Efficiency model.
    eff_plateau_barrel: float = 0.999
    eff_plateau_endcap: float = 0.996
    eff_turnon_pt_barrel: float = 0.20
    eff_turnon_pt_endcap: float = 0.35
    eff_width_pt_barrel: float = 0.06
    eff_width_pt_endcap: float = 0.08
    density_loss_scale: float = 0.015
    jet_quality_sigma: float = 0.025

    # Smearing and local confusion.
    smear_scale: float = 0.20
    tail_probability_base: float = 0.002
    tail_probability_eta: float = 0.0015
    tail_probability_density: float = 0.0015
    reassign_scale: float = 0.15
```

These are starting values, not sacred values. The calibration sweep decides
whether they are too strong or too weak.

### Scaling Semantics

`strength` scales away from the offline identity:

```text
strength = 0.0:
  exact copy of offline tokens and masks

strength = 1.0:
  use the realistic target values above

strength = s:
  threshold             = s * target_threshold
  merge_radius          = s * target_merge_radius
  merge_probability     = s * target_merge_probability
  plateau_loss          = s * target_plateau_loss
  turnon_pt             = s * target_turnon_pt
  width_pt              = max(s, eps) * target_width_pt when s > 0
  density_loss_scale    = s * target_density_loss_scale
  jet_quality_sigma     = s * target_jet_quality_sigma
  smear_scale           = s * target_smear_scale
  tail_probability      = s * target_tail_probability
  reassign_scale        = s * target_reassign_scale
```

For plateau values:

```text
plateau_barrel(s) = 1.0 - s * (1.0 - target_plateau_barrel)
plateau_endcap(s) = 1.0 - s * (1.0 - target_plateau_endcap)
```

For `s=0.0`, the builder should short-circuit to exact offline arrays rather
than relying on numeric formulas.

## Implementation Design

### Profile Selection

Add an explicit HLT profile argument everywhere we currently pass only
`hlt_degradation_strength`:

```text
--hlt-profile fixed_hlt_v1
--hlt-profile fixed_hlt_v2_realistic
```

Default behavior should be conservative:

```text
existing scripts:
  keep old profile unless intentionally changed

new PDV3/AV10 realistic scripts:
  require fixed_hlt_v2_realistic
```

This prevents accidental mixing of old and new results.

### Builder API

Add profile-aware helpers:

```python
fixed_hlt_params_from_profile(
    profile: str,
    strength: float,
) -> FixedHLTParams | FixedHLTV2Params

build_hlt_view_from_profile(
    tokens,
    mask,
    seed,
    profile,
    strength,
)
```

The old API can remain as a wrapper:

```python
fixed_hlt_params_from_strength(strength)
```

but new experiment code should call the profile-aware version.

### Cache Metadata

Every HLT cache metadata JSON should include:

```json
{
  "hlt_profile": "fixed_hlt_v2_realistic",
  "hlt_profile_version": "v1",
  "hlt_degradation_strength": 1.0,
  "hlt_params": {
    "...": "..."
  },
  "hlt_diagnostics_summary": {
    "drop_total_fraction": "...",
    "drop_eff_fraction": "...",
    "drop_merge_fraction": "...",
    "drop_threshold_fraction": "...",
    "mean_offline_constits": "...",
    "mean_hlt_constits": "...",
    "mean_merges_per_jet": "..."
  }
}
```

Loaders must check both:

```text
profile name
strength
```

not strength alone.

### Backward Compatibility

Old runs should remain readable:

```text
metadata missing hlt_profile -> interpret as fixed_hlt_v1
```

But new strict reports should require the expected profile explicitly.

## Calibration Plan

Before running expensive PDV3/AV10 ladders, run a calibration sweep.

### Cache-Only Sweep

Build HLT v2 caches for a small balanced subset:

```text
strengths:
  0.00
  0.50
  0.75
  1.00
  1.25
  1.50
```

Why these values?

```text
0.00:
  identity sanity check

0.50 / 0.75:
  possibly too mild, useful lower bounds

1.00:
  intended realistic target

1.25 / 1.50:
  stress tests in case v2 target is too mild
```

For each strength, report:

```text
drop_total_fraction
drop_eff_fraction
drop_merge_fraction
drop_threshold_fraction
mean_offline_constits
mean_hlt_constits
mean_merges_per_jet
pt response mean / p90 abs shift
eta/phi response p90 abs shift
per-class drop_total_fraction
per-class mean_hlt_constits
```

The first target is:

```text
strength=1.0:
  drop_total_fraction around 0.05 to 0.08
  no class with extremely outlying drop fraction
```

If strength 1.0 misses badly, adjust the target v2 parameters and rerun the
cache-only sweep. Do not tune using final-test accuracy.

### Small Baseline Sweep

Once cache diagnostics look reasonable, train small 10-class ParT baselines:

```text
offline ParT
HLT v2 strength 0.0 ParT
HLT v2 strength 0.75 ParT
HLT v2 strength 1.0 ParT
HLT v2 strength 1.25 ParT
```

Use the pilot split only and select checkpoints on model_val.

Expected checks:

```text
offline ParT ~= HLT v2 strength 0.0 ParT
HLT v2 strength 1.0 is ~2-3 points below offline on model_val
HLT v2 strength 1.25 is harsher than 1.0
```

If `strength=1.0` is too harsh, reduce target v2 parameters. If it is too mild,
increase them. The final chosen profile should still be called
`fixed_hlt_v2_realistic`; if the parameter values change after serious results
exist, bump:

```text
hlt_profile_version: v2
```

## Main Experimental Reset

Once HLT v2 is calibrated, rerun the serious ladders with:

```text
hlt_profile: fixed_hlt_v2_realistic
hlt_degradation_strength: 1.0
```

The first reruns should be:

```text
1. HLT ParT baseline
2. offline ParT reference
3. AV10 feature MLP adapter
4. AV10 LC MLP delta feature adapter
5. AV10 combined delta_F + delta_h adapter
6. PDV3 CE baseline
7. PDV3 V1 logit KD
8. PDV3 V2 logit + representation KD
9. PDV3 best combined adapter + V2 KD
```

The point is to reset the numbers under a realistic HLT gap, not to rerun every
old stress-test branch immediately.

## Reporting Requirements

Every report should include a protocol block:

```text
hlt_profile
hlt_profile_version
hlt_degradation_strength
hlt_params_hash
hlt_content_hash
split_manifest_hash
```

Every report should include HLT severity rows:

```text
drop_total_fraction
drop_eff_fraction
drop_merge_fraction
drop_threshold_fraction
mean_offline_constits
mean_hlt_constits
mean_merges_per_jet
per-class drop_total_fraction
```

Every final comparison table should label the profile:

```text
HLT v1 stress, strength=0.6
HLT v1 stress, strength=0.2
HLT v2 realistic, strength=1.0
```

Never compare these as if they are the same data regime.

## Success Criteria

HLT v2 is ready for serious model runs when all of the following are true:

```text
1. strength=0.0 produces exact offline arrays.
2. strength=1.0 cache diagnostics are mild:
   drop_total_fraction roughly 5-8%.
3. HLT v2 strength=1.0 ParT is 2-3 accuracy points below offline ParT
   on model_val.
4. profile metadata is strict enough that v1/v2 caches cannot be mixed.
5. pilot final-test is used only after strength/profile selection is frozen.
```

If the calibrated HLT gap lands closer to 1 point, the setting may be too easy.
If it lands closer to 5+ points, the setting is still too harsh for this
"realistic mild HLT" protocol.

## Implementation Steps

### Step 1: Add Profile-Aware HLT V2 Builder

Implement `fixed_hlt_v2_realistic` in `jetclass_fixed_hlt.py`.

Required behavior:

```text
strength=0.0 exact offline short-circuit
strength=1.0 target mild realistic profile
all degradation mechanisms scale with strength
profile/version included in parameter dictionary
```

Add tests for:

```text
strength=0 exact identity
strength increases drop/shift severity monotonically on a synthetic batch
v1 and v2 params produce different metadata hashes
```

### Step 2: Add Profile-Aware Cache CLI and Metadata Checks

Extend cache builders:

```text
scripts/build_fixed_hlt_cache.py
sbatch/run_pdv3_build_hlt_cache.sh
sbatch/run_pd10_build_hlt_cache.sh
```

with:

```text
--hlt-profile fixed_hlt_v2_realistic
--hlt-degradation-strength 1.0
```

Loaders and audits should validate:

```text
expected_hlt_profile
expected_hlt_profile_version
expected_hlt_degradation_strength
```

### Step 3: Add HLT V2 Calibration Sweep

Create a script such as:

```text
scripts/calibrate_hlt_v2_profile.py
```

It should build/read small cache subsets and write:

```text
hlt_v2_calibration_summary.csv
hlt_v2_calibration_summary.md
```

for strengths:

```text
0.00, 0.50, 0.75, 1.00, 1.25, 1.50
```

### Step 4: Add Pilot Baseline Sweep

Queue small ParT baselines for:

```text
offline
HLT v2 0.0
HLT v2 0.75
HLT v2 1.0
HLT v2 1.25
```

Report model-val first. Only use final-test after the v2 profile is frozen.

### Step 5: Reset PDV3 / AV10 Serious Runs to HLT V2

After profile selection, create new output roots named with `hltv2`:

```text
privileged_distill_v3_av10_adapter_hltv2real_pilot_<stamp>
privileged_distill_v3_av10_adapter_hltv2real_highdata_<stamp>
architecture_view_10class_ablation_hltv2real_pilot_<stamp>
architecture_view_10class_ablation_hltv2real_highdata_<stamp>
```

Do not reuse old `hlt0p2`, `hlt0p4`, or `hlt0p6` roots for HLT v2.

### Step 6: Compare Against Historical Stress-Test Results Carefully

Historical results remain useful, but only as stress-test context:

```text
old HLT v1 0.6:
  hard degradation benchmark

old HLT v1 0.2:
  still harsh due unscaled floors

new HLT v2 1.0:
  realistic mild benchmark
```

The primary claim should be made inside the HLT v2 protocol, not by mixing
performance numbers across v1 and v2.

