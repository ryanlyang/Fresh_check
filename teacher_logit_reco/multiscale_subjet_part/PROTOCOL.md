# Multi-Scale Subjet Particle Transformer Protocol

This package is frozen to the first serious multi-scale subjet experiment:

- Task: QCD vs Hgg
- Inference view: HLT only
- Offline at inference: not allowed
- HLT degradation strength: `0.6`
- Original JetClass labels: QCD=`0`, Hgg=`3`
- Binary-cache labels after filtering/remap: QCD=`0`, Hgg=`1`
- Primary metric: `fpr_at_signal_eff_0p50`
- Checkpoint selection: `model_val` FPR@50, lower is better
- Final comparison split: `final_test`
- Baseline: exact HLT ParT on the same HLT cache and splits
- Main variant: `multiscale_subjet_residual_part_adapter`
- Contract: `multiscale_subjet_part_qcd_hgg_hlt06_protocol_v1`

The main model is not a late-fusion ensemble. It is intended to be:

```text
raw HLT particles
  -> canonical ParT PF features F
  -> seeded multi-scale soft subjet tokens
  -> particle readback
  -> zero-initialized residual delta_F
  -> real HLT ParT backbone
```

Required first-run variants:

- `hlt_part_baseline`
- `multiscale_subjet_residual_part_adapter`
- `pure_perceiver_latent_control`
- `part_plus_random_subjet_control`

Default split caps:

```text
model_train: 500000
model_val:   150000
stack_train: 500000
stack_val:   150000
final_test:  500000
```

Offline information is not allowed at inference. Optional privileged losses can
only be added after the HLT-only architecture is stable and must not change the
deployed HLT-only input contract.
