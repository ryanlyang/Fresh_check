# Local Graph Particle Transformer Protocol

This note freezes the target experiment for the local graph Particle Transformer work. It is intentionally narrow so that all future runners, reports, and diagnostics compare the same thing.

## Frozen Task

- Task: QCD vs Hgg binary classification.
- Original JetClass labels: QCD is `0`, Hgg is `3`.
- Binary remapped labels after building the QCD/Hgg cache: QCD is `0`, Hgg is `1`.
- Inference view: HLT only.
- Offline information at inference: not allowed.
- HLT degradation strength: `0.6`.
- Goal: beat the HLT ParT baseline on the same HLT cache and identical splits.

This is not a reconstruction experiment. The local graph model is an HLT-side classifier meant to test whether local eta-phi structure before global ParT-style attention can extract information that the baseline HLT ParT underuses.

## Split Protocol

The split names and maximum jet counts are fixed:

| Split | Jets | Purpose |
| --- | ---: | --- |
| `model_train` | 500000 | Train model weights |
| `model_val` | 150000 | Select checkpoints |
| `stack_train` | 500000 | Reserved for stacked/comparison training |
| `stack_val` | 150000 | Report unbiased validation diagnostics |
| `final_test` | 500000 | Final held-out comparison |

Checkpoint selection must use `model_val` only. `final_test` must be evaluated only when `confirm_final_test` is explicitly true.

## Metric Protocol

Primary metric:

- `fpr_at_signal_eff_0p50`, minimized.

Secondary metrics:

- `background_rejection_at_signal_eff_0p50`, maximized.
- `fpr_at_signal_eff_0p30`, minimized.
- `auc`, maximized.
- `accuracy`, maximized as a sanity metric only.

Accuracy must not be used as the binary checkpoint-selection metric for this protocol.

## Baseline And Required Variants

The exact baseline is:

- `hlt_part_baseline`: HLT ParT trained on the same HLT cache, labels, split sizes, and selection metric.

The first serious local graph comparison should include:

- `local_edgeconv_adapter`
- `local_point_attention_adapter`
- `local_point_attention_adapter_warmstart`

The comparison is only meaningful if these variants use the same HLT degradation strength, same binary labels, same split names, and same final-test confirmation rule as the baseline.

## Machine-Readable Contract

The importable protocol lives in:

```text
teacher_logit_reco/local_graph_part/protocol.py
```

The canonical contract id is:

```text
local_graph_part_qcd_hgg_hlt06_protocol_v1
```
