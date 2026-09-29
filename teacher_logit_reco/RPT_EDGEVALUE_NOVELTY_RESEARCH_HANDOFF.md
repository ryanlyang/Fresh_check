# RPT EdgeValue: setup, results, and novelty research handoff

Prepared 2026-09-22. Purpose: assess prior art and the defensible novelty of this specific extension to Particle Transformer (ParT). **The generic operation of adding relation information to attention values is established prior art. First use in ParT or jet tagging has not been established.**

## 1. What exactly is EdgeValue?

An edge is an ordered particle pair `(i, j)`, where `i` receives information from `j`. EdgeValue is our implementation label for an additive, relation-conditioned value message; it is not a claim to have invented a new general attention family.

For one head of ordinary ParT, suppressing layer/head indices and output projections:

\[
\alpha_{ij}=\operatorname{softmax}_j\left(q_i^T k_j/\sqrt{d_h}+b_{ij}\right),
\qquad y_i=\sum_j\alpha_{ij}v_j.
\]

ParT embeds pairwise kinematics into a scalar attention bias per head. This changes how strongly a receiver weights each sender. Within a head and layer, the sender value `v_j` is the same vector for all receivers, although it can already contain context from previous layers. Original ParT shares its pair-bias tensor across particle-attention layers.

Our selected model constructs a learned pair embedding `r_ij` and uses it in two paths:

\[
b_{ij}^{\ell,h}=B^{\ell,h}(r_{ij}),\qquad
\alpha_{ij}^{\ell,h}=\operatorname{softmax}_j
\left((q_i^{\ell,h})^T k_j^{\ell,h}/\sqrt{d_h}+b_{ij}^{\ell,h}\right),
\]

\[
y_i^{\ell,h}=\sum_j\alpha_{ij}^{\ell,h}
\left(v_j^{\ell,h}+W_{\mathrm{edge}}^{\ell,h}r_{ij}\right).
\]

The bias changes the attention weights; the added vector carries information about the particular relationship into the output. For example, shared cluster membership can affect both how strongly two particles communicate and the content transmitted between them.

Implementation details relevant to equivalence with prior work:

- The pair stem is computed once per forward pass from input-derived pair features and reused across layers. It is not a recurrently updated edge state.
- Each particle-attention layer has an independent bias projection, including its supported Weaver projection/normalization tail. Each layer and head has a separate **linear, bias-free** edge-value projection.
- The added message is `W_edge r_ij`; it does not multiply `v_j`, use a pair-specific value matrix, or add new particle tokens.
- Ordinary and relation messages use the same attention weights, including attention dropout when active. Padding is masked. Heads are combined through the existing output projection; the class-attention readout and classifier are retained.
- By linearity, the relation term is computed as `W_edge(sum_j alpha_ij r_ij)`. This avoids allocating a `[batch, heads, particles, particles, head_width]` tensor. This algebraic rearrangement is not itself claimed as novel.

## 2. Pair information and experimental setup

The selected relations are **PT + TRACK + REGION**, chosen in the preceding HLT-like-input study and fixed before this offline replication:

- **Standard ParT base4:** log angular separation, log relative transverse-momentum scale, log momentum-sharing fraction, and log pair invariant mass squared.
- **PT:** directional momentum fractions, log ratios, asymmetry and particle momentum ranks; 10 raw features encoded to 8 channels.
- **TRACK:** impact parameters, uncertainties, significances, measurement-validity states and pair compatibility features; encoded to 12 channels. These are measurement-derived descriptors, not truth vertex labels.
- **REGION:** a deterministic beam-free angular clustering tree with exclusive resolutions `K = 2, 4, 8`. Its 41 raw features comprise 3 same-cluster indicators, 1 common-ancestor depth, 4 common-ancestor merge descriptors, 18 endpoint cluster descriptors, 6 within-cluster momentum fractions, 6 axis distances and 3 signed cluster-rank differences. They are encoded to 12 channels. Tree ancestry is reconstructed clustering ancestry, not generator ancestry.

The selected pair input therefore has `4 + 8 + 12 + 12 = 36` channels before the shared 64-channel pair stem. Pair information is derived from the same input view used by the classifier.

The offline study uses balanced ten-class JetClass data and exactly the parent HLT campaign's event identities and split assignments. The prescribed split counts are 1,000,000 training jets, 125,000 checkpoint-selection jets, 125,000 validation-reporting jets, and 500,000 final-test jets. This is a subset study, not a full 100M-training-jet benchmark.

All four configurations were trained from scratch at seeds **101, 202, 303**. The backbone uses 128-dimensional particle embeddings, 8 heads, 8 particle-attention layers and 2 class-attention layers. The common production protocol uses cross-entropy, AdamW, effective batch size 128, learning rate `1e-3` with warm-up/cosine decay to `1e-5`, weight decay `1e-4`, and a maximum of 40 epochs with early stopping (minimum 12 epochs, patience 8). Thus the protocol is shared, but realized training duration can differ.

Normalizers are fitted on offline training data. Checkpoints are selected using `model_val`; `stack_val` cannot remove any of the 12 predeclared final-test tasks. All checkpoints are locked before this campaign's final evaluation. The repository documents that these offline test identities may have appeared in older experiments: this is an offline-domain replication, not a claim of a globally untouched test set or transfer of pretrained HLT weights.

## 3. Reported offline final-test results

Source: the user's pasted output from `reports/offline_transfer_report.json`, supplied on 2026-08-13. These numbers have not been independently regenerated for this note. Campaign: `rpt_offline_transfer_20260809T195641Z_0737194d3c`.

| Configuration | Pair features | Bias across layers | Added edge values | Mean accuracy | Delta vs baseline | Seed sample SD |
|---|---|---|---|---:|---:|---:|
| `OFF_RPT_BASE` | base4 | Shared | No | 82.8519% | -- | 0.0982 pp |
| `OFF_RPT_BASE_EDGEVALUE` | base4 | Independent | Yes | 83.4722% | +0.6203 pp | 0.0510 pp |
| `OFF_RPT_SELECTED_LAYERWISE` | base4 + PT + TRACK + REGION | Independent | No | 83.5152% | +0.6633 pp | 0.0885 pp |
| `OFF_RPT_SELECTED_EDGEVALUE` | base4 + PT + TRACK + REGION | Independent | Yes | **83.8670%** | **+1.0151 pp** | 0.0385 pp |

| Configuration | Seed 101 | Seed 202 | Seed 303 |
|---|---:|---:|---:|
| Base | 82.9102% | 82.7386% | 82.9070% |
| Base EdgeValue | 83.4684% | 83.4232% | 83.5250% |
| Selected Layerwise | 83.4210% | 83.5966% | 83.5280% |
| Selected EdgeValue | 83.8568% | 83.9096% | 83.8346% |

Selected EdgeValue wins against each of the other configurations at all three seeds. It exceeds Selected Layerwise by **0.3518 pp**, and Base EdgeValue by **0.3948 pp** on average.

Selected QCD-rejection results at 50% signal efficiency (arithmetic means of the three per-seed rejections):

| Signal | Base | Selected EdgeValue |
|---|---:|---:|
| H4q | 750.41 | 1001.51 |
| Hbb | 4848.48 | 9074.07 |
| Hcc | 1411.91 | 1482.20 |
| Hgg | 96.28 | 101.20 |
| Tbqq | 5092.59 | 7229.44 |
| Wqq | 320.55 | 328.98 |
| Zqq | 236.62 | 265.12 |

Rejection is `1 / QCD false-positive rate`, using the signal-minus-QCD logit as discriminant. The threshold is the `ceil(target_efficiency * signal_count)`-th largest signal score in the evaluated split, and scores equal to the threshold pass. These are ROC operating points measured on test, not thresholds calibrated on validation. Hqql and Tbl have no finite aggregate at 50% because at least one seed admits zero QCD events. This is finite-sample saturation, not proof of zero population background. Counts and uncertainty are needed for strong rejection claims.

### What the comparisons establish

The most direct test of adding the value path is **Selected EdgeValue vs Selected Layerwise**: both use the same relation families and layerwise biases. Base EdgeValue vs Base changes both the value path and shared versus layerwise biases. Likewise, Selected Layerwise vs Base changes both the features and bias sharing. Therefore these four rows are **not a complete factorial isolation of all three choices**. Earlier conversational descriptions of a clean factorial decomposition were too strong.

The results support the utility of the combined design. They do not establish a unique causal mechanism, exclude additional capacity/compute as an explanation, establish state of the art, or prove architectural novelty. An offline base4 layerwise-bias-only row and capacity/compute-matched controls would improve attribution. Seed standard deviations above are not confidence intervals for the improvements.

## 4. Starting references and requested novelty assessment

1. [Qu, Li and Qian, Particle Transformer for Jet Tagging (ICML 2022)](https://proceedings.mlr.press/v162/qu22b.html): original shared pre-softmax pair bias and the baseline architecture.
2. [Shaw, Uszkoreit and Vaswani, Self-Attention with Relative Position Representations (2018)](https://aclanthology.org/N18-2074/): Equation 3 already adds an edge vector inside the attention-weighted value sum. This is direct prior art for the generic EdgeValue operation; their edge representations and score construction differ from this implementation.
3. [Wu et al., Jet tagging with more-interaction particle transformer (2025)](https://cpc.ihep.ac.cn/article/doi/10.1088/1674-1137/ad7f3d): a close ParT-specific comparator. Its published MIA equation uses interaction-derived weights times particle values; examine the full architecture and code for overlap.

Please investigate:

- Prior use of additive edge/relation-conditioned values in ParT variants and jet taggers, including mathematically equivalent formulations under different names.
- The closest graph-transformer, point-cloud and message-passing formulations; distinguish novelty of the operator from novelty of the physics representation, integration and empirical study.
- Prior combinations of track compatibility, directional momentum relations and multiscale clustering-tree descriptors with both attention biases and value messages.
- Which contribution claims survive comparison, what baselines/ablations are missing, and whether there is evidence supporting any specifically scoped first-use claim. Give primary sources and equation/code comparisons; absence from a quick search is insufficient.

A provisional description is: **a physics-informed adaptation of relation-aware value messages to ParT, combined with layerwise biases and selected particle-pair descriptors, with gains in an offline replication of an HLT-selected design.**

## 5. Implementation pointers

Paths below are relative to this Markdown file:

- [attention.py](relational_part/attention.py): `DirectionalPairStem`, `LayerwiseBiasProjection`, `EdgeValueAttention`, `efficient_edge_value_message`, and `ConfirmationArchitectureParticleTransformer`.
- [model.py](relational_part/model.py): `RPT_BASE_CONFIG` and `build_confirmation_architecture_model`.
- [pair_builder.py](relational_part/pair_builder.py), [relation_pt.py](relational_part/relation_pt.py), [relation_track.py](relational_part/relation_track.py), [relation_region.py](relational_part/relation_region.py): exact pair-feature definitions and encoders.
- [offline_transfer.py](relational_part/offline_transfer.py) and [offline study specification](RELATIONAL_PARTICLE_TRANSFORMER_OFFLINE_TRANSFER.md): fixed model matrix and experiment contract.
- [report producer](../scripts/write_relational_part_offline_report.py) and [evaluation.py](relational_part/evaluation.py): aggregation, rejection and paired statistics. Original campaign artifacts and its pinned source are authoritative for a publication audit.
