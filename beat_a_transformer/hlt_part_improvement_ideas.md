# Ideas For Beating HLT ParT

This note is an idea bank, not an implementation plan. The assumed target realm is:

- Task: QCD vs Hgg tagging
- Inference view: HLT only
- HLT degradation: 0.6 strength
- Baseline to beat: HLT Particle Transformer
- Goal: improve the HLT-side model itself, especially at high data counts, rather than relying on reconstruction as the main story

The guiding question is:

> What structure does HLT ParT not explicitly model, even though that structure may matter for QCD vs Hgg?

## 1. Local Graph Frontend Plus ParT Global Attention

The idea is to add a local geometric reasoning stage before the normal ParT-style global attention. ParT lets every particle attend to every other particle, with pairwise physics features, but it does not force a local neighborhood computation first. HLT degradation is often local: nearby constituents can be merged, dropped, smeared, or reassigned. A particle's reliability may depend strongly on what is around it.

The model shape would be:

```text
HLT particles
  -> local eta-phi kNN message passing or local attention
  -> enriched particle tokens
  -> ParT-style global pairwise attention
  -> classifier
```

This is inspired by GraphGPS and Point Transformer. GraphGPS combines local message passing with global transformer attention. Point Transformer improves over more generic point-set transformer baselines by using local neighborhoods and relative geometry. The particle-physics analogue would be a ParticleNet-like local frontend plus ParT-like global attention.

Why it might beat HLT ParT:

- HLT errors are local and geometry-dependent.
- QCD vs Hgg depends on local substructure and prong-like organization.
- The model would not ask global attention to rediscover all local consistency patterns from scratch.

## 2. Multi-Scale Particle And Subjet Tokens

The idea is to introduce intermediate tokens between individual particles and the whole jet. ParT tokenizes particles directly, but QCD vs Hgg is not just a particle-level problem. It is often a prong/subjet/substructure problem.

The model shape would be:

```text
particles
  -> local clusters or proto-subjets
  -> subjet tokens
  -> particle plus subjet interaction
  -> classifier
```

This is inspired by T2T-ViT, TNT, CrossViT, and hierarchical vision models. Vision transformers improved when researchers stopped treating fixed patches as the only token level and allowed local patches, subpatches, or multiple patch scales to interact.

Why it might beat HLT ParT:

- Hgg vs QCD may be easier if the model can explicitly form candidate two-prong or multi-core structure.
- Subjet tokens could summarize regions in a way individual particles do not.
- Particle-level attention plus subjet-level attention may expose structure that a flat particle sequence hides.

This is one of the most natural ideas for QCD vs Hgg specifically.

## 3. Dual-View Cross-Attention: ParT Branch Plus New Branch

The idea is not to replace ParT. Instead, keep a strong ParT-like particle branch and add a second branch that sees the same HLT jet through a different inductive bias. Then fuse them carefully.

Possible branches:

```text
Branch A: standard HLT ParT particle tokens
Branch B: local graph / subtoken / subjet / reliability tokens
Fusion: cross-attention, gated fusion, or late learned fusion
```

This is inspired by CrossViT, where small-patch and large-patch branches exchange information through cross-attention. It is also inspired by the broader idea that different tokenizations can be complementary if the fusion is strong enough.

Why it might beat HLT ParT:

- Our first subtoken-only runs were close but did not clearly beat HLT ParT.
- That suggests the new representation may be complementary rather than dominant.
- A dual-view model can preserve ParT's strengths while adding local/reliability/substructure information.

This may be a safer path than betting everything on a replacement architecture.

## 4. Reliability And Residual-Aware HLT Training

The idea is to train the HLT model to understand which HLT features are trustworthy and where the HLT view has likely drifted from offline. The final inference still uses only HLT, but offline can provide training-time signals about what HLT lost or distorted.

This is inspired less by one specific paper and more by the pattern that structured auxiliary losses can shape better representations. It is also motivated by our own result: the best QCD/Hgg subtoken run so far was not the plain HLT-only subtoken model. It was the residual-privileged version.

Relevant result:

```text
HLT ParT FPR@50:                 0.020384
subtoken_gate_context_residual:  0.020088
```

That is a small but real-looking improvement in the direction we care about.

Why it might beat HLT ParT:

- ParT only learns labels from HLT.
- A residual/reliability objective can teach the model where HLT is likely misleading.
- This may improve optimization or representation quality without adding offline information at inference.

Important caution:

- This does not break the Bayes limit.
- It can only help if it makes the fixed architecture/optimizer extract remaining HLT information better.

## 5. Stronger Within-Particle Subtokenization

The idea is to improve the way each particle token is formed. ParT embeds each particle's feature vector into a token through an input embedding. That may compress heterogeneous information too early.

A stronger subtoken model would let different groups inside the particle talk first:

```text
kinematics token
identity / PID token
track / displacement token
quality / reliability token
  -> local within-particle mixer
  -> enriched particle token
  -> ParT-style particle attention
```

This is inspired by TNT and FT-Transformer. TNT treats image patches as coarse tokens and subpatches as inner tokens. FT-Transformer treats fields in one tabular row as tokens. A particle is somewhat like a small structured row: kinematics, identity, tracking, displacement, and quality all mean different things.

Why it might beat HLT ParT:

- It gives the model a better internal representation of particle features before global attention.
- It can learn feature reliability and feature interactions inside a particle.
- It may be especially useful when HLT corrupts some modalities more than others.

What our current results suggest:

- Plain subtokenization alone was not enough.
- It may need to be paired with reliability objectives, local graph structure, or a standard ParT branch.

## 6. Local Compression Instead Of Global Scalar Attention

The tempting but dangerous idea is to tokenize every scalar feature. That would let every feature interact directly, but it would create too many tokens and might destroy the binding between features and particles.

The better version is local compression:

```text
feature or modality subtokens inside each particle
  -> local feature-level computation
  -> compressed particle token
  -> global particle-level computation
```

This is inspired by CANINE and MEGABYTE in NLP. Character-level or byte-level models can beat standard token models, but only when they locally compress fine tokens before global modeling. They do not usually run one flat global transformer over every byte.

Why it might beat HLT ParT:

- It exposes fine-grained feature structure.
- It avoids the cost and confusion of global scalar-token attention.
- It preserves particle identity while still allowing feature-level reasoning.

## 7. Modern ConvNet Lesson: Better Local Bias May Matter More Than More Attention

ConvNeXt beating Swin in some vision settings is a useful warning. The lesson is not "attention always wins." The lesson is that the right local and hierarchical bias can beat a transformer baseline.

For jets, that suggests trying a strong non-attention local module before or alongside ParT:

```text
EdgeConv / local MLP / convolution-like geometric block
  -> ParT global attention
```

Why it might beat HLT ParT:

- Local geometric interactions may be more naturally modeled by message passing than by global attention.
- QCD vs Hgg has local substructure that may benefit from a strong local operator.
- This would test whether ParT's weakness is not attention, but insufficient local bias.

## 8. State-Space Or Sequence Models

Mamba and RetNet are examples where non-transformer sequence models beat transformer baselines in language settings. They are exciting, but less directly aligned with jets because jets are unordered geometric sets, not natural one-dimensional sequences.

Possible jet versions could impose meaningful orders:

- radial order from jet axis
- angular scan order
- pT order
- subjet traversal order

Why it might be interesting:

- It could scale better than attention.
- It might model ordered radiation patterns if the ordering is meaningful.

Why it is lower priority:

- Bad ordering choices can inject arbitrary structure.
- ParT's set-like handling is already a strong prior.
- Local graph plus global attention is a more direct first attack.

## Current Best Bets

For QCD vs Hgg with HLT degradation 0.6, the strongest next ideas seem to be:

1. Local graph/EdgeConv frontend plus ParT global attention
2. Particle plus subjet multi-scale tokens
3. Standard ParT branch plus subtoken/reliability branch with cross-attention fusion
4. Residual/reliability auxiliary training for HLT-only inference
5. Stronger within-particle subtokenization, especially when paired with one of the above

The common theme is:

> Do not just build a bigger transformer. Add structure where ParT is still too flat: local HLT reliability, neighborhood geometry, and intermediate subjet/prong organization.

