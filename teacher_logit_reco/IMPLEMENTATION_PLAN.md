# Teacher-Logit Global Transformer Reconstructor Plan

This folder is reserved for a clean teacher-logit reconstruction line, separate
from the older `jetclass_fresh` m2/reco7 implementations.

The first target is a Global Transformer Reconstructor: a ParT-inspired,
paper-readable reconstructor that maps fixed-HLT constituent tokens into a soft
reconstructed particle view. The main training signal is not literal particle
matching. The main training signal is that a frozen offline teacher should
interpret the reconstructed view like it interprets the true offline view.

## Core Experiment

For each paired training jet:

```text
fixed HLT view  -> reconstructor -> reconstructed soft view -> frozen teacher -> reco logits
offline view                              -> frozen teacher -> offline logits
```

Train the reconstructor with:

```text
loss =
  teacher_KL(offline_teacher_logits, reco_teacher_logits)
+ CE(reco_teacher_logits, true_label)
+ correction_budget
+ weak_jet_summary_loss
```

The frozen teacher may be any compatible offline tagger architecture:

```text
ParT teacher
ParticleNet teacher
PFN teacher
PCNN teacher
```

The reconstructor itself, for this first line, is the global transformer family.
Future folders/modules can add graph, PFN/global-prior, and CNN/subjet
reconstructors while reusing the same teacher and evaluation interfaces.

## Design Principles

- Keep inference HLT-only.
- Keep offline available only during training, validation, and audits.
- Keep the reconstructor output constrained and physical enough to avoid
  teacher exploits.
- Prefer soft reconstructed views over hard particle matching.
- Make all teacher/tagger compatibility explicit and auditable.
- Preserve the five-way split discipline:
  `model_train`, `model_val`, `stack_train`, `stack_val`, `final_test`.

## Reconstructed View

The first decoder should output:

```text
corrected parent particles + fixed extra candidate particles
```

Parent-aligned outputs:

```text
pt'     = pt * exp(delta_logpt)
eta'    = eta + delta_eta
phi'    = phi + delta_phi
energy' = energy * exp(delta_loge)
weight  = sigmoid(parent_weight_logit)
```

Extra candidate outputs:

```text
K learned query slots attend to the encoded HLT jet.
Each slot predicts token features and a soft weight.
```

The teacher receives a single soft view:

```text
weighted corrected parents + weighted extra candidates
```

Weights are folded into `pt` and `energy` before building tagger inputs. This
matches the existing soft-view idea while keeping the output differentiable.

## Losses

### 1. Teacher KL

This is the primary loss.

```text
p_off = softmax(teacher(offline_view) / T)
p_rec = softmax(teacher(reconstructed_view) / T)
teacher_KL = T^2 * KL(p_off || p_rec)
```

Purpose:

```text
Make the reconstructed view preserve the offline teacher's discriminative
evidence, not necessarily every offline particle.
```

### 2. True-Label CE

```text
CE(teacher(reconstructed_view), true_label)
```

Purpose:

```text
Keep the reconstructed view directly class-useful even when the offline teacher
is uncertain or wrong.
```

### 3. Correction Budget

Penalize excessive edits:

```text
mean(delta_logpt^2)
mean(delta_eta^2)
mean(delta_phi^2)
mean(delta_loge^2)
mean((parent_weight - 1)^2)
mean(sum(extra_weights))
mean(extra_pt_or_energy_budget)
```

Purpose:

```text
Prevent the reconstructor from creating unphysical teacher-adversarial views.
```

### 4. Weak Jet Summary Loss

Compare reconstructed and offline coarse summaries:

```text
jet_pt
jet_mass
jet_energy
valid/weighted multiplicity
possibly leading few pt fractions
```

Purpose:

```text
Keep the reconstructed view physically anchored without returning to strict
particle-by-particle reconstruction.
```

## Implementation Steps

Current status:

```text
Step 1 complete: package boundary, paired views, soft-view conversion, tests.
Step 2 complete: frozen teacher adapters for ParT, PN, PFN, and PCNN.
Step 3 complete: global transformer forward model with bounded soft-view output.
Step 4 complete: KL/CE/budget/summary losses and one-batch training step.
Step 5 complete: model_train/model_val training loop with auditable checkpoints.
Step 6 complete: fusion-compatible prediction blocks for stack/final splits.
Step 7 complete: Slurm runners for train, predict, and independent fusion.
```

### Step 1: Define The New Package Boundary And Data Interfaces

Create a new Python package under this folder, for example:

```text
teacher_logit_reco/
  __init__.py
  views.py
  teachers.py
  losses.py
  global_transformer.py
  train_global_transformer.py
  predict_global_transformer.py
```

Implement only shared interfaces first:

- Paired HLT/offline batch loading from existing split/cache utilities.
- A `SoftReconstructedView` dataclass.
- Conversion from soft reconstructed tokens to tagger inputs.
- No model training yet.

Deliverables:

- Unit tests for shape/mask/weight behavior.
- A tiny smoke script that loads one paired batch and builds tagger inputs.

### Step 2: Implement Teacher Adapters

Create a common frozen-teacher interface:

```text
teacher.forward_view(tokens, mask, weights=None) -> logits
teacher.metadata -> architecture/config/checkpoint path
```

Adapters:

- ParT teacher adapter.
- PN teacher adapter.
- PFN teacher adapter.
- PCNN teacher adapter.

These should reuse the heterogeneous HLT/offline tagger builders where possible,
but live behind a clean teacher interface in this folder.

Deliverables:

- Load checkpoint.
- Freeze/eval mode.
- Forward offline view.
- Forward reconstructed soft view.
- Verify no teacher parameters require gradients.

### Step 3: Implement The Global Transformer Reconstructor

Model components:

- Token embedding for HLT token features.
- Transformer encoder over valid HLT constituents.
- Parent correction head.
- Learned extra-candidate query decoder.
- Output sanitizer/bounds for `pt`, `eta`, `phi`, `energy`, and weights.

Initial defaults:

```text
hidden_dim: 128 or 192
num_layers: 4
num_heads: 4 or 8
num_extra_candidates: 32
dropout: 0.05
```

Deliverables:

- Forward pass returns `SoftReconstructedView`.
- Unit tests for finite outputs, bounded corrections, correct masks.
- No teacher loss yet.

### Step 4: Implement Losses And Training Step

Implement:

- Temperature KL loss.
- CE loss on teacher logits from reconstructed view.
- Correction budget.
- Weak jet summary loss.
- Weighted loss combiner with logged components.

Recommended first loss weights:

```text
teacher_kl_weight: 1.0
ce_weight: 0.3
correction_budget_weight: 0.01
jet_summary_weight: 0.05
temperature: 2.0
```

Deliverables:

- Single-batch overfit/smoke training.
- Loss component JSON logging.
- Gradient check: gradients flow into reconstructor, not teacher.

### Step 5: Train Global Transformer Reconstructor

Build the first real training script:

```text
scripts/train_teacher_logit_global_transformer_reco.py
```

Inputs:

- HLT cache directory.
- split manifest / offline data directory.
- teacher checkpoint.
- teacher architecture: `part`, `pn`, `pfn`, or `pcnn`.
- output directory.
- max train/val jets.

Rules:

- Train only on `model_train`.
- Select only on `model_val`.
- Do not load `stack_train`, `stack_val`, or `final_test` during training.

Deliverables:

- `best_model_val.pt`
- `last.pt`
- `training_curves.json`
- `run_report.json`
- recorded teacher metadata and source hashes.

### Step 6: Prediction Blocks And Fusion Compatibility

Build evaluator:

```text
scripts/predict_teacher_logit_global_transformer_reco.py
```

For each requested split:

```text
HLT -> reconstructor -> reconstructed soft view -> frozen teacher -> logits
```

Save prediction blocks compatible with existing independent fusion:

```text
predictions/<model_name>/<split>_predictions.npz
predictions/<model_name>/<split>_predictions_metadata.json
```

Model naming:

```text
gt_reco_to_part_teacher
gt_reco_to_pn_teacher
gt_reco_to_pfn_teacher
gt_reco_to_pcnn_teacher
```

Deliverables:

- Prediction collection for `stack_train`, `stack_val`, `final_test`.
- Independent fusion can consume outputs without special code.

### Step 7: Slurm Runners

Add sbatch scripts after the local smoke path works:

```text
sbatch/run_train_teacher_logit_gt_reco.sh
sbatch/run_predict_teacher_logit_gt_reco.sh
sbatch/submit_teacher_logit_gt_reco_experiment.sh
```

The submitter should optionally queue:

```text
train GT reconstructor against ParT teacher
train GT reconstructor against PN teacher
train GT reconstructor against PFN teacher
train GT reconstructor against PCNN teacher
dependent prediction/fusion jobs
```

Do not add graph/PFN/CNN reconstructors until this path is tested.

### Step 8: First Experiment Matrix

Start small and interpretable:

```text
GT reconstructor -> ParT teacher
GT reconstructor -> PN teacher
GT reconstructor -> PFN teacher
GT reconstructor -> PCNN teacher
```

Compare:

- Raw HLT teacher/tagger logits.
- Reconstructed-view teacher logits.
- HLT + reconstructed-view fusion.
- Oracle/diversity/uncertainty reports.

Key questions:

```text
Does teacher-logit reconstruction beat raw HLT for any teacher?
Are different teachers complementary?
Does the reconstructed view help fusion even when it is not better alone?
```

## What Not To Do Yet

- Do not implement all four reconstructor families at once.
- Do not mix every reconstructor/tagger pair before the GT path is validated.
- Do not make particle Chamfer loss the main objective.
- Do not use `final_test` during training or model selection.
- Do not overwrite existing reco7/m2 experiment outputs.

## Success Criteria

The first global transformer reconstructor is worth extending if at least one is
true:

- Reconstructed-view teacher logits beat the raw HLT teacher/tagger on
  `stack_val` and hold most of the gain on `final_test`.
- Reconstructed-view logits are not better alone but improve stacked fusion.
- Diversity audits show the reconstructed-view model makes meaningfully
  different errors from raw HLT and heterogeneous HLT taggers.
