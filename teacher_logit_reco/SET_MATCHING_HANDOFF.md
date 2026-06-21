# Set-Matching Handoff

This is the current state of the set-matching multi-view branch.  It is separate
from the teacher-logit reconstruction work.

## Current Goal

The goal is to test whether HLT-only reconstructed particle views can improve a
final HLT-only classifier when the offline view is available only during
training.

The core idea is:

```text
offline particles are training targets
HLT particles are the only inference input
```

## Overall Scientific Goal

The broader project asks whether paired offline information can help build a
better deployed HLT classifier than a standard HLT-only tagger.

The deployed model must be a function only of the HLT observation:

```text
prediction = f(HLT jet)
```

Offline information is privileged training information.  It can shape
reconstructors, auxiliary losses, specialization, and model architecture, but it
cannot be used at inference.

The set-matching branch is one attempt to use the offline view as a structured
training target rather than only as teacher logits:

```text
HLT jet -> offline-like reconstructed particle views -> HLT-only final tagger
```

The hope is not that offline training restores information that is truly absent
from HLT.  The hope is that the paired offline target helps the model extract
more of the class-relevant information that is still present in the HLT view,
or gives the final tagger useful architecture-diverse hypotheses about the
underlying jet.

## Hard Constraints

The important constraints are:

```text
1. Inference can use only HLT-derived inputs.
2. Offline particles are allowed only during training/evaluation diagnostics.
3. HLT and offline jets are matched one-to-one.
4. Train/validation/test split boundaries must stay fixed.
5. final_test is only for final evaluation, not model selection.
6. Reconstructed views must be generated from HLT only.
7. Binary runs must remap selected labels to compact 0/1 labels.
8. Requested split caps are not guaranteed actual counts.
```

The actual number of usable binary jets is limited by the source manifest and
the downloaded JetClass parts.  For example, a run can request:

```text
500k train
150k val
500k test
```

but if the source manifest only contains 50k QCD and 50k Hbb in `model_train`,
then the filtered binary run really trains on only 100k total jets.

The set-matching branch should always be compared against:

```text
HLT-only tagger on the same filtered split
offline-only reference on the same filtered split
view-label shuffle controls
single-reconstructed-view ablations
```

The branch should not claim to beat the information-theoretic limit.  Any
claimed gain should be framed as an architecture/optimization gain for finite
models and finite data under the HLT-only inference constraint.

The set-matching branch trains four different HLT-to-offline reconstructors:

```text
HLT -> global-transformer reconstructor -> reconstructed view gt
HLT -> ParticleNet-style reconstructor  -> reconstructed view pn
HLT -> PFN-style reconstructor          -> reconstructed view pfn
HLT -> PCNN-style reconstructor         -> reconstructed view pcnn
```

Then it trains taggers on combinations of:

```text
original HLT view
gt reconstructed view
pn reconstructed view
pfn reconstructed view
pcnn reconstructed view
```

The main comparison is:

```text
HLT-only tagger
vs. HLT + one reconstructed view
vs. HLT + all four reconstructed views
vs. view-label shuffle control
vs. offline-only upper reference
```

For binary tasks such as `QCD vs Hbb` and `QCD vs Tbqq`, the key metrics are:

```text
accuracy
AUC
FPR @ 30% signal efficiency
FPR @ 50% signal efficiency
```

## What Has Been Implemented

### 1. Label-Filtered Binary Inputs

Code:

```text
scripts/build_label_filtered_split_manifest.py
sbatch/run_build_label_filtered_split_manifest.sh
sbatch/run_build_label_filtered_hlt_cache.sh
```

This builds a smaller manifest from a larger JetClass split manifest.  It can
keep only selected classes such as:

```text
QCD Hbb
QCD Tbqq
```

Important detail: labels can now be remapped to compact `0..N-1` labels with
`--remap-labels`.

This matters because `QCD,Hbb` are global JetClass labels `0,1`, but `QCD,Tbqq`
are global labels `0,8`.  A two-class training run needs compact labels `0,1`.

For binary submitters:

```text
manifest builder filter: QCD Hbb or QCD Tbqq
downstream model filter: 0 1
display label names:     QCD Hbb or QCD Tbqq
```

### 2. Set-Matching Reconstructors

Code:

```text
teacher_logit_reco/set_matching/reconstructors.py
teacher_logit_reco/set_matching/losses.py
teacher_logit_reco/set_matching/set_matching_losses.py
teacher_logit_reco/set_matching/data.py
teacher_logit_reco/set_matching/train.py
scripts/train_set_matching_reconstructor.py
sbatch/run_train_set_matching_reconstructor.sh
```

Each reconstructor consumes the fixed HLT view and predicts an offline-like
unordered particle set.

The output contract is roughly:

```text
tokens / features
mask
candidate weights / confidence
existence logits
diagnostics
```

The loss is not teacher-logit KL.  It is a direct offline set-matching objective:

```text
predicted set should match offline set up to permutation
```

The main loss components are:

```text
matched core feature loss
matched auxiliary feature loss
existence loss
count loss
weak jet-summary loss
correction-budget regularization
optional chamfer-style term
```

The intended story is that each architecture can produce a different plausible
offline reconstruction hypothesis.

### 3. Reconstructed View Caching

Code:

```text
teacher_logit_reco/set_matching/cache.py
scripts/cache_set_matching_reco_views.py
sbatch/run_cache_set_matching_multiview.sh
```

After a reconstructor is trained, this step runs it over:

```text
stack_train
stack_val
final_test
```

and writes fixed reconstructed-view caches.  The final taggers train from these
caches instead of rerunning the reconstructors every epoch.

The cache reports include useful diagnostics:

```text
identity hashes
label hashes
set-matching metrics
candidate count summaries
confidence summaries
HLT/offline/reco alignment audits
```

### 4. Five-View Tagger

Code:

```text
teacher_logit_reco/set_matching/five_view_data.py
teacher_logit_reco/set_matching/five_view_model.py
teacher_logit_reco/set_matching/five_view_attention.py
teacher_logit_reco/set_matching/five_view_train.py
scripts/train_five_view_tagger.py
sbatch/run_train_five_view_tagger.sh
```

The five-view tagger is a ParT-ish/two-stage transformer setup.

It can train these variants:

```text
hlt_only
hlt_plus_gt
hlt_plus_pn
hlt_plus_pfn
hlt_plus_pcnn
five_view_plain
five_view_geometry
five_view_no_confidence
view_label_shuffle_control
```

The intended input format is not "match particle i across all views."  Instead,
each view is encoded as a set of particle tokens, with view/source information
attached.  The model then learns cross-view evidence at the token/view level.

The control that matters most is:

```text
view_label_shuffle_control
```

If this performs near the real five-view model, the model may not be using view
identity in a meaningful way.

### 5. Ablation And Final Reports

Code:

```text
teacher_logit_reco/set_matching/five_view_ablation.py
scripts/evaluate_five_view_ablation.py
sbatch/run_audit_five_view_tagger.sh
scripts/write_set_matching_multiview_final_report.py
sbatch/run_write_set_matching_multiview_final_report.sh
```

The ablation runner evaluates trained taggers and writes summary diagnostics.
The final report collects the experiment into a compact JSON/CSV report.

### 6. Offline Binary Reference

Code:

```text
scripts/train_eval_set_matching_binary_offline_teacher.py
sbatch/run_train_eval_set_matching_binary_offline_teacher.sh
sbatch/submit_offline_binary_qcd_tbqq_reference.sh
```

This trains a fresh offline-only binary ParT on the same binary split.  It is an
upper-reference, not a deployable model.

It answers:

```text
If the tagger had offline particles at inference, how separable is this task?
```

For Hbb/QCD, the offline-only reference was very strong:

```text
final_test accuracy ~0.973
AUC ~0.996
FPR @ 30% signal efficiency ~0.0003
FPR @ 50% signal efficiency ~0.001
```

## Submitters

Main submitters:

```text
sbatch/submit_set_matching_multiview_experiment.sh
sbatch/submit_set_matching_multiview_smoke_test.sh
sbatch/submit_set_matching_hbb_qcd_binary_experiment.sh
sbatch/submit_set_matching_qcd_tbqq_binary_experiment.sh
sbatch/submit_offline_binary_qcd_tbqq_reference.sh
```

The binary submitters queue:

```text
1 binary manifest job
1 binary HLT cache job
1 offline-only reference job
4 reconstructor train jobs
4 reconstructed-view cache jobs
9 tagger/control jobs
1 audit job
1 final report job
```

So the full binary set-matching graph is 22 jobs by default.

## Current Important Findings

### The "500k" Hbb/QCD run was not really 500k after filtering

The requested caps were large, but the filtered manifest only contained:

```text
model_train = 100000  = 50k QCD + 50k Hbb
model_val   = 30000   = 15k QCD + 15k Hbb
stack_train = 50000   = 25k QCD + 25k Hbb
stack_val   = 10000   = 5k QCD + 5k Hbb
final_test  = 100000  = 50k QCD + 50k Hbb
```

The cap can say `500000`, but the actual number is limited by what exists in
the source manifest.

For larger binary runs, the source data/manifest must include more QCD and
signal jets, probably by adding more JetClass parts and building a larger
binary-only manifest.

### The suspicious Hbb/QCD 500k taggers were undertrained

The worrying run used only 12 tagger epochs.  The earlier better run used 30.

The bad run showed many models still improving at the final epoch:

```text
hlt_only                best_epoch=12/12 final_test_acc~0.885
five_view_no_confidence best_epoch=12/12 final_test_acc~0.853
five_view_plain         best_epoch=12/12 final_test_acc~0.749
hlt_plus_pfn            best_epoch=12/12 final_test_acc~0.694
hlt_plus_pn             best_epoch=12/12 final_test_acc~0.715
```

The submitter defaults have been changed back to 30 tagger epochs:

```text
submit_set_matching_hbb_qcd_binary_experiment.sh
submit_set_matching_qcd_tbqq_binary_experiment.sh
```

### Some reconstructed views are actively harmful alone

In the Hbb/QCD binary run, `hlt_plus_pfn` and `hlt_plus_pn` badly hurt QCD
classification.  This may mean those reconstructed views are adding misleading
tokens, or the tagger is over-trusting noisy reconstruction hypotheses.

This is not necessarily a fatal result for the full idea.  It means the five-view
tagger needs either:

```text
more training
better view gating
better confidence calibration
or stronger ablations before trusting any gain/drop
```

## Where To Look For Results

On the cluster:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/<experiment_name>
/home/ryreu/atlas/Fresh_check/fresh_check_logs
/home/ryreu/atlas/Fresh_check/fresh_check_diagnostics
```

Locally downloaded diagnostics:

```text
C:\Users\22rya\ComputerScience\CERN\a_download_checkpoints\fresh_check_diagnostics
C:\Users\22rya\ComputerScience\CERN\a_download_checkpoints\fresh_check_logs
```

Useful files inside each experiment:

```text
binary_inputs/filtered_manifest_report.json
binary_inputs/hlt_cache/*_fixed_hlt_metadata.json
reconstructors/<arch>/run_report.json
reconstructors/<arch>/diagnostics/epoch_metrics.csv
reconstructed_views/<arch>/cache_report.json
taggers/<variant>/run_report.json
taggers/<variant>/diagnostics/epoch_metrics.csv
taggers/<variant>/diagnostics/per_class_metrics.csv
taggers/<variant>/diagnostics/view_ablation_metrics.json
ablations/five_view_ablation_eval/summary.csv
final_report/final_report.json
offline_teacher_reference/<run>/run_report.json
```

## Current Next Moves

1. Do not interpret requested split caps as actual split sizes.
   Always check:

   ```text
   binary_inputs/filtered_manifest_report.json
   ```

2. For serious Hbb/QCD or QCD/Tbqq runs, first build or download enough
   binary-class data.  The current `part0`-based manifest is too small for real
   500k/2M binary studies.

3. Rerun the binary set-matching taggers with 30 epochs before concluding that
   the five-view setup collapsed.

4. Treat single-reco `hlt_plus_pfn` and `hlt_plus_pn` drops as a diagnostic
   target.  They may reveal poorly calibrated or misleading reconstructed views.

5. Use the offline-only binary reference as a sanity check for task difficulty,
   not as a deployable comparison.
