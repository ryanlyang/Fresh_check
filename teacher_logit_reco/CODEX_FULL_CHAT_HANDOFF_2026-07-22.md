# Fresh Check Handoff — 2026-07-22

This document is the working handoff for where the project stands after the long run of JetClass experiments around pseudo-HLT, privileged reconstruction, and multi-view/dual-view tagger studies.

## Current Objective (Current Contract)

- We are evaluating how to recover jet-tagging performance when only HLT-like inputs are available at inference.
- We use matched offline+HLT JetClass data for training/teacher supervision.
- At deployment, only HLT-view tokens can be used.
- We keep returning to a `set matching + reconstruction` style family, while testing stronger tagger baselines (especially HLT-style ParT quality).
- User-facing target remains practical gains over HLT ParT baseline at the same runtime budget, with special emphasis on QCD/Hgg and HLT degradation settings.

## Data Location / Cluster Layout

- Repo root: `C:\Users\22rya\ComputerScience\CERN\Fresh_check`
- Primary research path (cluster): `/home/ryreu/atlas/Fresh_check`
- Data path used in recent runs:
  - Older: `/home/ryreu/atlas/PracticeTagging/data/jetclass_part0`
  - On Tigris we shifted to: `/home/ryreu/atlas/PracticeTagging/data/jetclass_part1`
- Runtime diagnostics/log roots:
  - `/home/ryreu/atlas/Fresh_check/fresh_check_logs`
  - `/home/ryreu/atlas/Fresh_check/fresh_check_diagnostics`
- Download mirror used by user on Windows/WSL:
  - `C:\Users\22rya\ComputerScience\CERN\a_download_checkpoints\fresh_check_logs`
  - `C:\Users\22rya\ComputerScience\CERN\a_download_checkpoints\fresh_check_diagnostics`
- Active Slurm account/hostname context:
  - Account: `reu-aisocial`
  - Hostnames: `sporcsubmit` and `tigris`
- Conda:
  - `atlas_kd` on RIT cluster
  - `atlas_kd_tigris` on Tigris
- On Tigris set `PYTHONNOUSERSITE=1` to avoid user-site package shadowing (critical for uproot).

## Core Scientific Contracts Across Projects

- Offline model is teacher-only signal during training.
- Inference-time function is HLT-only only, so this is a privileged training setup.
- There are three recurring constraints:
  - Keep pseudo-HLT generation reproducible and fixed once built.
  - Maintain one explicit label pair per run contract (usually QCD/Hgg or earlier Hbb/QCD variants).
  - Keep split sizes and manifest/cache provenance versioned in `split_manifest.json.gz`, split manifests, and HLT cache paths.

## Pseudo-HLT implementation details (key points)

- Pseudo-HLT script area: `jetclass_fixed_hlt.py`.
- Raw particle fields for these pipelines are the 14-field JetClass raw token format.
- m2-style/v1/v2 settings in code use configurable strength.
- HLT v2 strength 2.5 in code is a stress-test and is applied in multiple recent runs.
- Typical knobs include:
  - Raised low-pT cutoff
  - pT-dependent efficiency (turn-on with plateau per eta region)
  - local density penalty term
  - merge radius and merge probability
  - optional reassignment/smearing steps
- Merge behavior:
  - Nearby particles in radius are merged by summing core kinematics and reweighting eta/phi by pT.
  - Non-kinematic values are typically inherited from the dominant-energy component in that merge.
- Important: never rely on hardcoded path assumptions if manifest/cache got deleted; every run checks for required `split_manifest.json.gz`, HLT cache files, and cache metadata.

## Architecture history and what was tried

### 1) Set-matching setup (five-view, HLT + reconstructed views)

- Reconstructors used: `gt`, `pn`, `pfn`, `pcnn`.
- Tagger variants include: `hlt_only`, `hlt_plus_*`, `five_view_plain`, `five_view_geometry`, `five_view_no_confidence`, shuffle-control and ablations.
- Output artifacts are heavy; cache + reconstructed-view directories are large.
- Most complete runs show:
  - offline-vs-HLT agreement is limited; offline-on-HLT is weak outside the easy subset.
  - standard five-view tags often did not beat strong HLT-only ParT in final full tests.
- Key issue seen: many queue/dependency failures from stale jobs and missing intermediate files, and occasional incomplete/failing cache stages.

### 2) DETR-style free slot reconstructor branch

- Added a free-slot Hungarian matcher and slot encoder family for gt/pn/pfn/pcnn.
- Major fixes already applied before this handoff:
  - 14-feature core contract over 19 where needed.
  - context path in heads/decoder fixed.
  - loss and cache interface compatibility.
  - aux-aware assignment and BCE for binary-like aux terms.
  - production-safe fallback behavior for assignment methods.
  - additional diagnostics (count MAE, precision/recall, p90 deltaR, jet summary).
  - report/final selection by binary FPR@50 preference (not AUC-only).
- Practical outcome at QCD/Hgg, HLT0.6:
  - DETR `hlt_plus_pn` underperformed HLT ParT.
  - The strongest baseline stayed vanilla HLT ParT and offline ParT.
- Operational lesson: this branch is promising in structure but still not yet a replacement winner.

### 3) Reliability-gated dual-view particle part branch

- Idea: keep HLT ParT as anchor and add PN-derived residual correction through learned gate.
- Final model form used: `final_logits = hlt_logits + gate * delta` with gate initially closed.
- Implemented:
  - anchor loader checks (including QCD/Hgg and HLT0.6 assumptions),
  - dataset contracts,
  - PN encoder with reliability features,
  - training/audit/report scripts and diagnostics,
  - control with shuffled PN.
- Repeatedly failed to get trustworthy full step-11 completion due missing/deleted split manifest and HLT cache files.
- Current status: implementation exists, but not yet a validated, clean full-run set in this phase.

### 4) Local residual field / localgraph / multiscale exploration

- New campaign (`LOCAL_RESIDUAL_FIELD_*` and related docs) introduced prediction-anchored bridge and local residual fields.
- Current notable pilot numbers from P7b-like stages and localfield experiments:
  - A0 74.946%
  - P7b 75.328%
  - G0 mean-logits 75.8273%
- These are exploratory and not the final production recommendation without full comparison at equal baseline controls.

### 5) AV10 / early residual fusion work

- 10-class baseline improvements were modest (~0.5–1.5 pts in some settings).
- Good finding: per-particle residual adapters can help, but not enough to supersede HLT baseline by large margin in all settings.
- Some control experiments (feature shuffle) still competitive, which means some gains likely from architecture/regularization effects.

### 6) Older original m2/reco7 branch

- Still relevant as fallback baseline.
- User repeatedly asked to rerun with faster settings and stronger HLT v2.5/1.5 settings on Tigris.
- Known submitter file: `sbatch/submit_reco7_v2_hlt1p5_3m_fast.sh` and variants.
- Current data moved from part0 to part1 on Tigris; stale scripts pointing to part0 fail.
- Must avoid stale caches after user deletes checkpoint roots.

## Major failure modes repeatedly observed

- `DependencyNeverSatisfied` in Slurm due upstream failed/cancelled job or deleted artifact.
- Job failures from missing required files:
  - missing `stack_train_fixed_hlt.npz`
  - missing `split_manifest.json.gz`
- Non-finite logits in HLT/teacher prediction steps.
- Mismatched class/filter assumptions in loaders.
- Out-of-space causing mid-run or post-run failures.
- Cluster/host path mismatch (sporcsubmit vs tigris assumptions).
- Tar/rsync transfer edge cases in WSL due metadata permissions (`utime` warnings), usually non-fatal but confusing.

## What is currently happening now

- Latest strong signal is that many custom five-view/bridge variants are being dominated by a strong standalone HLT ParT branch.
- User is now expressing frustration: current dual-view and some custom fusion variants are not consistently beating strong HLT-only baseline.
- Working direction has shifted toward preserving HLT ParT quality and only adding auxiliary views if they can clearly pass strong head-to-head checks.

## Important file map for onboarding (where to continue)

- Handoff docs and plans:
  - [teacher_logit_reco/CODEX_HANDOFF_LOCAL_RESIDUAL_FIELD.md](/Users/22rya/ComputerScience/CERN/Fresh_check/teacher_logit_reco/CODEX_HANDOFF_LOCAL_RESIDUAL_FIELD.md)
  - [teacher_logit_reco/SET_MATCHING_HANDOFF.md](/Users/22rya/ComputerScience/CERN/Fresh_check/teacher_logit_reco/SET_MATCHING_HANDOFF.md)
  - [teacher_logit_reco/DETR_FREE_SLOT_RECONSTRUCTOR_PLAN.md](/Users/22rya/ComputerScience/CERN/Fresh_check/teacher_logit_reco/DETR_FREE_SLOT_RECONSTRUCTOR_PLAN.md)
  - [teacher_logit_reco/RELIABILITY_GATED_DUALVIEW_PART_PLAN.md](/Users/22rya/ComputerScience/CERN/Fresh_check/teacher_logit_reco/RELIABILITY_GATED_DUALVIEW_PART_PLAN.md)
  - [teacher_logit_reco/ARCHITECTURE_VIEW_10CLASS_ENSEMBLE_PLAN.md](/Users/22rya/ComputerScience/CERN/Fresh_check/teacher_logit_reco/ARCHITECTURE_VIEW_10CLASS_ENSEMBLE_PLAN.md)
  - [teacher_logit_reco/ARCHITECTURE_VIEW_RESIDUAL_PART_PLAN.md](/Users/22rya/ComputerScience/CERN/Fresh_check/teacher_logit_reco/ARCHITECTURE_VIEW_RESIDUAL_PART_PLAN.md)
  - [teacher_logit_reco/CONSTRAINED_COARSE_TO_FINE_RUNTIME_ACCELERATION_PLAN.md](/Users/22rya/ComputerScience/CERN/Fresh_check/teacher_logit_reco/CONSTRAINED_COARSE_TO_FINE_RUNTIME_ACCELERATION_PLAN.md)
  - [teacher_logit_reco/MULTISCALE_SUBJET_PARTICLE_TRANSFORMER_PLAN.md](/Users/22rya/ComputerScience/CERN/Fresh_check/teacher_logit_reco/MULTISCALE_SUBJET_PARTICLE_TRANSFORMER_PLAN.md)
  - [teacher_logit_reco/LOCAL_RESIDUAL_FIELD_P7B_FUSION_IMPLEMENTATION.md](/Users/22rya/ComputerScience/CERN/Fresh_check/teacher_logit_reco/LOCAL_RESIDUAL_FIELD_P7B_FUSION_IMPLEMENTATION.md)
  - [teacher_logit_reco/LOCAL_RESIDUAL_FIELD_P7B_FUSION_IMPLEMENTATION_PLAN.md](/Users/22rya/ComputerScience/CERN/Fresh_check/teacher_logit_reco/LOCAL_RESIDUAL_FIELD_P7B_FUSION_IMPLEMENTATION_PLAN.md)
  - [teacher_logit_reco/PREDICTION_ANCHORED_BRIDGE_STEP10_IMPLEMENTATION.md](/Users/22rya/ComputerScience/CERN/Fresh_check/teacher_logit_reco/PREDICTION_ANCHORED_BRIDGE_STEP10_IMPLEMENTATION.md)

- Runner and submit entry points:
  - `sbatch/run_write_set_matching_multiview_*`
  - `sbatch/run_write_detr_slot_final_report.sh`
  - `sbatch/run_train_dualview_part_residual.sh`
  - `sbatch/run_detr_slot_*`
  - `sbatch/submit_reco7_v2_hlt1p5_3m_fast.sh`
  - `sbatch/run_...local_residual_field...` (multiple untracked additions currently in working tree)

- HLT implementation and data utilities:
  - [jetclass_fixed_hlt.py](/Users/22rya/ComputerScience/CERN/Fresh_check/jetclass_fixed_hlt.py)
  - [teacher_logit_reco/views.py](/Users/22rya/ComputerScience/CERN/Fresh_check/teacher_logit_reco/views.py)
  - [teacher_logit_reco/set_matching/detr_slots/*](/Users/22rya/ComputerScience/CERN/Fresh_check/teacher_logit_reco/set_matching/detr_slots)
  - [teacher_logit_reco/dualview_part/*](/Users/22rya/ComputerScience/CERN/Fresh_check/teacher_logit_reco/dualview_part)

## Quantitative snapshot to carry forward

- 10-class discrepancy diagnostic on 50k split (offline teacher vs HLT):
  - HLT on HLT 0.72538
  - offline-on-HLT 0.58108
  - offline-on-offline 0.80688
  - agreement 0.64838
  - oracle both 0.78
  - higher-confidence heuristic 0.68234

- Av10/old fusion bests (indicative, not final):
  - standard HLT ParT ~0.745424
  - best fusion ~0.752929
  - gain not large enough to supersede strong baseline universally

- Setmatching offline reference (QCD/Hgg/Hbb checks)
  - offline models on clean offline references frequently report AUC ~0.96+ with much stronger acc/FPR than custom five-view tags in matching runs

- DETR QCD/Hgg HLT0.6 sample (important warning):
  - DETR hlt_plus_pn acc 0.836764, AUC 0.913003, FPR30 0.020284, FPR50 0.048936
  - DETR hlt_only ablation acc 0.824812, AUC 0.902448, FPR30 0.023936, FPR50 0.059248
  - standard HLT ParT acc 0.883348, AUC 0.951628, FPR30 0.007076, FPR50 0.020384

- Dual-view residual / two-stage concept showed promise structurally, but no verified step-11 completion after repeated missing-cache reruns.

## Current git/worktree hygiene notes

- Worktree is intentionally dirty with active implementation and untracked campaign files.
- Active untracked/modified files include local residual field modules, prediction-anchored bridge scripts, and slide notes (`HLT_V2_STRENGTH_2P5_SLIDE_NOTES.md`).
- Do not revert unrelated changes unless explicitly requested by user.

## What a new agent should do first

- Confirm cluster state and whether step dependencies are satisfied before requeue:
  - check `squeue --me`
  - confirm required upstream artifacts exist in manifest/cache/binary trees
  - cancel stale dependency chains and rerun from first missing stage
- Rebuild pseudo-HLT cache and splits together if any files were deleted.
- Keep one campaign active at a time to control disk.
- Validate with final report stage (including skipped/dependency metadata) before launching interpretation.
- Run with `PYTHONNOUSERSITE=1` on Tigris and correct data partition path for that host.

## Why we are here now

- We are at a scientific fork point where custom reconstructor/two-view architectures are not consistently beating direct strong HLT ParT, especially at QCD/Hgg HLT degradation around 0.6–2.5.
- Most practical next steps are either:
  - restart a clean, strong HLT-parity baseline then add one constrained auxiliary signal,
  - or return to earlier reproducible m2/reco7 style with corrected throughput and robust checks.

## Suggested next actions

1. Decide final target experiment: 10-class full setup vs QCD/Hgg binary.
2. Pick one run family (setmatching, dualview residual, or local residual) and remove all stale jobs/old checkpoints before relaunch.
3. Rebuild from scratch with explicit checkpoints and deterministic labels.
4. Require both best model and final-test-confirmation artifacts before claiming results.

