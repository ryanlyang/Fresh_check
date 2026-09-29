# Experimental-design draft: author notes

This file is not part of the rendered paper. Open `main.tex` as the Overleaf
main document and upload the entire `EdgeValue` folder, including `sections/`
and `references.bib`. The experimental section is in
`sections/experimental_design.tex`. The results draft is in
`sections/results.tex`, with offline details in `sections/results_appendix.tex`
and the full reduced-view study in `sections/supporting_hlt.tex`. Introduction
and the pending full-test results remain unfinished.

## Status and interpretation

- The main narrative is an offline architecture study: should pair
  information affect attention weights only, or also value messages?
  Shared versus layerwise bias is a separate architectural factor;
  selected additional relations are a secondary extension.
- Detailed HLT-like methods, all screening results, and all confirmation
  results are in the supporting appendix. The main text retains a short
  summary and the feature-selection provenance. Do not turn the synthetic,
  hand-degraded setting into a real-trigger benchmark or an independently
  sampled replication.
- The standard-feature 2x2 architecture matrix is planned, not complete.
  The existing selected-layerwise pair with/without EdgeValue is the direct
  value-path comparison. Baseline-to-best changes multiple factors; the
  combined gain must not be described as an isolated EdgeValue effect.
- HLT has 21 single-seed validation screening configurations and 18
  three-seed final-test configurations, with overlap: 26 unique trained
  configurations in all. The six-row architecture summary is a subset of
  the final-test comparisons, not the whole screen.
- Four offline configurations (12 checkpoints) have completed training. Three
  additional configurations (nine checkpoints) are planned, not implemented
  or launched by this paper-writing task. The current 20M inference array
  covers only the original twelve checkpoints.
- The seven-row offline matrix is intentional. Selected features with shared
  bias and no EdgeValue would be the eighth factorial cell, but the user has
  not requested adding it. Do not call the current matrix fully factorial.
- New ablations follow inspection of the earlier 500k offline test. Preserve
  this chronology; do not describe them as part of the original preregistered
  four-row study. Update status/tense once the new runs and evaluation finish.
- HLT-like is synthetic, not detector-realistic HLT reconstruction. Offline
  and HLT-like development samples share identities, and offline weights are
  trained from scratch. Avoid claims of independent development datasets,
  pretrained-weight transfer, a globally untouched original test, or trigger
  readiness.
- The results draft uses completed HLT results and one offline presentation
  targeting the official 20M test. Four offline rows contain explicitly
  dagger-marked provisional 500k values; three untrained rows remain TBD.
  These are drafting placeholders, not measured 20M results. Replace the
  numbers and derived interpretations when authenticated results arrive.
  See `RESULTS_NOTES.md` for provenance.
- Rendered prose and captions use one common seed or three matched seeds,
  without numerical seed IDs. Per-seed columns are Replica A/B/C; the
  source mapping and fixed generation/bootstrap seeds remain in the author
  notes and campaign artifacts rather than the paper's narrative.

## Reproducibility details omitted from the rendered prose

- Replica A/B/C correspond to training seeds 101/202/303, respectively;
  this ordering also applies to listed paired intervals and count triples.
- The HLT-like initial screen uses seed 101, reused in matching
  confirmation configurations; confirmation uses all three seeds.
- Original reduced-view cache seeds: training 1053, checkpoint validation
  1054, comparison validation 1056, earlier test 1057.
- The reported paired-bootstrap protocol uses PCG64 seed 917301.
- These identifiers have not changed in code or artifacts. Their removal
  from rendered text is editorial, not a new randomization protocol.

## Sources checked

Downloaded HLT root (relative to `C:/Users/22rya/ComputerScience/CERN/a_download_checkpoints/`):

`rpt_attention_bias_20260729T155648Z_bfcda5bd6f_049b2555e6`

- `inputs/hlt_expectation.json`, `inputs/hlt_cache_audit.json`: split sizes,
  generator configuration, cached seeds and active input view.
- `registry/screening_registry.json`, `registry/relation_family_registry.json`:
  screening matrix and encoded channel dimensions.
- `selection/screening_summary.json`, `selection/confirmation_registry.json`:
  the validation ranking, selected families and 17-row confirmation matrix.
- `registry/semantic_control_registry.json`, `reports/relational_part_report.json`:
  unary control, perturbations, and 18 final-test configurations.
- Generator behavior was inspected at frozen commit
  `bfcda5bd6fd48037ab198bcca1eadf6f4d87e131`, especially
  `jetclass_fixed_hlt.py::apply_hlt_single_jet_m2style`.

Downloaded offline root:

`rpt_offline_transfer_20260809T195641Z_0737194d3c`

- `campaign_spec.json`: four original rows, seeds, split relationship,
  training/selection policies, and prior-test-exposure caveat.
- `runs/*/seed_*/training_curves.json` and checkpoint registrations: planned
  optimizer-update counts and recorded training precision.
- Frozen source commit `0737194d3c77d03826b306f03b986b278ecec7df`:
  `teacher_logit_reco/relational_part/train.py` and `determinism.py` for the
  optimizer, checkpoint rule, stopping and schedule;
  `model.py` for backbone widths and attention blocks;
  `evaluation.py` for metrics;
  `jetclass_fresh/part_inputs.py` for the 17 input features.
- HLT architecture-recovery contracts supersede some original runtime
  metadata. Do not infer exact final recovered Weaver versions from the
  first-generation architecture registry alone.

Official full-test evaluation driver:

- `scripts/evaluate_relational_part_offline_full_test.py`: full-test coverage,
  immutable checkpoint binding, classwise paired bootstrap, score and tie
  conventions, Wilson intervals, additional operating points and seed means.
- The current driver supports four configurations. Adding the three planned
  ablations requires separate implementation/evaluation work, not edits to
  the running evaluation.

Bibliographic metadata was checked against the official
[ParT proceedings page](https://proceedings.mlr.press/v162/qu22b.html),
[JetClass/ParT repository](https://github.com/jet-universe/particle_transformer),
and [Shaw et al. (2018), ACL Anthology](https://aclanthology.org/N18-2074/).
The methods section cites `Shaw2018Relative` for the value-side mechanism
(their Eq. 3), not the entire local attention rule. EdgeValue is our label,
not a first-invention claim. ParT-specific novelty still requires the wider
related-work assessment; this citation does not establish first use in jets.

## Draft checks

The HLT design received an independent read-only factual review against the
frozen campaign records. Static checks passed for LaTeX input paths, balanced
environments and braces, unique labels, resolved references and bibliography
keys, the seven-row offline matrix, provisional-source labeling, and the
three untrained TBD rows. No accessible local LaTeX compiler was available, so PDF compilation
and rendered table/page layout still need checking in Overleaf.
