# Results draft: provenance and status

This file is not part of the rendered paper. The main results text is in
`sections/results.tex`; offline seed and uncertainty details are in
`sections/results_appendix.tex`, and the complete supporting HLT-like
methods and results are in `sections/supporting_hlt.tex`. Numeric tables are separate files in
`tables/` and are included from those sections. Upload the entire
`EdgeValue` directory to Overleaf, with `main.tex` as the main document.

## What is and is not available

- Main results lead with offline architectural comparisons, separating the
  direct value-path comparison from the combined-design gain. A short HLT
  summary follows as exploratory support, not a co-equal trigger benchmark.
- The supporting HLT appendix contains the 21-row single-seed **validation**
  screen, the six-row three-seed **test** architecture summary, and all
  18 three-seed HLT test configurations; 13 overlap with the screen,
  yielding 26 distinct trained configurations. No configurations or metric
  rows were dropped to change the narrative.
- All accuracy comparison tables now include macro one-versus-rest AUC.
  The main text gives QCD rejection at both 30% and 50% signal efficiency
  for all nine classes in the four completed offline configurations.
  The supporting appendix contains the six-core-architecture HLT rejection
  tables and complete panels for every HLT screen and test configuration.
- The paper now has **one offline results presentation targeting the 20M
  test**, rather than separate initial-offline and full-test sections.
  At the author's request, existing **500,000-jet** results are temporary
  numerical placeholders in that presentation, including macro AUC,
  rejection, seed details, paired intervals and the associated prose.
  Their background support remains 50,000 QCD jets, not two million.
  Dagger markers and explicit captions identify borrowed values. The four
  trained configurations, all seeds, and every signal at 30% and 50% are
  retained, including SAT and cases where selected EdgeValue is not best.
- No authenticated official **20M-test** metrics have been incorporated.
  The seven-row summary has four provisional rows and three `TBD` rows;
  no populated cell should be interpreted as a 20M measurement.
- The three new offline ablations remain planned. No jobs or training code
  were changed by writing this section.
- After the full-test results are downloaded and authenticated, replace all
  provisional offline values and revise derived comparisons and prose.
  Never simply relabel placeholder numbers as measured on 20M or imply the
  additional nine checkpoints have been evaluated.
- The original offline paired intervals condition on each fixed trained
  checkpoint pair. They are not training-seed confidence intervals.
- There are no conditional rejection intervals in the original reports.
  The large Hbb mean rejection increase is explicitly qualified by its
  low counts (baseline 10/10/11 versus selected EdgeValue 9/10/3 at 50%).
- All table rounding is performed from original unrounded means. In
  particular, HLT selected EdgeValue minus selected layerwise is **0.2633 pp**,
  not 0.2634 pp obtained by subtracting rounded displayed accuracies.
- No hypothesis-testing significance claim or complete capacity/compute
  attribution is asserted.
- HLT BASE_LAYERWISE has slightly lower macro AUC than BASE despite its small
  accuracy improvement. Selected HLT EdgeValue also has lower rejection than
  baseline for Hbb, Tbqq and Zqq at 30% efficiency. These counterexamples are
  retained in the prose and tables; improved accuracy is not described as a
  universal improvement in every operating point.
- Numerical random-seed IDs are omitted from rendered prose and captions;
  matched-seed count and uncertainty definitions are retained. Offline
  columns now use Replica A/B/C in their original order; exact identifiers
  are retained in `EXPERIMENTAL_DESIGN_NOTES.md` and the source artifacts.

## Frozen source artifacts

Relative to `C:/Users/22rya/ComputerScience/CERN/a_download_checkpoints/`:

1. `rpt_attention_bias_20260729T155648Z_bfcda5bd6f_049b2555e6/selection/screening_summary.json`
   - File SHA-256: `74e1c38e20a607c228b37399a77c29e311a3f1c552f79a96f2543adf85e83940`.
   - 21 rows; seed 101; comparison validation.
2. `rpt_attention_bias_20260729T155648Z_bfcda5bd6f_049b2555e6/reports/relational_part_report.json`
   - File SHA-256: `cc4ee22e4c1398657ed749b7517cc525c656578c6540e76fbf85dc634239d4a3`.
   - 18 rows; 500k final test; seeds 101/202/303.
3. `rpt_offline_transfer_20260809T195641Z_0737194d3c/reports/offline_transfer_report.json`
   - File SHA-256: `b6bdcb57364c152502b9d4132af73394af4fb4b30eaf9b77c9981b6c0e15fd56`.
   - Four models, three seeds, original 500k test; accuracy, rejection and
     paired accuracy intervals versus baseline.
4. `rpt_offline_transfer_20260809T195641Z_0737194d3c/final_test/<MODEL>/seed_<SEED>/metrics.json`
   - Supporting per-seed QCD counts, exact achieved efficiencies and
     saturation checks, plus per-class one-versus-rest AUCs for macro AUC.
5. `rpt_attention_bias_20260729T155648Z_bfcda5bd6f_049b2555e6/selection/screening_results.json`
   - File SHA-256: `8d6d761d8968b167561c2533ca526f157f6276082ae6601b4c772f5aea56c3c9`.
   - Full metrics for all 21 validation checkpoints, including AUC and
     classwise rejection. Each nested metric hash matches its row binding;
     each accuracy matches the screening summary.
   - HLT three-seed AUC and rejection use the final report's embedded
     `per_seed_complete_metrics`, not the earlier screening checkpoints.

## Numeric table conventions

Each numerical row has a LaTeX comment of the form
`% data:<table_group>:<source_run_id_or_signal>` for auditability.
Numeric table files are generated from the frozen JSON values rather than
copied from rounded chat summaries. Panels group three signal classes, each
with adjacent 30% and 50% columns, so all HLT configurations fit readably.
The offline summary is now `tables/results_offline_full_test.tex`; its
borrowed rows use the tag `offline_placeholder_500k`. Their numbers are
unchanged from the former `results_offline_initial.tex`. The
`\provisionalresult` macro in `main.tex` adds a visible dagger to borrowed
summary cells or to column headers when the whole column is provisional.
The macro is not a switch to hide provenance: remove markers only after
replacing and checking the underlying full-test measurements.

Accuracy is displayed in percent, SD and accuracy differences in percentage
points, and rejection as the arithmetic mean of three separately evaluated
rejections. Rejection changes are 100*(mean_candidate/mean_baseline - 1).
If any seed has zero passing QCD, a finite three-seed rejection mean is not
reported; the entry is `SAT`, not zero, a finite-only mean, or proof of
population-infinite rejection.

Macro AUC is the unweighted average over all ten stored one-versus-rest
class AUCs, including QCD, using the multiclass softmax probability for each
class versus all remaining classes. It is not a nine-signal-versus-QCD mean,
a micro AUC, or the macro accuracy field. The average is calculated within
each seed first; then the three-seed mean and sample SD (denominator n-1)
are reported on the 0--1 scale. No spread across classes is substituted for
training-seed variation. The original model-selection rule remains unchanged.

## Checks

Numerical tables are checked against their source values, with complete
21/18/4-configuration coverage: 43 unique macro-AUC summaries and 774 unique
configuration/class/efficiency rejection entries, based on 87 checkpoint
metric records. Repeated architecture summaries account for additional
display rows. The seven-row offline comparison contains four marked
provisional 500k rows and three TBD rows, including macro AUC; the seed
table follows the same configuration coverage. Static LaTeX checks cover included files,
balanced environments and braces, labels, references and citation keys.
The scientific text is reviewed against the original artifacts. No accessible
local LaTeX compiler is available; rendered PDF layout still needs checking
in Overleaf.

## Replacing the provisional offline numbers

1. Check the 20M evaluation's checkpoint binding, complete event coverage,
   and final report; keep the one-million-jet training description unchanged.
2. Replace accuracy, seed SD, matched differences and macro AUC in
   `tables/results_offline_full_test.tex`, and per-seed accuracies in
   `tables/results_offline_seeds.tex`. Keep untrained rows TBD.
3. Replace both offline rejection tables, including SAT flags and percentage
   changes. Recompute from unrounded per-seed results, not displayed values.
4. Update all offline numerical prose in `sections/results.tex` and
   `sections/results_appendix.tex`: error reduction, ordering, seed gains,
   paired intervals, passing counts and achieved efficiencies. Do not carry
   500k low-count caveats or confidence intervals over as 20M facts.
5. Add available full-test rejection intervals; remove the provisional
   notices and dagger markers only for results actually replaced. Update
   source hashes, data tags, design status and these notes together.
