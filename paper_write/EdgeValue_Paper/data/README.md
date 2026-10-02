# Results and provenance

These are derived summaries, not new training or evaluation results. Four offline
configurations were evaluated with training seeds 101, 202, and 303; the other
four configurations remain explicitly pending in every table.

## Reproduce and validate

From the repository root, with Python 3.10 or later and no third-party packages:

```powershell
python paper_write/EdgeValue_Paper/tools/generate_tables.py
python paper_write/EdgeValue_Paper/tools/generate_tables.py --check
python -m unittest discover -s paper_write/EdgeValue_Paper/tools -p "test_*.py"
```

The default read-only archive is
`C:/Users/22rya/ComputerScience/CERN/a_download_checkpoints/rpt_offline_test20m_20260925T235646Z`.
For a relocated archive, pass `--source-root PATH`. The two inputs are
`reports/full_test_report.json` and `evaluation_plan.json` at the archive root.
The report is read from `per_model_results[*].metrics`, with each record matched
to `task.run_id` and `task.seed`; the top-level accuracy and finite-rejection
summaries are used only as cross-checks.

The generator writes `tables/overview.tex`, `tables/rejection_30.tex`,
`tables/rejection_50.tex`, and `data/derived_metrics.json`. `--check` writes nothing:
it validates all source/content hashes, task/seed coverage, class support,
confusion-matrix accuracy, and count-derived rejection, then checks all four
generated files byte-for-byte. The source inputs are never changed. TeX requires
`booktabs` and a `\pending` macro defined by the main document; no rotation
package is required.

Each rejection table is an upright portrait `table[p]` with two vertically
stacked panels: (a) standard pair inputs and (b) selected pair inputs. Each panel
has columns Shared, Layerwise, Shared + EV, and Layerwise + EV; the overview uses
the more explicit `Std-*` and `Sel-*` manuscript labels listed below.
The main benchmark table lists all nine signal classes, at 50% signal efficiency
for the seven hadronic classes, 99% for Hqql, and 99.5% for Tbl, following the
original ParT/JetClass benchmark convention. The historical filename
`rejection_50.tex` is retained, but its caption and label
`tab:rejection-benchmark` explicitly describe these mixed working points.
The supplementary `rejection_30.tex` lists only the seven hadronic classes;
Hqql and Tbl have zero accepted QCD events in all completed configurations and
seeds at 30%, so they are omitted from this display, not from the backing data.
Each panel has an explicit signal-efficiency column and mean plus/minus sample
standard deviation in each finite result cell. These row-wise efficiencies are
identical for every model and seed, including the pending configurations.
All three tables have a concise caption ending with "Placeholder."
Current displayed cells have finite rejection values, so captions omit the
zero-background convention. If future displayed cells contain zero accepted
QCD counts, the generator automatically restores the `ZB k/3` caption explanation.
The panels share one label per table, use readable footnote-size text
without scaling, and preserve pending and zero-background cells. Rejection
percentage gains are retained only in JSON, not displayed in the tables. The
overview retains its accuracy and macro-AUROC percentage-point difference columns.
Layout regression tests check all 8 configurations and each displayed row's
working point even when the external evaluation archive is unavailable. A
regression fingerprint verifies that all original accuracy/AUROC and 30%/50%
rejection measurements remain exactly unchanged.

The JSON records the input file SHA-256 hashes separately from their canonical
JSON content hashes, plus per-seed checkpoint, model-contract, result, and metric
content hashes. Canonical hashes use sorted keys, compact separators, and omit
the document's own `content_hash` field. The archive's `plan_sha256` is this
canonical content hash, not the hash of the raw JSON file bytes.

Derived schema version 2 added 99% and 99.5% to the retained working points, with
all nine signal classes retained at all four targets (30%, 50%, 99%, 99.5%).
It also serializes the global table presentation policy separately from the
unchanged source metric policy. The source evaluation plan, report, models,
checkpoints, and training contracts are not modified or re-evaluated.
The existing schema-3 generator uses internal `E-*` rather than `R-*` IDs for
selected-input configurations. Manuscript display labels are derived separately
from configuration metadata; relabeling them does not change these internal IDs,
source run IDs, measured values, or provenance. Tests check both the historical
measurement fingerprint (canonicalizing only that prior ID change) and a full
metric/provenance fingerprint across all four retained targets.

## Configuration mapping

The first paper ID component denotes standard pair inputs (`Std`) or standard
pair inputs plus selected PT, TRACK, and REGION families (`Sel`); the second
denotes shared (S) or layerwise (L)
pair bias. `+EV` denotes EdgeValue: these configurations are ParT-EV variants,
while those without `+EV` are controls without EdgeValue. The scientific
configuration IDs and source run IDs are unchanged. Existing run names do not
consistently encode all three choices: notably, `OFF_RPT_BASE_EDGEVALUE` is
**layerwise**.

| Paper ID | Internal ID | Source/planned run ID | Status |
| --- | --- | --- | --- |
| Std-S | S-S | OFF_RPT_BASE | Evaluated baseline |
| Std-L | S-L | OFF_RPT_BASE_LAYERWISE | Pending |
| Std-S+EV | S-S+EV | OFF_RPT_BASE_SHARED_EDGEVALUE | Pending |
| Std-L+EV | S-L+EV | OFF_RPT_BASE_EDGEVALUE | Evaluated |
| Sel-S | E-S | OFF_RPT_SELECTED_SHARED | Pending |
| Sel-L | E-L | OFF_RPT_SELECTED_LAYERWISE | Evaluated |
| Sel-S+EV | E-S+EV | OFF_RPT_SELECTED_SHARED_EDGEVALUE | Pending |
| Sel-L+EV | E-L+EV | OFF_RPT_SELECTED_EDGEVALUE | Evaluated |

The four pending identifiers describe the intended follow-up matrix; the
supplied evaluation plan contains only the twelve already evaluated run/seed
tasks. The generator deliberately rejects changed source coverage until it is
explicitly updated for a new evaluation artifact.

## Metric conventions and limits

- All central values are arithmetic means over the three seeds. Uncertainties
  are sample standard deviations (`ddof=1`), not standard errors or confidence
  intervals. They describe seed variation, not the full experimental uncertainty.
- Macro one-versus-rest AUROC is the unweighted average of **all ten** class
  AUROCs, including QCD, within each seed, then summarized across seeds. It is
  not a QCD-versus-signal binary AUC or a pooled prediction AUC.
- Accuracy and AUROC are percentages. Their baseline differences are percentage
  points. QCD-rejection gains are `100 * (model mean / baseline mean - 1)`, not
  the mean of three seedwise percentage gains; these gains are retained only
  in JSON.
- Rejection uses the archive's signal-minus-QCD logit score, empirical signal
  threshold, and inclusive `>=` pass rule. Each seed has 20 million test jets:
  two million QCD jets and two million jets from each of nine signal classes.
- If **any** seed accepts zero QCD events, the aggregate cell is `ZB k/3`, where
  `k` is the zero-count seed count. No finite-only mean, aggregate standard
  deviation, or baseline percentage increase is reported. Per-seed counts and
  finite/nonfinite values are retained in JSON. This is not evidence of infinite
  population rejection.
- The archive supplies conditional Wilson intervals that exclude empirical
  signal-threshold uncertainty. Some zero-count endpoints contain tiny positive
  floating-point residuals, yielding enormous finite upper rejection endpoints.
  Those bounds are neither displayed nor reinterpreted here; no population-level
  inference is made from them.
- This is a supplemental, inference-only test of models trained on one million
  jets, not training on 20 million jets and not a new model-selection round.
  Missing cells prevent a complete factorial attribution of pair-bias sharing,
  relation features, and EdgeValue. The available Sel-L versus Sel-L+EV comparison
  holds selected relation inputs and layerwise bias fixed.

The JSON retains full precision. TeX rounds accuracy/AUROC and their differences
to three decimals; finite rejection is rounded to an integer at or above 1000
and to one decimal below 1000. All calculations precede display rounding.
