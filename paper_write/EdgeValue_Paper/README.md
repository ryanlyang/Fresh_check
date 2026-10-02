# ParT-EV paper

Open `main.tex` to build the paper. The written sections are **Method** and
**Experimental Setup**. `sections/results.tex` currently includes only the
result tables, without a Results heading or narrative. Each table has a concise
caption ending with "Placeholder." Introduction,
Related Work, and Conclusion are deliberately empty. The author field preserves the starting
template's name and should be replaced with the final author list before release.

**ParT-EV** names the Particle Transformer family augmented with edge-conditioned
value messages; **EdgeValue** names the message mechanism. Relation inputs and
bias sharing remain explicit variant choices. The baseline and no-EdgeValue
controls are not renamed as ParT-EV. Existing run IDs and artifact paths are
unchanged, including the `EdgeValue_Paper` directory name.

The results use the **official 20-million-jet test evaluation**
of models trained on **one million jets**.
The four completed configurations each have three training seeds. The other
four configurations appear as explicit `TBD` placeholders throughout the tables;
there are no estimated or invented results. There are no HLT result tables.
Paper labels use `Std` for standard pair inputs and `Sel` for standard inputs
plus PT, TRACK, and REGION. For example, `Std-S` is the shared-bias baseline
and `Sel-L+EV` uses selected relations, layerwise bias, and EdgeValue.
Rejection-table panels identify the feature set, so their columns use
`Shared`, `Layerwise`, `Shared + EV`, and `Layerwise + EV`.
Internal configuration and source run IDs are retained in the backing data.

## Build

With a standard TeX distribution including `latexmk` and BibTeX:

```powershell
cd paper_write/EdgeValue_Paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Alternatively run `pdflatex main`, `bibtex main`, and `pdflatex main` twice.
The two per-class rejection tables are upright portrait tables. Each contains
two stacked panels (standard and selected relations), with signal classes in
rows, an explicit signal-efficiency column, and four configuration columns.
The main rejection table follows the JetClass benchmark: 50% for the seven
hadronic classes, 99% for `Hqql`, and 99.5% for `Tbl`. The additional 30% table
shows only the seven hadronic classes. The document uses standard article,
math, table, typography, and numeric
citation packages. No external figure assets or shell escape are required.

The paper's references separately cite ParT and the JetClass dataset,
and cover multihead attention, prior
relation-aware value messages, angular clustering, AdamW, and ROC analysis.
It does not claim that relation-dependent
value messages are themselves a new general attention mechanism.

## Result tables and audit

The tables and their full-precision backing data are generated rather than
manually transcribed. From the repository root:

```powershell
python paper_write/EdgeValue_Paper/tools/generate_tables.py --check
python -m unittest discover -s paper_write/EdgeValue_Paper/tools -p "test_*.py"
```

See [data/README.md](data/README.md) for source paths, checkpoint/result hashes,
configuration-to-run-ID mappings, generation commands, and metric conventions.
The LaTeX and derived JSON are self-contained for reading and compiling; checking
them against the original artifacts requires the downloaded report and evaluation
plan. No training or evaluation artifacts are modified by the generator.

The current displayed operating points have finite rejection values; the
manuscript omits the zero-background convention until it is needed in a table.
The generator and backing data retain that handling. Unfinished measurements
remain `TBD`. Rejection tables show only mean rejection and seed sample
standard deviation, with no percentage-gain lines. Accuracy and macro-AUROC
tables retain their percentage-point differences from baseline. Full-precision
rejection gains remain in the backing JSON but are not displayed in the paper.
Reported spreads are seed sample standard deviations, not confidence intervals.
The omitted leptonic 30% results and their original 50% results remain in the
backing data, including zero-background counts. The higher-efficiency values
come from the same saved 20M evaluation, so no inference was rerun. Operating
points are chosen per class, identically for all configurations and seeds,
following the cited ParT/JetClass benchmark. For compatibility, the mixed
benchmark table retains the filename `tables/rejection_50.tex`; its caption,
row efficiencies, and label identify the mixed operating points explicitly.

## Before replacing placeholders or releasing the paper

- Add the completed follow-up evaluation artifact as a separately authenticated
  source, update the generator's declared coverage, and regenerate all tables.
  Do not replace `TBD` with guessed values or reuse a different test population.
- Write the Results narrative once all eight configurations are measured.
  In particular, shared bias plus EdgeValue is not yet a completed result.
- Preserve the distinction between combined configuration gains and isolated
  effects. The historical shared/layerwise paths differ in trimming and pair
  normalization; this is not universally parameter- or compute-matched.
- Complete the intentionally blank sections and final author information only
  when requested.

The sibling deleted `paper_write/EdgeValue` files and existing ZIP archives have
not been restored or rewritten by these paper updates.
