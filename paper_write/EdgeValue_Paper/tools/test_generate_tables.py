"""Small regression tests; source-dependent tests skip if the archive is absent."""

import copy
import hashlib
import json
import math
import unittest

import generate_tables as tables


class AggregationTests(unittest.TestCase):
    def test_sample_standard_deviation(self):
        result = tables.summary([1.0, 2.0, 3.0])
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["sample_sd"], 1.0)
        self.assertEqual(set(result["per_seed"]), {"101", "202", "303"})

    def test_nonfinite_values_cannot_enter_a_summary(self):
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                tables.summary([1.0, 2.0, value])

    def test_zero_background_cell_is_not_a_finite_point(self):
        for zero_count in (1, 2, 3):
            with self.subTest(zero_count=zero_count):
                cell = {"zero_background_seed_count": zero_count, "mean": None,
                        "sample_sd": None, "gain_percent": None}
                self.assertEqual(tables.rejection_cell(cell),
                                 "ZB " + str(zero_count) + "/3")

    def test_finite_rejection_cell_has_only_mean_and_sample_sd(self):
        cell = {"zero_background_seed_count": 0, "mean": 17415.4,
                "sample_sd": 784.2, "gain_percent": 0.0}
        self.assertEqual(tables.rejection_cell(cell), r"$17415 \pm 784$")
        cell.update(mean=95.43, sample_sd=1.04, gain_percent=5.65)
        self.assertEqual(tables.rejection_cell(cell), r"$95.4 \pm 1.0$")
        cell["gain_percent"] = None
        self.assertEqual(tables.rejection_cell(cell), r"$95.4 \pm 1.0$")


class TableLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Layout checks also run when the external evaluation archive is absent.
        cls.data = json.loads((tables.PAPER_ROOT / "data/derived_metrics.json")
                              .read_text(encoding="utf-8"))

    def assert_placeholder_caption(self, output, label):
        self.assertEqual(output.count(r"\caption{"), 1)
        self.assertEqual(output.count("Placeholder."), 1)
        self.assertRegex(output, r"\\caption\{[\s\S]*?Placeholder\.\}")
        self.assertNotIn(r"\refstepcounter", output)
        self.assertEqual(output.count(r"\label{"), 1)
        self.assertIn(label, output)
        self.assertLess(output.index("Placeholder.}") + len("Placeholder.}"),
                        output.index(label))

    def test_rejection_tables_are_upright_with_placeholder_captions(self):
        for display in tables.DISPLAY_TABLES:
            with self.subTest(display=display):
                output = tables.rejection_tex(self.data, display)
                self.assertEqual(output.count(r"\begin{table}[p]"), 1)
                self.assertEqual(output.count(r"\end{table}"), 1)
                self.assert_placeholder_caption(
                    output, r"\label{tab:rejection-" +
                    ("benchmark" if display == "benchmark" else "30") + "}")
                self.assertIn(r"\footnotesize", output)
                for forbidden in ("sidewaystable", "landscape", r"\rotatebox",
                                  r"\resizebox", r"\scalebox", r"\tiny", r"\scriptsize",
                                  r"\shortstack", "---", "baseline", "gain"):
                    self.assertNotIn(forbidden, output)

    def test_panels_preserve_every_configuration_signal_and_target(self):
        covered = []
        expected_panels = (
            ("(a) Standard pair inputs", ("S-S", "S-L", "S-S+EV", "S-L+EV")),
            ("(b) Selected pair inputs (standard + PT + TRACK + REGION)",
             ("E-S", "E-L", "E-S+EV", "E-L+EV")),
        )
        by_id = {config["id"]: config for config in self.data["configurations"]}
        for display in tables.DISPLAY_TABLES:
            signals = tables.SIGNALS if display == "benchmark" else tables.HADRONIC_SIGNALS
            targets = (tables.BENCHMARK_TARGETS if display == "benchmark" else
                       {signal: "0.3" for signal in signals})
            with self.subTest(display=display):
                output = tables.rejection_tex(self.data, display)
                panels = output.split(r"\begin{tabular*}")[1:]
                self.assertEqual(len(panels), 2)
                self.assertEqual(output.count(r"\end{tabular*}"), 2)
                self.assertEqual(output.count(r"\par\vspace{1em}"), 1)
                for panel, (heading, ids) in zip(panels, expected_panels):
                    body = panel.split(r"\end{tabular*}")[0]
                    self.assertTrue(body.startswith(
                        r"{\textwidth}{@{\extracolsep{\fill}}lc*{4}{c}@{}}"))
                    title = ("QCD rejection at class-specific signal efficiencies" if
                             display == "benchmark" else r"QCD rejection at 30\% signal efficiency")
                    self.assertIn(r"\multicolumn{6}{@{}l}{" + title + "}", body)
                    self.assertIn(r"\textbf{" + heading + "}", body)
                    header = (r"Signal & $\epsilon_s$ (\%) & Shared & Layerwise & "
                              r"Shared + EV & Layerwise + EV \\")
                    self.assertIn(header, body)
                    rows = [line for line in body.splitlines()
                            if line.startswith("$") and " & " in line]
                    self.assertEqual(len(rows), len(signals))
                    for signal, row in zip(signals, rows):
                        target = targets[signal]
                        self.assertTrue(row.endswith(r" \\"))
                        cells = row[:-3].split(" & ")
                        self.assertEqual(cells[0], tables.CLASS_TEX[signal])
                        self.assertEqual(cells[1], f"{100 * float(target):g}")
                        self.assertEqual(len(cells), 6)
                        for config_id, cell in zip(ids, cells[2:]):
                            config = by_id[config_id]
                            expected = (r"\pending" if config["status"] == "pending" else
                                        tables.rejection_cell(config["rejection"][target][signal]))
                            self.assertEqual(cell, expected)
                            self.assertNotIn(r"\%", cell)
                            self.assertNotIn(r"\shortstack", cell)
                            covered.append((config_id, signal, target))
                self.assertEqual(output.count(r"\pending &") +
                                 output.count(r"\pending \\"), 4 * len(signals))
        expected_coverage = {(config[0], signal, tables.BENCHMARK_TARGETS[signal])
                             for config in tables.CONFIGS for signal in tables.SIGNALS}
        expected_coverage |= {(config[0], signal, "0.3") for config in tables.CONFIGS
                              for signal in tables.HADRONIC_SIGNALS}
        self.assertEqual(set(covered), expected_coverage)
        self.assertEqual(len(covered), len(expected_coverage))

    def test_overview_has_placeholder_caption_and_preserves_delta_columns(self):
        output = tables.overview_tex(self.data)
        self.assert_placeholder_caption(output, r"\label{tab:overview}")
        self.assertIn(r"ID & Accuracy (\%) & $\Delta$ (pp) & Macro AUROC (\%) & $\Delta$ (pp)",
                      output)
        rows = [line for line in output.splitlines() if line.startswith(r"\texttt{")]
        self.assertEqual(len(rows), 8)
        expected_ids = ("Std-S", "Std-L", "Std-S+EV", "Std-L+EV",
                        "Sel-S", "Sel-L", "Sel-S+EV", "Sel-L+EV")
        for config, row, expected_id in zip(self.data["configurations"], rows, expected_ids):
            cells = row[:-3].split(" & ")
            self.assertEqual(cells[0], r"\texttt{" + expected_id + "}")
            if config["status"] == "pending":
                self.assertEqual(cells[1:], [r"\pending"] * 4)
                continue
            for metric, gain, column in (("accuracy_percent", "accuracy_gain_pp", 1),
                                         ("macro_ovr_auroc_percent", "macro_ovr_auroc_gain_pp", 3)):
                values = config[metric]
                self.assertEqual(cells[column],
                                 f"${values['mean']:.3f} \\pm {values['sample_sd']:.3f}$")
                delta = config[gain]
                self.assertEqual(cells[column + 1], f"${delta:+.3f}$" if delta else "$0.000$")

    def test_display_labels_follow_architecture_not_historical_ids(self):
        for config in self.data["configurations"]:
            changed = {**config, "id": "unrelated-id", "source_run_id": "historical-name"}
            self.assertEqual(tables.display_id(changed), tables.display_id(config))
            self.assertEqual(tables.rejection_column(changed), tables.rejection_column(config))

    def test_zero_background_caption_is_included_only_for_displayed_zero_counts(self):
        for display in tables.DISPLAY_TABLES:
            original = tables.rejection_tex(self.data, display)
            self.assertNotIn("ZB", original)
            self.assertNotIn("zero accepted", original)
            changed = copy.deepcopy(self.data)
            target = "0.5" if display == "benchmark" else "0.3"
            changed["configurations"][0]["rejection"][target]["Hbb"].update(
                zero_background_seed_count=1, mean=None, sample_sd=None, gain_percent=None)
            output = tables.rejection_tex(changed, display)
            self.assertIn("ZB 1/3", output)
            self.assertIn(r"ZB $k/3$ denotes zero accepted QCD jets", output)
            self.assertIn(r"\pending\ denotes pending results. Placeholder.", output)

    def test_rejection_display_is_independent_of_stored_gains(self):
        changed = copy.deepcopy(self.data)
        for config in changed["configurations"]:
            if config["status"] == "evaluated":
                for target in tables.TARGETS:
                    for cell in config["rejection"][target].values():
                        cell["gain_percent"] = 123456.789
        for display in tables.DISPLAY_TABLES:
            self.assertEqual(tables.rejection_tex(changed, display),
                             tables.rejection_tex(self.data, display))

    def test_original_metrics_remain_exactly_unchanged(self):
        # Fingerprint of every configuration, seed provenance, accuracy/AUC and
        # original 30%/50% rejection cell before adding the new presentation.
        original = copy.deepcopy(self.data["configurations"])
        for config in original:
            # The pre-existing schema-3 generator renamed only internal R-* IDs
            # to E-*. Canonicalize that metadata to the legacy fingerprint; no
            # measured value or source-run identifier is modified.
            if config["id"].startswith("E-"):
                config["id"] = "R-" + config["id"][2:]
            if config["rejection"] is not None:
                config["rejection"] = {target: config["rejection"][target]
                                       for target in ("0.3", "0.5")}
        canonical = json.dumps(original, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(),
                         "2995abf525aa16d13ed3d4db78f053e79e69dc1cbe4c0e674011259dd7a08d54")

    def test_all_metrics_and_seed_provenance_survive_display_relabeling(self):
        original = copy.deepcopy(self.data["configurations"])
        for config in original:
            del config["id"]
        canonical = json.dumps(original, sort_keys=True, separators=(",", ":"), allow_nan=False)
        # Captured before changing labels, includes all four retained targets,
        # all source-run identifiers, full-precision metrics and seed provenance.
        self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(),
                         "6c78bfc18b20ceee70cf1689a8897ae38f2d54f397af5adc64462cecb782c813")

    def test_leptonic_classes_are_omitted_only_from_30_percent_display(self):
        output = tables.rejection_tex(self.data, "0.3")
        benchmark = tables.rejection_tex(self.data, "benchmark")
        self.assertIn(r"\cite{Qu2022ParticleTransformer}", benchmark)
        for signal in ("Hqql", "Tbl"):
            self.assertNotIn(tables.CLASS_TEX[signal] + " & ", output)
            self.assertIn(tables.CLASS_TEX[signal] + " & ", benchmark)
            for config in self.data["configurations"]:
                if config["status"] == "evaluated":
                    self.assertEqual(config["rejection"]["0.3"][signal]
                                     ["zero_background_seed_count"], 3)
                    selected = config["rejection"][tables.BENCHMARK_TARGETS[signal]][signal]
                    self.assertEqual(selected["zero_background_seed_count"], 0)

    def test_presentation_policy_is_global_and_rejects_drift(self):
        policy = self.data["presentation_policy"]
        self.assertEqual(self.data["schema_version"], 3)
        self.assertTrue(policy["working_points_apply_identically_to_all_configurations_and_seeds"])
        targets = policy["benchmark"]["signal_efficiency_by_class"]
        self.assertEqual(targets, {signal: "0.99" if signal == "Hqql" else
                                  "0.995" if signal == "Tbl" else "0.5"
                                  for signal in tables.SIGNALS})
        changed = copy.deepcopy(self.data)
        changed["presentation_policy"]["benchmark"]["signal_efficiency_by_class"]["Hqql"] = "0.5"
        with self.assertRaisesRegex(ValueError, "presentation policy drift"):
            tables.validate_derived(changed)
        with self.assertRaisesRegex(ValueError, "Unknown rejection table"):
            tables.rejection_tex(self.data, "0.5")


@unittest.skipUnless(tables.DEFAULT_SOURCE_ROOT.exists(), "Read-only source archive unavailable")
class SourceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.plan, provenance = tables.read_sources(tables.DEFAULT_SOURCE_ROOT)
        cls.data = tables.derive(cls.report, cls.plan, provenance)

    def test_exact_coverage_and_arithmetic(self):
        tables.validate_derived(self.data)
        self.assertEqual(len(self.data["configurations"]), 8)
        self.assertEqual(sum(c["status"] == "pending" for c in self.data["configurations"]), 4)
        self.assertAlmostEqual(self.data["configurations"][0]["accuracy_percent"]["mean"],
                               82.81902666666666, places=10)
        self.assertAlmostEqual(self.data["configurations"][-1]["macro_ovr_auroc_percent"]["mean"],
                               98.3992349356175, places=10)

    def test_macro_auc_is_averaged_within_seed(self):
        for config in self.data["configurations"]:
            if config["status"] == "pending":
                continue
            for record in self.report["per_model_results"]:
                if record["task"]["run_id"] != config["source_run_id"]:
                    continue
                seed = str(record["task"]["seed"])
                expected = sum(record["metrics"]["one_vs_rest_auc"].values()) / 10 * 100
                self.assertAlmostEqual(config["macro_ovr_auroc_percent"]["per_seed"][seed],
                                       expected, places=12)

    def test_high_efficiency_rejections_match_saved_counts_for_all_seeds(self):
        expected_counts = {
            "S-S": {"Hqql": [1260, 1281, 1297], "Tbl": [502, 538, 499]},
            "S-L+EV": {"Hqql": [1033, 1056, 1153], "Tbl": [444, 448, 522]},
            "E-L": {"Hqql": [1176, 1060, 1251], "Tbl": [566, 506, 522]},
            "E-L+EV": {"Hqql": [1050, 1020, 1037], "Tbl": [406, 404, 455]},
        }
        for config in self.data["configurations"]:
            if config["status"] == "pending":
                continue
            self.assertEqual(set(config["rejection"]), {"0.3", "0.5", "0.99", "0.995"})
            for signal, counts in expected_counts[config["id"]].items():
                target = tables.BENCHMARK_TARGETS[signal]
                cell = config["rejection"][target][signal]
                self.assertEqual(cell["per_seed_qcd_false_positive_count"],
                                 dict(zip(map(str, tables.SEEDS), counts)))
                expected = tables.summary([tables.CLASS_SUPPORT / count for count in counts])
                for seed in map(str, tables.SEEDS):
                    self.assertAlmostEqual(cell["per_seed_rejection"][seed],
                                           expected["per_seed"][seed], delta=1e-9)
                self.assertAlmostEqual(cell["mean"], expected["mean"], delta=1e-9)
                self.assertAlmostEqual(cell["sample_sd"], expected["sample_sd"], delta=1e-9)

    def test_fabricated_pending_result_is_rejected(self):
        changed = copy.deepcopy(self.data)
        changed["configurations"][1]["accuracy_gain_pp"] = 0.0
        with self.assertRaisesRegex(ValueError, "Fabricated pending"):
            tables.validate_derived(changed)

    def test_finite_only_zero_background_mean_is_rejected(self):
        changed = copy.deepcopy(self.data)
        changed["configurations"][0]["rejection"]["0.5"]["Hqql"]["mean"] = 833333.3333333333
        with self.assertRaisesRegex(ValueError, "Zero-background"):
            tables.validate_derived(changed)

    def test_changed_source_content_is_rejected(self):
        changed = copy.deepcopy(self.report)
        changed["event_count_per_model"] += 1
        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            tables.validate_content_hash(changed, "Changed fixture")


if __name__ == "__main__":
    unittest.main()
