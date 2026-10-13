"""Stdlib regression tests for the complete Tigris factorial manuscript tables."""

import copy
import hashlib
import json
import math
import subprocess
import sys
import unittest

import generate_tables as tables


def rehash(document):
    payload = {key: value for key, value in document.items() if key != "content_hash"}
    document["content_hash"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class AggregationTests(unittest.TestCase):
    def test_sample_standard_deviation(self):
        result = tables.summary([1.0, 2.0, 3.0])
        self.assertEqual(result, {"mean": 2.0, "sample_sd": 1.0,
                                  "per_seed": {"101": 1.0, "202": 2.0, "303": 3.0}})

    def test_nonfinite_and_incomplete_summary_rejected(self):
        for values in ([1.0, 2.0], [1.0, 2.0, math.inf], [1.0, 2.0, math.nan]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                tables.summary(values)

    def test_rejection_cell_keeps_zero_background_distinct(self):
        for n in (1, 2, 3):
            self.assertEqual(tables.rejection_cell({"zero_background_seed_count": n}), f"ZB {n}/3")
        cell = {"zero_background_seed_count": 0, "mean": 17415.4, "sample_sd": 784.2}
        self.assertEqual(tables.rejection_cell(cell), r"$17415 \pm 784$")
        cell.update(mean=95.43, sample_sd=1.04)
        self.assertEqual(tables.rejection_cell(cell), r"$95.4 \pm 1.0$")


class TableLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((tables.PAPER_ROOT / "data/derived_metrics.json").read_text())
        cls.by_id = {c["id"]: c for c in cls.data["configurations"]}

    def test_schema_coverage_and_arithmetic(self):
        tables.validate_derived(self.data)
        self.assertEqual(self.data["schema_version"], 4)
        self.assertEqual(len(self.data["configurations"]), 8)
        self.assertEqual({c["status"] for c in self.data["configurations"]}, {"evaluated"})
        self.assertEqual(sum(len(c["seed_provenance"]) for c in self.data["configurations"]), 24)

    def test_every_table_has_real_caption_and_is_upright(self):
        outputs = {
            "overview": tables.overview_tex(self.data),
            "edgevalue-baseline": tables.edgevalue_baseline_tex(self.data),
            "factorial": tables.factorial_tex(self.data),
            "rejection-main": tables.rejection_main_tex(self.data),
            "rejection-benchmark": tables.rejection_tex(self.data, "benchmark"),
            "rejection-30": tables.rejection_tex(self.data, "0.3"),
        }
        for label, output in outputs.items():
            with self.subTest(label=label):
                self.assertEqual(output.count(r"\caption{"), 1)
                self.assertEqual(output.count(r"\label{"), 1)
                self.assertIn(r"\label{tab:" + label + "}", output)
                self.assertIn("20M", output)
                self.assertIn("1M", output)
                self.assertIn("sample standard", output)
                self.assertIn("three", output)
                for forbidden in ("placeholder", "pending", "tbd", "sidewaystable", "landscape",
                                  r"\rotatebox", r"\resizebox", r"\scalebox", r"\tiny"):
                    self.assertNotIn(forbidden, output.lower())

    def test_overview_keeps_all_eight_cells_and_correct_deltas(self):
        rows = [line for line in tables.overview_tex(self.data).splitlines()
                if line.startswith(r"\texttt{")]
        self.assertEqual(len(rows), 8)
        for config, row in zip(self.data["configurations"], rows):
            cells = row[:-3].split(" & ")
            self.assertEqual(cells[0], r"\texttt{" + tables.display_id(config) + "}")
            for column, metric, gain in ((1, "accuracy_percent", "accuracy_gain_pp"),
                                         (3, "macro_ovr_auroc_percent", "macro_ovr_auroc_gain_pp")):
                self.assertEqual(cells[column], tables.metric_cell(config[metric]))
                value = config[gain]
                self.assertEqual(cells[column + 1], f"${value:+.3f}$" if value else "$0.000$")

    def test_opening_comparison_uses_standard_shared_not_historical_layerwise(self):
        output = tables.edgevalue_baseline_tex(self.data)
        self.assertIn("\nParT & ", output)
        self.assertIn("\nParT-EV & ", output)
        self.assertNotIn("Std-", output)
        for key in ("S-S", "S-S+EV"):
            for metric in ("accuracy_percent", "macro_ovr_auroc_percent"):
                self.assertIn(tables.metric_cell(self.by_id[key][metric]), output)
        self.assertIn("$+0.655$", output)

    def test_factorial_panels_match_all_eight_standard_metric_cells(self):
        panels = tables.factorial_tex(self.data).split(r"\begin{tabular*}")[1:]
        self.assertEqual(len(panels), 2)
        for metric, panel in zip(("accuracy_percent", "macro_ovr_auroc_percent"), panels):
            rows = [line for line in panel.splitlines() if line.startswith(("Shared &", "Layerwise &"))]
            self.assertEqual(len(rows), 2)
            for row, (bias, negative, positive) in zip(rows, tables.MAIN_FACTORIAL_ROWS):
                a, b = self.by_id[negative][metric], self.by_id[positive][metric]
                self.assertEqual(row[:-3].split(" & "), [bias, tables.metric_cell(a),
                    tables.metric_cell(b), f"${b['mean'] - a['mean']:+.3f}$"])

    def test_main_rejection_keeps_two_matched_comparisons(self):
        output = tables.rejection_main_tex(self.data)
        self.assertIn(r" & & \multicolumn{2}{c}{Shared bias} & \multicolumn{2}{c}{Layerwise bias} \\",
                      output)
        self.assertIn(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}", output)
        self.assertIn(r"Signal & $\epsilon_s$ (\%) & ParT & ParT-EV & ParT & ParT-EV \\", output)
        rows = [line for line in output.splitlines() if line.startswith("$") and " & " in line]
        self.assertEqual(len(rows), 9)
        for row, cls in zip(rows, tables.SIGNALS):
            target = tables.BENCHMARK_TARGETS[cls]
            cells = [tables.CLASS_TEX[cls], f"{100 * float(target):g}"]
            for key in ("S-S", "S-S+EV", "S-L", "S-L+EV"):
                cells.append(tables.rejection_cell(self.by_id[key]["rejection"][target][cls]))
            self.assertEqual(row[:-3].split(" & "), cells)

    def test_main_tables_exclude_richer_relations_without_losing_appendix_evidence(self):
        self.assertEqual(tables.MAIN_CONFIGURATION_IDS, ("S-S", "S-S+EV", "S-L", "S-L+EV"))
        changed = copy.deepcopy(self.data)
        for config in changed["configurations"][4:]:
            config["accuracy_percent"]["mean"] = 0
            config["macro_ovr_auroc_percent"]["mean"] = 0
            for target in tables.TARGETS:
                for cell in config["rejection"][target].values():
                    cell.update(mean=0, sample_sd=0, zero_background_seed_count=0)
        for render in (tables.edgevalue_baseline_tex, tables.factorial_tex, tables.rejection_main_tex):
            output = render(self.data)
            self.assertEqual(output, render(changed))
            for excluded in ("selected", "Sel-", "Std-", "PT, TRACK", "REGION"):
                self.assertNotIn(excluded.lower(), output.lower())
        for render in (tables.overview_tex, lambda d: tables.rejection_tex(d, "benchmark"),
                       lambda d: tables.rejection_tex(d, "0.3")):
            self.assertNotEqual(render(self.data), render(changed))

    def test_complete_rejection_panels_cover_every_model_signal_and_working_point(self):
        covered = []
        for display in tables.DISPLAY_TABLES:
            signals = tables.SIGNALS if display == "benchmark" else tables.HADRONIC_SIGNALS
            targets = tables.BENCHMARK_TARGETS if display == "benchmark" else {s: "0.3" for s in signals}
            output = tables.rejection_tex(self.data, display)
            panels = output.split(r"\begin{tabular*}")[1:]
            self.assertEqual(len(panels), 2)
            self.assertNotIn("gain", output)
            for panel, configs in zip(panels, (self.data["configurations"][:4], self.data["configurations"][4:])):
                rows = [line for line in panel.splitlines() if line.startswith("$") and " & " in line]
                self.assertEqual(len(rows), len(signals))
                for cls, row in zip(signals, rows):
                    cells = row[:-3].split(" & ")
                    self.assertEqual(cells[:2], [tables.CLASS_TEX[cls], f"{100 * float(targets[cls]):g}"])
                    for c, cell in zip(configs, cells[2:]):
                        self.assertEqual(cell, tables.rejection_cell(c["rejection"][targets[cls]][cls]))
                        covered.append((c["id"], cls, targets[cls]))
        expected = {(c[0], s, tables.BENCHMARK_TARGETS[s]) for c in tables.CONFIGS for s in tables.SIGNALS}
        expected |= {(c[0], s, "0.3") for c in tables.CONFIGS for s in tables.HADRONIC_SIGNALS}
        self.assertEqual(set(covered), expected)
        self.assertEqual(len(covered), len(expected))

    def test_presentation_is_global_and_leptonic_measurements_are_retained(self):
        output = tables.rejection_tex(self.data, "0.3")
        for signal in ("Hqql", "Tbl"):
            self.assertNotIn(tables.CLASS_TEX[signal] + " & ", output)
            for config in self.data["configurations"]:
                self.assertIn(signal, config["rejection"]["0.3"])
                self.assertEqual(config["rejection"][tables.BENCHMARK_TARGETS[signal]][signal]
                                 ["zero_background_seed_count"], 0)
        changed = copy.deepcopy(self.data)
        changed["presentation_policy"]["benchmark"]["signal_efficiency_by_class"]["Hqql"] = "0.5"
        with self.assertRaisesRegex(ValueError, "presentation policy drift"):
            tables.validate_derived(changed)

    def test_conditional_zero_note_without_finite_only_averaging(self):
        self.assertNotIn("ZB", tables.rejection_tex(self.data, "benchmark"))
        changed = copy.deepcopy(self.data)
        cell = changed["configurations"][0]["rejection"]["0.5"]["Hbb"]
        cell.update(zero_background_seed_count=1, mean=None, sample_sd=None, gain_percent=None)
        cell["per_seed_qcd_false_positive_count"]["101"] = 0
        cell["per_seed_rejection"]["101"] = None
        tables.validate_derived(changed)
        self.assertIn("ZB 1/3", tables.rejection_main_tex(changed))
        self.assertIn(r"ZB $k/3$", tables.rejection_main_tex(changed))
        cell["mean"] = 5000
        with self.assertRaisesRegex(ValueError, "Zero-background"):
            tables.validate_derived(changed)

    def test_sample_sd_and_seed_coverage_drift_rejected(self):
        changed = copy.deepcopy(self.data)
        changed["configurations"][0]["accuracy_percent"]["sample_sd"] *= math.sqrt(2 / 3)
        with self.assertRaisesRegex(ValueError, "Sample SD"):
            tables.validate_derived(changed)
        changed = copy.deepcopy(self.data)
        del changed["configurations"][0]["accuracy_percent"]["per_seed"]["101"]
        with self.assertRaisesRegex(ValueError, "seed coverage"):
            tables.validate_derived(changed)


@unittest.skipUnless(tables.DEFAULT_SOURCE_ROOT.exists(), "Read-only source archive unavailable")
class SourceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents, cls.provenance = tables.read_sources(tables.DEFAULT_SOURCE_ROOT)
        cls.data = tables.derive(cls.documents, cls.provenance)

    def test_exact_24_result_coverage_and_unchanged_original_measurements(self):
        report = self.documents["report"]
        self.assertEqual(len(report["per_model_results"]), 24)
        records = {(r["task"]["run_id"], r["task"]["seed"]): r for r in report["per_model_results"]}
        for record in self.documents["reference_report"]["per_model_results"]:
            self.assertEqual(records[record["task"]["run_id"], record["task"]["seed"]], record)
        self.assertAlmostEqual(self.data["configurations"][0]["accuracy_percent"]["mean"], 82.81902666666666)
        self.assertAlmostEqual(self.data["configurations"][-1]["macro_ovr_auroc_percent"]["mean"], 98.3992349356175)

    def test_macro_auc_averaged_within_seed_then_across_seeds(self):
        configs = {c["source_run_id"]: c for c in self.data["configurations"]}
        for record in self.documents["report"]["per_model_results"]:
            config = configs[record["task"]["run_id"]]
            seed = str(record["task"]["seed"])
            expected = sum(record["metrics"]["one_vs_rest_auc"].values()) / 10 * 100
            self.assertAlmostEqual(config["macro_ovr_auroc_percent"]["per_seed"][seed], expected, places=12)

    def test_all_ev_accuracy_contrasts_win_three_seeds(self):
        for inputs in ("standard", "selected"):
            for bias in ("shared", "layerwise"):
                contrast = self.data["contrasts"][inputs]["EV_at_" + bias]
                self.assertEqual(contrast["accuracy_seeds_improved"], 3)

    def assert_rejected(self, documents, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            tables.derive(documents, self.provenance)

    def test_top_level_and_nested_hashes_authenticated(self):
        for name in tables.SOURCE_PATHS:
            changed = copy.deepcopy(self.documents)
            changed[name]["corrupt"] = True
            self.assert_rejected(changed, "content hash mismatch")
        changed = copy.deepcopy(self.documents)
        changed["report"]["per_model_results"][-1]["metrics"]["accuracy"] = 0.1
        rehash(changed["report"])
        self.assert_rejected(changed, "content hash mismatch")

    def test_duplicate_and_missing_records_rejected(self):
        for duplicate in (False, True):
            changed = copy.deepcopy(self.documents)
            records = changed["report"]["per_model_results"]
            if duplicate:
                records.append(copy.deepcopy(records[-1]))
            else:
                records.pop()
            rehash(changed["report"])
            self.assert_rejected(changed, "Duplicate run/seed|Result seed coverage")

    def test_nested_plan_binding_and_arithmetic_rejected(self):
        for mutation, pattern in (
            (lambda r: r.update(plan_sha256="0" * 64), "Nested plan hash"),
            (lambda r: r["metrics"].update(event_count=500000), "Event count"),
            (lambda r: r["metrics"].update(accuracy=0.1), "Confusion/accuracy"),
            (lambda r: r["metrics"]["confusion_matrix"][0].__setitem__(0, -1), "Confusion support"),
            (lambda r: r["metrics"]["qcd_signal_rejection"]["Hbb"]["0.5"].update(qcd_support=500000), "Hbb/0.5"),
        ):
            changed = copy.deepcopy(self.documents)
            record = next(r for r in changed["report"]["per_model_results"]
                          if r["task"]["run_id"] == "OFF_RPT_BASE_SHARED_EDGEVALUE")
            mutation(record)
            rehash(record["metrics"])
            rehash(record)
            rehash(changed["report"])
            self.assert_rejected(changed, pattern)

    def test_reference_protocol_and_identity_binding(self):
        changed = copy.deepcopy(self.documents)
        changed["plan"]["files"][0]["sha256"] = "0" * 64
        rehash(changed["plan"])
        self.assert_rejected(changed, "protocol mismatch: files")
        changed = copy.deepcopy(self.documents)
        changed["plan"]["reference_plan_sha256"] = "0" * 64
        rehash(changed["plan"])
        self.assert_rejected(changed, "Reference plan mismatch")

    def test_check_mode_reproduces_all_generated_files_without_writing(self):
        proc = subprocess.run([sys.executable, "-B", str(tables.PAPER_ROOT / "tools/generate_tables.py"),
                               "--check"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("Checked 7 artifacts", proc.stdout)


if __name__ == "__main__":
    unittest.main()
