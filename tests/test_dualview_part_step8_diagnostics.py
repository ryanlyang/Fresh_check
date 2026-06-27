import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from teacher_logit_reco.dualview_part import (
    DUALVIEW_PART_DIAGNOSTICS_CONTRACT,
    DUALVIEW_PART_STEP8,
    build_residual_case_rows,
    summarize_residual_behavior,
    write_residual_diagnostics,
)


class DualViewPartStep8DiagnosticsTests(unittest.TestCase):
    def make_analysis(self):
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        hlt_logits = np.asarray(
            [
                [3.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [0.0, 3.0],
            ],
            dtype=np.float32,
        )
        logits = np.asarray(
            [
                [3.0, 0.0],
                [0.0, 3.0],
                [0.0, 3.0],
                [0.0, 3.0],
            ],
            dtype=np.float32,
        )
        gate = np.asarray([[0.1], [0.8], [0.9], [0.2]], dtype=np.float32)
        delta = logits - hlt_logits
        residual = gate * delta
        return summarize_residual_behavior(
            logits=logits,
            hlt_logits=hlt_logits,
            labels=labels,
            gate=gate,
            delta_logits=delta,
            residual_logits=residual,
            label_names=("QCD", "Hgg"),
        )

    def test_summary_counts_fixes_breaks_and_buckets(self):
        analysis = self.make_analysis()

        self.assertEqual(analysis["experiment_step"], DUALVIEW_PART_STEP8)
        self.assertEqual(analysis["output_contract"], DUALVIEW_PART_DIAGNOSTICS_CONTRACT)
        self.assertEqual(analysis["prediction_changes"]["changed_count"], 2)
        self.assertEqual(analysis["prediction_changes"]["fix_count"], 1)
        self.assertEqual(analysis["prediction_changes"]["break_count"], 1)
        self.assertAlmostEqual(analysis["prediction_changes"]["hlt_accuracy"], 0.75)
        self.assertAlmostEqual(analysis["prediction_changes"]["final_accuracy"], 0.75)
        self.assertEqual(len(analysis["gate_by_class"]), 2)
        self.assertEqual(len(analysis["gate_by_hlt_confidence_bucket"]), 5)
        self.assertEqual(len(analysis["gate_by_hlt_correctness"]), 2)

    def test_writer_emits_json_and_csv_summaries(self):
        analysis = self.make_analysis()
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        hlt_logits = np.asarray([[3.0, 0.0], [2.0, 0.0], [3.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        logits = np.asarray([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0], [0.0, 3.0]], dtype=np.float32)
        gate = np.asarray([[0.1], [0.8], [0.9], [0.2]], dtype=np.float32)
        case_rows = build_residual_case_rows(
            split="stack_val",
            logits=logits,
            hlt_logits=hlt_logits,
            labels=labels,
            gate=gate,
            delta_logits=logits - hlt_logits,
            residual_logits=gate * (logits - hlt_logits),
            sample_indices=np.asarray([10, 11, 12, 13], dtype=np.int64),
            jet_ids=[
                {"file": "a.root", "entry": 10, "label": 0},
                {"file": "b.root", "entry": 11, "label": 1},
                {"file": "c.root", "entry": 12, "label": 0},
                {"file": "d.root", "entry": 13, "label": 1},
            ],
            label_names=("QCD", "Hgg"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            files = write_residual_diagnostics(tmpdir, {"stack_val": analysis}, {"stack_val": case_rows})
            paths = {name: Path(path) for name, path in files.items()}
            for path in paths.values():
                self.assertTrue(path.exists(), path)

            payload = json.loads(paths["residual_diagnostics_json"].read_text())
            self.assertEqual(payload["output_contract"], DUALVIEW_PART_DIAGNOSTICS_CONTRACT)
            self.assertIn("stack_val", payload["splits"])

            with paths["prediction_change_summary_csv"].open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["split"], "stack_val")
            self.assertEqual(rows[0]["fix_count"], "1")
            self.assertEqual(rows[0]["break_count"], "1")

            with paths["fix_break_cases_csv"].open(newline="", encoding="utf-8") as handle:
                case_csv_rows = list(csv.DictReader(handle))
            self.assertEqual({row["case_type"] for row in case_csv_rows}, {"fix", "break"})
            self.assertIn("jet_file", case_csv_rows[0])
            self.assertIn("hlt_logits_json", case_csv_rows[0])
            self.assertIn("residual_logits_json", case_csv_rows[0])


if __name__ == "__main__":
    unittest.main()
