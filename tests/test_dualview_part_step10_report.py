import json
from pathlib import Path
import tempfile
import unittest

from teacher_logit_reco.dualview_part import (
    DUALVIEW_PART_REPORT_CONTRACT,
    DualViewPartReportConfig,
    build_dualview_part_report,
)


def _write_report(path: Path, *, shuffle: bool, fpr50: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "shuffle_pn_view": bool(shuffle),
        "initialization_check_passed": True,
        "best_epoch": 2,
        "selection_metric": "fpr_at_signal_eff_0p50",
        "final_test_evaluated": True,
        "best_stack_val_metrics": {
            "accuracy": 0.8,
            "binary_metrics": {
                "auc": 0.9,
                "fpr_at_signal_eff_0p50": fpr50 + 0.01,
            },
        },
        "final_test_metrics": {
            "accuracy": 0.82,
            "binary_metrics": {
                "auc": 0.91,
                "fpr_at_signal_eff_0p30": fpr50 / 2.0,
                "fpr_at_signal_eff_0p50": fpr50,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class DualViewPartStep10ReportTests(unittest.TestCase):
    def test_report_requires_real_to_beat_shuffled_on_fpr50(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            taggers = root / "taggers"
            _write_report(taggers / "frozen_anchor_pn_residual" / "run_report.json", shuffle=False, fpr50=0.02)
            _write_report(
                taggers / "frozen_anchor_shuffled_pn_control" / "run_report.json",
                shuffle=True,
                fpr50=0.05,
            )

            report = build_dualview_part_report(
                DualViewPartReportConfig(
                    output_dir=str(root / "final_report"),
                    experiment_dir=str(root),
                    tagger_root=str(taggers),
                    confirm_final_test=True,
                    require_real_beats_shuffled=True,
                )
            )

            self.assertTrue(report["ok"])
            self.assertEqual(report["output_contract"], DUALVIEW_PART_REPORT_CONTRACT)
            self.assertTrue(report["real_vs_shuffled"]["real_beats_shuffled"])
            self.assertTrue((root / "final_report" / "dualview_part_report.json").exists())
            self.assertTrue((root / "final_report" / "metric_table.csv").exists())

    def test_report_flags_shuffled_beating_real(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            taggers = root / "taggers"
            _write_report(taggers / "frozen_anchor_pn_residual" / "run_report.json", shuffle=False, fpr50=0.06)
            _write_report(
                taggers / "frozen_anchor_shuffled_pn_control" / "run_report.json",
                shuffle=True,
                fpr50=0.05,
            )

            report = build_dualview_part_report(
                DualViewPartReportConfig(
                    output_dir=str(root / "final_report"),
                    experiment_dir=str(root),
                    tagger_root=str(taggers),
                    require_real_beats_shuffled=True,
                )
            )

            self.assertFalse(report["ok"])
            self.assertFalse(report["real_vs_shuffled"]["real_beats_shuffled"])
            self.assertTrue(any("real PN did not beat shuffled PN" in problem for problem in report["problems"]))


if __name__ == "__main__":
    unittest.main()
