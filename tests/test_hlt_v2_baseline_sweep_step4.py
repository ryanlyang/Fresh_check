import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_report_module():
    spec = importlib.util.spec_from_file_location(
        "write_hlt_v2_baseline_sweep_report_test_module",
        REPO_ROOT / "scripts" / "write_hlt_v2_baseline_sweep_report.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class HLTv2BaselineSweepStep4Tests(unittest.TestCase):
    def test_report_summarizes_model_val_gaps(self):
        module = load_report_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "offline_reference" / "teachers" / "offline_part_teacher_10class" / "model_val_report.json",
                {
                    "best_model_val_accuracy": 0.80,
                    "best_model_val_loss": 0.55,
                    "final_epoch": {"model_val": {"n_jets": 150000}},
                },
            )
            for strength, accuracy in [(0.0, 0.799), (1.0, 0.775)]:
                tag = str(strength).replace(".", "p")
                write_json(
                    root / f"hlt_v2_strength_{tag}" / "teachers" / "hlt_part_teacher_10class" / "model_val_report.json",
                    {
                        "best_model_val_accuracy": accuracy,
                        "best_model_val_loss": 0.60,
                        "final_epoch": {"model_val": {"n_jets": 150000}},
                    },
                )
                write_json(
                    root / f"hlt_v2_strength_{tag}" / "hlt_cache" / "model_val_fixed_hlt_metadata.json",
                    {
                        "hlt_profile": "fixed_hlt_v2_realistic",
                        "hlt_profile_version": "v1",
                        "hlt_degradation_strength": strength,
                        "hlt_content_hash": f"hash-{tag}",
                        "offline_constit_count_summary": {"mean": 70.0, "p50": 70.0},
                        "hlt_constit_count_summary": {"mean": 68.0, "p50": 68.0},
                        "hlt_diagnostics_summary": {
                            "mean_offline_constits": 70.0,
                            "mean_hlt_constits": 68.0,
                            "drop_eff_fraction": 0.003,
                            "drop_threshold_fraction": 0.004,
                            "drop_merge_fraction": 0.002,
                            "drop_total_fraction": 0.01,
                            "mean_merges_per_jet": 0.30,
                        },
                    },
                )

            report = module.build_sweep_report(root, strengths=(0.0, 1.0))
            self.assertTrue(report["ok"])
            self.assertEqual(len(report["rows"]), 3)
            strength_one = next(row for row in report["rows"] if row["strength"] == 1.0)
            self.assertAlmostEqual(strength_one["accuracy_gap_vs_offline"], 0.025)
            self.assertEqual(strength_one["hlt_profile"], "fixed_hlt_v2_realistic")
            self.assertAlmostEqual(strength_one["drop_fraction_mean"], 0.01)
            self.assertAlmostEqual(strength_one["merge_fraction_mean"], 0.002)
            self.assertAlmostEqual(strength_one["mean_merges_per_jet"], 0.30)
            self.assertEqual(module.strength_tag(0.0), "0p0")
            self.assertEqual(module.strength_tag(1.0), "1p0")
            self.assertEqual(module.strength_tag("1.25"), "1p25")

            output_dir = root / "report"
            outputs = module.write_outputs(report, output_dir)
            self.assertTrue(Path(outputs["json"]).exists())
            self.assertTrue(Path(outputs["csv"]).exists())
            self.assertTrue(Path(outputs["markdown"]).exists())
            self.assertIn("accuracy_gap_vs_offline", Path(outputs["csv"]).read_text(encoding="utf-8"))

    def test_sbatch_contracts_are_model_val_only_and_profile_aware(self):
        train_wrapper = (REPO_ROOT / "sbatch" / "run_pd10_train_teacher.sh").read_text(encoding="utf-8")
        submitter = (REPO_ROOT / "sbatch" / "submit_hlt_v2_baseline_sweep.sh").read_text(encoding="utf-8")
        reporter = (REPO_ROOT / "sbatch" / "run_hlt_v2_baseline_sweep_report.sh").read_text(encoding="utf-8")

        self.assertIn("PD10_TEACHER_SKIP_FINAL_TEST", train_wrapper)
        self.assertIn("--skip-final-test", train_wrapper)
        self.assertIn("fixed_hlt_v2_realistic", submitter)
        self.assertIn("HLT_V2_BASELINE_HLT_SPLITS:=model_train model_val", submitter)
        self.assertIn("PD10_TEACHER_SKIP_FINAL_TEST=1", submitter)
        self.assertIn("write_hlt_v2_baseline_sweep_report.py", reporter)


if __name__ == "__main__":
    unittest.main()
