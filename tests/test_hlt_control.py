import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from jetclass_fresh.hlt_control import (
    HLT5FusionRunConfig,
    compare_hlt5_to_reco7,
    compare_hlt5_to_reco7_reports,
    default_hlt5_specs,
    hlt_seed_checkpoint_path,
    hlt_seed_model_name,
    run_hlt5_fusion,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fusion_report(model_names, *, final_accuracy, final_cross_entropy):
    return {
        "model_names": list(model_names),
        "uniform_probability_average": {
            "final_test": {
                "accuracy": final_accuracy - 0.01,
                "cross_entropy": final_cross_entropy + 0.02,
                "n_jets": 30,
            }
        },
        "weighted_probability_average": {
            "metrics": {
                "final_test": {
                    "accuracy": final_accuracy - 0.005,
                    "cross_entropy": final_cross_entropy + 0.01,
                    "n_jets": 30,
                }
            }
        },
        "weighted_logit_average": {
            "metrics": {
                "final_test": {
                    "accuracy": final_accuracy,
                    "cross_entropy": final_cross_entropy,
                    "n_jets": 30,
                }
            }
        },
        "stacked_logistic_regression": {
            "metrics": {
                "stack_val": {
                    "accuracy": final_accuracy - 0.02,
                    "cross_entropy": final_cross_entropy + 0.04,
                    "n_jets": 20,
                },
                "final_test": {
                    "accuracy": final_accuracy,
                    "cross_entropy": final_cross_entropy,
                    "n_jets": 30,
                },
            }
        },
    }


class HLTControlStep11Tests(unittest.TestCase):
    def test_default_hlt5_specs_use_canonical_names_and_paths(self):
        specs = default_hlt5_specs(checkpoint_root="hlt_root", seeds=[101, 202])

        self.assertEqual([spec.name for spec in specs], ["hlt_seed101", "hlt_seed202"])
        self.assertEqual([spec.kind for spec in specs], ["hlt", "hlt"])
        self.assertEqual(specs[0].checkpoint, str(Path("hlt_root") / "seed101" / "best_model_val.pt"))
        self.assertEqual(hlt_seed_model_name(505), "hlt_seed505")
        self.assertEqual(hlt_seed_checkpoint_path("root", 303), Path("root") / "seed303" / "best_model_val.pt")

    def test_final_test_requires_explicit_confirmation(self):
        cfg = HLT5FusionRunConfig(
            output_dir="/tmp/unused_hlt5",
            hlt_cache_dir="/tmp/hlt_cache",
            splits=["stack_train", "stack_val", "final_test"],
            confirm_final_test=False,
        )
        with self.assertRaises(ValueError):
            run_hlt5_fusion(cfg)

    def test_compare_hlt5_to_reco7_reports_computes_deltas(self):
        hlt5 = fusion_report(["hlt_seed101", "hlt_seed202"], final_accuracy=0.70, final_cross_entropy=0.90)
        reco7 = fusion_report(["hlt_baseline", "m2_base"], final_accuracy=0.76, final_cross_entropy=0.82)

        comparison = compare_hlt5_to_reco7_reports(hlt5, reco7)
        stack_final = comparison["methods"]["stacked_logistic_regression"]["final_test"]

        self.assertAlmostEqual(stack_final["accuracy_delta_reco7_minus_hlt5"], 0.06)
        self.assertAlmostEqual(stack_final["cross_entropy_delta_reco7_minus_hlt5"], -0.08)
        self.assertEqual(
            comparison["summary"]["stacked_logistic_final_test_accuracy_delta_reco7_minus_hlt5"],
            stack_final["accuracy_delta_reco7_minus_hlt5"],
        )

    def test_compare_hlt5_to_reco7_reads_and_writes_json(self):
        hlt5 = fusion_report(["hlt_seed101"], final_accuracy=0.71, final_cross_entropy=0.89)
        reco7 = fusion_report(["hlt_baseline", "m2_base"], final_accuracy=0.75, final_cross_entropy=0.83)
        with tempfile.TemporaryDirectory() as tmp:
            hlt5_path = Path(tmp) / "hlt5.json"
            reco7_path = Path(tmp) / "reco7.json"
            output_path = Path(tmp) / "comparison.json"
            hlt5_path.write_text(json.dumps(hlt5), encoding="utf-8")
            reco7_path.write_text(json.dumps(reco7), encoding="utf-8")

            comparison = compare_hlt5_to_reco7(hlt5_path, reco7_path, output_path=output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["hlt5_report_path"], str(hlt5_path))
            self.assertEqual(comparison["reco7_report_path"], str(reco7_path))

    def test_train_hlt5_control_builds_seed_train_and_fusion_commands(self):
        script = load_script_module("train_hlt5_control_test_module", "scripts/train_hlt5_control.py")
        args = script.parse_args(
            [
                "--stage",
                "both",
                "--dry-run",
                "--hlt-cache-dir",
                "cache",
                "--baseline-root",
                "baselines",
                "--fusion-output-dir",
                "fusion",
                "--seeds",
                "101",
                "202",
                "--confirm-final-test",
                "--model-size",
                "tiny",
                "--max-train-jets",
                "12",
                "--max-jets-per-split",
                "8",
            ]
        )

        commands = script.build_commands(args)

        self.assertEqual(len(commands), 3)
        self.assertIn("train_hlt_baseline.py", commands[0][1])
        self.assertIn(str(Path("baselines") / "seed101"), commands[0])
        self.assertIn("101", commands[0])
        self.assertIn(str(Path("baselines") / "seed202"), commands[1])
        self.assertIn("run_hlt5_fusion.py", commands[2][1])
        self.assertIn("--confirm-final-test", commands[2])
        self.assertIn("--max-jets-per-split", commands[2])
        self.assertIn("tiny", commands[0])


if __name__ == "__main__":
    unittest.main()
