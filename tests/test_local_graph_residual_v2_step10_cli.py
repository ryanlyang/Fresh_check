import importlib.util
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LocalGraphResidualV2Step10CliTest(unittest.TestCase):
    def test_cache_cli_builds_embedding_cache_config_and_rejects_final_metric_split(self):
        module = _load_script("cache_local_graph_residual_v2_embeddings.py")
        args = module.parse_args(
            [
                "--output-dir",
                "out",
                "--hlt-cache-dir",
                "hlt",
                "--checkpoint",
                "best_model_val.pt",
                "--splits",
                "model_train model_val",
                "--metric-splits",
                "model_train",
                "--max-model-train-jets",
                "10",
            ]
        )
        config = module.build_config(args)

        self.assertEqual(config.splits, ("model_train", "model_val"))
        self.assertEqual(config.metric_splits, ("model_train",))
        self.assertEqual(config.max_jets_by_split["model_train"], 10)
        self.assertEqual(config.checkpoint_path, "best_model_val.pt")

        bad = module.parse_args(
            [
                "--output-dir",
                "out",
                "--hlt-cache-dir",
                "hlt",
                "--checkpoint",
                "best_model_val.pt",
                "--metric-splits",
                "model_train final_test",
            ]
        )
        with self.assertRaisesRegex(ValueError, "final_test metrics"):
            module.build_config(bad)

    def test_train_cli_builds_strict_v2_train_config(self):
        module = _load_script("train_local_graph_residual_expert_v2.py")
        args = module.parse_args(
            [
                "--output-dir",
                "out",
                "--hlt-cache-dir",
                "hlt",
                "--baseline-embedding-cache-dir",
                "emb",
                "--confirm-split-settings",
                "--loss-mode",
                "D",
                "--epochs",
                "2",
                "--baseline-embedding-dim",
                "128",
                "--disable-gamma-learnable",
                "--residual-input-mode",
                "embedding_only",
                "--condition-control-mode",
                "shuffled",
                "--label-control-mode",
                "shuffled",
            ]
        )
        config = module.build_config(args)

        self.assertEqual(config.train_split, "model_train")
        self.assertEqual(config.val_split, "model_val")
        self.assertEqual(config.epochs, 2)
        self.assertFalse(config.gamma_learnable)
        self.assertEqual(config.loss_mode, "residual_v2_boundary_pairwise_soft_fpr_bce_anchor")
        self.assertEqual(config.residual_input_mode, "embedding_only")
        self.assertEqual(config.condition_control_mode, "shuffled")
        self.assertEqual(config.label_control_mode, "shuffled")

    def test_report_cli_builds_full_step12_config_and_guards_final_test(self):
        module = _load_script("write_local_graph_residual_expert_v2_report.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experts"
            args = module.parse_args(
                [
                    "--output-dir",
                    str(Path(tmp) / "report"),
                    "--hlt-cache-dir",
                    str(Path(tmp) / "hlt_cache"),
                    "--baseline-embedding-cache-dir",
                    str(Path(tmp) / "baseline_embeddings"),
                    "--residual-expert-root",
                    str(root),
                    "--residual-variants",
                    "mode_d",
                    "--comparison-split",
                    "final_test",
                    "--confirm-final-test",
                    "--max-final-test-jets",
                    "32",
                    "--disable-calibration-control",
                ]
            )
            config = module.build_config(args)

            self.assertEqual(config.comparison_split, "final_test")
            self.assertTrue(config.confirm_final_test)
            self.assertEqual(config.max_final_test_jets, 32)
            self.assertFalse(config.include_calibration_control)
            self.assertEqual(config.residual_variants, ("mode_d",))
            self.assertEqual(config.primary_metric, "fpr_at_signal_eff_0p50")

            final_args = module.parse_args(
                [
                    "--output-dir",
                    str(Path(tmp) / "report_final"),
                    "--hlt-cache-dir",
                    str(Path(tmp) / "hlt_cache"),
                    "--baseline-embedding-cache-dir",
                    str(Path(tmp) / "baseline_embeddings"),
                    "--residual-expert-root",
                    str(root),
                    "--comparison-split",
                    "final_test",
                ]
            )
            with self.assertRaisesRegex(ValueError, "confirm_final_test"):
                module.build_config(final_args)


if __name__ == "__main__":
    unittest.main()
