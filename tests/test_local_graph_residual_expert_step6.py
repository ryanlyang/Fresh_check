import importlib.util
from pathlib import Path
import unittest

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_PART_PRIMARY_METRIC,
    LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE,
    LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
    LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
    LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE,
)


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_local_graph_residual_expert.py"
    spec = importlib.util.spec_from_file_location("train_local_graph_residual_expert", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LocalGraphResidualExpertStep6CliTests(unittest.TestCase):
    def minimal_args(self, *extra):
        return [
            "--output-dir",
            "out",
            "--hlt-cache-dir",
            "hlt_cache",
            "--baseline-logit-cache-dir",
            "baseline_logits",
            "--confirm-split-settings",
            *extra,
        ]

    def test_cli_builds_protocol_config(self):
        script = _load_script_module()
        args = script.parse_args(
            self.minimal_args(
                "--loss-mode",
                "D",
                "--local-adapter",
                "point_attention",
                "--disable-alpha-learnable",
                "--disable-alpha-max",
                "--freeze-part-epochs",
                "2",
                "--confirm-final-test",
            )
        )
        config = script.build_config(args)

        self.assertEqual(config.selection_metric, LOCAL_GRAPH_PART_PRIMARY_METRIC)
        self.assertEqual(config.loss_mode, LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR)
        self.assertEqual(config.train_split, "model_train")
        self.assertEqual(config.val_split, "model_val")
        self.assertFalse(config.alpha_learnable)
        self.assertIsNone(config.alpha_max)
        self.assertEqual(config.freeze_part_epochs, 2)

    def test_cli_supports_loss_ladder_aliases(self):
        script = _load_script_module()
        expected = {
            "A": LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE,
            "B": LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE,
            "C": LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
            "D": LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
        }

        for alias, canonical in expected.items():
            with self.subTest(alias=alias):
                args = script.parse_args(self.minimal_args("--loss-mode", alias))
                config = script.build_config(args)
                self.assertEqual(config.loss_mode, canonical)

        args = script.parse_args(self.minimal_args("--loss-mode", "E"))
        with self.assertRaisesRegex(ValueError, "no longer a separate training loss"):
            script.build_config(args)

    def test_cli_keeps_split_confirmation_required(self):
        script = _load_script_module()
        args = script.parse_args(
            [
                "--output-dir",
                "out",
                "--hlt-cache-dir",
                "hlt_cache",
                "--baseline-logit-cache-dir",
                "baseline_logits",
            ]
        )
        with self.assertRaisesRegex(ValueError, "confirm-split-settings"):
            script.build_config(args)


if __name__ == "__main__":
    unittest.main()
