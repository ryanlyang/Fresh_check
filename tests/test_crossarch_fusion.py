import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block, softmax_np
from jetclass_fresh.jetclass_data import JetIdentity
from teacher_logit_reco.crossarch_fusion import (
    CrossArchFusionFeatureBuildConfig,
    CrossArchFusionFitConfig,
    DEFAULT_CROSSARCH_FUSERS,
    assign_bins_from_specs,
    bin_assignment_name_for_fuser,
    build_group_feature_report,
    build_split_feature_set,
    build_uncertainty_feature_matrix,
    entropy_from_probs,
    fit_group_fusers,
    fit_bin_specs,
    pairwise_disagreement_fraction,
    quantile_edges,
    run_crossarch_feature_builder,
    run_crossarch_fusers,
    top1_margin_from_probs,
)


def _block(model_name: str, split: str, logits: np.ndarray, labels: np.ndarray) -> PredictionBlock:
    jet_ids = [
        JetIdentity(file=f"{split}.root", entry=index, label=int(label))
        for index, label in enumerate(labels)
    ]
    return PredictionBlock(
        model_name=model_name,
        split=split,
        logits=logits.astype(np.float32),
        probs=softmax_np(logits),
        labels=labels.astype(np.int64),
        jet_ids=jet_ids,
        metadata={"model_kind": "test", "allowed_inputs": "frozen_prediction_block_test"},
    )


def _block_with_keys(
    model_name: str,
    split: str,
    logits: np.ndarray,
    labels: np.ndarray,
    keys: list[tuple[str, int]],
) -> PredictionBlock:
    jet_ids = [
        JetIdentity(file=str(file_name), entry=int(entry), label=int(label))
        for (file_name, entry), label in zip(keys, labels)
    ]
    return PredictionBlock(
        model_name=model_name,
        split=split,
        logits=logits.astype(np.float32),
        probs=softmax_np(logits),
        labels=labels.astype(np.int64),
        jet_ids=jet_ids,
        metadata={"model_kind": "test", "allowed_inputs": "frozen_prediction_block_test"},
    )


def _write_prediction_fixture(prediction_dir: Path, model_names: list[str]) -> None:
    labels_by_split = {
        "stack_train": np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
        "stack_val": np.asarray([0, 1, 2, 1], dtype=np.int64),
        "final_test": np.asarray([2, 1, 0, 2], dtype=np.int64),
    }
    base_logits = {
        "stack_train": np.asarray(
            [
                [3.0, 1.0, 0.0],
                [0.2, 2.0, 0.1],
                [0.1, 0.4, 2.4],
                [2.5, 0.6, 0.1],
                [0.1, 2.2, 0.3],
                [0.2, 0.5, 2.0],
            ],
            dtype=np.float32,
        ),
        "stack_val": np.asarray(
            [
                [2.0, 0.5, 0.1],
                [0.2, 1.8, 0.4],
                [0.3, 0.7, 2.1],
                [0.4, 1.9, 0.2],
            ],
            dtype=np.float32,
        ),
        "final_test": np.asarray(
            [
                [0.2, 0.5, 2.4],
                [0.1, 2.0, 0.3],
                [2.3, 0.4, 0.2],
                [0.4, 0.5, 1.8],
            ],
            dtype=np.float32,
        ),
    }
    for model_index, model_name in enumerate(model_names):
        for split, labels in labels_by_split.items():
            logits = base_logits[split] + 0.15 * float(model_index)
            logits[:, model_index % 3] += 0.2
            save_prediction_block(_block(model_name, split, logits, labels), prediction_dir)


def _write_prediction_fixture_with_split_overlap(prediction_dir: Path, model_names: list[str]) -> None:
    labels_by_split = {
        "stack_train": np.asarray([0, 1, 2], dtype=np.int64),
        "stack_val": np.asarray([0, 1, 2], dtype=np.int64),
        "final_test": np.asarray([2, 1, 0], dtype=np.int64),
    }
    keys_by_split = {
        "stack_train": [("shared.root", 0), ("train.root", 1), ("train.root", 2)],
        "stack_val": [("shared.root", 0), ("val.root", 1), ("val.root", 2)],
        "final_test": [("test.root", 0), ("test.root", 1), ("test.root", 2)],
    }
    logits_by_split = {
        split: np.eye(3, dtype=np.float32) * 2.0 + 0.1
        for split in labels_by_split
    }
    for model_index, model_name in enumerate(model_names):
        for split, labels in labels_by_split.items():
            logits = logits_by_split[split].copy()
            logits += 0.05 * float(model_index)
            save_prediction_block(
                _block_with_keys(model_name, split, logits, labels, keys_by_split[split]),
                prediction_dir,
            )


class CrossArchFusionFeatureTests(unittest.TestCase):
    def test_uncertainty_and_bin_helpers(self):
        probs = np.asarray([[0.7, 0.2, 0.1], [0.34, 0.33, 0.33]], dtype=np.float32)
        entropy = entropy_from_probs(probs)
        margin = top1_margin_from_probs(probs)
        self.assertEqual(entropy.shape, (2,))
        self.assertLess(entropy[0], entropy[1])
        self.assertAlmostEqual(float(margin[0]), 0.5, places=6)
        preds = np.asarray([[0, 0, 1], [2, 2, 2]], dtype=np.int64)
        disagreement = pairwise_disagreement_fraction(preds)
        self.assertGreater(float(disagreement[0]), 0.0)
        self.assertEqual(float(disagreement[1]), 0.0)
        edges = quantile_edges(np.asarray([0.0, 1.0, 2.0, 3.0]), n_bins=3)
        self.assertEqual(edges.ndim, 1)

    def test_builds_aligned_raw_uncertainty_and_combined_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            prediction_dir = Path(tmp) / "predictions"
            model_names = ["gt_reco_to_part_teacher", "pn_reco_to_pfn_teacher"]
            _write_prediction_fixture(prediction_dir, model_names)
            feature_set = build_split_feature_set(
                prediction_dir,
                group_name="tiny_group",
                model_names=model_names,
                split="stack_train",
            )
            self.assertEqual(feature_set.labels.shape, (6,))
            self.assertEqual(feature_set.feature_matrices["logits"].values.shape, (6, 6))
            self.assertEqual(feature_set.feature_matrices["probs"].values.shape, (6, 6))
            self.assertEqual(feature_set.feature_matrices["logits_probs"].values.shape, (6, 12))
            uncertainty = feature_set.feature_matrices["uncertainty"]
            self.assertEqual(uncertainty.values.shape, (6, 28))
            self.assertIn("pairwise_disagreement_fraction", uncertainty.names)
            self.assertEqual(
                feature_set.feature_matrices["logits_probs_uncertainty"].values.shape,
                (6, 40),
            )
            self.assertEqual(feature_set.anchor_model_name, model_names[0])

    def test_group_report_fits_train_bins_and_can_write_matrices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_dir = root / "predictions"
            output_dir = root / "features"
            model_names = ["gt_reco_to_part_teacher", "pn_reco_to_pfn_teacher"]
            _write_prediction_fixture(prediction_dir, model_names)
            report = build_group_feature_report(
                prediction_dir,
                group_name="tiny_group",
                model_names=model_names,
                output_dir=output_dir,
                write_matrices=True,
            )
            self.assertEqual(report["n_models"], 2)
            self.assertIn("anchor_entropy_quantile3", report["bin_specs"])
            self.assertTrue((output_dir / "features" / "tiny_group" / "stack_train_feature_matrices.npz").exists())
            self.assertEqual(report["splits"]["final_test"]["features"]["logits_probs"]["shape"], [4, 12])

    def test_run_feature_builder_guardrail_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_dir = root / "predictions"
            output_dir = root / "feature_builder"
            model_names = ["gt_reco_to_part_teacher", "pn_reco_to_pfn_teacher"]
            _write_prediction_fixture(prediction_dir, model_names)
            with self.assertRaises(ValueError):
                CrossArchFusionFeatureBuildConfig(
                    prediction_dir=str(prediction_dir),
                    output_dir=str(output_dir),
                    groups={"tiny_group": model_names},
                    splits=["stack_train", "stack_val", "final_test"],
                    confirm_final_test=False,
                )
            config = CrossArchFusionFeatureBuildConfig(
                prediction_dir=str(prediction_dir),
                output_dir=str(output_dir),
                groups={"tiny_group": model_names},
                splits=["stack_train", "stack_val", "final_test"],
                confirm_final_test=True,
            )
            report = run_crossarch_feature_builder(config)
            self.assertEqual(report["experiment_step"], "crossarch_step7_fusion_feature_builder")
            self.assertIn("tiny_group", report["groups"])
            self.assertTrue((output_dir / "feature_build_report.json").exists())
            self.assertFalse(report["write_feature_matrices"])

    def test_bin_specs_assign_all_splits(self):
        scores = {
            "anchor_entropy": np.asarray([0.1, 0.2, 0.3, 0.9], dtype=np.float32),
            "anchor_margin": np.asarray([0.9, 0.7, 0.2, 0.1], dtype=np.float32),
            "disagreement_fraction": np.asarray([0.0, 0.0, 0.5, 1.0], dtype=np.float32),
            "anchor_predicted_class": np.asarray([0, 1, 1, 2], dtype=np.int64),
        }
        specs = fit_bin_specs(scores, n_bins=3)
        assignments = assign_bins_from_specs(scores, specs)
        self.assertEqual(assignments["anchor_entropy_quantile3"].shape, (4,))
        self.assertEqual(assignments["anchor_predicted_class"].tolist(), [0, 1, 1, 2])

    def test_step8_mean_and_logistic_fusers_run_on_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            prediction_dir = Path(tmp) / "predictions"
            model_names = ["gt_reco_to_part_teacher", "pn_reco_to_pfn_teacher"]
            _write_prediction_fixture(prediction_dir, model_names)
            report = fit_group_fusers(
                prediction_dir,
                group_name="tiny_group",
                model_names=model_names,
                fusers=[
                    "mean_logits",
                    "mean_probs",
                    "logistic_logits_probs",
                    "uncertainty_logistic_logits_probs",
                ],
                c_grid=[0.1, 1.0],
                max_iter=25,
                run_controls=False,
            )
            self.assertEqual(report["n_models"], 2)
            self.assertIn("mean_logits", report["fusers"])
            self.assertIn("logistic_logits_probs", report["fusers"])
            for fuser_name, fuser_report in report["fusers"].items():
                self.assertIn("final_test", fuser_report["metrics"], msg=fuser_name)
                self.assertIn("accuracy", fuser_report["metrics"]["stack_val"], msg=fuser_name)

    def test_step8_bin_gated_fusers_and_multiplicity_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            prediction_dir = Path(tmp) / "predictions"
            model_names = ["gt_reco_to_part_teacher", "pn_reco_to_pfn_teacher"]
            _write_prediction_fixture(prediction_dir, model_names)
            self.assertEqual(
                bin_assignment_name_for_fuser("entropy_bin_gated_logistic"),
                "anchor_entropy_quantile3",
            )
            report = fit_group_fusers(
                prediction_dir,
                group_name="tiny_group",
                model_names=model_names,
                fusers=[
                    "entropy_bin_gated_logistic",
                    "margin_bin_gated_logistic",
                    "disagreement_bin_gated_logistic",
                    "predicted_class_bin_gated_logistic",
                    "multiplicity_bin_gated_logistic",
                ],
                c_grid=[0.1],
                max_iter=20,
                min_bin_train_rows=1,
                run_controls=False,
            )
            fusers = report["fusers"]
            self.assertEqual(fusers["entropy_bin_gated_logistic"]["status"], "ok")
            self.assertIn("final_test", fusers["entropy_bin_gated_logistic"]["metrics"])
            self.assertEqual(fusers["multiplicity_bin_gated_logistic"]["status"], "skipped")
            self.assertIn("multiplicity", fusers["multiplicity_bin_gated_logistic"]["reason"])

    def test_step8_run_crossarch_fusers_writes_report_and_stackers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_dir = root / "predictions"
            output_dir = root / "fusion"
            model_names = ["gt_reco_to_part_teacher", "pn_reco_to_pfn_teacher"]
            _write_prediction_fixture(prediction_dir, model_names)
            config = CrossArchFusionFitConfig(
                prediction_dir=str(prediction_dir),
                output_dir=str(output_dir),
                groups={"tiny_group": model_names},
                confirm_final_test=True,
                fusers=["mean_logits", "logistic_logits_probs", "multiplicity_bin_gated_logistic"],
                c_grid=[0.1],
                max_iter=20,
            )
            report = run_crossarch_fusers(config)
            self.assertEqual(report["experiment_step"], "crossarch_step8_fusers")
            self.assertTrue((output_dir / "fusion_config.json").exists())
            self.assertTrue((output_dir / "fusion_report.json").exists())
            self.assertTrue(
                (output_dir / "stackers" / "group__tiny_group__logistic_logits_probs.npz").exists()
            )
            self.assertEqual(
                report["groups"]["tiny_group"]["fusers"]["multiplicity_bin_gated_logistic"]["status"],
                "skipped",
            )

    def test_step8_config_requires_all_stack_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                CrossArchFusionFitConfig(
                    prediction_dir=str(Path(tmp) / "predictions"),
                    output_dir=str(Path(tmp) / "fusion"),
                    groups={"tiny_group": ["model_a"]},
                    splits=["stack_train"],
                    confirm_final_test=True,
                )
            self.assertIn("mean_logits", DEFAULT_CROSSARCH_FUSERS)

    def test_step9_controls_and_audits_are_in_fusion_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_dir = root / "predictions"
            output_dir = root / "fusion"
            model_names = ["gt_reco_to_part_teacher", "pn_reco_to_pfn_teacher"]
            _write_prediction_fixture(prediction_dir, model_names)
            config = CrossArchFusionFitConfig(
                prediction_dir=str(prediction_dir),
                output_dir=str(output_dir),
                groups={"tiny_group": model_names},
                confirm_final_test=True,
                fusers=["mean_logits"],
                c_grid=[0.1],
                max_iter=10,
                control_feature_modes=["logits"],
                control_warning_min_accuracy=1.0,
            )
            report = run_crossarch_fusers(config)
            group = report["groups"]["tiny_group"]
            self.assertTrue(report["ok"])
            self.assertTrue(report["audit_summary"]["ok"])
            self.assertTrue(report["controls_summary"]["ok"])
            self.assertIn("audits", group)
            self.assertIn("controls", group)
            self.assertTrue(group["audits"]["source_alignment"]["ok"])
            self.assertTrue(group["audits"]["split_leakage"]["ok"])
            self.assertTrue(group["controls"]["enabled"])
            self.assertIn("logits", group["controls"]["mode_reports"])

    def test_step9_split_leakage_audit_detects_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            prediction_dir = Path(tmp) / "predictions"
            model_names = ["gt_reco_to_part_teacher", "pn_reco_to_pfn_teacher"]
            _write_prediction_fixture_with_split_overlap(prediction_dir, model_names)
            report = fit_group_fusers(
                prediction_dir,
                group_name="leaky_group",
                model_names=model_names,
                fusers=["mean_logits"],
                run_controls=False,
            )
            leakage = report["audits"]["split_leakage"]
            self.assertFalse(report["ok"])
            self.assertFalse(leakage["ok"])
            self.assertEqual(leakage["cross_split_overlaps"]["stack_train__stack_val"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
