import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block, softmax_np
from jetclass_fresh.jetclass_data import JetIdentity
from teacher_logit_reco.conditional_evidence_fusers import (
    ConditionalEvidenceFuserConfig,
    build_conditional_features,
    run_conditional_evidence_fusers,
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


def _write_fixture(prediction_dir: Path, model_names: list[str]) -> None:
    labels_by_split = {
        "stack_train": np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64),
        "stack_val": np.asarray([0, 1, 2, 1, 0, 2], dtype=np.int64),
        "final_test": np.asarray([2, 1, 0, 2, 1, 0], dtype=np.int64),
    }
    logits_by_split = {
        "stack_train": np.asarray(
            [
                [2.5, 0.2, 0.0],
                [0.1, 2.2, 0.2],
                [0.2, 0.3, 2.4],
                [2.2, 0.4, 0.1],
                [0.0, 2.0, 0.4],
                [0.3, 0.2, 2.1],
                [2.4, 0.3, 0.2],
                [0.2, 2.3, 0.0],
                [0.1, 0.5, 2.3],
            ],
            dtype=np.float32,
        ),
        "stack_val": np.asarray(
            [
                [2.0, 0.5, 0.1],
                [0.2, 1.8, 0.4],
                [0.3, 0.7, 2.1],
                [0.4, 1.9, 0.2],
                [2.2, 0.3, 0.1],
                [0.2, 0.4, 2.0],
            ],
            dtype=np.float32,
        ),
        "final_test": np.asarray(
            [
                [0.2, 0.5, 2.4],
                [0.1, 2.0, 0.3],
                [2.3, 0.4, 0.2],
                [0.4, 0.5, 1.8],
                [0.3, 1.9, 0.2],
                [2.1, 0.2, 0.5],
            ],
            dtype=np.float32,
        ),
    }
    for model_index, model_name in enumerate(model_names):
        for split, labels in labels_by_split.items():
            logits = logits_by_split[split].copy()
            logits[:, model_index % 3] += 0.15
            logits += 0.02 * float(model_index)
            save_prediction_block(_block(model_name, split, logits, labels), prediction_dir)


class ConditionalEvidenceFuserTests(unittest.TestCase):
    def test_build_conditional_features_shapes(self):
        labels = np.asarray([0, 1, 2, 0], dtype=np.int64)
        adapted = [
            _block("gt_reco_to_part_adapted_tagger", "stack_train", np.eye(3, dtype=np.float32)[labels] * 2.0, labels),
            _block("pn_reco_to_pn_adapted_tagger", "stack_train", np.eye(3, dtype=np.float32)[labels] * 1.5, labels),
        ]
        hlt = [_block("hlt_part", "stack_train", np.eye(3, dtype=np.float32)[labels] * 1.8, labels)]
        anchor_logits = hlt[0].logits
        features, names = build_conditional_features(
            adapted,
            hlt,
            anchor_logits=anchor_logits,
            anchor_probs=softmax_np(anchor_logits),
        )
        self.assertEqual(features.shape[0], len(labels))
        self.assertEqual(features.shape[1], len(names))
        self.assertIn("adapted_pairwise_disagreement", names)
        self.assertIn("anchor_entropy", names)

    def test_run_linear_suite_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            prediction_dir = Path(tmp) / "predictions"
            output_dir = Path(tmp) / "conditional"
            hlt_models = ["hlt_part", "hlt_pn"]
            adapted_models = [
                "gt_reco_to_part_adapted_tagger",
                "pn_reco_to_pn_adapted_tagger",
            ]
            _write_fixture(prediction_dir, [*hlt_models, *adapted_models])
            config = ConditionalEvidenceFuserConfig(
                prediction_dir=str(prediction_dir),
                output_dir=str(output_dir),
                suite="linear",
                hlt_model_names=tuple(hlt_models),
                adapted_model_names=tuple(adapted_models),
                c_grid=(0.1,),
                max_iter=20,
                confirm_final_test=True,
                run_controls=False,
                residual_penalties=(0.0,),
                weight_decays=(0.0,),
                confusion_pair_counts=(2,),
                neural_epochs=1,
                neural_batch_size=4,
            )
            report = run_conditional_evidence_fusers(config)
            self.assertTrue(report["ok"])
            self.assertEqual(report["suite"], "linear")
            self.assertIn("hlt4_anchor", report["methods"])
            self.assertIn("plain_logistic_conditional_features", report["methods"])
            self.assertTrue((output_dir / "conditional_fuser_report.json").exists())
            self.assertTrue((output_dir / "method_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
