import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM
from teacher_logit_reco.set_matching.experiment import (
    SOURCE_TYPE_ORIGINAL_HLT,
    SOURCE_TYPE_RECONSTRUCTED,
    VIEW_NAMES,
)
from teacher_logit_reco.set_matching.five_view_data import FiveViewJetDataset, make_five_view_loader
from teacher_logit_reco.set_matching.five_view_model import build_five_view_tagger
from teacher_logit_reco.set_matching.five_view_train import (
    FiveViewTaggerTrainConfig,
    classification_metrics_from_predictions,
    five_view_tagger_training_checkpoint_payload,
    infer_experiment_dir_from_tagger_output,
    run_five_view_tagger_epoch,
    train_five_view_tagger,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


def make_five_view_dataset(*, split="stack_train", n_jets=4, n_tokens=5):
    rng = np.random.default_rng(17 if split == "stack_train" else 23)
    features = rng.normal(size=(n_jets, 5, n_tokens, RAW_TOKEN_DIM)).astype(np.float32)
    masks = np.ones((n_jets, 5, n_tokens), dtype=bool)
    masks[:, :, -1] = False
    confidence = np.where(masks, 0.8, 0.0).astype(np.float32)
    confidence[:, 0] = np.where(masks[:, 0], 1.0, 0.0)
    labels = (np.arange(n_jets, dtype=np.int64) % 3).astype(np.int64)
    for jet_index, label in enumerate(labels):
        features[jet_index, :, :, int(label)] += 1.0
    jet_ids = [
        JetIdentity(file=f"synthetic_{index % 2}.root", entry=index, label=int(label))
        for index, label in enumerate(labels)
    ]
    source_types = [SOURCE_TYPE_ORIGINAL_HLT] + [SOURCE_TYPE_RECONSTRUCTED] * 4
    return FiveViewJetDataset(
        view_features=features,
        view_masks=masks,
        view_confidence=confidence,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        view_names=VIEW_NAMES,
        source_types=source_types,
        view_ids=np.arange(5, dtype=np.int64),
        source_type_ids=np.asarray([0, 1, 1, 1, 1], dtype=np.int64),
        metadata={"split": split, "n_jets": n_jets, "synthetic": True},
    )


def tiny_train_config(output_dir, **overrides):
    payload = {
        "output_dir": str(output_dir),
        "hlt_cache_dir": "unused_hlt_cache",
        "experiment_dir": str(Path(output_dir).parent.parent),
        "confirm_split_settings": True,
        "batch_size": 2,
        "epochs": 1,
        "early_stop_patience": 3,
        "num_workers": 0,
        "device": "cpu",
        "amp": False,
        "embed_dim": 32,
        "stage1_layers": 1,
        "stage1_heads": 4,
        "stage2_layers": 1,
        "stage2_heads": 4,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "max_tokens_per_view": 5,
        "min_tokens_per_view": 0,
        "confidence_threshold": 0.0,
    }
    payload.update(overrides)
    return FiveViewTaggerTrainConfig(**payload)


class FiveViewTaggerTrainConfigTests(unittest.TestCase):
    def test_config_requires_explicit_split_confirmation(self):
        with self.assertRaises(ValueError):
            FiveViewTaggerTrainConfig(output_dir="out", hlt_cache_dir="hlt_cache")

    def test_infers_experiment_root_from_canonical_tagger_path(self):
        root = Path("checkpoints/set_matching_multiview_500k")
        output_dir = root / "taggers" / "five_view_tagger"
        self.assertEqual(infer_experiment_dir_from_tagger_output(output_dir), root)
        config = tiny_train_config(output_dir, experiment_dir=None)
        self.assertEqual(config.experiment_root, root)
        self.assertEqual(config.dataset_config("stack_train").output_dir, str(root))

    def test_metrics_report_confusion_and_per_class_accuracy(self):
        metrics = classification_metrics_from_predictions(
            preds=np.asarray([0, 1, 1, 2]),
            labels=np.asarray([0, 1, 2, 2]),
            loss_sum=2.0,
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["n_jets"], 4)
        self.assertEqual(metrics["confusion_matrix"][2][1], 1)
        self.assertAlmostEqual(metrics["loss"], 0.5)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class FiveViewTaggerTrainingLoopTests(unittest.TestCase):
    def test_epoch_runner_returns_basic_metrics(self):
        dataset = make_five_view_dataset()
        loader = make_five_view_loader(dataset, batch_size=2, shuffle=False, num_workers=0)
        config = tiny_train_config("out/taggers/five_view_tagger")
        model = build_five_view_tagger(config.model_config(particle_feature_dim=RAW_TOKEN_DIM))
        metrics = run_five_view_tagger_epoch(
            model,
            loader,
            device=torch.device("cpu"),
            criterion=torch.nn.CrossEntropyLoss(),
            amp=False,
            collect_predictions=True,
        )
        self.assertEqual(metrics["n_jets"], len(dataset))
        self.assertIn("loss", metrics)
        self.assertIn("per_class_accuracy", metrics)
        self.assertTrue(np.isfinite(metrics["loss"]))

    def test_checkpoint_payload_includes_training_metadata(self):
        config = tiny_train_config("out/taggers/five_view_tagger")
        model = build_five_view_tagger(config.model_config(particle_feature_dim=RAW_TOKEN_DIM))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        payload = five_view_tagger_training_checkpoint_payload(
            model,
            optimizer,
            epoch=2,
            config=config,
            metrics={"stack_val": {"accuracy": 0.25}},
            source={"commit": "test"},
        )
        self.assertIn("model_state_dict", payload)
        self.assertIn("optimizer_state_dict", payload)
        self.assertEqual(payload["epoch"], 2)
        self.assertEqual(payload["metrics"]["stack_val"]["accuracy"], 0.25)

    def test_train_five_view_tagger_writes_report_without_final_test(self):
        train_dataset = make_five_view_dataset(split="stack_train", n_jets=4)
        val_dataset = make_five_view_dataset(split="stack_val", n_jets=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "experiment" / "taggers" / "five_view_tagger"
            config = tiny_train_config(output_dir)
            report = train_five_view_tagger(config, train_dataset=train_dataset, val_dataset=val_dataset)

            self.assertTrue((output_dir / "best_model_val.pt").exists())
            self.assertTrue((output_dir / "run_report.json").exists())
            self.assertTrue((output_dir / "diagnostics" / "per_class_metrics.csv").exists())
            self.assertTrue((output_dir / "diagnostics" / "view_ablation_metrics.json").exists())
            self.assertFalse(report["final_test_evaluated"])
            self.assertTrue(report["no_final_test_evaluation"])
            self.assertEqual(report["best_stack_val_metrics"]["n_jets"], len(val_dataset))


if __name__ == "__main__":
    unittest.main()
