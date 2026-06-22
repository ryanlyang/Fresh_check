import tempfile
import unittest
from pathlib import Path

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM

from teacher_logit_reco.subtoken_part import (
    SUBTOKEN_PART_TRAIN_STEP,
    SubtokenHLTJetDataset,
    SubtokenTaggerTrainConfig,
    collate_subtoken_hlt_batch,
    load_subtoken_tagger_checkpoint,
    make_subtoken_hlt_loader,
    run_subtoken_tagger_epoch,
    train_subtoken_tagger,
)


class SubtokenPartStep12TrainingTests(unittest.TestCase):
    def make_view(self, *, split="model_train", n_jets=8):
        torch = require_torch()
        tokens = torch.zeros((n_jets, 6, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.ones((n_jets, 6), dtype=torch.bool)
        labels = torch.tensor([0, 3] * ((n_jets + 1) // 2), dtype=torch.long)[:n_jets]
        for index in range(n_jets):
            label = int(labels[index].item())
            base = 30.0 + 10.0 * label + float(index)
            tokens[index, :, 0] = base - torch.arange(6, dtype=torch.float32)
            tokens[index, :, 1] = torch.linspace(-0.5, 0.5, 6) + 0.1 * label
            tokens[index, :, 2] = torch.linspace(-2.0, 2.0, 6)
            tokens[index, :, 3] = tokens[index, :, 0] + 20.0
            tokens[index, :, 4] = 1.0 if label == 3 else -1.0
            tokens[index, :, 5 + (label % 5)] = 1.0
            tokens[index, :, 10] = 0.1 * (label + 1)
            tokens[index, :, 11] = 0.2
            tokens[index, :, 12] = -0.1 * (label + 1)
            tokens[index, :, 13] = 0.3
        mask[1, 4:] = False
        mask[3, 5:] = False
        tokens = tokens * mask[:, :, None]
        jet_ids = [
            JetIdentity(file=f"{split}_{index}.root", entry=index, label=int(labels[index].item()))
            for index in range(n_jets)
        ]
        return JetView(
            tokens=tokens.numpy(),
            mask=mask.numpy(),
            labels=labels.numpy(),
            jet_ids=jet_ids,
            split=split,
            metadata={"view": "fixed_hlt", "hlt_content_hash": f"{split}_hash", "jet_identity_hash": f"{split}_ids"},
        )

    def make_dataset(self, split="model_train", n_jets=8):
        return SubtokenHLTJetDataset(
            self.make_view(split=split, n_jets=n_jets),
            label_filter=(0, 3),
            label_names=("QCD", "Hgg"),
        )

    def make_config(self, output_dir):
        return SubtokenTaggerTrainConfig(
            output_dir=str(output_dir),
            hlt_cache_dir="unused",
            confirm_split_settings=True,
            confirm_final_test=False,
            seed=17,
            batch_size=4,
            eval_batch_size=4,
            epochs=1,
            lr=1.0e-3,
            weight_decay=0.0,
            num_workers=0,
            device="cpu",
            amp=False,
            max_train_batches=1,
            max_val_batches=1,
            max_stack_val_batches=1,
            selection_metric="accuracy",
            num_classes=2,
            label_names=("QCD", "Hgg"),
            label_filter=(0, 3),
            embed_dim=8,
            local_layers=1,
            local_heads=2,
            context_layers=1,
            context_heads=2,
            global_layers=1,
            global_heads=2,
            dropout=0.0,
            attention_dropout=0.0,
            use_pairwise_bias=True,
        )

    def test_dataset_filters_and_remaps_labels(self):
        dataset = self.make_dataset(n_jets=6)

        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset.label_counts(), {"QCD": 3, "Hgg": 3})
        self.assertEqual(set(dataset.labels.tolist()), {0, 1})
        self.assertEqual(dataset.metadata["label_filter"], [0, 3])
        self.assertEqual(dataset.metadata["raw_token_dim"], RAW_TOKEN_DIM)

    def test_config_can_infer_label_filter_from_label_names(self):
        config = SubtokenTaggerTrainConfig(
            output_dir="unused",
            hlt_cache_dir="unused",
            confirm_split_settings=True,
            num_classes=2,
            label_names=("QCD", "Hgg"),
        )

        self.assertEqual(config.resolved_label_filter, (0, 3))
        self.assertEqual(config.resolved_label_names, ("QCD", "Hgg"))

    def test_binary_selection_metric_rejects_multiclass_setup(self):
        config = SubtokenTaggerTrainConfig(
            output_dir="unused",
            hlt_cache_dir="unused",
            confirm_split_settings=True,
            selection_metric="fpr_at_signal_eff_0p50",
            num_classes=3,
        )

        with self.assertRaises(ValueError):
            config.validate_label_metadata()

    def test_collate_subtoken_hlt_batch_shapes(self):
        torch = require_torch()
        dataset = self.make_dataset(n_jets=4)
        batch = collate_subtoken_hlt_batch([dataset[0], dataset[1]])

        self.assertEqual(tuple(batch["tokens"].shape), (2, 6, RAW_TOKEN_DIM))
        self.assertEqual(tuple(batch["mask"].shape), (2, 6))
        self.assertEqual(tuple(batch["labels"].shape), (2,))
        self.assertTrue(bool(torch.isfinite(batch["tokens"]).all()))

    def test_run_epoch_collects_binary_metrics_and_diagnostics(self):
        torch = require_torch()
        dataset = self.make_dataset(n_jets=6)
        loader = make_subtoken_hlt_loader(dataset, batch_size=3, shuffle=False, num_workers=0, seed=5)
        config = self.make_config(Path("unused")).model_config()
        from teacher_logit_reco.subtoken_part import SubtokenParticleTransformerClassifier

        model = SubtokenParticleTransformerClassifier(config)
        criterion = torch.nn.CrossEntropyLoss()

        metrics = run_subtoken_tagger_epoch(
            model,
            loader,
            device=torch.device("cpu"),
            criterion=criterion,
            amp=False,
            collect_predictions=True,
            collect_diagnostics=True,
            label_names=("QCD", "Hgg"),
        )

        self.assertEqual(metrics["n_jets"], 6)
        self.assertIn("binary_metrics", metrics)
        self.assertIn("fpr_at_signal_eff_0p50", metrics["binary_metrics"])
        self.assertIn("diagnostics", metrics)

    def test_train_subtoken_tagger_writes_reports_and_reloadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "subtoken_train"
            config = self.make_config(output_dir)
            report = train_subtoken_tagger(
                config,
                train_dataset=self.make_dataset(split="model_train", n_jets=8),
                val_dataset=self.make_dataset(split="model_val", n_jets=8),
                stack_val_dataset=self.make_dataset(split="stack_val", n_jets=8),
            )

            self.assertEqual(report["experiment_step"], SUBTOKEN_PART_TRAIN_STEP)
            self.assertTrue((output_dir / "run_report.json").exists())
            self.assertTrue((output_dir / "best_model_val.pt").exists())
            self.assertTrue((output_dir / "diagnostics" / "epoch_metrics.csv").exists())
            model, payload = load_subtoken_tagger_checkpoint(output_dir / "best_model_val.pt")
            self.assertEqual(payload["num_classes"], 2)
            self.assertEqual(model.config.num_classes, 2)


if __name__ == "__main__":
    unittest.main()
