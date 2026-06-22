import tempfile
import unittest
from pathlib import Path

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM

from teacher_logit_reco.subtoken_part import (
    ModalityResidualHead,
    SUBTOKEN_PART_RESIDUAL_STEP,
    SubtokenDistillJetDataset,
    SubtokenDistillTrainConfig,
    SubtokenHLTJetDataset,
    build_subtoken_particle_transformer_classifier,
    compute_modality_residual_loss,
    compute_modality_residual_targets,
    load_subtoken_tagger_checkpoint,
    train_subtoken_distilled_tagger,
)


class FakeOfflineTeacher:
    def __init__(self):
        self.calls = 0
        self.device = require_torch().device("cpu")
        self.metadata = {
            "architecture": "fake_part",
            "checkpoint_path": "fake_offline_teacher.pt",
            "frozen": True,
        }

    def forward_view_no_grad(self, tokens, mask, *, weights=None):
        torch = require_torch()
        self.calls += 1
        score = torch.where(mask, tokens[:, :, 0], torch.zeros_like(tokens[:, :, 0])).sum(dim=1) / 100.0
        return torch.stack([-score, score], dim=1)


class SubtokenPartStep17ResidualTests(unittest.TestCase):
    def make_view(self, *, split="model_train", n_jets=8, view="fixed_hlt", offset=0.0):
        torch = require_torch()
        tokens = torch.zeros((n_jets, 6, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.ones((n_jets, 6), dtype=torch.bool)
        labels = torch.tensor([0, 3] * ((n_jets + 1) // 2), dtype=torch.long)[:n_jets]
        for index in range(n_jets):
            label = int(labels[index].item())
            base = 28.0 + 7.0 * label + float(index) + float(offset)
            tokens[index, :, 0] = base - torch.arange(6, dtype=torch.float32)
            tokens[index, :, 1] = torch.linspace(-0.45, 0.45, 6) + 0.02 * offset
            tokens[index, :, 2] = torch.linspace(-1.5, 1.5, 6) + 0.01 * offset
            tokens[index, :, 3] = tokens[index, :, 0] + 18.0
            tokens[index, :, 4] = 1.0 if label == 3 else -1.0
            tokens[index, :, 5 + (label % 5)] = 1.0
            tokens[index, :, 10] = 0.08 * (label + 1) + 0.01 * offset
            tokens[index, :, 11] = 0.2
            tokens[index, :, 12] = -0.08 * (label + 1) - 0.01 * offset
            tokens[index, :, 13] = 0.3
        mask[1, 4:] = False
        mask[3, 5:] = False
        tokens = tokens * mask[:, :, None]
        jet_ids = [
            JetIdentity(file=f"{split}_{index}.root", entry=index, label=int(labels[index].item()))
            for index in range(n_jets)
        ]
        metadata = {
            "view": view,
            "hlt_content_hash": f"{split}_hlt_hash",
            "jet_identity_hash": f"{split}_ids",
            "source_manifest_hash": "manifest_hash",
        }
        if view == "offline":
            metadata = {"view": "offline", "source_manifest_hash": "manifest_hash"}
        return JetView(
            tokens=tokens.numpy(),
            mask=mask.numpy(),
            labels=labels.numpy(),
            jet_ids=jet_ids,
            split=split,
            metadata=metadata,
        )

    def make_paired_dataset(self, split="model_train", n_jets=8):
        return SubtokenDistillJetDataset(
            self.make_view(split=split, n_jets=n_jets, view="fixed_hlt", offset=0.0),
            self.make_view(split=split, n_jets=n_jets, view="offline", offset=1.5),
            label_filter=(0, 3),
            label_names=("QCD", "Hgg"),
        )

    def make_hlt_dataset(self, split="model_val", n_jets=8):
        return SubtokenHLTJetDataset(
            self.make_view(split=split, n_jets=n_jets, view="fixed_hlt", offset=0.0),
            label_filter=(0, 3),
            label_names=("QCD", "Hgg"),
        )

    def make_config(self, output_dir):
        return SubtokenDistillTrainConfig(
            output_dir=str(output_dir),
            hlt_cache_dir="unused_hlt_cache",
            manifest_path="unused_manifest.json.gz",
            offline_teacher_checkpoint="unused_teacher.pt",
            confirm_split_settings=True,
            confirm_final_test=False,
            seed=31,
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
            distillation_temperature=2.0,
            distillation_weight=0.25,
            classification_weight=1.0,
            modality_residual_weight=0.1,
            modality_residual_target_mode="jet_plus_nearest",
            gate_residual_regularization_weight=0.05,
        )

    def test_residual_targets_have_expected_shape_and_modes(self):
        torch = require_torch()
        dataset = self.make_paired_dataset(n_jets=4)
        hlt_tokens = torch.from_numpy(dataset.tokens).float()
        hlt_mask = torch.from_numpy(dataset.mask).bool()
        offline_tokens = torch.from_numpy(dataset.offline_tokens).float()
        offline_mask = torch.from_numpy(dataset.offline_mask).bool()

        jet_targets = compute_modality_residual_targets(
            hlt_tokens,
            hlt_mask,
            offline_tokens,
            offline_mask,
            target_mode="jet",
        )
        nearest_targets = compute_modality_residual_targets(
            hlt_tokens,
            hlt_mask,
            offline_tokens,
            offline_mask,
            target_mode="nearest",
        )
        combined_targets = compute_modality_residual_targets(
            hlt_tokens,
            hlt_mask,
            offline_tokens,
            offline_mask,
            target_mode="jet_plus_nearest",
        )

        self.assertEqual(jet_targets.target_mode, "jet")
        self.assertEqual(tuple(jet_targets.targets.shape), (4, 6, 3))
        self.assertEqual(tuple(nearest_targets.targets.shape), (4, 6, 3))
        self.assertEqual(tuple(combined_targets.targets.shape), (4, 6, 3))
        self.assertTrue(bool(torch.isfinite(combined_targets.targets).all()))
        self.assertGreater(float(combined_targets.targets[combined_targets.mask].mean().item()), 0.0)

    def test_residual_head_and_loss_are_finite(self):
        torch = require_torch()
        dataset = self.make_paired_dataset(n_jets=4)
        config = self.make_config(Path("unused")).model_config()
        model = build_subtoken_particle_transformer_classifier(config)
        head = ModalityResidualHead(config)
        hlt_tokens = torch.from_numpy(dataset.tokens).float()
        hlt_mask = torch.from_numpy(dataset.mask).bool()
        offline_tokens = torch.from_numpy(dataset.offline_tokens).float()
        offline_mask = torch.from_numpy(dataset.offline_mask).bool()

        output = model(hlt_tokens, hlt_mask, return_outputs=True)
        targets = compute_modality_residual_targets(
            hlt_tokens,
            hlt_mask,
            offline_tokens,
            offline_mask,
            feature_config=config.feature_config,
            target_mode="jet_plus_nearest",
        )
        prediction = head(output)
        loss = compute_modality_residual_loss(prediction, targets)

        self.assertEqual(tuple(prediction.residual_pred_by_modality.shape), (4, 6, 3))
        self.assertTrue(bool(torch.isfinite(prediction.residual_pred_by_modality).all()))
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_distilled_training_records_residual_auxiliary_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "distill_residual"
            teacher = FakeOfflineTeacher()
            report = train_subtoken_distilled_tagger(
                self.make_config(output_dir),
                teacher=teacher,
                train_dataset=self.make_paired_dataset(split="model_train", n_jets=8),
                val_dataset=self.make_hlt_dataset(split="model_val", n_jets=8),
                stack_val_dataset=self.make_hlt_dataset(split="stack_val", n_jets=8),
            )

            self.assertTrue(report["modality_residual_supervision"]["enabled"])
            self.assertEqual(report["modality_residual_supervision"]["target_mode"], "jet_plus_nearest")
            self.assertIn("modality_residual_loss", report["final_epoch"]["train"])
            self.assertTrue((output_dir / "best_model_val.pt").exists())
            _, payload = load_subtoken_tagger_checkpoint(output_dir / "best_model_val.pt")
            self.assertEqual(report["modality_residual_supervision"]["step"], SUBTOKEN_PART_RESIDUAL_STEP)
            self.assertEqual(payload["experiment_step"], "subtoken_part_step16_offline_teacher_distillation")
            self.assertIsNotNone(payload["residual_head_state_dict"])
            self.assertEqual(payload["model_config"]["version"], "privileged_offline")


if __name__ == "__main__":
    unittest.main()
