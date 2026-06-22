import tempfile
import unittest
from pathlib import Path

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM

from teacher_logit_reco.subtoken_part import (
    MaskedSubtokenPredictionHead,
    MaskedSubtokenPredictionOutput,
    SUBTOKEN_PART_MASKED_STEP,
    SubtokenDistillJetDataset,
    SubtokenDistillTrainConfig,
    SubtokenHLTJetDataset,
    build_masked_subtoken_targets,
    build_subtoken_particle_transformer_classifier,
    compute_masked_subtoken_loss,
    load_subtoken_tagger_checkpoint,
    sample_masked_subtoken_modality_mask,
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


class SubtokenPartStep18MaskedTests(unittest.TestCase):
    def make_view(self, *, split="model_train", n_jets=8, view="fixed_hlt", offset=0.0):
        torch = require_torch()
        tokens = torch.zeros((n_jets, 6, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.ones((n_jets, 6), dtype=torch.bool)
        labels = torch.tensor([0, 3] * ((n_jets + 1) // 2), dtype=torch.long)[:n_jets]
        for index in range(n_jets):
            label = int(labels[index].item())
            base = 32.0 + 6.0 * label + float(index) + float(offset)
            tokens[index, :, 0] = base - torch.arange(6, dtype=torch.float32)
            tokens[index, :, 1] = torch.linspace(-0.4, 0.4, 6) + 0.02 * offset
            tokens[index, :, 2] = torch.linspace(-1.7, 1.7, 6) + 0.01 * offset
            tokens[index, :, 3] = tokens[index, :, 0] + 16.0
            tokens[index, :, 4] = 1.0 if label == 3 else -1.0
            tokens[index, :, 5 + (label % 5)] = 1.0
            tokens[index, :, 10] = 0.07 * (label + 1) + 0.01 * offset
            tokens[index, :, 11] = 0.2
            tokens[index, :, 12] = -0.07 * (label + 1) - 0.01 * offset
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
            self.make_view(split=split, n_jets=n_jets, view="offline", offset=2.0),
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
            seed=41,
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
            masked_subtoken_weight=0.1,
            masked_subtoken_target_mode="offline",
            masked_subtoken_probability=0.5,
            masked_subtoken_max_match_delta_r=0.4,
        )

    def test_mask_sampler_masks_one_modality_and_keeps_input_valid(self):
        torch = require_torch()
        particle_mask = torch.ones((3, 5), dtype=torch.bool)

        mask_output = sample_masked_subtoken_modality_mask(
            particle_mask,
            num_modalities=3,
            mask_probability=0.5,
            modality_names=("kinematics", "identity", "track"),
            force_at_least_one=True,
        )

        self.assertEqual(tuple(mask_output.prediction_mask.shape), (3, 5, 3))
        self.assertTrue(bool(mask_output.prediction_mask.any()))
        self.assertTrue(bool((mask_output.input_modality_mask | mask_output.prediction_mask).all()))
        self.assertTrue(bool(mask_output.input_modality_mask.any(dim=2).all()))

    def test_masked_forward_hides_requested_modality(self):
        torch = require_torch()
        dataset = self.make_paired_dataset(n_jets=4)
        config = self.make_config(Path("unused")).model_config()
        model = build_subtoken_particle_transformer_classifier(config)
        tokens = torch.from_numpy(dataset.tokens).float()
        mask = torch.from_numpy(dataset.mask).bool()
        mask_output = sample_masked_subtoken_modality_mask(
            mask,
            num_modalities=3,
            mask_probability=0.75,
            modality_names=config.feature_config.modality_names,
            force_at_least_one=True,
        )

        output = model(tokens, mask, return_outputs=True, modality_mask_override=mask_output.input_modality_mask)

        self.assertTrue(bool(torch.equal(output.mixed.modality_mask, mask_output.input_modality_mask)))
        self.assertTrue(bool((output.mixed.local_tokens[mask_output.prediction_mask] == 0.0).all()))

    def test_masked_prediction_loss_is_finite_for_hlt_and_offline_targets(self):
        torch = require_torch()
        dataset = self.make_paired_dataset(n_jets=4)
        config = self.make_config(Path("unused")).model_config()
        model = build_subtoken_particle_transformer_classifier(config)
        head = MaskedSubtokenPredictionHead(config)
        tokens = torch.from_numpy(dataset.tokens).float()
        mask = torch.from_numpy(dataset.mask).bool()
        offline_tokens = torch.from_numpy(dataset.offline_tokens).float()
        offline_mask = torch.from_numpy(dataset.offline_mask).bool()
        mask_output = sample_masked_subtoken_modality_mask(
            mask,
            num_modalities=3,
            mask_probability=0.75,
            modality_names=config.feature_config.modality_names,
            force_at_least_one=True,
        )
        output = model(tokens, mask, return_outputs=True, modality_mask_override=mask_output.input_modality_mask)
        prediction = head(output, mask_output)
        hlt_targets = build_masked_subtoken_targets(
            tokens,
            mask,
            mask_output.prediction_mask,
            feature_config=config.feature_config,
            target_mode="hlt_self",
        )
        offline_targets = build_masked_subtoken_targets(
            offline_tokens,
            offline_mask,
            mask_output.prediction_mask,
            feature_config=config.feature_config,
            target_mode="offline",
            reference_tokens=tokens,
            reference_mask=mask,
        )

        hlt_loss = compute_masked_subtoken_loss(prediction, hlt_targets)
        offline_loss = compute_masked_subtoken_loss(prediction, offline_targets)

        self.assertTrue(bool(torch.isfinite(hlt_loss)))
        self.assertTrue(bool(torch.isfinite(offline_loss)))
        self.assertGreaterEqual(float(hlt_loss.item()), 0.0)
        self.assertGreaterEqual(float(offline_loss.item()), 0.0)
        self.assertEqual(offline_targets.matching_metadata["matching"], "nearest_delta_r")

    def test_masked_prediction_calibration_preserves_bounded_channel_gradients(self):
        torch = require_torch()
        config = self.make_config(Path("unused")).model_config()
        tokens = torch.zeros((1, 1, RAW_TOKEN_DIM), dtype=torch.float32)
        tokens[0, 0, :4] = torch.tensor([30.0, 0.0, 0.0, 42.0])
        tokens[0, 0, 5] = 1.0
        mask = torch.ones((1, 1), dtype=torch.bool)
        prediction_mask = torch.zeros((1, 1, 3), dtype=torch.bool)
        prediction_mask[0, 0, 1] = True
        targets = build_masked_subtoken_targets(
            tokens,
            mask,
            prediction_mask,
            feature_config=config.feature_config,
            target_mode="hlt_self",
        )
        pred_identity = torch.full((1, 1, 6), -5.0, dtype=torch.float32, requires_grad=True)
        prediction = MaskedSubtokenPredictionOutput(
            predictions={
                "kinematics": torch.zeros((1, 1, 11), dtype=torch.float32),
                "identity": pred_identity,
                "track": torch.zeros((1, 1, 8), dtype=torch.float32),
            },
            prediction_mask=prediction_mask,
            particle_mask=mask,
            modality_names=("kinematics", "identity", "track"),
        )

        loss = compute_masked_subtoken_loss(prediction, targets)
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(pred_identity.grad)
        self.assertGreater(abs(float(pred_identity.grad[0, 0, 1].item())), 0.0)

    def test_offline_masked_targets_are_nearest_matched_not_slot_aligned(self):
        torch = require_torch()
        config = self.make_config(Path("unused")).model_config()
        hlt_tokens = torch.zeros((1, 3, RAW_TOKEN_DIM), dtype=torch.float32)
        hlt_mask = torch.tensor([[True, True, False]], dtype=torch.bool)
        hlt_tokens[0, 0, :4] = torch.tensor([50.0, 0.0, 0.0, 60.0])
        hlt_tokens[0, 1, :4] = torch.tensor([40.0, 1.0, 1.0, 50.0])
        offline_tokens = torch.zeros((1, 3, RAW_TOKEN_DIM), dtype=torch.float32)
        offline_mask = torch.tensor([[True, True, False]], dtype=torch.bool)
        # Deliberately reverse the offline slot order.  Nearest matching should
        # recover HLT slot 0 -> offline slot 1 and HLT slot 1 -> offline slot 0.
        offline_tokens[0, 0, :4] = torch.tensor([400.0, 1.01, 1.02, 410.0])
        offline_tokens[0, 1, :4] = torch.tensor([500.0, 0.01, 0.02, 510.0])
        prediction_mask = torch.zeros((1, 3, 3), dtype=torch.bool)
        prediction_mask[0, 0, 0] = True
        prediction_mask[0, 1, 0] = True

        targets = build_masked_subtoken_targets(
            offline_tokens,
            offline_mask,
            prediction_mask,
            feature_config=config.feature_config,
            target_mode="offline",
            reference_tokens=hlt_tokens,
            reference_mask=hlt_mask,
            max_match_delta_r=0.2,
        )

        kin = targets.target_values["kinematics"]
        self.assertAlmostEqual(float(kin[0, 0, 0].item()), 500.0)
        self.assertAlmostEqual(float(kin[0, 1, 0].item()), 400.0)
        self.assertEqual(targets.matching_metadata["matching"], "nearest_delta_r")

    def test_offline_masked_targets_require_reference_slots(self):
        torch = require_torch()
        config = self.make_config(Path("unused")).model_config()
        tokens = torch.zeros((1, 2, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.ones((1, 2), dtype=torch.bool)
        prediction_mask = torch.zeros((1, 2, 3), dtype=torch.bool)
        prediction_mask[0, 0, 0] = True

        with self.assertRaises(ValueError):
            build_masked_subtoken_targets(
                tokens,
                mask,
                prediction_mask,
                feature_config=config.feature_config,
                target_mode="offline",
            )

    def test_distilled_training_records_masked_objective_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "distill_masked"
            report = train_subtoken_distilled_tagger(
                self.make_config(output_dir),
                teacher=FakeOfflineTeacher(),
                train_dataset=self.make_paired_dataset(split="model_train", n_jets=8),
                val_dataset=self.make_hlt_dataset(split="model_val", n_jets=8),
                stack_val_dataset=self.make_hlt_dataset(split="stack_val", n_jets=8),
            )

            self.assertTrue(report["masked_subtoken_objective"]["enabled"])
            self.assertEqual(report["masked_subtoken_objective"]["step"], SUBTOKEN_PART_MASKED_STEP)
            self.assertEqual(report["masked_subtoken_objective"]["target_mode"], "offline")
            self.assertEqual(report["masked_subtoken_objective"]["offline_target_alignment"], "nearest_delta_r")
            self.assertAlmostEqual(report["masked_subtoken_objective"]["masked_subtoken_max_match_delta_r"], 0.4)
            self.assertIn("masked_subtoken_loss", report["final_epoch"]["train"])
            masked_diagnostics = report["final_epoch"]["train"]["masked_subtoken_diagnostics"]
            self.assertIn("target_matched_fraction", masked_diagnostics)
            self.assertIn("target_mean_nearest_delta_r", masked_diagnostics)
            _, payload = load_subtoken_tagger_checkpoint(output_dir / "best_model_val.pt")
            self.assertIsNotNone(payload["masked_subtoken_head_state_dict"])
            self.assertEqual(payload["masked_subtoken_objective"]["target_mode"], "offline")
            self.assertEqual(payload["masked_subtoken_objective"]["offline_target_alignment"], "nearest_delta_r")
            self.assertAlmostEqual(payload["masked_subtoken_objective"]["masked_subtoken_max_match_delta_r"], 0.4)
            self.assertTrue(payload["inference_consumes_hlt_only"])


if __name__ == "__main__":
    unittest.main()
