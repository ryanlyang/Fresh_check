import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fixed_hlt import HLT_PROFILE_V2_REALISTIC, HLT_PROFILE_V2_REALISTIC_VERSION
from jetclass_fresh.hlt_cache import fixed_hlt_params_dict, fixed_hlt_params_from_profile
from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, JetView
from teacher_logit_reco.target_denoising_part import (
    DENOISING_TARGET_NAMES,
    TARGET_DENOISING_TRAINING_CONTRACT,
    TargetDenoisingDatasetConfig,
    TargetDenoisingPairedDataset,
    TargetDenoisingPretrainConfig,
    TargetConditionedDenoiserConfig,
    TargetConditionedPairwiseDenoiser,
    collate_target_denoising_batch,
    target_denoising_loss,
    train_target_conditioned_denoiser,
)


def _tokens(n_jets=4, n_particles=5):
    tokens = np.zeros((n_jets, n_particles, 14), dtype=np.float32)
    for row in range(n_jets):
        for col in range(n_particles):
            tokens[row, col, 0] = 10.0 + row + 0.5 * col
            tokens[row, col, 1] = -0.15 + 0.02 * row + 0.03 * col
            tokens[row, col, 2] = ((-2.8 + 0.35 * col + 0.04 * row + math.pi) % (2.0 * math.pi)) - math.pi
            tokens[row, col, 3] = tokens[row, col, 0] + 2.0
            tokens[row, col, 4] = 1.0 if col % 2 == 0 else -1.0
            tokens[row, col, 5 + (col % 5)] = 1.0
    return tokens


def _view(tokens, mask, *, split, view, manifest_hash="manifest-step3", labels=None):
    labels = np.arange(tokens.shape[0], dtype=np.int64) % 10 if labels is None else np.asarray(labels, dtype=np.int64)
    jet_ids = [
        JetIdentity(file=f"class_{int(label)}.root", entry=2000 + index, label=int(label))
        for index, label in enumerate(labels)
    ]
    metadata = {
        "view": view,
        "source_manifest_hash": manifest_hash,
    }
    if view == "fixed_hlt":
        params = fixed_hlt_params_from_profile(HLT_PROFILE_V2_REALISTIC, 1.0)
        metadata.update(
            {
                "hlt_profile": HLT_PROFILE_V2_REALISTIC,
                "hlt_profile_version": HLT_PROFILE_V2_REALISTIC_VERSION,
                "hlt_degradation_strength": 1.0,
                "hlt_params": fixed_hlt_params_dict(params),
                "hlt_content_hash": f"fake-hlt-{split}",
                "target_denoising_order_preserving": True,
            }
        )
    return JetView(
        tokens=tokens.astype(np.float32),
        mask=mask.astype(bool),
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata=metadata,
    )


def _dataset(split):
    offline = _tokens()
    hlt = offline.copy()
    hlt[:, :, 0] *= 0.94
    hlt[:, :, 1] += 0.006
    hlt[:, :, 2] = ((hlt[:, :, 2] - 0.011 + math.pi) % (2.0 * math.pi)) - math.pi
    hlt[:, :, 3] *= 0.96
    mask = np.ones(offline.shape[:2], dtype=bool)
    mask[:, -1] = False
    config = TargetDenoisingDatasetConfig(
        manifest_path="unused.json.gz",
        hlt_cache_dir="unused_cache",
        split=split,
        expected_hlt_profile=HLT_PROFILE_V2_REALISTIC,
        expected_hlt_profile_version=HLT_PROFILE_V2_REALISTIC_VERSION,
        expected_hlt_degradation_strength=1.0,
    )
    return TargetDenoisingPairedDataset(
        _view(hlt, mask, split=split, view="fixed_hlt"),
        _view(offline, mask, split=split, view="offline"),
        config=config,
        expected_manifest_hash="manifest-step3",
    )


class TargetDenoisingPartStep3TrainTests(unittest.TestCase):
    def test_loss_is_finite_on_synthetic_batch(self):
        torch = require_torch()
        dataset = _dataset("model_train")
        batch = collate_target_denoising_batch([dataset[0], dataset[1]])
        model = TargetConditionedPairwiseDenoiser(
            TargetConditionedDenoiserConfig(embed_dim=16, num_heads=4, pair_hidden_dim=8, head_hidden_dim=16)
        )
        output = model(batch["hlt_tokens"], batch["hlt_constituent_mask"])
        config = TargetDenoisingPretrainConfig(output_dir="unused", manifest_path="unused", hlt_cache_dir="unused")

        loss, diagnostics = target_denoising_loss(output, batch, config)

        self.assertTrue(bool(torch.isfinite(loss).all()))
        self.assertGreater(float(diagnostics["target_count"].detach().cpu().item()), 0.0)
        for name in DENOISING_TARGET_NAMES:
            self.assertIn(f"rmse_{name}", diagnostics)

    def test_train_loop_writes_step3_artifacts(self):
        require_torch()
        train_dataset = _dataset("model_train")
        val_dataset = _dataset("model_val")
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "denoiser"
            config = TargetDenoisingPretrainConfig(
                output_dir=str(output_dir),
                manifest_path="unused.json.gz",
                hlt_cache_dir="unused_cache",
                seed=123,
                batch_size=2,
                eval_batch_size=2,
                epochs=2,
                lr=1.0e-3,
                weight_decay=0.0,
                num_workers=0,
                device="cpu",
                amp=False,
                grad_clip_norm=1.0,
                early_stop_patience=0,
                max_train_batches=1,
                max_val_batches=1,
                embed_dim=16,
                num_heads=4,
                pair_hidden_dim=8,
                head_hidden_dim=16,
                dropout=0.0,
                attention_dropout=0.0,
            )

            report = train_target_conditioned_denoiser(
                config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )

            expected = [
                "best_denoiser_model_val.pt",
                "last.pt",
                "config.json",
                "training_curves.json",
                "model_val_diagnostics.json",
                "run_report.json",
                "diagnostics/epoch_metrics.csv",
                "diagnostics/train_dataset_metadata.json",
                "diagnostics/model_val_dataset_metadata.json",
            ]
            for relative in expected:
                self.assertTrue((output_dir / relative).exists(), relative)

            self.assertTrue(report["ok"])
            self.assertEqual(report["output_contract"], TARGET_DENOISING_TRAINING_CONTRACT)
            self.assertGreaterEqual(report["best_epoch"], 0)
            self.assertLess(report["best_epoch"], 2)
            self.assertFalse(report["final_test_evaluated"])
            self.assertEqual(report["train_dataset"]["split"], "model_train")
            self.assertEqual(report["model_val_dataset"]["split"], "model_val")

            saved_report = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_report["checkpoint"], str(output_dir / "best_denoiser_model_val.pt"))
            torch = require_torch()
            checkpoint_payload = torch.load(
                output_dir / "best_denoiser_model_val.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint_payload["train_dataset"]["split"], "model_train")
            self.assertEqual(checkpoint_payload["model_val_dataset"]["split"], "model_val")
            self.assertEqual(checkpoint_payload["train_dataset"]["source_manifest_hash"], "manifest-step3")
            curves = json.loads((output_dir / "training_curves.json").read_text(encoding="utf-8"))
            self.assertEqual(len(curves["epochs"]), 2)
            model_val = json.loads((output_dir / "model_val_diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(model_val["split"], "model_val")

    def test_shuffled_target_control_is_recorded_in_checkpoint_config(self):
        require_torch()
        train_dataset = _dataset("model_train")
        val_dataset = _dataset("model_val")
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "denoiser_shuffled"
            config = TargetDenoisingPretrainConfig(
                output_dir=str(output_dir),
                manifest_path="unused.json.gz",
                hlt_cache_dir="unused_cache",
                seed=321,
                batch_size=2,
                eval_batch_size=2,
                epochs=1,
                lr=1.0e-3,
                weight_decay=0.0,
                num_workers=0,
                device="cpu",
                amp=False,
                max_train_batches=1,
                max_val_batches=1,
                embed_dim=16,
                num_heads=4,
                pair_hidden_dim=8,
                head_hidden_dim=16,
                shuffle_target_residuals=True,
                target_shuffle_seed=99,
            )

            train_target_conditioned_denoiser(config, train_dataset=train_dataset, val_dataset=val_dataset)

            saved_report = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
            self.assertTrue(saved_report["config"]["shuffle_target_residuals"])
            self.assertEqual(saved_report["config"]["target_shuffle_seed"], 99)


if __name__ == "__main__":
    unittest.main()
