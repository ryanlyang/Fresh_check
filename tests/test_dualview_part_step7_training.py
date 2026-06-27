import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.dualview_part import (
        DUALVIEW_PART_PRIMARY_METRIC,
        DUALVIEW_PART_SHUFFLED_PN_CONTRACT,
        DUALVIEW_PART_STEP7,
        DUALVIEW_PART_STEP9,
        DUALVIEW_PART_TRAINING_CONTRACT,
        DualViewPartJetDataset,
        DualViewResidualParTConfig,
        DualViewResidualTrainConfig,
        HLTPartAnchorConfig,
        PNMemoryEncoderConfig,
        build_dualview_residual_model_from_config,
        build_dualview_residual_part,
        build_hlt_part_anchor,
        build_pn_memory_encoder,
        shuffle_dualview_part_pn_view,
        train_dualview_residual_part,
    )
else:  # pragma: no cover - environment dependent
    torch = None


if TORCH_AVAILABLE:

    class DummyPartModel(torch.nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.config = {"num_classes": int(num_classes), "input_dim": 17}
            self.proj = torch.nn.Linear(17, int(num_classes))

        def forward(self, points, features, lorentz_vectors, mask):
            del points, lorentz_vectors
            token_mask = mask[:, 0, :].float()
            denom = token_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (features * token_mask[:, None, :]).sum(dim=2) / denom
            return self.proj(pooled)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DualViewPartStep7TrainingTests(unittest.TestCase):
    def make_dataset(self, *, n_jets: int, split: str) -> DualViewPartJetDataset:
        rng = np.random.default_rng(1234 + n_jets)
        labels = (np.arange(n_jets) % 2).astype(np.int64)
        n_hlt = 10
        n_pn = 12
        hlt_tokens = rng.normal(0.0, 0.02, size=(n_jets, n_hlt, RAW_TOKEN_DIM)).astype(np.float32)
        pn_tokens = rng.normal(0.0, 0.02, size=(n_jets, n_pn, RAW_TOKEN_DIM)).astype(np.float32)
        hlt_mask = np.ones((n_jets, n_hlt), dtype=bool)
        pn_mask = np.ones((n_jets, n_pn), dtype=bool)
        pn_mask[::3, -2:] = False
        for index, label in enumerate(labels):
            base_pt = 40.0 if int(label) == 0 else 90.0
            hlt_tokens[index, :, 0] = np.linspace(base_pt, base_pt / 4.0, n_hlt)
            hlt_tokens[index, :, 1] = np.linspace(-0.3, 0.3, n_hlt)
            hlt_tokens[index, :, 2] = np.linspace(-0.2, 0.2, n_hlt)
            hlt_tokens[index, :, 3] = hlt_tokens[index, :, 0] * 1.5
            pn_tokens[index, :, 0] = np.linspace(base_pt * 1.02, base_pt / 5.0, n_pn)
            pn_tokens[index, :, 1] = np.linspace(-0.28, 0.28, n_pn)
            pn_tokens[index, :, 2] = np.linspace(-0.18, 0.18, n_pn)
            pn_tokens[index, :, 3] = pn_tokens[index, :, 0] * 1.45
        pn_confidence = np.where(pn_mask, 0.85, 0.0).astype(np.float32)
        jet_ids = [JetIdentity(file=f"{split}.root", entry=index, label=int(label)) for index, label in enumerate(labels)]
        return DualViewPartJetDataset(
            hlt_tokens=hlt_tokens,
            hlt_mask=hlt_mask,
            pn_reco_tokens=pn_tokens,
            pn_reco_mask=pn_mask,
            pn_reco_confidence=pn_confidence,
            labels=labels,
            jet_ids=jet_ids,
            split=split,
            metadata={"split": split, "n_jets": int(n_jets), "synthetic": True},
        )

    def make_model(self):
        anchor = build_hlt_part_anchor(
            DummyPartModel(num_classes=2),
            config=HLTPartAnchorConfig(
                num_classes=2,
                context_dim=8,
                summary_hidden_dim=16,
                freeze_anchor=True,
                label_names=("QCD", "Hgg"),
            ),
        )
        pn_encoder = build_pn_memory_encoder(
            PNMemoryEncoderConfig(
                embed_dim=8,
                num_layers=1,
                num_heads=2,
                dropout=0.0,
                attention_dropout=0.0,
            )
        )
        model = build_dualview_residual_part(
            anchor,
            pn_encoder,
            config=DualViewResidualParTConfig(
                num_classes=2,
                hlt_context_dim=8,
                pn_context_dim=8,
                hidden_dim=16,
                num_hidden_layers=1,
                dropout=0.0,
                gate_bias_init=-6.0,
            ),
            infer_dims_from_modules=False,
        )
        return model

    def test_config_requires_split_confirmation(self):
        with self.assertRaises(ValueError):
            DualViewResidualTrainConfig(output_dir="unused")

    def test_build_model_from_config_freezes_only_hlt_backbone(self):
        anchor = build_hlt_part_anchor(
            DummyPartModel(num_classes=2),
            config=HLTPartAnchorConfig(
                num_classes=2,
                context_dim=8,
                summary_hidden_dim=16,
                freeze_anchor=False,
                label_names=("QCD", "Hgg"),
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DualViewResidualTrainConfig(
                output_dir=tmpdir,
                hlt_anchor_checkpoint="dummy_anchor.pt",
                confirm_split_settings=True,
                device="cpu",
                amp=False,
                anchor_context_dim=8,
                anchor_summary_hidden_dim=16,
                pn_embed_dim=8,
                pn_layers=1,
                pn_heads=2,
                residual_hidden_dim=16,
                residual_layers=1,
                residual_dropout=0.0,
                freeze_anchor=True,
            )
            with patch(
                "teacher_logit_reco.dualview_part.training.load_hlt_part_anchor",
                return_value=anchor,
            ):
                model = build_dualview_residual_model_from_config(config, device=torch.device("cpu"))

        self.assertEqual(model.pn_encoder.context_dim, 8)
        self.assertTrue(model.hlt_anchor.anchor_parameters_frozen())
        self.assertTrue(any(param.requires_grad for param in model.hlt_anchor.summary_encoder.parameters()))
        self.assertTrue(any(param.requires_grad for param in model.pn_encoder.parameters()))
        self.assertTrue(any(param.requires_grad for param in model.delta_mlp.parameters()))

    def test_one_epoch_training_writes_reports_and_final_metrics(self):
        train_dataset = self.make_dataset(n_jets=16, split="stack_train")
        val_dataset = self.make_dataset(n_jets=8, split="stack_val")
        final_dataset = self.make_dataset(n_jets=8, split="final_test")
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DualViewResidualTrainConfig(
                output_dir=tmpdir,
                confirm_split_settings=True,
                confirm_final_test=True,
                seed=17,
                batch_size=4,
                eval_batch_size=4,
                epochs=1,
                lr=1.0e-3,
                num_workers=0,
                device="cpu",
                amp=False,
                early_stop_patience=-1,
                selection_metric=DUALVIEW_PART_PRIMARY_METRIC,
                anchor_context_dim=8,
                pn_embed_dim=8,
                pn_layers=1,
                pn_heads=2,
                residual_hidden_dim=16,
                residual_layers=1,
                residual_dropout=0.0,
                freeze_anchor=True,
            )
            report = train_dualview_residual_part(
                config,
                model=self.make_model(),
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                final_test_dataset=final_dataset,
            )
            output_dir = Path(tmpdir)
            self.assertTrue((output_dir / "best_model_val.pt").exists())
            self.assertTrue((output_dir / "run_report.json").exists())
            self.assertTrue((output_dir / "diagnostics" / "epoch_metrics.csv").exists())
            self.assertTrue((output_dir / "diagnostics" / "residual_diagnostics.json").exists())
            self.assertTrue((output_dir / "diagnostics" / "gate_by_class.csv").exists())
            self.assertTrue((output_dir / "diagnostics" / "gate_by_hlt_confidence.csv").exists())
            self.assertTrue((output_dir / "diagnostics" / "gate_by_hlt_correctness.csv").exists())
            self.assertTrue((output_dir / "diagnostics" / "prediction_change_summary.csv").exists())

            self.assertEqual(report["experiment_step"], DUALVIEW_PART_STEP7)
            self.assertEqual(report["output_contract"], DUALVIEW_PART_TRAINING_CONTRACT)
            self.assertEqual(report["selection_metric"], DUALVIEW_PART_PRIMARY_METRIC)
            self.assertTrue(report["final_test_evaluated"])
            self.assertIn("residual_diagnostics", report)
            self.assertIn("binary_metrics", report["best_stack_val_metrics"])
            self.assertIn("binary_metrics", report["final_test_metrics"])
            self.assertIn("residual_analysis", report["best_stack_val_metrics"])

    def test_shuffle_pn_view_keeps_hlt_and_labels_but_moves_pn_rows(self):
        dataset = self.make_dataset(n_jets=10, split="stack_val")
        shuffled = shuffle_dualview_part_pn_view(dataset, seed=99, split="stack_val")
        control = shuffled.metadata["pn_view_shuffle_control"]
        permutation = np.asarray(control["permutation_preview"], dtype=np.int64)

        np.testing.assert_array_equal(shuffled.hlt_tokens, dataset.hlt_tokens)
        np.testing.assert_array_equal(shuffled.hlt_mask, dataset.hlt_mask)
        np.testing.assert_array_equal(shuffled.labels, dataset.labels)
        self.assertEqual(shuffled.jet_ids, dataset.jet_ids)
        self.assertTrue(control["enabled"])
        self.assertEqual(control["experiment_step"], DUALVIEW_PART_STEP9)
        self.assertEqual(control["output_contract"], DUALVIEW_PART_SHUFFLED_PN_CONTRACT)
        self.assertLess(control["identity_fraction"], 1.0)
        np.testing.assert_array_equal(shuffled.pn_reco_tokens[: len(permutation)], dataset.pn_reco_tokens[permutation])
        np.testing.assert_array_equal(shuffled.pn_reco_mask[: len(permutation)], dataset.pn_reco_mask[permutation])
        np.testing.assert_array_equal(
            shuffled.pn_reco_confidence[: len(permutation)],
            dataset.pn_reco_confidence[permutation],
        )

    def test_training_flag_applies_shuffled_pn_control_to_reported_datasets(self):
        train_dataset = self.make_dataset(n_jets=12, split="stack_train")
        val_dataset = self.make_dataset(n_jets=8, split="stack_val")
        final_dataset = self.make_dataset(n_jets=8, split="final_test")
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DualViewResidualTrainConfig(
                output_dir=tmpdir,
                confirm_split_settings=True,
                confirm_final_test=True,
                seed=19,
                batch_size=4,
                eval_batch_size=4,
                epochs=1,
                lr=1.0e-3,
                num_workers=0,
                device="cpu",
                amp=False,
                early_stop_patience=-1,
                selection_metric=DUALVIEW_PART_PRIMARY_METRIC,
                anchor_context_dim=8,
                pn_embed_dim=8,
                pn_layers=1,
                pn_heads=2,
                residual_hidden_dim=16,
                residual_layers=1,
                residual_dropout=0.0,
                freeze_anchor=True,
                shuffle_pn_view=True,
                pn_view_shuffle_seed=111,
            )
            report = train_dualview_residual_part(
                config,
                model=self.make_model(),
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                final_test_dataset=final_dataset,
            )

        self.assertTrue(report["shuffle_pn_view"])
        self.assertEqual(report["negative_control_step"], DUALVIEW_PART_STEP9)
        self.assertEqual(report["negative_control_contract"], DUALVIEW_PART_SHUFFLED_PN_CONTRACT)
        self.assertEqual(report["source"]["negative_control_contract"], DUALVIEW_PART_SHUFFLED_PN_CONTRACT)
        self.assertTrue(report["train_dataset"]["pn_view_shuffle_control"]["enabled"])
        self.assertTrue(report["val_dataset"]["pn_view_shuffle_control"]["enabled"])
        self.assertTrue(report["final_test_dataset"]["pn_view_shuffle_control"]["enabled"])


if __name__ == "__main__":
    unittest.main()
