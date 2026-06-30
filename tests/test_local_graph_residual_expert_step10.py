import unittest

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
    LOCAL_GRAPH_RESIDUAL_LOSS_MODES,
    LocalGraphBaselineLogitBlock,
    LocalGraphResidualExpertTrainConfig,
    LocalGraphResidualLossConfig,
    baseline_condition_reference_from_block,
    baseline_condition_features,
    compute_local_graph_residual_loss,
    residual_boundary_masks,
    select_alpha_shrinkage,
    verify_baseline_logit_block_alignment,
    verify_baseline_logit_cache_family,
)
from teacher_logit_reco.local_graph_part.residual_train import _baseline_arrays_for_batch


_TORCH = require_torch()


def make_block(labels=(0, 1, 0, 1), margins=(-1.0, 0.8, 0.2, 0.5)):
    labels_np = np.asarray(labels, dtype=np.int64)
    margins_np = np.asarray(margins, dtype=np.float32)
    logits = np.stack((-0.5 * margins_np, 0.5 * margins_np), axis=1).astype(np.float32)
    block = LocalGraphBaselineLogitBlock(
        split="model_train",
        logits=logits,
        labels=labels_np,
        indices=np.arange(labels_np.shape[0], dtype=np.int64),
        metadata={
            "source": "step10_unit_test",
            "checkpoint": "toy_hlt_part_baseline.pt",
            "checkpoint_variant": "hlt_part_baseline",
            "checkpoint_epoch": 1,
            "run_report": "toy_hlt_part_baseline/run_report.json",
            "dataset": {
                "split": "model_train",
                "n_jets": int(labels_np.shape[0]),
                "hlt_content_hash": "hlt-a",
                "jet_identity_hash": "jets-a",
                "hlt_params": {"strength": 0.6},
            },
        },
    )
    block.metadata["condition_reference"] = baseline_condition_reference_from_block(block, source_split="model_train")
    return block


class LocalGraphResidualExpertStep10ContractTests(unittest.TestCase):
    def test_baseline_cache_alignment_uses_indices_and_rejects_label_mismatch(self):
        torch = require_torch()
        block = make_block()

        z_base, condition = _baseline_arrays_for_batch(
            block,
            torch.tensor([2, 0], dtype=torch.long),
            labels=torch.tensor([0, 0], dtype=torch.long),
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(z_base.shape), (2,))
        self.assertEqual(tuple(condition.shape), (2, 6))
        self.assertAlmostEqual(float(z_base[0]), float(block.z_base[2]), places=6)
        self.assertAlmostEqual(float(z_base[1]), float(block.z_base[0]), places=6)

        with self.assertRaisesRegex(ValueError, "labels do not align"):
            _baseline_arrays_for_batch(
                block,
                torch.tensor([1], dtype=torch.long),
                labels=torch.tensor([0], dtype=torch.long),
                device=torch.device("cpu"),
            )

        with self.assertRaisesRegex(IndexError, "missing dataset indices"):
            _baseline_arrays_for_batch(
                block,
                torch.tensor([99], dtype=torch.long),
                labels=torch.tensor([0], dtype=torch.long),
                device=torch.device("cpu"),
            )

    def test_baseline_cache_alignment_rejects_hash_mismatch(self):
        block = make_block()
        ok = verify_baseline_logit_block_alignment(
            block,
            {
                "split": "model_train",
                "n_jets": 4,
                "hlt_content_hash": "hlt-a",
                "jet_identity_hash": "jets-a",
                "hlt_params": {"strength": 0.6},
            },
            split="model_train",
            dataset_length=4,
        )
        self.assertTrue(ok["ok"])

        with self.assertRaisesRegex(ValueError, "hlt_content_hash mismatch"):
            verify_baseline_logit_block_alignment(
                block,
                {
                    "split": "model_train",
                    "n_jets": 4,
                    "hlt_content_hash": "wrong",
                    "jet_identity_hash": "jets-a",
                    "hlt_params": {"strength": 0.6},
                },
                split="model_train",
                dataset_length=4,
            )

    def test_condition_reference_must_come_from_model_train(self):
        block = make_block()
        bad_block = make_block()
        bad_block.metadata["condition_reference"] = baseline_condition_reference_from_block(
            bad_block,
            source_split="final_test",
        )
        with self.assertRaisesRegex(ValueError, "source_split"):
            bad_block.condition_reference(require=True)
        with self.assertRaisesRegex(ValueError, "source_split"):
            verify_baseline_logit_cache_family((block, bad_block), require_condition_reference=True)

    def test_condition_feature_slice_uses_cached_array(self):
        block = make_block()
        reference = block.condition_reference(require=True)
        features, _ = baseline_condition_features(
            block.z_base,
            tau50=float(reference["tau50"]),
            tau30=float(reference["tau30"]),
            near_scale=float(reference["near_tau50_scale"]),
        )
        cached = LocalGraphBaselineLogitBlock(
            split=block.split,
            logits=block.logits,
            labels=block.labels,
            indices=block.indices,
            metadata=dict(block.metadata),
            condition_features_array=features,
        )
        sliced, _ = cached.condition_features_for_positions(np.asarray([2, 0]), require_reference=True)
        np.testing.assert_allclose(sliced, features[[2, 0]])

    def test_baseline_cache_shape_validation_rejects_bad_logits(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            LocalGraphBaselineLogitBlock(
                split="model_train",
                logits=np.zeros((4,), dtype=np.float32),
                labels=np.asarray([0, 1, 0, 1], dtype=np.int64),
                indices=np.arange(4, dtype=np.int64),
            )

        with self.assertRaisesRegex(ValueError, "length mismatch"):
            LocalGraphBaselineLogitBlock(
                split="model_train",
                logits=np.zeros((4, 2), dtype=np.float32),
                labels=np.asarray([0, 1], dtype=np.int64),
                indices=np.arange(4, dtype=np.int64),
            )

    def test_all_loss_modes_are_finite_and_backpropagate_to_residual_and_alpha(self):
        torch = require_torch()
        labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32)
        baseline = torch.tensor([1.0, 0.2, -0.5, 0.8, 0.1, -1.0], dtype=torch.float32)
        for mode in LOCAL_GRAPH_RESIDUAL_LOSS_MODES:
            with self.subTest(mode=mode):
                residual = torch.tensor([0.1, 0.2, 0.3, -0.4, -0.2, 0.1], dtype=torch.float32, requires_grad=True)
                alpha = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
                fused = baseline + alpha * residual
                loss = compute_local_graph_residual_loss(
                    fused_logit=fused,
                    labels=labels,
                    baseline_logit=baseline,
                    residual_logit=residual,
                    alpha=alpha,
                    tau50=0.2,
                    config=LocalGraphResidualLossConfig(
                        mode=mode,
                        default_tau50=0.2,
                        cvar_top_fraction=1.0,
                        bce_boundary_scale=1.0,
                    ),
                )
                self.assertTrue(torch.isfinite(loss.total_loss))
                loss.total_loss.backward()
                self.assertIsNotNone(residual.grad)
                self.assertIsNotNone(alpha.grad)
                self.assertTrue(torch.isfinite(residual.grad).all())
                self.assertTrue(torch.isfinite(alpha.grad).all())

    def test_boundary_selection_picks_signal_band_and_hard_background(self):
        torch = require_torch()
        labels = torch.tensor([1, 1, 1, 1, 0, 0, 0], dtype=torch.float32)
        baseline = torch.tensor([2.0, 1.0, 0.0, -1.0, 0.8, 0.1, -1.5], dtype=torch.float32)
        config = LocalGraphResidualLossConfig(
            mode=LOCAL_GRAPH_RESIDUAL_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
            signal_boundary_quantile_low=0.25,
            signal_boundary_quantile_high=0.75,
            hard_background_fraction=0.2,
        )

        signal_mask, background_mask = residual_boundary_masks(labels, baseline, tau50=0.5, config=config)

        self.assertEqual(torch.nonzero(signal_mask, as_tuple=False).reshape(-1).tolist(), [1, 2])
        self.assertEqual(torch.nonzero(background_mask, as_tuple=False).reshape(-1).tolist(), [4])

    def test_boundary_selection_falls_back_to_top_background_when_none_cross_threshold(self):
        torch = require_torch()
        labels = torch.tensor([1, 1, 0, 0, 0], dtype=torch.float32)
        baseline = torch.tensor([1.0, 0.0, 0.3, 0.1, -1.0], dtype=torch.float32)
        config = LocalGraphResidualLossConfig(hard_background_fraction=0.2)

        _signal_mask, background_mask = residual_boundary_masks(labels, baseline, tau50=0.5, config=config)

        self.assertEqual(torch.nonzero(background_mask, as_tuple=False).reshape(-1).tolist(), [2])

    def test_alpha_shrinkage_collapses_to_zero_when_residual_hurts(self):
        report = select_alpha_shrinkage(
            labels=[1, 1, 0, 0],
            baseline_logit=[0.6, 0.5, 0.2, -1.0],
            residual_logit=[-1.0, -1.0, 2.0, 0.0],
            alpha_grid=(0.0, 0.5, 1.0),
            target_signal_efficiency=0.5,
        )

        self.assertEqual(report["selected_alpha"], 0.0)
        self.assertTrue(report["collapsed_to_zero"])
        self.assertEqual(report["selected_fpr"], report["baseline_fpr"])

    def test_residual_training_config_rejects_stack_or_final_split_leakage(self):
        with self.assertRaisesRegex(ValueError, "model_train and selects only on model_val"):
            LocalGraphResidualExpertTrainConfig(
                output_dir="out",
                hlt_cache_dir="hlt",
                baseline_logit_cache_dir="baseline",
                train_split="stack_train",
                val_split="model_val",
                confirm_split_settings=True,
            )

        with self.assertRaisesRegex(ValueError, "model_train and selects only on model_val"):
            LocalGraphResidualExpertTrainConfig(
                output_dir="out",
                hlt_cache_dir="hlt",
                baseline_logit_cache_dir="baseline",
                train_split="model_train",
                val_split="final_test",
                confirm_split_settings=True,
            )


if __name__ == "__main__":
    unittest.main()
