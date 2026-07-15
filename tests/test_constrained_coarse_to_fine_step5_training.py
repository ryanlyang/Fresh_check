from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from scripts import train_constrained_coarse_to_fine as training_cli

from teacher_logit_reco.constrained_coarse_to_fine import (
    ACCOUNTING_FIELD_NAMES,
    B1_GLOBAL_8,
    B4_NO_MOMENTS,
    B5_NO_COMPOSITION,
    B6_NO_COUNTS,
    MOMENT_FIELD_NAMES,
    PID_COUNT_INDICES,
    PID_PT_INDICES,
    CoarseToFineReconstructorConfig,
    CoarseToFineTrainConfig,
    HIERARCHY_RECONSTRUCTION_LOSS_CONTRACT,
    HierarchyReconstructionLossConfig,
    ParticleSlotDecoderConfig,
    ParticleSlotLossConfig,
    build_coarse_to_fine_reconstructor,
    compute_hierarchy_reconstruction_loss,
)
from teacher_logit_reco.constrained_coarse_to_fine.train import (
    _grad_norm_if_finite,
    _loss_configs,
    _write_curves_csv,
)


def _toy_model_and_inputs():
    torch.manual_seed(507)
    batch, particles = 2, 6
    points = 0.15 * torch.randn(batch, 2, particles)
    features = torch.randn(batch, 17, particles)
    features[:, 6:11, :] = 0.0
    features[:, 6, :] = 1.0
    mask = torch.ones(batch, 1, particles, dtype=torch.bool)
    pt = torch.linspace(5.0, 20.0, particles).expand(batch, -1)
    eta = points[:, 0]
    phi = points[:, 1]
    vectors = torch.stack(
        (
            pt * torch.cos(phi),
            pt * torch.sin(phi),
            pt * torch.sinh(eta),
            pt * torch.cosh(eta),
        ),
        dim=1,
    )
    config = CoarseToFineReconstructorConfig(
        variant=B1_GLOBAL_8,
        d_model=32,
        num_heads=4,
        encoder_layers=1,
        pool_layers=1,
        decoder_layers_per_level=1,
        pair_hidden_dim=16,
        ffn_multiplier=2.0,
        dropout=0.0,
        attention_dropout=0.0,
    )
    return build_coarse_to_fine_reconstructor(config), (points, features, vectors, mask)


class ConstrainedCoarseToFineStep5TrainingTests(unittest.TestCase):
    def test_hierarchy_loss_is_exact_at_target_and_reports_required_diagnostics(self):
        model, inputs = _toy_model_and_inputs()
        output = model(*inputs)
        targets = {
            "global_accounting": output.global_accounting.detach().clone(),
            "level1_accounting": output.level(1).accounting.detach().clone(),
        }
        config = HierarchyReconstructionLossConfig(uncertainty_weight=0.0)
        exact = compute_hierarchy_reconstruction_loss(output, targets, config)
        shifted = compute_hierarchy_reconstruction_loss(
            output,
            {name: value * 1.25 for name, value in targets.items()},
            config,
        )
        self.assertEqual(exact.detached_summary()["contract"], HIERARCHY_RECONSTRUCTION_LOSS_CONTRACT)
        self.assertLess(float(exact.loss.detach()), 1.0e-7)
        self.assertGreater(float(shifted.loss.detach()), float(exact.loss.detach()))
        required = {
            "global_total_pT_mae",
            "global_total_pT_relative_mae",
            "global_count_mae",
            "global_composition_mae",
            "global_axis_delta_mae",
            "global_width_mae",
            "level1_pT_allocation_kl",
            "level1_high_pT_cell_recall",
            "level1_parent_child_consistency_max",
        }
        self.assertTrue(required.issubset(exact.metrics))

    def test_losses_do_not_bypass_channel_ablation_masks(self):
        _, inputs = _toy_model_and_inputs()
        field_index = {name: index for index, name in enumerate(ACCOUNTING_FIELD_NAMES)}
        cases = (
            (B4_NO_MOMENTS, [field_index[name] for name in MOMENT_FIELD_NAMES], "global_relative"),
            (
                B6_NO_COUNTS,
                [field_index["expected_constituent_count"], *PID_COUNT_INDICES],
                "global_relative",
            ),
        )
        for variant, changed_indices, _ in cases:
            with self.subTest(variant=variant):
                config = CoarseToFineReconstructorConfig(
                    variant=variant,
                    d_model=32,
                    num_heads=4,
                    encoder_layers=1,
                    pool_layers=1,
                    decoder_layers_per_level=1,
                    pair_hidden_dim=16,
                    ffn_multiplier=2.0,
                    dropout=0.0,
                    attention_dropout=0.0,
                )
                output = build_coarse_to_fine_reconstructor(config)(*inputs)
                targets = {"global_accounting": output.global_accounting.detach().clone()}
                targets["global_accounting"][:, changed_indices] += 100.0
                for level in output.levels:
                    target = level.accounting.detach().clone()
                    target[..., changed_indices] += 100.0
                    targets[f"level{level.level}_accounting"] = target
                loss = compute_hierarchy_reconstruction_loss(
                    output,
                    targets,
                    HierarchyReconstructionLossConfig(uncertainty_weight=0.0),
                )
                self.assertLess(float(loss.loss.detach()), 1.0e-7)

        composition_model = build_coarse_to_fine_reconstructor(
            CoarseToFineReconstructorConfig(
                variant=B5_NO_COMPOSITION,
                d_model=32,
                num_heads=4,
                encoder_layers=1,
                pool_layers=1,
                decoder_layers_per_level=1,
                pair_hidden_dim=16,
                ffn_multiplier=2.0,
                dropout=0.0,
                attention_dropout=0.0,
            )
        )
        output = composition_model(*inputs)
        targets = {"global_accounting": output.global_accounting.detach().clone()}
        targets["global_accounting"][:, list(PID_PT_INDICES)] += 25.0
        # Preserve total pT while perturbing the explicitly unsupervised composition channels.
        for level in output.levels:
            target = level.accounting.detach().clone()
            target[..., list(PID_PT_INDICES)] += 25.0
            targets[f"level{level.level}_accounting"] = target
        loss = compute_hierarchy_reconstruction_loss(
            output,
            targets,
            HierarchyReconstructionLossConfig(uncertainty_weight=0.0),
        )
        self.assertEqual(float(loss.components["global_relative"].detach()), 0.0)

    def test_hierarchy_loss_backpropagates_and_gradient_guard_rejects_nan(self):
        model, inputs = _toy_model_and_inputs()
        output = model(*inputs)
        targets = {
            "global_accounting": output.global_accounting.detach() * 1.1,
            "level1_accounting": output.level(1).accounting.detach() * 1.1,
        }
        loss = compute_hierarchy_reconstruction_loss(
            output,
            targets,
            HierarchyReconstructionLossConfig(uncertainty_weight=0.0),
        ).loss
        loss.backward()
        norm = _grad_norm_if_finite(model)
        self.assertIsNotNone(norm)
        self.assertTrue(torch.isfinite(norm))
        first = next(parameter for parameter in model.parameters() if parameter.grad is not None)
        first.grad.flatten()[0] = float("nan")
        self.assertIsNone(_grad_norm_if_finite(model))

    def test_config_forbids_final_test_supervision(self):
        base = dict(
            output_dir="out",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt",
            offline_cache_dir="offline",
            target_cache_dir="targets",
        )
        with self.assertRaisesRegex(ValueError, "final_test"):
            CoarseToFineTrainConfig(**base, stack_val_split="final_test")
        config = CoarseToFineTrainConfig(**base, variant="C5-B2", amp=False)
        self.assertEqual(config.to_dict()["resolved_variant"], "C5-B2")
        no_slot = CoarseToFineTrainConfig(**base, variant="C5", slot_loss_weight=0.0, amp=False)
        self.assertEqual(no_slot.slot_loss_weight, 0.0)
        direct = CoarseToFineTrainConfig(
            **base,
            variant="C5",
            constrain_slot_accounting=False,
            direct_particle_decoding=True,
            hierarchy_loss_weight=0.0,
            amp=False,
        )
        self.assertTrue(direct.direct_particle_decoding)
        self.assertEqual(direct.hierarchy_loss_weight, 0.0)
        _, direct_slot_loss = _loss_configs(direct, "C", "C5_uncertainty")
        self.assertEqual(direct_slot_loss.accounting_consistency_weight, 0.0)
        self.assertEqual(direct_slot_loss.dust_weight, 0.0)
        with self.assertRaisesRegex(ValueError, "direct particle decoding"):
            CoarseToFineTrainConfig(**base, variant="C5", direct_particle_decoding=True, amp=False)

    def test_full_precision_is_the_default_and_amp_is_explicit(self):
        base = dict(
            output_dir="out",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt",
            offline_cache_dir="offline",
            target_cache_dir="targets",
        )
        self.assertFalse(CoarseToFineTrainConfig(**base).amp)
        parser = training_cli.build_parser()
        required = [
            "--output-dir", "out",
            "--manifest", "manifest.json.gz",
            "--hlt-cache-dir", "hlt",
            "--offline-cache-dir", "offline",
            "--target-cache-dir", "targets",
        ]
        self.assertFalse(parser.parse_args(required).amp)
        self.assertTrue(parser.parse_args([*required, "--amp"]).amp)
        self.assertFalse(parser.parse_args([*required, "--no-amp"]).amp)

    def test_uncertainty_precision_is_bounded_for_stable_real_batch_training(self):
        config = HierarchyReconstructionLossConfig()
        self.assertEqual(config.uncertainty_log_sigma_floor, -2.0)
        self.assertEqual(config.uncertainty_weight, 0.01)
        self.assertLessEqual(
            float(torch.exp(torch.tensor(-2.0 * config.uncertainty_log_sigma_floor))),
            55.0,
        )
        self.assertEqual(ParticleSlotDecoderConfig().uncertainty_min, -2.0)
        self.assertEqual(ParticleSlotLossConfig().uncertainty_weight, 0.05)

    def test_diagnostic_csv_flattens_train_and_model_val_metrics(self):
        curves = [
            {
                "epoch": 0,
                "train": {"loss.total": 2.0, "nonfinite_batches_skipped": 1},
                "model_val": {"loss.total": 1.5, "hierarchy.metric.global_count_mae": 0.4},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostics" / "reconstruction_metrics.csv"
            _write_curves_csv(path, curves)
            text = path.read_text(encoding="utf-8")
        self.assertIn("model_val.loss.total", text)
        self.assertIn("hierarchy.metric.global_count_mae", text)
        self.assertIn("nonfinite_batches_skipped", text)


if __name__ == "__main__":
    unittest.main()
