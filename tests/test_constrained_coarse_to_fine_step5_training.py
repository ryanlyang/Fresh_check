from __future__ import annotations

import tempfile
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from scripts import train_constrained_coarse_to_fine as training_cli

from teacher_logit_reco.constrained_coarse_to_fine import (
    ACCOUNTING_FIELD_NAMES,
    B1_GLOBAL_8,
    B4_NO_MOMENTS,
    B5_NO_COMPOSITION,
    B6_NO_COUNTS,
    C4_HUNGARIAN,
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
    _SplitSource,
    _build_shard_loader,
    _grad_norm_if_finite,
    _input_pipeline_metadata,
    _loader_order_seed,
    _loader_worker_seed,
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
    def test_streamed_split_source_exposes_hlt_row_count_without_labels_attribute(self):
        source = _SplitSource(
            split="model_train",
            hlt_view=SimpleNamespace(labels=np.zeros(7, dtype=np.int64)),
            offline_view=None,
            target_metadata={},
            layout=None,
            provenance={},
        )
        self.assertFalse(hasattr(source, "labels"))
        self.assertEqual(source.n_jets, 7)

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

    def test_b4_zero_moment_diagnostics_have_finite_backward_gradients(self):
        """B4's exact zero moments must not expose sqrt(0)'s infinite derivative."""

        _, inputs = _toy_model_and_inputs()
        config = CoarseToFineReconstructorConfig(
            variant=B4_NO_MOMENTS,
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
        model = build_coarse_to_fine_reconstructor(config)
        output = model(*inputs)
        targets = {"global_accounting": output.global_accounting.detach() * 1.2 + 0.01}
        for level in output.levels:
            targets[f"level{level.level}_accounting"] = level.accounting.detach() * 1.2 + 0.01
        loss = compute_hierarchy_reconstruction_loss(output, targets).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        norm = _grad_norm_if_finite(model)
        self.assertIsNotNone(norm)
        self.assertTrue(torch.isfinite(norm))

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

    def test_config_requires_positive_progress_heartbeat_interval(self):
        base = dict(
            output_dir="out",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt",
            offline_cache_dir="offline",
            target_cache_dir="targets",
        )
        with self.assertRaisesRegex(ValueError, "progress_interval_batches"):
            CoarseToFineTrainConfig(**base, progress_interval_batches=0)

    def test_prefetch_requires_workers_and_per_shard_loader_keeps_the_established_order_seed(self):
        base = dict(
            output_dir="out",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt",
            offline_cache_dir="offline",
            target_cache_dir="targets",
            seed=412,
            batch_size=3,
            eval_batch_size=4,
            pin_memory=False,
        )
        with self.assertRaisesRegex(ValueError, "prefetch_factor requires"):
            CoarseToFineTrainConfig(**base, num_workers=0, prefetch_factor=2)

        config = CoarseToFineTrainConfig(**base, num_workers=0)
        dataset = TensorDataset(torch.arange(11, dtype=torch.int64))
        epoch, shard_index = 3, 5
        first_loader = _build_shard_loader(
            dataset,
            config,
            train=True,
            epoch=epoch,
            shard_index=shard_index,
        )
        second_loader = _build_shard_loader(
            dataset,
            config,
            train=True,
            epoch=epoch,
            shard_index=shard_index,
        )
        first_rows = torch.cat([batch[0] for batch in first_loader])
        second_rows = torch.cat([batch[0] for batch in second_loader])
        # DataLoader consumes one generator draw to seed its iterator before
        # RandomSampler produces the shuffled order.  Match the established
        # pre-Step-3 loader behavior exactly rather than asserting a raw
        # randperm from the initial seed.
        reference_generator = torch.Generator().manual_seed(
            _loader_order_seed(config, epoch=epoch, shard_index=shard_index)
        )
        torch.empty((), dtype=torch.int64).random_(generator=reference_generator)
        expected = torch.randperm(len(dataset), generator=reference_generator)
        self.assertTrue(torch.equal(first_rows, expected))
        self.assertTrue(torch.equal(second_rows, expected))

    def test_prefetch_loader_has_reproducible_worker_seeds_without_persistent_workers(self):
        config = CoarseToFineTrainConfig(
            output_dir="out",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt",
            offline_cache_dir="offline",
            target_cache_dir="targets",
            seed=912,
            batch_size=2,
            eval_batch_size=2,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=False,
        )
        captured: dict[str, object] = {}

        def fake_loader(_dataset, **kwargs):
            captured.update(kwargs)
            return object()

        with patch("teacher_logit_reco.constrained_coarse_to_fine.train.DataLoader", fake_loader):
            _build_shard_loader(
                TensorDataset(torch.arange(4, dtype=torch.int64)),
                config,
                train=True,
                epoch=2,
                shard_index=7,
            )

        self.assertEqual(captured["prefetch_factor"], 2)
        self.assertNotIn("persistent_workers", captured)
        self.assertIn("worker_init_fn", captured)
        self.assertEqual(
            _loader_worker_seed(config, epoch=2, shard_index=7, worker_id=3),
            _loader_worker_seed(config, epoch=2, shard_index=7, worker_id=3),
        )
        self.assertNotEqual(
            _loader_worker_seed(config, epoch=2, shard_index=7, worker_id=3),
            _loader_worker_seed(config, epoch=2, shard_index=8, worker_id=3),
        )

        py_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            expected_seed = _loader_worker_seed(config, epoch=2, shard_index=7, worker_id=3)
            random.seed(expected_seed)
            np.random.seed(expected_seed)
            torch.manual_seed(expected_seed)
            expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
            worker_init = captured["worker_init_fn"]
            assert callable(worker_init)
            worker_init(3)
            first = (random.random(), float(np.random.rand()), float(torch.rand(())))
            worker_init(3)
            second = (random.random(), float(np.random.rand()), float(torch.rand(())))
        finally:
            random.setstate(py_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
        self.assertEqual(first, expected)
        self.assertEqual(first, second)
        self.assertEqual(
            _input_pipeline_metadata(config),
            {
                "contract": "constrained_c2f_per_shard_loader_v1",
                "loader_lifecycle": "one_loader_per_target_shard",
                "shuffle_seed": "run_seed + epoch * 1009 + shard_index",
                "worker_seed": "shuffle_seed + worker_id modulo 2**32",
                "num_workers": 4,
                "prefetch_factor": 2,
                "persistent_workers": False,
            },
        )

    def test_c4_slot_loss_receives_the_explicit_hungarian_execution_settings(self):
        config = CoarseToFineTrainConfig(
            output_dir="out",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt",
            offline_cache_dir="offline",
            target_cache_dir="targets",
            hungarian_workers=4,
            hungarian_executor="thread",
        )
        _, slot_config = _loss_configs(config, "C", C4_HUNGARIAN)
        self.assertIsNotNone(slot_config)
        assert slot_config is not None
        self.assertEqual(slot_config.matching_mode, "hungarian")
        self.assertEqual(slot_config.hungarian_workers, 4)
        self.assertEqual(slot_config.hungarian_executor, "thread")

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
