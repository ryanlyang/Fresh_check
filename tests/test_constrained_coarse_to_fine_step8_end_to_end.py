from __future__ import annotations

from dataclasses import replace
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from teacher_logit_reco.constrained_coarse_to_fine import (
    A0_HLT_BASELINE,
    A1_LARGE_HLT_BASELINE,
    A2_OFFLINE_REFERENCE,
    A4_EXTRA_ATTENTION_HLT,
    C5_B1,
    COARSE_TO_FINE_TRAIN_CONTRACT,
    D5_B1,
    D8_MULTIDEPTH,
    PHASE_FROZEN_RECONSTRUCTOR,
    PHASE_FUSION_WARMUP,
    PHASE_TERMINAL_DECODER,
    PHASE_UPPER_HIERARCHY,
    EndToEndLossConfig,
    EndToEndScheduleConfig,
    EndToEndTrainConfig,
    ParticleStreamInput,
    ReconstructorSourceSpec,
    apply_end_to_end_phase,
    build_c_tier_reconstructor,
    build_end_to_end_optimizer,
    build_end_to_end_tagger,
    compute_end_to_end_loss,
    default_hierarchy_target_layout,
    end_to_end_phase,
    resolve_reconstructor_sources,
)


def _small_reconstructor():
    layout = default_hierarchy_target_layout(radial_boundary=0.16, coordinate_extent=0.8)
    return build_c_tier_reconstructor(
        C5_B1,
        hierarchy_overrides={
            "d_model": 32,
            "num_heads": 4,
            "encoder_layers": 1,
            "pool_layers": 1,
            "decoder_layers_per_level": 1,
            "ffn_multiplier": 2.0,
            "pair_hidden_dim": 8,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        },
        slot_overrides={
            "ffn_multiplier": 2.0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        },
        layout=layout,
    )


def _write_checkpoint(path: Path, model, *, seed: int = 0) -> None:
    if seed:
        with torch.no_grad():
            next(model.parameters()).add_(float(seed) * 1.0e-4)
    provenance = {
        "model_train": {
            "source_manifest_hash": "manifest",
            "hlt_content_hash": "hlt",
            "layout": model.hierarchy.layout.to_dict(),
        },
        "model_val": {
            "source_manifest_hash": "manifest",
            "hlt_content_hash": "hlt-val",
            "layout": model.hierarchy.layout.to_dict(),
        },
    }
    torch.save(
        {
            "checkpoint_contract": COARSE_TO_FINE_TRAIN_CONTRACT,
            "checkpoint_role": "best_model_val",
            "model_state_dict": model.state_dict(),
            "model": {
                "family": "C",
                "variant": model.slot_decoder.config.variant,
                "hierarchy_config": model.hierarchy.config.to_dict(),
                "slot_config": model.slot_decoder.config.to_dict(),
            },
            "provenance": provenance,
        },
        path,
    )


def _toy_hlt() -> tuple[ParticleStreamInput, torch.Tensor, torch.Tensor]:
    torch.manual_seed(808)
    batch, particles = 2, 6
    points = 0.12 * torch.randn(batch, 2, particles)
    features = torch.randn(batch, 17, particles)
    features[:, 6:11] = 0.0
    features[:, 6] = 1.0
    mask = torch.ones(batch, 1, particles, dtype=torch.bool)
    pt = torch.linspace(6.0, 24.0, particles).expand(batch, -1)
    eta, phi = points[:, 0], points[:, 1]
    vectors = torch.stack(
        (pt * torch.cos(phi), pt * torch.sin(phi), pt * torch.sinh(eta), pt * torch.cosh(eta)),
        dim=1,
    )
    return ParticleStreamInput(points, features, vectors, mask), torch.zeros(batch), torch.zeros(batch)


class ConstrainedCoarseToFineStep8Tests(unittest.TestCase):
    def test_a0_cli_accepts_an_empty_reconstructor_source_list(self):
        from scripts import train_constrained_coarse_to_fine_end_to_end as cli

        argv = [
            "train_constrained_coarse_to_fine_end_to_end.py",
            "--output-dir", "out",
            "--manifest", "manifest.json.gz",
            "--hlt-cache-dir", "hlt",
            "--offline-cache-dir", "offline",
            "--target-cache-dir", "targets",
            "--variant", "A0",
            "--allow-random-hlt-start",
        ]
        with patch.object(sys, "argv", argv), patch.object(
            cli,
            "train_end_to_end_tagger",
            return_value={"ok": True},
        ) as mocked:
            self.assertEqual(cli.main(), 0)
        config = mocked.call_args.args[0]
        self.assertEqual(config.variant, A0_HLT_BASELINE)
        self.assertEqual(config.reconstructor_sources, ())

    def test_a_tier_controls_are_source_free_and_architecturally_explicit(self):
        overrides = {
            "d_model": 32,
            "num_heads": 4,
            "hlt_encoder_layers": 1,
            "hlt_pool_layers": 1,
            "pair_hidden_dim": 8,
            "ffn_multiplier": 2.0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        }
        models = {
            variant: build_end_to_end_tagger(variant, (), fusion_overrides=overrides)[0]
            for variant in (
                A0_HLT_BASELINE,
                A1_LARGE_HLT_BASELINE,
                A2_OFFLINE_REFERENCE,
                A4_EXTRA_ATTENTION_HLT,
            )
        }
        self.assertTrue(all(not model.reconstructors for model in models.values()))
        self.assertIsNone(models[A0_HLT_BASELINE].tagger.extra_hlt_attention)
        self.assertIsNotNone(models[A4_EXTRA_ATTENTION_HLT].tagger.extra_hlt_attention)
        self.assertGreater(
            sum(parameter.numel() for parameter in models[A4_EXTRA_ATTENTION_HLT].parameters()),
            sum(parameter.numel() for parameter in models[A0_HLT_BASELINE].parameters()),
        )

    def test_training_config_requires_explicit_trusted_or_random_hlt_start(self):
        kwargs = dict(
            output_dir="out",
            manifest_path="manifest.json.gz",
            hlt_cache_dir="hlt",
            offline_cache_dir="offline",
            target_cache_dir="targets",
            reconstructor_sources=(
                ReconstructorSourceSpec("c5_b1", "checkpoint.pt", ("c5_b1",)),
            ),
            variant=D5_B1,
        )
        with self.assertRaisesRegex(ValueError, "trusted HLT warm start"):
            EndToEndTrainConfig(**kwargs)
        config = EndToEndTrainConfig(**kwargs, allow_random_hlt_start=True)
        self.assertTrue(config.to_dict()["allow_random_hlt_start"])

    def test_declared_alias_requires_exact_checkpoint_reuse_and_drops_duplicate_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "shared.pt"
            different = root / "different.pt"
            _write_checkpoint(shared, _small_reconstructor())
            _write_checkpoint(different, _small_reconstructor(), seed=1)
            target = ReconstructorSourceSpec("c5_b3", str(shared), ("c5_b3",))
            alias = ReconstructorSourceSpec(
                "canonical", str(shared), ("canonical",), alias_of="c5_b3"
            )
            resolved = resolve_reconstructor_sources((alias, target))
            self.assertEqual(resolved.view_names, ("c5_b3",))
            self.assertEqual(len(resolved.modules), 1)
            self.assertEqual(resolved.aliases, {"canonical": "c5_b3"})
            with self.assertRaisesRegex(ValueError, "does not reuse"):
                resolve_reconstructor_sources(
                    (
                        ReconstructorSourceSpec(
                            "canonical", str(different), ("canonical",), alias_of="c5_b3"
                        ),
                        target,
                    )
                )

    def test_phase_schedule_and_optimizer_use_explicit_decoder_lr(self):
        schedule = EndToEndScheduleConfig(
            fusion_only_warmup_epochs=1,
            frozen_reconstructor_epochs=2,
            terminal_decoder_epochs=2,
            upper_hierarchy_epochs=1,
            terminal_decoder_lr_scale=0.075,
        )
        self.assertEqual(end_to_end_phase(0, D8_MULTIDEPTH, schedule).name, PHASE_FUSION_WARMUP)
        self.assertEqual(end_to_end_phase(0, D5_B1, schedule).name, PHASE_FROZEN_RECONSTRUCTOR)
        self.assertEqual(end_to_end_phase(2, D5_B1, schedule).name, PHASE_TERMINAL_DECODER)
        self.assertEqual(end_to_end_phase(4, D5_B1, schedule).name, PHASE_UPPER_HIERARCHY)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "source.pt"
            _write_checkpoint(checkpoint, _small_reconstructor())
            model, _ = build_end_to_end_tagger(
                D5_B1,
                (ReconstructorSourceSpec("c5_b1", str(checkpoint), ("c5_b1",)),),
                fusion_overrides={
                    "d_model": 32,
                    "num_heads": 4,
                    "hlt_encoder_layers": 1,
                    "hlt_pool_layers": 1,
                    "pseudo_local_layers": 1,
                    "pseudo_global_layers": 1,
                    "fusion_layers": 1,
                    "pair_hidden_dim": 8,
                    "ffn_multiplier": 2.0,
                    "dropout": 0.0,
                    "attention_dropout": 0.0,
                    "pseudo_view_dropout": 0.0,
                },
            )
            phase = end_to_end_phase(2, D5_B1, schedule)
            metadata = apply_end_to_end_phase(model, phase)
            optimizer = build_end_to_end_optimizer(model, schedule)
            groups = {row["group_name"]: row for row in optimizer.param_groups}
            slot = next(row for name, row in groups.items() if name.endswith("slot_decoder"))
            self.assertAlmostEqual(slot["lr"], schedule.tagger_learning_rate * 0.075)
            self.assertGreater(metadata["group_parameter_counts"]["tagger.fusion_and_head"], 0)
            self.assertTrue(all(not parameter.requires_grad for parameter in model.reconstructors["reconstructor_0"].hierarchy.parameters()))

    def test_live_render_path_backpropagates_tagging_gradient_to_terminal_decoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "source.pt"
            _write_checkpoint(checkpoint, _small_reconstructor())
            model, _ = build_end_to_end_tagger(
                D5_B1,
                (ReconstructorSourceSpec("c5_b1", str(checkpoint), ("c5_b1",)),),
                fusion_overrides={
                    "d_model": 32,
                    "num_heads": 4,
                    "hlt_encoder_layers": 1,
                    "hlt_pool_layers": 1,
                    "pseudo_local_layers": 1,
                    "pseudo_global_layers": 1,
                    "fusion_layers": 1,
                    "pair_hidden_dim": 8,
                    "ffn_multiplier": 2.0,
                    "dropout": 0.0,
                    "attention_dropout": 0.0,
                    "pseudo_view_dropout": 0.0,
                },
            )
            schedule = EndToEndScheduleConfig(
                fusion_only_warmup_epochs=0,
                frozen_reconstructor_epochs=0,
                terminal_decoder_epochs=2,
                upper_hierarchy_epochs=0,
                reconstruction_weight=0.0,
            )
            phase = end_to_end_phase(0, D5_B1, schedule)
            apply_end_to_end_phase(model, phase)
            hlt, reference_eta, reference_phi = _toy_hlt()
            output = model.forward_detailed(
                hlt,
                reference_eta=reference_eta,
                reference_phi=reference_phi,
            )
            self.assertEqual(tuple(output.tagger.diagnostics["view_names"]), ("c5_b1",))
            reconstruction_output = output.reconstructors["reconstructor_0"]
            rendered = output.renders["reconstructor_0"]
            field_dim = int(reconstruction_output.hierarchy.global_accounting.shape[-1])
            reconstruction_batch = {
                "labels": torch.tensor([0, 1]),
                "global_accounting": reconstruction_output.hierarchy.global_accounting.detach(),
                "level1_accounting": reconstruction_output.hierarchy.level(1).accounting.detach(),
                "level2_accounting": torch.zeros(2, 32, field_dim),
                "level3_accounting": torch.zeros(2, 128, field_dim),
                "offline_tokens": rendered.tokens[:, 0].detach(),
                "offline_mask": rendered.mask[:, 0].detach(),
                "final_cell_indices": (
                    rendered.token_cell_indices.long()[None, :].expand(2, -1) * 16
                ),
                "reference_eta": reference_eta,
                "reference_phi": reference_phi,
            }
            reconstruction_loss = compute_end_to_end_loss(
                output,
                reconstruction_batch,
                replace(phase, reconstruction_weight=0.1),
                EndToEndLossConfig(),
            )
            self.assertTrue(torch.isfinite(reconstruction_loss.loss))
            self.assertIn("loss.reconstruction_unique_source_mean", reconstruction_loss.metrics)
            compute_end_to_end_loss(
                output,
                {"labels": torch.tensor([0, 1])},
                phase,
                EndToEndLossConfig(),
            ).loss.backward()
            slot_grad = sum(
                float(parameter.grad.detach().abs().sum())
                for parameter in model.reconstructors["reconstructor_0"].slot_decoder.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(slot_grad, 0.0)
            self.assertTrue(
                all(parameter.grad is None for parameter in model.reconstructors["reconstructor_0"].hierarchy.parameters())
            )
            with self.assertRaisesRegex(ValueError, "teacher_logits"):
                compute_end_to_end_loss(
                    output,
                    {"labels": torch.tensor([0, 1])},
                    phase,
                    EndToEndLossConfig(kd_loss_weight=0.5),
                )


if __name__ == "__main__":
    unittest.main()
