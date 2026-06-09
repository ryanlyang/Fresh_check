import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.jetclass_data import JetIdentity, JetView
from jetclass_fresh.reconstructor import (
    RECONSTRUCTOR_VARIANT_NAMES,
    ReconstructorVariantConfig,
    StageAReconstructorTrainConfig,
    all_reconstructor_variant_configs,
    detect_reconstructor_family_from_state_dict,
    get_reconstructor_variant_config,
    m2_base_variant_config,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from jetclass_fresh.reconstructor import (
        LegacyM2BaseReconstructor,
        ReconstructionOutput,
        PairedReconstructionDataset,
        build_reconstructor_for_state_dict,
        build_reconstructor,
        reconstruction_loss,
        train_stage_a_reconstructor,
    )
else:  # pragma: no cover - environment dependent
    torch = None


def make_views(n_jets=4, n_constits=5):
    hlt_tokens = np.zeros((n_jets, n_constits, 14), dtype=np.float32)
    offline_tokens = np.zeros((n_jets, n_constits, 14), dtype=np.float32)
    hlt_mask = np.zeros((n_jets, n_constits), dtype=bool)
    offline_mask = np.zeros((n_jets, n_constits), dtype=bool)
    labels = np.zeros((n_jets,), dtype=np.int64)
    jet_ids = [JetIdentity(file="paired.root", entry=i, label=0) for i in range(n_jets)]

    for jet_index in range(n_jets):
        hlt_mask[jet_index, :3] = True
        offline_mask[jet_index, :4] = True
        for part_index in range(4):
            pt = 2.0 + part_index + 0.1 * jet_index
            eta = 0.04 * part_index
            phi = 0.20 * part_index
            offline_tokens[jet_index, part_index, 0] = pt
            offline_tokens[jet_index, part_index, 1] = eta
            offline_tokens[jet_index, part_index, 2] = phi
            offline_tokens[jet_index, part_index, 3] = pt * np.cosh(eta)
            offline_tokens[jet_index, part_index, 4] = 1.0
            offline_tokens[jet_index, part_index, 5] = 1.0
            if part_index < 3:
                hlt_tokens[jet_index, part_index, 0] = pt * 0.85
                hlt_tokens[jet_index, part_index, 1] = eta + 0.01
                hlt_tokens[jet_index, part_index, 2] = phi - 0.01
                hlt_tokens[jet_index, part_index, 3] = pt * 0.85 * np.cosh(eta + 0.01)
                hlt_tokens[jet_index, part_index, 4] = 1.0
                hlt_tokens[jet_index, part_index, 5] = 1.0

    hlt_view = JetView(
        tokens=hlt_tokens,
        mask=hlt_mask,
        labels=labels,
        jet_ids=jet_ids,
        split="model_train",
        metadata={"view": "fixed_hlt", "hlt_content_hash": "synthetic_hlt"},
    )
    offline_view = JetView(
        tokens=offline_tokens,
        mask=offline_mask,
        labels=labels,
        jet_ids=jet_ids,
        split="model_train",
        metadata={"view": "offline", "source_manifest_hash": "synthetic_manifest"},
    )
    return hlt_view, offline_view


class ReconstructorStep7ConfigTests(unittest.TestCase):
    def test_detects_legacy_m2_simple_checkpoint_keys(self):
        state_dict = {
            "generated_query": object(),
            "edit_head.weight": object(),
            "token_encoder.0.weight": object(),
        }

        family = detect_reconstructor_family_from_state_dict(state_dict)

        self.assertEqual(family, "legacy_m2_simple")

    def test_detects_original_mechanism_checkpoint_keys(self):
        state_dict = {
            "token_encoder.layers.0.qkv.weight": object(),
            "generator_decoder.queries": object(),
        }

        family = detect_reconstructor_family_from_state_dict(state_dict)

        self.assertEqual(family, "m2_hybrid_original_mechanism")

    def test_step9_registers_all_seven_variants(self):
        configs = all_reconstructor_variant_configs()
        self.assertEqual(list(configs), RECONSTRUCTOR_VARIANT_NAMES)
        self.assertEqual(set(configs), set(RECONSTRUCTOR_VARIANT_NAMES))
        for name in RECONSTRUCTOR_VARIANT_NAMES:
            self.assertEqual(get_reconstructor_variant_config(name).name, name)

    def test_variant_knobs_match_protocol_intent(self):
        cfg = m2_base_variant_config()
        self.assertEqual(cfg.name, "m2_base")
        self.assertEqual(cfg.max_generated, 56)
        self.assertEqual(cfg.max_split_children, 2)
        self.assertEqual(cfg.matching_mode, "hungarian")
        self.assertAlmostEqual(cfg.set_matching_weight, 1.0)
        configs = all_reconstructor_variant_configs()
        self.assertLess(configs["m2_genlow"].max_generated, configs["m2_base"].max_generated)
        self.assertGreater(configs["m2_genhigh"].max_generated, configs["m2_base"].max_generated)
        self.assertEqual(configs["m2_topk60ish"].max_generated, 60)
        self.assertLess(configs["m2_budgetlite"].budget_count_weight, configs["m2_base"].budget_count_weight)
        self.assertGreater(configs["m2_consstrong"].pt_ratio_weight, configs["m2_base"].pt_ratio_weight)
        self.assertGreater(configs["m2_antioverlap"].anti_overlap_weight, 0.0)
        with self.assertRaises(ValueError):
            get_reconstructor_variant_config("not_a_variant")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ReconstructorStep7TorchTests(unittest.TestCase):
    def tiny_config(self):
        return ReconstructorVariantConfig(
            name="m2_base",
            max_generated=3,
            max_split_children=2,
            hidden_dim=16,
            global_dim=16,
            num_heads=4,
            num_encoder_layers=2,
            feedforward_dim=32,
            dropout=0.0,
            max_hlt_constits=5,
        )

    def test_build_reconstructor_for_legacy_state_dict_roundtrip(self):
        hlt_view, _ = make_views(n_jets=2, n_constits=5)
        cfg = self.tiny_config()
        legacy = LegacyM2BaseReconstructor(cfg)
        state_dict = legacy.state_dict()
        loaded = build_reconstructor_for_state_dict(state_dict, cfg)
        loaded.load_state_dict(state_dict, strict=True)

        hlt_tokens = torch.from_numpy(hlt_view.tokens).float()
        hlt_mask = torch.from_numpy(hlt_view.mask).bool()
        output = loaded(hlt_tokens, hlt_mask)

        self.assertEqual(output.tokens.shape, (2, 13, 14))
        self.assertEqual(output.weights.shape, (2, 13))
        self.assertEqual(output.candidate_mask.shape, (2, 13))
        self.assertEqual(output.edited_tokens.shape, (2, 5, 14))
        self.assertEqual(output.split_tokens.shape, (2, 5, 14))
        self.assertEqual(output.generated_tokens.shape, (2, 3, 14))
        self.assertTrue(torch.isfinite(output.tokens).all())
        self.assertTrue(torch.isfinite(output.weights).all())

    def test_forward_shapes_and_finite_loss(self):
        hlt_view, offline_view = make_views(n_jets=2, n_constits=5)
        cfg = self.tiny_config()
        model = build_reconstructor(cfg)
        hlt_tokens = torch.from_numpy(hlt_view.tokens).float()
        hlt_mask = torch.from_numpy(hlt_view.mask).bool()
        offline_tokens = torch.from_numpy(offline_view.tokens).float()
        offline_mask = torch.from_numpy(offline_view.mask).bool()

        output = model(hlt_tokens, hlt_mask)
        self.assertEqual(output.tokens.shape, (2, 18, 14))
        self.assertEqual(output.weights.shape, (2, 18))
        self.assertEqual(output.candidate_mask.shape, (2, 18))
        self.assertEqual(output.generated_tokens.shape, (2, 3, 14))
        self.assertEqual(output.split_child_tokens.shape, (2, 5, 2, 14))
        self.assertEqual(output.split_child_weights.shape, (2, 5, 2))
        self.assertEqual(output.split_parent_added_support.shape, (2, 5))
        self.assertEqual(output.generator_to_parent_assignment.shape, (2, 3, 5))
        self.assertEqual(output.generator_parent_added_support.shape, (2, 5))
        self.assertEqual(output.parent_added_support.shape, (2, 5))
        self.assertEqual(output.budget_efficiency_share.shape, (2, 5))
        self.assertEqual(output.corrected_parent_tokens.shape, (2, 5, 14))
        self.assertEqual(output.corrected_parent_weights.shape, (2, 5))
        self.assertTrue(torch.allclose(output.generator_to_parent_assignment.sum(dim=2), torch.ones(2, 3), atol=1.0e-5))
        for candidate_tokens in (
            output.edited_tokens,
            output.split_tokens,
            output.split_child_tokens.reshape(2, 10, 14),
            output.generated_tokens,
        ):
            pt = torch.clamp(candidate_tokens[:, :, 0], min=0.0)
            eta = candidate_tokens[:, :, 1]
            energy = candidate_tokens[:, :, 3]
            self.assertTrue(torch.all(energy + 1.0e-6 >= pt * torch.cosh(torch.clamp(eta, -5.0, 5.0))))
            self.assertTrue(torch.isfinite(candidate_tokens).all())
        self.assertTrue(torch.isfinite(output.weights).all())
        self.assertTrue(torch.isfinite(output.total_count_pred).all())
        self.assertTrue(torch.isfinite(output.added_count_pred).all())

        loss, diagnostics = reconstruction_loss(
            output,
            hlt_tokens=hlt_tokens,
            hlt_mask=hlt_mask,
            offline_tokens=offline_tokens,
            offline_mask=offline_mask,
            config=cfg,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("set_loss", diagnostics)
        self.assertIn("hungarian_set_loss", diagnostics)
        self.assertIn("matched_weight_loss", diagnostics)
        self.assertIn("weighted_chamfer_loss", diagnostics)
        self.assertIn("split_sparsity_loss", diagnostics)
        self.assertIn("generated_sparsity_loss", diagnostics)
        self.assertIn("nonfinite_penalty", diagnostics)
        self.assertIn("matching_candidate_count", diagnostics)

    def test_reconstruction_loss_backward_with_generated_weight_views(self):
        hlt_view, offline_view = make_views(n_jets=2, n_constits=5)
        cfg = self.tiny_config()
        model = build_reconstructor(cfg)
        hlt_tokens = torch.from_numpy(hlt_view.tokens).float()
        hlt_mask = torch.from_numpy(hlt_view.mask).bool()
        offline_tokens = torch.from_numpy(offline_view.tokens).float()
        offline_mask = torch.from_numpy(offline_view.mask).bool()

        output = model(hlt_tokens, hlt_mask)
        loss, diagnostics = reconstruction_loss(
            output,
            hlt_tokens=hlt_tokens,
            hlt_mask=hlt_mask,
            offline_tokens=offline_tokens,
            offline_mask=offline_mask,
            config=cfg,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(output.generated_weights.shape[1], 0)
        loss.backward()
        finite_grad_count = 0
        for param in model.parameters():
            if param.grad is not None:
                finite_grad_count += 1
                self.assertTrue(torch.isfinite(param.grad).all())
        self.assertGreater(finite_grad_count, 0)
        self.assertTrue(torch.isfinite(diagnostics["generated_sparsity_loss"]))

    def test_forward_sanitizes_nonfinite_or_empty_inputs(self):
        cfg = self.tiny_config()
        model = build_reconstructor(cfg)
        hlt_tokens = torch.zeros(2, 5, 14)
        hlt_mask = torch.zeros(2, 5, dtype=torch.bool)
        hlt_mask[0, :2] = True
        hlt_tokens[0, 0, 0] = float("nan")
        hlt_tokens[0, 0, 3] = float("inf")
        hlt_tokens[0, 1, 0] = 2.0
        hlt_tokens[0, 1, 3] = 2.0

        output = model(hlt_tokens, hlt_mask)
        self.assertTrue(torch.isfinite(output.tokens).all())
        self.assertTrue(torch.isfinite(output.weights).all())
        self.assertEqual(output.sanitized_hlt_mask[0].sum().item(), 1)
        self.assertEqual(output.sanitized_hlt_mask[1].sum().item(), 1)
        self.assertEqual(output.diagnostics["forced_nonempty_mask"][1].item(), 1.0)

    def test_loss_sanitizes_nonfinite_candidates_before_matching(self):
        hlt_tokens = torch.zeros(1, 2, 14)
        hlt_mask = torch.ones(1, 2, dtype=torch.bool)
        offline_tokens = torch.zeros(1, 2, 14)
        offline_mask = torch.ones(1, 2, dtype=torch.bool)
        for idx in range(2):
            pt = 2.0 + idx
            hlt_tokens[0, idx, 0] = pt
            hlt_tokens[0, idx, 2] = 0.1 * idx
            hlt_tokens[0, idx, 3] = pt
            offline_tokens[0, idx, 0] = pt
            offline_tokens[0, idx, 2] = 0.1 * idx
            offline_tokens[0, idx, 3] = pt

        edited_tokens = hlt_tokens.clone()
        split_tokens = hlt_tokens.clone()
        generated_tokens = hlt_tokens[:, :1, :].clone()
        edited_tokens[0, 0, 0] = float("nan")
        split_tokens[0, 1, 3] = float("inf")
        generated_tokens[0, 0, 1] = float("-inf")
        output = ReconstructionOutput(
            tokens=torch.cat([edited_tokens, split_tokens, generated_tokens], dim=1),
            weights=torch.tensor([[1.0, float("nan"), 0.5, 0.5, float("inf")]]),
            candidate_mask=torch.ones(1, 5, dtype=torch.bool),
            edited_tokens=edited_tokens,
            split_tokens=split_tokens,
            generated_tokens=generated_tokens,
            edited_weights=torch.tensor([[1.0, float("nan")]]),
            split_weights=torch.tensor([[0.5, 0.5]]),
            generated_weights=torch.tensor([[float("inf")]]),
            total_count_pred=torch.ones(1) * 2.0,
            added_count_pred=torch.ones(1),
        )
        loss, diagnostics = reconstruction_loss(
            output,
            hlt_tokens=hlt_tokens,
            hlt_mask=hlt_mask,
            offline_tokens=offline_tokens,
            offline_mask=offline_mask,
            config=self.tiny_config(),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(diagnostics["nonfinite_candidate_count"]), 0.0)
        self.assertGreaterEqual(float(diagnostics["nonfinite_penalty"]), 0.0)
        self.assertTrue(torch.isfinite(diagnostics["hungarian_set_loss"]))

    def test_reconstruction_global_response_losses_are_bounded(self):
        hlt_tokens = torch.zeros(1, 1, 14)
        hlt_mask = torch.ones(1, 1, dtype=torch.bool)
        offline_tokens = torch.zeros(1, 1, 14)
        offline_mask = torch.ones(1, 1, dtype=torch.bool)
        hlt_tokens[0, 0, 0] = 1.0
        hlt_tokens[0, 0, 3] = 1.0
        offline_tokens[0, 0, 0] = 1.0
        offline_tokens[0, 0, 3] = 1.0

        edited_tokens = hlt_tokens.clone()
        split_tokens = hlt_tokens.clone()
        edited_tokens[0, 0, 0] = 1.0e8
        edited_tokens[0, 0, 3] = 1.0e8
        split_tokens[0, 0, 0] = 1.0e8
        split_tokens[0, 0, 2] = torch.pi
        split_tokens[0, 0, 3] = 1.0e8
        output = ReconstructionOutput(
            tokens=torch.cat([edited_tokens, split_tokens], dim=1),
            weights=torch.ones(1, 2),
            candidate_mask=torch.ones(1, 2, dtype=torch.bool),
            edited_tokens=edited_tokens,
            split_tokens=split_tokens,
            generated_tokens=torch.zeros(1, 0, 14),
            edited_weights=torch.ones(1, 1),
            split_weights=torch.ones(1, 1),
            generated_weights=torch.zeros(1, 0),
            total_count_pred=torch.ones(1),
            added_count_pred=torch.ones(1),
        )
        loss, diagnostics = reconstruction_loss(
            output,
            hlt_tokens=hlt_tokens,
            hlt_mask=hlt_mask,
            offline_tokens=offline_tokens,
            offline_mask=offline_mask,
            config=self.tiny_config(),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertLessEqual(float(diagnostics["mass_ratio_loss"]), 36.0)

    def test_paired_dataset_requires_matching_identities(self):
        hlt_view, offline_view = make_views()
        dataset = PairedReconstructionDataset(hlt_view, offline_view)
        self.assertEqual(len(dataset), 4)
        bad_offline = make_views()[1]
        bad_offline.jet_ids[0] = JetIdentity(file="different.root", entry=0, label=0)
        with self.assertRaises(ValueError):
            PairedReconstructionDataset(hlt_view, bad_offline)

    def test_stage_a_training_smoke(self):
        train_hlt, train_offline = make_views(n_jets=4)
        val_hlt, val_offline = make_views(n_jets=2)
        val_hlt.split = "model_val"
        val_offline.split = "model_val"
        cfg = self.tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            train_cfg = StageAReconstructorTrainConfig(
                output_dir=tmp,
                manifest_path="/unused",
                hlt_cache_dir="/unused",
                epochs=1,
                batch_size=2,
                device="cpu",
                amp=False,
                early_stop_patience=2,
            )
            report = train_stage_a_reconstructor(
                train_cfg,
                model=build_reconstructor(cfg),
                train_hlt_view=train_hlt,
                train_offline_view=train_offline,
                val_hlt_view=val_hlt,
                val_offline_view=val_offline,
            )
            self.assertTrue((Path(tmp) / "best_model_val.pt").exists())
            self.assertTrue((Path(tmp) / "model_val_reconstruction_report.json").exists())
        self.assertEqual(report["experiment_step"], "step7_stage_a_reconstructor")
        self.assertEqual(report["variant"], "m2_base")
        self.assertTrue(report["no_final_test_evaluation"])
        self.assertTrue(report["not_a_classifier"])


if __name__ == "__main__":
    unittest.main()
