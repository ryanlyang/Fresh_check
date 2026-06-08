import inspect
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.dual_view import (
    CORRECTED_VIEW_FEATURE_NAMES,
    DUAL_VIEW_EXPERIMENT_STEP,
    DualViewTaggerTrainConfig,
    build_dual_view_tagger,
    build_soft_corrected_view_torch,
)
from jetclass_fresh.jetclass_data import JetIdentity, JetView
from jetclass_fresh.reconstructor import RECONSTRUCTOR_VARIANT_NAMES

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from jetclass_fresh.dual_view import (
        HLTTokenDataset,
        build_part_inputs_torch,
        make_hlt_token_loader,
        run_dual_view_epoch,
        train_dual_view_tagger,
    )
    from jetclass_fresh.reconstructor import ReconstructionOutput
else:  # pragma: no cover - environment dependent
    torch = None


def make_hlt_view(split="model_train", n_jets=6, n_constits=5):
    tokens = np.zeros((n_jets, n_constits, 14), dtype=np.float32)
    mask = np.zeros((n_jets, n_constits), dtype=bool)
    labels = np.zeros((n_jets,), dtype=np.int64)
    for jet_index in range(n_jets):
        mask[jet_index, :3] = True
        for part_index in range(3):
            pt = 2.0 + part_index + 0.1 * jet_index
            eta = 0.03 * part_index
            phi = 0.18 * part_index
            tokens[jet_index, part_index, 0] = pt
            tokens[jet_index, part_index, 1] = eta
            tokens[jet_index, part_index, 2] = phi
            tokens[jet_index, part_index, 3] = pt * np.cosh(eta)
            tokens[jet_index, part_index, 4] = 1.0
            tokens[jet_index, part_index, 5] = 1.0
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=[JetIdentity(file="hlt.root", entry=i, label=0) for i in range(n_jets)],
        split=split,
        metadata={"view": "fixed_hlt", "hlt_content_hash": f"{split}_hash"},
    )


class DualViewStep8ConfigTests(unittest.TestCase):
    def test_config_defaults_to_m2_base_and_model_val_only(self):
        cfg = DualViewTaggerTrainConfig(
            output_dir="/tmp/out",
            hlt_cache_dir="/tmp/hlt",
            reconstructor_checkpoint="/tmp/reco.pt",
        )
        self.assertEqual(cfg.variant, "m2_base")
        self.assertEqual(cfg.train_split, "model_train")
        self.assertEqual(cfg.val_split, "model_val")
        self.assertEqual(cfg.max_constits, 128)
        self.assertIn("m2_antioverlap", RECONSTRUCTOR_VARIANT_NAMES)

    def test_soft_corrected_view_builder_has_no_offline_arguments(self):
        params = inspect.signature(build_soft_corrected_view_torch).parameters
        self.assertNotIn("offline_tokens", params)
        self.assertNotIn("offline_mask", params)
        self.assertEqual(list(CORRECTED_VIEW_FEATURE_NAMES[-3:]), [
            "token_weight",
            "parent_added_support",
            "budget_efficiency_share",
        ])


if TORCH_AVAILABLE:
    def make_parent_aligned_reconstruction_output(hlt_tokens, hlt_mask, *, zero_weights=False):
        corrected = hlt_tokens.clone()
        corrected[:, :, 0] = torch.clamp(corrected[:, :, 0] + 0.25 * hlt_mask.float(), min=0.0)
        corrected[:, :, 3] = corrected[:, :, 0] * torch.cosh(corrected[:, :, 1]) + 1.0e-4
        weights = hlt_mask.float()
        if zero_weights:
            weights = torch.zeros_like(weights)
        else:
            weights = weights * torch.linspace(0.35, 0.95, steps=hlt_tokens.shape[1])[None, :]
        split_support = 0.20 * hlt_mask.float()
        generator_support = 0.15 * hlt_mask.float()
        parent_added_support = split_support + generator_support
        budget_efficiency_share = 0.40 * parent_added_support
        return ReconstructionOutput(
            tokens=corrected,
            weights=weights,
            candidate_mask=hlt_mask,
            edited_tokens=corrected,
            split_tokens=corrected,
            generated_tokens=corrected[:, :0],
            edited_weights=weights,
            split_weights=weights,
            generated_weights=weights[:, :0],
            total_count_pred=weights.sum(dim=1),
            added_count_pred=parent_added_support.sum(dim=1),
            corrected_parent_tokens=corrected,
            corrected_parent_weights=weights,
            split_parent_added_support=split_support,
            generator_parent_added_support=generator_support,
            parent_added_support=parent_added_support,
            budget_efficiency_share=budget_efficiency_share,
        )

    class DummyReconstructor(torch.nn.Module):
        def forward(self, hlt_tokens, hlt_mask):
            return make_parent_aligned_reconstruction_output(hlt_tokens, hlt_mask)

    class DummyDualTagger(torch.nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(num_classes))

        def forward(self, hlt_inputs, reco_inputs):
            batch_size = hlt_inputs["features"].shape[0]
            self._last_hlt_shape = tuple(hlt_inputs["features"].shape)
            self._last_reco_shape = tuple(reco_inputs.features.shape)
            return self.logits.unsqueeze(0).expand(batch_size, -1)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DualViewStep8TorchTests(unittest.TestCase):
    def test_build_soft_corrected_view_parent_aligned_features(self):
        view = make_hlt_view(n_jets=2, n_constits=5)
        hlt_tokens = torch.from_numpy(view.tokens).float()
        hlt_mask = torch.from_numpy(view.mask).bool()
        output = make_parent_aligned_reconstruction_output(hlt_tokens, hlt_mask)

        corrected_view = build_soft_corrected_view_torch(
            hlt_tokens,
            hlt_mask,
            output,
            weight_threshold=0.05,
        )

        self.assertEqual(corrected_view.features.shape, (2, len(CORRECTED_VIEW_FEATURE_NAMES), 5))
        self.assertEqual(corrected_view.mask.shape, (2, 1, 5))
        self.assertEqual(corrected_view.tokens.shape, (2, 5, 14))
        self.assertEqual(corrected_view.feature_names, list(CORRECTED_VIEW_FEATURE_NAMES))
        self.assertTrue(corrected_view.metadata["parent_aligned"])
        self.assertFalse(corrected_view.metadata["uses_offline_constituents"])
        self.assertTrue(torch.isfinite(corrected_view.features).all())

        valid = corrected_view.mask.squeeze(1)
        token_idx = CORRECTED_VIEW_FEATURE_NAMES.index("token_weight")
        support_idx = CORRECTED_VIEW_FEATURE_NAMES.index("parent_added_support")
        budget_idx = CORRECTED_VIEW_FEATURE_NAMES.index("budget_efficiency_share")
        self.assertTrue(torch.allclose(corrected_view.features[:, token_idx, :][valid], output.corrected_parent_weights[valid]))
        self.assertTrue(torch.allclose(corrected_view.token_weight[valid], output.corrected_parent_weights[valid]))
        self.assertTrue(torch.allclose(corrected_view.features[:, support_idx, :][valid], output.parent_added_support[valid]))
        self.assertTrue(torch.allclose(corrected_view.parent_added_support[valid], output.parent_added_support[valid]))
        self.assertTrue(torch.allclose(corrected_view.features[:, budget_idx, :][valid], output.budget_efficiency_share[valid]))
        self.assertTrue(torch.allclose(corrected_view.budget_efficiency_share[valid], output.budget_efficiency_share[valid]))

    def test_build_soft_corrected_view_forces_nonempty_mask(self):
        view = make_hlt_view(n_jets=2, n_constits=5)
        hlt_tokens = torch.from_numpy(view.tokens).float()
        hlt_mask = torch.from_numpy(view.mask).bool()
        output = make_parent_aligned_reconstruction_output(hlt_tokens, hlt_mask, zero_weights=True)

        corrected_view = build_soft_corrected_view_torch(
            hlt_tokens,
            hlt_mask,
            output,
            weight_threshold=0.05,
        )

        self.assertTrue(torch.equal(corrected_view.mask.sum(dim=2).squeeze(1), torch.ones(2, dtype=torch.long)))
        self.assertEqual(corrected_view.metadata["forced_nonempty_count"], 2)

    def test_build_soft_corrected_view_changes_when_hlt_zeroed(self):
        view = make_hlt_view(n_jets=2, n_constits=5)
        hlt_tokens = torch.from_numpy(view.tokens).float()
        hlt_mask = torch.from_numpy(view.mask).bool()
        output = make_parent_aligned_reconstruction_output(hlt_tokens, hlt_mask)
        baseline = build_soft_corrected_view_torch(hlt_tokens, hlt_mask, output)

        zero_hlt_tokens = torch.zeros_like(hlt_tokens)
        zero_output = make_parent_aligned_reconstruction_output(zero_hlt_tokens, hlt_mask)
        zero_view = build_soft_corrected_view_torch(zero_hlt_tokens, hlt_mask, zero_output)

        self.assertFalse(torch.allclose(baseline.features, zero_view.features))

    def test_cross_attention_tagger_forward_pass(self):
        view = make_hlt_view(n_jets=2, n_constits=5)
        hlt_tokens = torch.from_numpy(view.tokens).float()
        hlt_mask = torch.from_numpy(view.mask).bool()
        hlt_inputs = build_part_inputs_torch(hlt_tokens, hlt_mask, max_constits=5)
        reco_output = make_parent_aligned_reconstruction_output(hlt_tokens, hlt_mask)
        corrected_inputs = build_soft_corrected_view_torch(hlt_tokens, hlt_mask, reco_output)

        tagger = build_dual_view_tagger(model_size="tiny", num_classes=10)
        logits = tagger(hlt_inputs, corrected_inputs)

        self.assertEqual(logits.shape, (2, 10))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(tagger.config["architecture"], "cross_attention_fusion")
        self.assertEqual(tagger.config["corrected_view_feature_names"], list(CORRECTED_VIEW_FEATURE_NAMES))

    def test_cross_attention_tagger_checkpoint_roundtrip(self):
        view = make_hlt_view(n_jets=2, n_constits=5)
        hlt_tokens = torch.from_numpy(view.tokens).float()
        hlt_mask = torch.from_numpy(view.mask).bool()
        hlt_inputs = build_part_inputs_torch(hlt_tokens, hlt_mask, max_constits=5)
        reco_output = make_parent_aligned_reconstruction_output(hlt_tokens, hlt_mask)
        corrected_inputs = build_soft_corrected_view_torch(hlt_tokens, hlt_mask, reco_output)
        tagger = build_dual_view_tagger(model_size="tiny", num_classes=10)
        tagger.eval()
        with torch.no_grad():
            expected = tagger(hlt_inputs, corrected_inputs)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tagger.pt"
            torch.save({"model_state_dict": tagger.state_dict(), "model_config": tagger.config}, path)
            payload = torch.load(path, map_location="cpu")
        cfg = dict(payload["model_config"])
        loaded = build_dual_view_tagger(
            architecture=cfg["architecture"],
            model_size=cfg["model_size"],
            num_classes=cfg["num_classes"],
            hidden_dim=cfg["hidden_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            feedforward_dim=cfg["feedforward_dim"],
            dropout=cfg["dropout"],
        )
        loaded.load_state_dict(payload["model_state_dict"], strict=True)
        loaded.eval()
        with torch.no_grad():
            actual = loaded(hlt_inputs, corrected_inputs)

        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-6))

    def test_build_part_inputs_torch_topk_reco_candidates(self):
        tokens = torch.zeros(2, 7, 14)
        mask = torch.ones(2, 7, dtype=torch.bool)
        weights = torch.linspace(0.1, 0.7, steps=7).unsqueeze(0).repeat(2, 1)
        for index in range(7):
            tokens[:, index, 0] = float(index + 1)
            tokens[:, index, 1] = 0.01 * index
            tokens[:, index, 2] = 0.1 * index
            tokens[:, index, 3] = float(index + 1)
            tokens[:, index, 5] = 1.0

        inputs = build_part_inputs_torch(tokens, mask, weights=weights, max_constits=4)
        self.assertEqual(inputs["features"].shape, (2, 17, 4))
        self.assertEqual(inputs["lorentz_vectors"].shape, (2, 4, 4))
        self.assertEqual(inputs["mask"].shape, (2, 1, 4))
        self.assertTrue(torch.all(inputs["mask"]))

    def test_build_part_inputs_torch_sanitizes_reco_four_vectors(self):
        tokens = torch.zeros(1, 3, 14)
        mask = torch.ones(1, 3, dtype=torch.bool)
        tokens[0, 0, 0] = 10.0
        tokens[0, 0, 1] = 5.0
        tokens[0, 0, 2] = 0.2
        tokens[0, 0, 3] = 1.0
        tokens[0, 0, 5] = 1.0
        tokens[0, 1, 0] = float("nan")
        tokens[0, 1, 3] = 4.0
        tokens[0, 2, 0] = 3.0
        tokens[0, 2, 1] = 0.1
        tokens[0, 2, 2] = -0.2
        tokens[0, 2, 3] = 3.0
        weights = torch.tensor([[1.0, 1.0, float("nan")]])

        inputs = build_part_inputs_torch(tokens, mask, weights=weights, max_constits=3)
        vectors = inputs["lorentz_vectors"]
        out_mask = inputs["mask"]
        self.assertTrue(torch.isfinite(inputs["features"]).all())
        self.assertTrue(torch.isfinite(vectors).all())
        self.assertEqual(int(out_mask.sum().item()), 1)
        px, py, pz, energy = vectors[0, :, 0]
        momentum = torch.sqrt(px * px + py * py + pz * pz)
        self.assertGreaterEqual(float(energy + 1.0e-6), float(momentum))

    def test_run_dual_view_epoch_with_frozen_dummy_reconstructor(self):
        dataset = HLTTokenDataset(make_hlt_view(n_jets=4))
        loader = make_hlt_token_loader(dataset, batch_size=2, shuffle=False, num_workers=0, seed=909)
        tagger = DummyDualTagger()
        metrics = run_dual_view_epoch(
            tagger,
            DummyReconstructor(),
            loader,
            device=torch.device("cpu"),
            criterion=torch.nn.CrossEntropyLoss(),
            max_constits=5,
        )
        self.assertEqual(metrics["n_jets"], 4)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(tagger._last_hlt_shape, (2, 17, 5))
        self.assertEqual(tagger._last_reco_shape, (2, len(CORRECTED_VIEW_FEATURE_NAMES), 5))

    def test_train_cross_attention_tagger_tiny_subset_with_injected_reconstructor(self):
        train_view = make_hlt_view("model_train", n_jets=6)
        val_view = make_hlt_view("model_val", n_jets=4)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = DualViewTaggerTrainConfig(
                output_dir=tmp,
                hlt_cache_dir="/unused",
                reconstructor_checkpoint="/unused",
                epochs=1,
                batch_size=2,
                device="cpu",
                amp=False,
                early_stop_patience=2,
                max_constits=5,
            )
            report = train_dual_view_tagger(
                cfg,
                tagger=build_dual_view_tagger(model_size="tiny", num_classes=10),
                reconstructor=DummyReconstructor(),
                train_view=train_view,
                val_view=val_view,
            )
            self.assertTrue((Path(tmp) / "best_model_val.pt").exists())
            self.assertTrue((Path(tmp) / "model_val_report.json").exists())
        self.assertEqual(report["experiment_step"], DUAL_VIEW_EXPERIMENT_STEP)
        self.assertEqual(report["variant"], "m2_base")
        self.assertEqual(report["dual_view_architecture"], "cross_attention_fusion")
        self.assertTrue(report["no_final_test_evaluation"])
        self.assertTrue(report["reconstructor_frozen"])
        self.assertGreaterEqual(report["best_model_val_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
