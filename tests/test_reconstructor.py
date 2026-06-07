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
    get_reconstructor_variant_config,
    m2_base_variant_config,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from jetclass_fresh.reconstructor import (
        PairedReconstructionDataset,
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
            hidden_dim=16,
            global_dim=16,
            max_hlt_constits=5,
        )

    def test_forward_shapes_and_finite_loss(self):
        hlt_view, offline_view = make_views(n_jets=2, n_constits=5)
        cfg = self.tiny_config()
        model = build_reconstructor(cfg)
        hlt_tokens = torch.from_numpy(hlt_view.tokens).float()
        hlt_mask = torch.from_numpy(hlt_view.mask).bool()
        offline_tokens = torch.from_numpy(offline_view.tokens).float()
        offline_mask = torch.from_numpy(offline_view.mask).bool()

        output = model(hlt_tokens, hlt_mask)
        self.assertEqual(output.tokens.shape, (2, 13, 14))
        self.assertEqual(output.weights.shape, (2, 13))
        self.assertEqual(output.generated_tokens.shape, (2, 3, 14))
        for candidate_tokens in (output.edited_tokens, output.split_tokens, output.generated_tokens):
            pt = torch.clamp(candidate_tokens[:, :, 0], min=0.0)
            eta = candidate_tokens[:, :, 1]
            energy = candidate_tokens[:, :, 3]
            self.assertTrue(torch.all(energy + 1.0e-6 >= pt * torch.cosh(torch.clamp(eta, -5.0, 5.0))))

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
