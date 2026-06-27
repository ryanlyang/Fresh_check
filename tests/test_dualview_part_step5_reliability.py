import importlib.util
import math
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.dualview_part import (
        DUALVIEW_PART_RELIABILITY_CONTRACT,
        DUALVIEW_PART_RELIABILITY_FEATURE_NAMES,
        DUALVIEW_PART_STEP5,
        ReliabilityFeatureConfig,
        build_reliability_features,
        reliability_feature_dim,
    )
else:  # pragma: no cover - environment dependent
    torch = None


def feature_column(features, name):
    return features[:, DUALVIEW_PART_RELIABILITY_FEATURE_NAMES.index(name)]


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DualViewPartStep5ReliabilityTests(unittest.TestCase):
    def make_tokens(self, *, batch_size=2, n_tokens=4, eta_shift=0.0):
        tokens = torch.zeros(batch_size, n_tokens, RAW_TOKEN_DIM)
        mask = torch.ones(batch_size, n_tokens, dtype=torch.bool)
        tokens[:, :, 0] = torch.tensor([100.0, 50.0, 20.0, 10.0])[:n_tokens]
        tokens[:, :, 1] = torch.linspace(-0.2, 0.2, n_tokens) + float(eta_shift)
        tokens[:, :, 2] = torch.linspace(-0.1, 0.1, n_tokens)
        tokens[:, :, 3] = tokens[:, :, 0] * 1.4
        if batch_size > 1:
            mask[1, -1] = False
        return tokens, mask

    def test_config_and_names_are_stable(self):
        cfg = ReliabilityFeatureConfig()

        self.assertEqual(cfg.raw_token_dim, RAW_TOKEN_DIM)
        self.assertEqual(cfg.feature_dim, len(DUALVIEW_PART_RELIABILITY_FEATURE_NAMES))
        self.assertEqual(reliability_feature_dim(), len(DUALVIEW_PART_RELIABILITY_FEATURE_NAMES))
        self.assertEqual(cfg.output_contract, DUALVIEW_PART_RELIABILITY_CONTRACT)
        self.assertEqual(cfg.experiment_step, DUALVIEW_PART_STEP5)
        self.assertIn("hlt_margin", DUALVIEW_PART_RELIABILITY_FEATURE_NAMES)
        self.assertIn("pn_to_hlt_delta_r_mean_norm", DUALVIEW_PART_RELIABILITY_FEATURE_NAMES)

    def test_config_rejects_bad_values(self):
        bad_configs = [
            {"raw_token_dim": 0},
            {"max_constituents": 0},
            {"pt_index": RAW_TOKEN_DIM},
            {"low_confidence_threshold": -0.1},
            {"eta_scale": 0.0},
            {"log_kinematic_scale": 0.0},
            {"nearest_neighbor_delta_r_scale": 0.0},
        ]
        for kwargs in bad_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ReliabilityFeatureConfig(**kwargs)

    def test_hlt_confidence_features_match_known_logits(self):
        logits = torch.tensor([[4.0, 0.0], [0.0, 0.0]])
        output = build_reliability_features(hlt_logits=logits, return_diagnostics=True)
        features = output.features

        expected_prob = torch.softmax(logits, dim=-1)
        self.assertEqual(tuple(features.shape), (2, reliability_feature_dim()))
        self.assertAlmostEqual(float(feature_column(features, "hlt_top1_prob")[0]), float(expected_prob[0, 0]), places=6)
        self.assertAlmostEqual(float(feature_column(features, "hlt_top2_prob")[0]), float(expected_prob[0, 1]), places=6)
        self.assertAlmostEqual(
            float(feature_column(features, "hlt_margin")[0]),
            float(expected_prob[0, 0] - expected_prob[0, 1]),
            places=6,
        )
        self.assertAlmostEqual(float(feature_column(features, "hlt_entropy_norm")[1]), 1.0, places=6)
        self.assertEqual(output.diagnostics["output_contract"], DUALVIEW_PART_RELIABILITY_CONTRACT)

    def test_pn_count_and_confidence_features_are_masked(self):
        hlt_tokens, hlt_mask = self.make_tokens()
        pn_tokens, pn_mask = self.make_tokens()
        pn_confidence = torch.tensor(
            [
                [1.0, 0.5, 0.0, 0.25],
                [0.8, 0.4, 0.2, 1.0],
            ]
        )
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        output = build_reliability_features(
            hlt_logits=logits,
            hlt_tokens=hlt_tokens,
            hlt_mask=hlt_mask,
            pn_tokens=pn_tokens,
            pn_mask=pn_mask,
            pn_confidence=pn_confidence,
        )
        features = output.features

        self.assertTrue(bool(torch.isfinite(features).all()))
        self.assertAlmostEqual(float(feature_column(features, "pn_conf_mean")[0]), 0.4375, places=6)
        self.assertAlmostEqual(float(feature_column(features, "pn_conf_max")[1]), 0.8, places=6)
        self.assertGreater(float(feature_column(features, "pn_low_conf_frac")[0]), 0.0)
        self.assertEqual(float(feature_column(features, "pn_count_minus_hlt_norm")[0]), 0.0)

    def test_summary_and_geometry_disagreement_increase_when_pn_is_shifted(self):
        hlt_tokens, hlt_mask = self.make_tokens(batch_size=1, n_tokens=4)
        same_pn, same_mask = self.make_tokens(batch_size=1, n_tokens=4)
        shifted_pn, shifted_mask = self.make_tokens(batch_size=1, n_tokens=4, eta_shift=2.0)
        logits = torch.tensor([[1.5, -0.5]])

        same = build_reliability_features(
            hlt_logits=logits,
            hlt_tokens=hlt_tokens,
            hlt_mask=hlt_mask,
            pn_tokens=same_pn,
            pn_mask=same_mask,
        ).features
        shifted = build_reliability_features(
            hlt_logits=logits,
            hlt_tokens=hlt_tokens,
            hlt_mask=hlt_mask,
            pn_tokens=shifted_pn,
            pn_mask=shifted_mask,
        ).features

        self.assertLess(
            float(feature_column(same, "jet_eta_abs_diff_norm")[0]),
            float(feature_column(shifted, "jet_eta_abs_diff_norm")[0]),
        )
        self.assertLess(
            float(feature_column(same, "pn_to_hlt_delta_r_mean_norm")[0]),
            float(feature_column(shifted, "pn_to_hlt_delta_r_mean_norm")[0]),
        )

    def test_missing_pn_inputs_keep_shape_and_finite_values(self):
        logits = torch.tensor([[0.0, 0.0], [3.0, -1.0], [-2.0, 2.0]])
        output = build_reliability_features(hlt_logits=logits)

        self.assertEqual(tuple(output.features.shape), (3, reliability_feature_dim()))
        self.assertTrue(bool(torch.isfinite(output.features).all()))
        self.assertTrue(bool((feature_column(output.features, "pn_count_norm") == 0.0).all()))
        self.assertTrue(bool((feature_column(output.features, "pn_to_hlt_delta_r_mean_norm") == 0.0).all()))

    def test_feature_builder_keeps_logits_gradient_path(self):
        logits = torch.tensor([[1.0, -0.5], [-0.5, 1.0]], requires_grad=True)
        output = build_reliability_features(hlt_logits=logits)
        loss = output.features[:, :4].sum()
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
