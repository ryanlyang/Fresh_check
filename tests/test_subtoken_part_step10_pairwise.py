import math
import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.subtoken_part import (
    SUBTOKEN_PART_PAIRWISE_CONTRACT,
    SUBTOKEN_PART_PAIRWISE_FEATURE_DIM,
    SUBTOKEN_PART_PAIRWISE_FEATURE_NAMES,
    PairwiseBiasConfig,
    PairwiseBiasEncoder,
    PairwiseBiasedAttentionBlock,
    PairwiseFeatureBuilder,
    PairwiseFeatureConfig,
    wrap_pairwise_delta_phi,
)


class SubtokenPartStep10PairwiseTests(unittest.TestCase):
    def make_tokens(self):
        torch = require_torch()
        tokens = torch.zeros((1, 4, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.tensor([[True, True, False, True]], dtype=torch.bool)
        tokens[0, :, 0] = torch.tensor([50.0, 20.0, 900.0, 10.0])
        tokens[0, :, 1] = torch.tensor([0.2, -0.3, 9.0, 1.2])
        tokens[0, :, 2] = torch.tensor([math.pi - 0.05, -math.pi + 0.05, 7.0, 0.4])
        tokens[0, :, 3] = tokens[0, :, 0] + 10.0
        return tokens, mask

    def test_delta_phi_wraparound(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        output = PairwiseFeatureBuilder(PairwiseFeatureConfig(include_cls_token=False))(tokens, mask)
        names = list(output.feature_names)
        sin_index = names.index("sin_delta_phi")
        abs_index = names.index("abs_delta_phi")
        log_dr_index = names.index("log_delta_r")

        delta = wrap_pairwise_delta_phi(tokens[0, 0, 2] - tokens[0, 1, 2])
        delta_eta = tokens[0, 0, 1] - tokens[0, 1, 1]
        delta_r = torch.sqrt(delta_eta * delta_eta + delta * delta)

        self.assertLess(float(delta.abs().detach().cpu().item()), 0.11)
        self.assertAlmostEqual(
            float(output.pair_features[0, 0, 1, sin_index].detach().cpu().item()),
            float(torch.sin(delta).detach().cpu().item()),
            places=6,
        )
        self.assertAlmostEqual(
            float(output.pair_features[0, 0, 1, abs_index].detach().cpu().item()),
            float(delta.abs().div(math.pi).detach().cpu().item()),
            places=6,
        )
        self.assertAlmostEqual(
            float(output.pair_features[0, 0, 1, log_dr_index].detach().cpu().item()),
            float(torch.log(delta_r + 1.0e-6).detach().cpu().item()),
            places=6,
        )

    def test_pair_mass_relative_kt_and_z_features(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        output = PairwiseFeatureBuilder(PairwiseFeatureConfig(include_cls_token=False))(tokens, mask)
        names = list(output.feature_names)
        mass_index = names.index("log_pair_mass")
        kt_index = names.index("relative_kt")
        log_kt_index = names.index("log_relative_kt")
        z_index = names.index("z")

        left = tokens[0, 0]
        right = tokens[0, 1]
        left_pt, left_eta, left_phi, left_e = left[0], left[1], left[2], left[3]
        right_pt, right_eta, right_phi, right_e = right[0], right[1], right[2], right[3]
        pair_px = left_pt * torch.cos(left_phi) + right_pt * torch.cos(right_phi)
        pair_py = left_pt * torch.sin(left_phi) + right_pt * torch.sin(right_phi)
        pair_pz = left_pt * torch.sinh(left_eta) + right_pt * torch.sinh(right_eta)
        pair_e = left_e + right_e
        pair_mass = torch.sqrt(torch.clamp(pair_e * pair_e - pair_px * pair_px - pair_py * pair_py - pair_pz * pair_pz, min=1.0e-12))
        delta_phi = wrap_pairwise_delta_phi(left_phi - right_phi)
        delta_r = torch.sqrt((left_eta - right_eta) ** 2 + delta_phi ** 2)
        relative_kt = torch.minimum(left_pt, right_pt) * delta_r
        z = torch.minimum(left_pt, right_pt) / (left_pt + right_pt)

        self.assertAlmostEqual(
            float(output.pair_features[0, 0, 1, mass_index].detach().cpu().item()),
            float(torch.log(pair_mass + 1.0e-6).clamp(-14.0, 14.0).detach().cpu().item()),
            places=5,
        )
        self.assertAlmostEqual(
            float(output.pair_features[0, 0, 1, kt_index].detach().cpu().item()),
            float((relative_kt / 1000.0).clamp(0.0, 10.0).detach().cpu().item()),
            places=6,
        )
        self.assertAlmostEqual(
            float(output.pair_features[0, 0, 1, log_kt_index].detach().cpu().item()),
            float(torch.log(relative_kt + 1.0e-6).clamp(-14.0, 14.0).detach().cpu().item()),
            places=6,
        )
        self.assertAlmostEqual(
            float(output.pair_features[0, 0, 1, z_index].detach().cpu().item()),
            float(z.detach().cpu().item()),
            places=6,
        )

    def test_pairwise_mask_with_cls_token(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        output = PairwiseFeatureBuilder(PairwiseFeatureConfig(include_cls_token=True))(tokens, mask)

        expected_token_mask = torch.tensor([[True, True, True, False, True]], dtype=torch.bool)
        expected_pair_mask = expected_token_mask[:, :, None] & expected_token_mask[:, None, :]

        self.assertEqual(output.summary()["contract"], SUBTOKEN_PART_PAIRWISE_CONTRACT)
        self.assertTrue(bool(torch.equal(output.token_mask, expected_token_mask)))
        self.assertTrue(bool(torch.equal(output.pair_mask, expected_pair_mask)))
        self.assertTrue(bool((output.pair_features[~expected_pair_mask] == 0.0).all()))
        self.assertTrue(bool((output.pair_features[:, 0, :, :] == 0.0).all()))
        self.assertTrue(bool((output.pair_features[:, :, 0, :] == 0.0).all()))

    def test_pairwise_symmetry_and_antisymmetry(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        output = PairwiseFeatureBuilder(PairwiseFeatureConfig(include_cls_token=False))(tokens, mask)
        features = output.pair_features
        names = list(output.feature_names)

        for name in ("delta_eta", "sin_delta_phi", "delta_log_pt"):
            index = names.index(name)
            self.assertTrue(bool(torch.allclose(features[0, :, :, index], -features[0, :, :, index].T, atol=1.0e-6)))
        for name in (
            "cos_delta_phi",
            "delta_r",
            "log_delta_r",
            "log_pair_mass",
            "log_relative_kt",
            "relative_kt",
            "z",
            "abs_delta_eta",
            "abs_delta_phi",
        ):
            index = names.index(name)
            self.assertTrue(bool(torch.allclose(features[0, :, :, index], features[0, :, :, index].T, atol=1.0e-6)))
        same_index = names.index("same_particle")
        valid_diag = torch.diagonal(features[0, :, :, same_index])[mask[0]]
        self.assertTrue(bool(torch.allclose(valid_diag, torch.ones_like(valid_diag))))

    def test_bias_encoder_shape_and_pair_mask(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        pairwise = PairwiseFeatureBuilder(PairwiseFeatureConfig(include_cls_token=True))(tokens, mask)
        encoder = PairwiseBiasEncoder(PairwiseBiasConfig(num_heads=4, hidden_dim=16))

        bias = encoder(pairwise)

        self.assertEqual(tuple(bias.shape), (1, 4, 5, 5))
        self.assertTrue(bool(torch.isfinite(bias).all()))
        self.assertTrue(bool((bias.permute(0, 2, 3, 1)[~pairwise.pair_mask] == 0.0).all()))

    def test_attention_block_respects_masks_and_bias_changes_output(self):
        torch = require_torch()
        torch.manual_seed(123)
        tokens = torch.randn((1, 4, 8), dtype=torch.float32)
        token_mask = torch.tensor([[True, True, True, False]], dtype=torch.bool)
        block = PairwiseBiasedAttentionBlock(
            embed_dim=8,
            num_heads=2,
            mlp_ratio=2.0,
            dropout=0.0,
            attention_dropout=0.0,
        )
        block.eval()
        zero_bias = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
        changed_bias = zero_bias.clone()
        changed_bias[:, :, 0, 1] = 6.0
        changed_bias[:, :, 0, 2] = -6.0

        baseline = block(tokens, zero_bias, token_mask, need_weights=True)
        changed = block(tokens, changed_bias, token_mask, need_weights=True)

        delta = (baseline.tokens[:, 0] - changed.tokens[:, 0]).abs().sum()
        self.assertGreater(float(delta.detach().cpu().item()), 1.0e-6)
        self.assertTrue(bool((baseline.tokens[~token_mask] == 0.0).all()))
        self.assertIsNotNone(baseline.attention_weights)
        self.assertTrue(bool((baseline.attention_weights[:, :, :, ~token_mask[0]] == 0.0).all()))

    def test_bias_encoder_receives_gradients(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        pairwise = PairwiseFeatureBuilder(PairwiseFeatureConfig(include_cls_token=True))(tokens, mask)
        encoder = PairwiseBiasEncoder(PairwiseBiasConfig(num_heads=2, hidden_dim=16))

        bias = encoder(pairwise)
        loss = bias.square().mean()
        loss.backward()

        grad_sum = sum(
            float(param.grad.detach().abs().sum().cpu().item())
            for param in encoder.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_sum, 0.0)

    def test_feature_dim_constant_matches_names(self):
        self.assertEqual(SUBTOKEN_PART_PAIRWISE_FEATURE_DIM, len(SUBTOKEN_PART_PAIRWISE_FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
