import math
import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.target_denoising_part import (
    DENOISING_TARGET_NAMES,
    PAIRWISE_DENOISING_FEATURE_DIM,
    PAIRWISE_DENOISING_FEATURE_NAMES,
    DenoisingPairBiasConfig,
    DenoisingPairBiasEncoder,
    PairwiseDenoisingFeatureBuilder,
    PairwiseDenoisingFeatureConfig,
    TargetConditionedDenoiserConfig,
    TargetConditionedPairwiseDenoiser,
    wrap_delta_phi_torch,
)


class TargetDenoisingPartStep2ModelTests(unittest.TestCase):
    def make_tokens(self, *, batch=2, particles=5):
        torch = require_torch()
        tokens = torch.zeros((batch, particles, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.ones((batch, particles), dtype=torch.bool)
        mask[:, -1] = False
        for row in range(batch):
            for col in range(particles):
                tokens[row, col, 0] = 20.0 + 3.0 * row + col
                tokens[row, col, 1] = -0.2 + 0.04 * col + 0.01 * row
                tokens[row, col, 2] = -math.pi + 0.03 if col == 0 else -2.0 + 0.31 * col
                tokens[row, col, 3] = tokens[row, col, 0] + 5.0
                tokens[row, col, 4] = 1.0 if col % 2 == 0 else -1.0
                tokens[row, col, 5 + (col % 5)] = 1.0
        tokens[:, 1, 2] = math.pi - 0.04
        tokens = tokens * mask[:, :, None].float()
        return tokens, mask

    def test_pairwise_feature_builder_shapes_and_wraparound(self):
        torch = require_torch()
        tokens, mask = self.make_tokens(batch=1, particles=4)
        output = PairwiseDenoisingFeatureBuilder(PairwiseDenoisingFeatureConfig())(tokens, mask)
        names = list(output.feature_names)

        self.assertEqual(PAIRWISE_DENOISING_FEATURE_DIM, len(PAIRWISE_DENOISING_FEATURE_NAMES))
        self.assertEqual(tuple(output.pair_features.shape), (1, 4, 4, PAIRWISE_DENOISING_FEATURE_DIM))
        self.assertEqual(tuple(output.pair_mask.shape), (1, 4, 4))
        self.assertTrue(bool((output.pair_features[~output.pair_mask] == 0.0).all()))

        sin_index = names.index("sin_delta_phi")
        abs_index = names.index("abs_delta_phi")
        delta = wrap_delta_phi_torch(tokens[0, 0, 2] - tokens[0, 1, 2])
        self.assertLess(float(delta.abs().detach().cpu().item()), 0.12)
        self.assertAlmostEqual(
            float(output.pair_features[0, 0, 1, sin_index].detach().cpu().item()),
            float(torch.sin(delta).detach().cpu().item()),
            places=6,
        )
        self.assertAlmostEqual(
            float(output.pair_features[0, 0, 1, abs_index].detach().cpu().item()),
            float((delta.abs() / math.pi).detach().cpu().item()),
            places=6,
        )

    def test_pair_bias_encoder_masks_and_can_be_nonzero(self):
        torch = require_torch()
        tokens, mask = self.make_tokens(batch=1, particles=4)
        pairwise = PairwiseDenoisingFeatureBuilder()(tokens, mask)
        encoder = DenoisingPairBiasEncoder(
            DenoisingPairBiasConfig(num_heads=2, hidden_dim=16, zero_init=False, max_abs_bias=2.0)
        )

        bias = encoder(pairwise)

        self.assertEqual(tuple(bias.shape), (1, 2, 4, 4))
        self.assertTrue(bool(torch.isfinite(bias).all()))
        self.assertTrue(bool((bias.permute(0, 2, 3, 1)[~pairwise.pair_mask] == 0.0).all()))
        self.assertGreater(float(bias.abs().sum().detach().cpu().item()), 0.0)

    def test_denoiser_zero_init_outputs_masked_finite_baseline(self):
        torch = require_torch()
        torch.manual_seed(7)
        tokens, mask = self.make_tokens()
        model = TargetConditionedPairwiseDenoiser(
            TargetConditionedDenoiserConfig(embed_dim=32, num_heads=4, zero_init=True, dropout=0.0)
        )
        model.eval()

        output = model(tokens, mask, need_weights=True)

        self.assertEqual(tuple(output.deltas.shape), (2, 5, len(DENOISING_TARGET_NAMES)))
        self.assertEqual(tuple(output.log_variances.shape), (2, 5, len(DENOISING_TARGET_NAMES)))
        self.assertEqual(tuple(output.reliability.shape), (2, 5))
        self.assertEqual(tuple(output.reliability_logits.shape), (2, 5))
        self.assertTrue(bool(torch.isfinite(output.deltas).all()))
        self.assertTrue(bool(torch.isfinite(output.log_variances).all()))
        self.assertTrue(bool(torch.isfinite(output.reliability).all()))
        self.assertTrue(bool(torch.isfinite(output.reliability_logits).all()))
        self.assertTrue(bool(torch.allclose(output.deltas, torch.zeros_like(output.deltas))))
        self.assertTrue(bool(torch.allclose(output.log_variances, torch.zeros_like(output.log_variances))))
        self.assertTrue(bool((output.deltas[~mask] == 0.0).all()))
        self.assertTrue(bool((output.reliability[~mask] == 0.0).all()))
        self.assertTrue(bool((output.reliability_logits[~mask] == 0.0).all()))
        self.assertTrue(bool(torch.allclose(output.reliability[mask], torch.full_like(output.reliability[mask], 0.5))))
        self.assertTrue(bool(torch.allclose(output.reliability_logits[mask], torch.zeros_like(output.reliability_logits[mask]))))
        self.assertIsNotNone(output.attention_weights)
        self.assertTrue(bool((output.attention_weights[:, :, :, ~mask[0]] == 0.0).all()))

    def test_pair_bias_changes_attention_weights(self):
        torch = require_torch()
        torch.manual_seed(11)
        tokens, mask = self.make_tokens(batch=1, particles=4)
        model = TargetConditionedPairwiseDenoiser(
            TargetConditionedDenoiserConfig(embed_dim=32, num_heads=4, zero_init=True, dropout=0.0)
        )
        model.eval()

        baseline = model(tokens, mask, need_weights=True)
        with torch.no_grad():
            model.local_kernel_gate.fill_(4.0)
        changed = model(tokens, mask, need_weights=True)

        self.assertIsNotNone(baseline.attention_weights)
        self.assertIsNotNone(changed.attention_weights)
        diff = (baseline.attention_weights - changed.attention_weights).abs().sum()
        self.assertGreater(float(diff.detach().cpu().item()), 1.0e-6)
        self.assertTrue(bool(torch.allclose(changed.deltas, torch.zeros_like(changed.deltas))))

    def test_no_pair_bias_variant_still_forwards(self):
        torch = require_torch()
        tokens, mask = self.make_tokens(batch=1, particles=3)
        model = TargetConditionedPairwiseDenoiser(
            TargetConditionedDenoiserConfig(embed_dim=16, num_heads=4, use_pair_bias=False, zero_init=True)
        )
        output = model(tokens, mask)
        self.assertEqual(tuple(output.attention_bias.shape), (1, 4, 3, 3))
        self.assertTrue(bool(torch.isfinite(output.deltas).all()))
        self.assertEqual(output.summary()["contract"], "target_conditioned_pairwise_denoising_model_v1")


if __name__ == "__main__":
    unittest.main()
