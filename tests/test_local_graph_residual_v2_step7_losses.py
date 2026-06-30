import types
import unittest

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE,
    LocalGraphResidualV2LossConfig,
    boundary_pairwise_v2_loss,
    compute_local_graph_residual_v2_loss,
    compute_local_graph_residual_v2_loss_from_output,
    normalize_local_graph_residual_v2_loss_mode,
    residual_v2_bce_weights,
    soft_fpr50_v2_loss,
)

from teacher_logit_reco.local_graph_part import LocalGraphResidualExpertV2Config, build_local_graph_residual_expert_v2


def make_tokens(batch_size: int = 4, particles: int = 7):
    torch = require_torch()
    tokens = np.zeros((batch_size, particles, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((batch_size, particles), dtype=bool)
    for batch in range(batch_size):
        valid = particles - (batch % 3)
        mask[batch, :valid] = True
        for particle in range(valid):
            tokens[batch, particle, 0] = 16.0 + batch + 0.75 * particle
            tokens[batch, particle, 1] = -0.4 + 0.12 * particle
            tokens[batch, particle, 2] = -0.25 + 0.15 * particle
            tokens[batch, particle, 3] = tokens[batch, particle, 0] * np.cosh(tokens[batch, particle, 1]) + 0.2
            tokens[batch, particle, 4] = -1.0 + (particle % 3)
            tokens[batch, particle, 5 + (particle % 5)] = 1.0
            tokens[batch, particle, 10] = 0.03 * particle
            tokens[batch, particle, 11] = 0.05
            tokens[batch, particle, 12] = -0.02 * particle
            tokens[batch, particle, 13] = 0.04
    return torch.from_numpy(tokens), torch.from_numpy(mask)


def make_baseline_inputs(batch_size: int, embedding_dim: int = 9):
    torch = require_torch()
    z_base = torch.linspace(-1.4, 1.2, steps=batch_size, dtype=torch.float32)
    embedding = torch.randn(batch_size, embedding_dim, generator=torch.Generator().manual_seed(1401))
    condition = torch.stack(
        (
            z_base,
            torch.sigmoid(z_base),
            z_base - 0.2,
            torch.abs(z_base - 0.2),
            z_base - 0.8,
            torch.exp(-torch.abs(z_base - 0.2)),
        ),
        dim=1,
    )
    return z_base, embedding, condition


class LocalGraphResidualV2Step7LossesTest(unittest.TestCase):
    def test_mode_aliases_and_active_weights_are_v2_specific(self):
        self.assertEqual(normalize_local_graph_residual_v2_loss_mode("A"), LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE)
        self.assertEqual(
            normalize_local_graph_residual_v2_loss_mode("boundary-pairwise-bce-anchor"),
            LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
        )
        with self.assertRaisesRegex(ValueError, "E is not a training loss"):
            normalize_local_graph_residual_v2_loss_mode("E")

        weights_a = LocalGraphResidualV2LossConfig(mode="A").active_weights()
        weights_c = LocalGraphResidualV2LossConfig(mode="C").active_weights()
        weights_d = LocalGraphResidualV2LossConfig(mode="D").active_weights()

        self.assertGreater(weights_a["weighted_bce"], 0.0)
        self.assertEqual(weights_a["boundary_pairwise"], 0.0)
        self.assertGreater(weights_c["boundary_pairwise"], 0.0)
        self.assertGreater(weights_c["weighted_bce"], 0.0)
        self.assertEqual(weights_c["soft_fpr50"], 0.0)
        self.assertGreater(weights_d["soft_fpr50"], 0.0)

    def test_bce_weights_emphasize_baseline_false_positives_and_boundary_signal(self):
        torch = require_torch()
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        baseline = torch.tensor([0.7, -1.0, 0.45, 1.2])
        config = LocalGraphResidualV2LossConfig(mode="A", normalize_bce_weights=False, bce_boundary_scale=0.25)

        weights = residual_v2_bce_weights(labels, baseline, tau50=0.5, config=config)

        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertGreater(float(weights[2]), float(weights[3]))
        self.assertTrue(bool(torch.isfinite(weights).all()))

    def test_pairwise_and_soft_fpr_terms_are_finite_and_directional(self):
        torch = require_torch()
        labels = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        baseline = torch.tensor([0.8, 0.4, -0.8, 0.35, 0.55, 1.2])
        bad_fused = torch.tensor([1.2, 0.2, -0.6, 0.1, 0.3, 0.9])
        good_fused = torch.tensor([-0.5, -0.7, -1.0, 0.6, 0.8, 1.3])
        config = LocalGraphResidualV2LossConfig(mode=LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE)

        bad_pairwise, pair_diag = boundary_pairwise_v2_loss(bad_fused, labels, baseline, tau50=0.5, config=config)
        good_pairwise, _ = boundary_pairwise_v2_loss(good_fused, labels, baseline, tau50=0.5, config=config)
        bad_soft, soft_diag = soft_fpr50_v2_loss(bad_fused, labels, tau50=0.5, config=config)
        good_soft, _ = soft_fpr50_v2_loss(good_fused, labels, tau50=0.5, config=config)

        self.assertTrue(bool(torch.isfinite(bad_pairwise)))
        self.assertTrue(bool(torch.isfinite(bad_soft)))
        self.assertGreater(float(bad_pairwise), float(good_pairwise))
        self.assertGreater(float(bad_soft), float(good_soft))
        self.assertGreater(float(pair_diag["pair_count"]), 0.0)
        self.assertGreater(float(soft_diag["soft_fpr_background_count"]), 0.0)

    def test_compute_loss_components_and_gradient_into_correction(self):
        torch = require_torch()
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        baseline = torch.tensor([0.8, -0.4, 0.2, 0.9, 0.6, 0.45])
        correction = torch.zeros_like(baseline, requires_grad=True)
        fused = baseline.detach() + correction
        config = LocalGraphResidualV2LossConfig(mode=LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR)

        loss = compute_local_graph_residual_v2_loss(
            fused_logit=fused,
            labels=labels,
            baseline_logit=baseline,
            correction_logit=correction,
            tau50=0.5,
            config=config,
        )
        loss.total_loss.backward()

        self.assertEqual(loss.diagnostics["contract"], LOCAL_GRAPH_RESIDUAL_V2_LOSS_CONTRACT)
        self.assertEqual(loss.config.mode, LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR)
        self.assertGreater(loss.weights["boundary_pairwise"], 0.0)
        self.assertGreater(loss.weights["soft_fpr50"], 0.0)
        self.assertIn("correction_l2", loss.components)
        self.assertIsNotNone(correction.grad)
        self.assertGreater(float(correction.grad.detach().abs().sum()), 0.0)

    def test_compute_loss_from_v2_model_output(self):
        torch = require_torch()
        torch.manual_seed(2707)
        model = build_local_graph_residual_expert_v2(
            LocalGraphResidualExpertV2Config(
                baseline_embedding_dim=9,
                max_constits=7,
                k=3,
                local_embed_dim=16,
                local_heads=4,
                local_hidden_dim=24,
                local_context_dim=12,
                condition_embed_dim=5,
                residual_hidden_dim=18,
                dropout=0.0,
                attention_dropout=0.0,
                residual_dropout=0.0,
                gamma_initial=0.1,
                gamma_learnable=False,
            )
        )
        tokens, mask = make_tokens(batch_size=4)
        z_base, embedding, condition = make_baseline_inputs(batch_size=4, embedding_dim=9)
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
        output = model(
            tokens.float(),
            mask.bool(),
            baseline_logit=z_base,
            baseline_embedding=embedding,
            baseline_condition_features=condition,
            return_outputs=True,
        )

        loss = compute_local_graph_residual_v2_loss_from_output(
            output,
            labels,
            tau50=0.2,
            config=LocalGraphResidualV2LossConfig(mode=LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE),
        )
        loss.total_loss.backward()

        self.assertTrue(bool(torch.isfinite(loss.total_loss)))
        self.assertGreater(float(model.delta_head.weight.grad.detach().abs().sum()), 0.0)
        self.assertEqual(loss.config.mode, LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE)

    def test_compute_from_plain_output_object(self):
        torch = require_torch()
        labels = torch.tensor([0.0, 1.0])
        baseline = torch.tensor([-0.5, 0.5])
        correction = torch.tensor([0.1, -0.1], requires_grad=True)
        output = types.SimpleNamespace(
            fused_logit=baseline + correction,
            baseline_logit=baseline,
            correction_logit=correction,
        )

        loss = compute_local_graph_residual_v2_loss_from_output(output, labels, tau50=0.0)
        self.assertTrue(bool(torch.isfinite(loss.total_loss)))


if __name__ == "__main__":
    unittest.main()
