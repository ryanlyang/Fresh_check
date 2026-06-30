import unittest

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES,
    LOCAL_GRAPH_RESIDUAL_V2_MODEL_CONTRACT,
    LocalGraphResidualExpertV2Config,
    build_local_graph_residual_expert_v2,
)


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


class LocalGraphResidualV2Step6ModelTest(unittest.TestCase):
    def config(self) -> LocalGraphResidualExpertV2Config:
        return LocalGraphResidualExpertV2Config(
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
            local_adapter_gamma_init=1.0,
            gamma_initial=0.1,
            gamma_learnable=False,
            delta_init_std=1.0e-3,
        )

    def test_v2_outputs_additive_residual_around_frozen_baseline(self):
        torch = require_torch()
        torch.manual_seed(2201)
        config = self.config()
        model = build_local_graph_residual_expert_v2(config)
        tokens, mask = make_tokens()
        z_base, embedding, condition = make_baseline_inputs(batch_size=int(tokens.shape[0]), embedding_dim=9)
        embedding.requires_grad_(True)

        output = model(
            tokens.float(),
            mask.bool(),
            baseline_logit=z_base,
            baseline_embedding=embedding,
            baseline_condition_features=condition,
            return_outputs=True,
        )

        self.assertEqual(output.summary()["contract"], LOCAL_GRAPH_RESIDUAL_V2_MODEL_CONTRACT)
        self.assertEqual(tuple(output.fused_logits.shape), (int(tokens.shape[0]), 2))
        self.assertEqual(tuple(output.residual_logits.shape), (int(tokens.shape[0]), 2))
        self.assertEqual(tuple(output.correction_logits.shape), (int(tokens.shape[0]), 2))
        self.assertEqual(tuple(output.baseline_embedding.shape), (int(tokens.shape[0]), 9))
        self.assertEqual(tuple(output.local_context.shape), (int(tokens.shape[0]), config.local_context_dim))
        self.assertEqual(
            tuple(output.condition_features.shape),
            (int(tokens.shape[0]), len(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES)),
        )
        torch.testing.assert_close(output.correction_logit, output.gamma * output.delta)
        torch.testing.assert_close(output.fused_logit, z_base + output.correction_logit)
        self.assertFalse(output.baseline_embedding.requires_grad)
        self.assertLess(float((output.fused_logit - z_base).detach().abs().mean()), 1.0e-2)

    def test_v2_accepts_mapping_inputs(self):
        config = self.config()
        model = build_local_graph_residual_expert_v2(config)
        tokens, mask = make_tokens(batch_size=3)
        z_base, embedding, condition = make_baseline_inputs(batch_size=3, embedding_dim=9)

        output = model(
            {
                "tokens": tokens.float(),
                "mask": mask.bool(),
                "z_base": z_base,
                "baseline_embedding": embedding,
                "condition_features": condition,
            },
            return_outputs=True,
        )

        self.assertEqual(tuple(output.fused_logits.shape), (3, 2))
        self.assertTrue(bool(require_torch().isfinite(output.fused_logits).all()))

    def test_v2_residual_path_receives_gradients(self):
        torch = require_torch()
        torch.manual_seed(2203)
        config = self.config()
        model = build_local_graph_residual_expert_v2(config)
        tokens, mask = make_tokens(batch_size=3)
        z_base, embedding, condition = make_baseline_inputs(batch_size=3, embedding_dim=9)

        output = model(
            tokens.float(),
            mask.bool(),
            baseline_logit=z_base,
            baseline_embedding=embedding,
            baseline_condition_features=condition,
            return_outputs=True,
        )
        loss = output.fused_logits[:, 1].sum()
        loss.backward()

        self.assertIsNotNone(model.delta_head.weight.grad)
        self.assertTrue(bool(torch.isfinite(model.delta_head.weight.grad).all()))
        self.assertGreater(float(model.delta_head.weight.grad.detach().abs().sum()), 0.0)
        local_grad = sum(
            float(param.grad.detach().abs().sum())
            for name, param in model.named_parameters()
            if name.startswith("raw_embed") and param.grad is not None
        )
        self.assertGreater(local_grad, 0.0)

    def test_v2_requires_baseline_embedding_and_shape_contract(self):
        config = self.config()
        model = build_local_graph_residual_expert_v2(config)
        tokens, mask = make_tokens(batch_size=2)
        z_base, embedding, condition = make_baseline_inputs(batch_size=2, embedding_dim=9)

        with self.assertRaisesRegex(ValueError, "baseline_embedding is required"):
            model(
                tokens.float(),
                mask.bool(),
                baseline_logit=z_base,
                baseline_condition_features=condition,
            )
        with self.assertRaisesRegex(ValueError, "baseline_embedding dim"):
            model(
                tokens.float(),
                mask.bool(),
                baseline_logit=z_base,
                baseline_embedding=embedding[:, :4],
                baseline_condition_features=condition,
            )


if __name__ == "__main__":
    unittest.main()
