import unittest

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES,
    LOCAL_GRAPH_RESIDUAL_EXPERT_CONTRACT,
    LocalGraphResidualExpertConfig,
    build_local_graph_residual_expert,
)


_TORCH = require_torch()


class TinyEmbeddingPartBackbone(_TORCH.nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        torch = require_torch()
        self.config = {"architecture": "tiny_embedding_part_backbone", "output_dim": int(output_dim)}
        self.proj = torch.nn.Linear(17 + 4 + 1, int(output_dim))

    def __call__(self, points, features, lorentz_vectors, mask):
        torch = require_torch()
        del points
        particle_mask = mask.float()
        packed = torch.cat([features, lorentz_vectors, particle_mask], dim=1).transpose(1, 2)
        denom = torch.clamp(particle_mask.squeeze(1).sum(dim=1, keepdim=True), min=1.0)
        pooled = (packed * particle_mask.transpose(1, 2)).sum(dim=1) / denom
        return self.proj(pooled)

    def no_weight_decay(self):
        return set()


def make_tokens(batch_size: int = 4, particles: int = 6):
    tokens = np.zeros((batch_size, particles, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((batch_size, particles), dtype=bool)
    for batch in range(batch_size):
        valid = particles - (batch % 2)
        mask[batch, :valid] = True
        for particle in range(valid):
            tokens[batch, particle, 0] = 20.0 + 0.5 * batch + particle
            tokens[batch, particle, 1] = -0.3 + 0.1 * particle
            tokens[batch, particle, 2] = -0.2 + 0.2 * particle
            tokens[batch, particle, 3] = tokens[batch, particle, 0] * np.cosh(tokens[batch, particle, 1]) + 0.1
            tokens[batch, particle, 4] = -1.0 + (particle % 3)
            tokens[batch, particle, 5 + (particle % 5)] = 1.0
            tokens[batch, particle, 10] = 0.02 * particle
            tokens[batch, particle, 11] = 0.04
            tokens[batch, particle, 12] = -0.03 * particle
            tokens[batch, particle, 13] = 0.08
    torch = require_torch()
    return torch.from_numpy(tokens), torch.from_numpy(mask)


class LocalGraphResidualExpertStep2Tests(unittest.TestCase):
    def config(self) -> LocalGraphResidualExpertConfig:
        return LocalGraphResidualExpertConfig(
            model_size="tiny",
            max_constits=6,
            local_adapter="point_attention",
            k=2,
            local_embed_dim=16,
            local_heads=2,
            dropout=0.0,
            attention_dropout=0.0,
            residual_gamma_init=0.01,
            backbone_output_dim=8,
            condition_embed_dim=4,
            residual_hidden_dim=12,
            residual_dropout=0.0,
            alpha_initial=0.25,
            alpha_learnable=False,
        )

    def test_residual_expert_outputs_additive_fused_logits(self):
        torch = require_torch()
        config = self.config()
        model = build_local_graph_residual_expert(
            config,
            part_model=TinyEmbeddingPartBackbone(output_dim=config.backbone_output_dim),
        )
        tokens, mask = make_tokens()
        z_base = torch.linspace(-1.0, 1.0, steps=int(tokens.shape[0]))
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

        output = model(
            tokens.float(),
            mask.bool(),
            baseline_logit=z_base,
            baseline_condition_features=condition,
            return_outputs=True,
        )

        self.assertEqual(output.summary()["contract"], LOCAL_GRAPH_RESIDUAL_EXPERT_CONTRACT)
        self.assertEqual(tuple(output.fused_logits.shape), (int(tokens.shape[0]), 2))
        self.assertEqual(tuple(output.residual_logit.shape), (int(tokens.shape[0]),))
        self.assertEqual(tuple(output.backbone_embedding.shape), (int(tokens.shape[0]), config.backbone_output_dim))
        self.assertEqual(tuple(output.condition_features.shape), (int(tokens.shape[0]), len(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES)))
        torch.testing.assert_close(output.fused_logit, z_base + output.alpha * output.residual_logit)

    def test_forward_accepts_mapping_and_fallback_condition_features(self):
        config = self.config()
        model = build_local_graph_residual_expert(
            config,
            part_model=TinyEmbeddingPartBackbone(output_dim=config.backbone_output_dim),
        )
        tokens, mask = make_tokens(batch_size=3)
        z_base = require_torch().tensor([-0.5, 0.1, 1.0], dtype=require_torch().float32)

        output = model({"tokens": tokens.float(), "mask": mask.bool(), "z_base": z_base}, return_outputs=True)

        self.assertEqual(tuple(output.condition_features.shape), (3, len(LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES)))
        self.assertTrue(require_torch().isfinite(output.fused_logits).all())

    def test_residual_head_has_gradient_path(self):
        torch = require_torch()
        config = self.config()
        model = build_local_graph_residual_expert(
            config,
            part_model=TinyEmbeddingPartBackbone(output_dim=config.backbone_output_dim),
        )
        tokens, mask = make_tokens(batch_size=3)
        z_base = torch.tensor([-0.5, 0.2, 0.8], dtype=torch.float32)

        output = model(tokens.float(), mask.bool(), baseline_logit=z_base, return_outputs=True)
        loss = output.fused_logits[:, 1].sum()
        loss.backward()

        grad = model.raw_residual_head.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
