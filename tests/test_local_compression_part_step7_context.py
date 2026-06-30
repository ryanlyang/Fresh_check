import math
import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_compression_part import (
    LOCAL_COMPRESSION_CONTEXT_CONTRACT,
    LocalCompressionPartConfig,
    LocalCompressionProvisionalPooler,
    LocalCompressionSubtokenEncoder,
    LocalModalityCompressor,
    ParticleContextBlock,
    build_local_compression_modalities_from_tokens,
)


torch = require_torch()


def make_tokens(num_particles: int = 6, *, include_empty_batch: bool = False):
    batch_size = 4 if include_empty_batch else 3
    tokens = torch.zeros((batch_size, num_particles, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.zeros((batch_size, num_particles), dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :4] = True
    mask[2, :3] = True
    if include_empty_batch:
        mask[3, :] = False

    for batch_index in range(batch_size):
        for particle_index in range(int(mask[batch_index].sum().item())):
            pt = 10.0 + 3.0 * particle_index + 0.9 * batch_index
            eta = -0.3 + 0.09 * particle_index
            phi = -math.pi + 0.17 * particle_index
            tokens[batch_index, particle_index, 0] = pt
            tokens[batch_index, particle_index, 1] = eta
            tokens[batch_index, particle_index, 2] = phi
            tokens[batch_index, particle_index, 3] = pt * math.cosh(eta) + 0.4
            tokens[batch_index, particle_index, 4] = 1.0 if particle_index % 2 == 0 else -1.0
            tokens[batch_index, particle_index, 5 + (particle_index % 5)] = 1.0
            tokens[batch_index, particle_index, 10] = 0.02 * particle_index
            tokens[batch_index, particle_index, 11] = 0.05 + 0.01 * particle_index
            tokens[batch_index, particle_index, 12] = -0.03 * particle_index
            tokens[batch_index, particle_index, 13] = 0.07 + 0.01 * batch_index
    return tokens, mask


def make_pooled(embed_dim: int = 16, *, include_empty_batch: bool = False):
    tokens, mask = make_tokens(include_empty_batch=include_empty_batch)
    canonical, modalities = build_local_compression_modalities_from_tokens(
        tokens,
        mask,
        max_constits=tokens.shape[1],
    )
    config = LocalCompressionPartConfig(
        embed_dim=embed_dim,
        local_layers=1,
        local_heads=4,
        context_layers=1,
        context_heads=4,
        dropout=0.0,
        attention_dropout=0.0,
    )
    subtokens = LocalCompressionSubtokenEncoder(config)(canonical, modalities)
    compressed = LocalModalityCompressor(config)(subtokens)
    pooled = LocalCompressionProvisionalPooler(config)(compressed)
    return config, pooled


class LocalCompressionStep7ContextTests(unittest.TestCase):
    def test_context_shapes_contract_and_masks(self):
        config, pooled = make_pooled(embed_dim=16)
        context = ParticleContextBlock(config)

        output = context(pooled)

        self.assertEqual(tuple(output.context_tokens.shape), tuple(pooled.local_particle_token.shape))
        self.assertTrue(torch.equal(output.mask, pooled.mask))
        self.assertEqual(output.summary()["contract"], LOCAL_COMPRESSION_CONTEXT_CONTRACT)
        self.assertTrue(torch.isfinite(output.context_tokens).all())
        self.assertEqual(float(output.context_tokens[0, 5].abs().sum().item()), 0.0)

    def test_context_is_finite_with_empty_particles_and_empty_batch_rows(self):
        config, pooled = make_pooled(embed_dim=16, include_empty_batch=True)
        context = ParticleContextBlock(config)

        output = context(pooled)

        self.assertTrue(torch.isfinite(output.context_tokens).all())
        self.assertEqual(float(output.context_tokens[3].abs().sum().item()), 0.0)
        self.assertEqual(float(output.input_particles[3].abs().sum().item()), 0.0)

    def test_context_changes_when_neighboring_particle_changes(self):
        config, pooled = make_pooled(embed_dim=16)
        context = ParticleContextBlock(config)
        context.eval()
        particles = pooled.local_particle_token.detach().clone()
        changed = particles.clone()
        changed[0, 1, 0] = changed[0, 1, 0] + 4.0

        with torch.no_grad():
            base = context(particles, mask=pooled.mask).context_tokens
            altered = context(changed, mask=pooled.mask).context_tokens

        self.assertGreater(float((base[0, 0] - altered[0, 0]).abs().sum().item()), 1.0e-5)

    def test_invalid_particles_remain_zero_even_with_nonzero_input(self):
        config, pooled = make_pooled(embed_dim=16)
        context = ParticleContextBlock(config)
        particles = pooled.local_particle_token.detach().clone()
        particles[0, 5] = 100.0

        output = context(particles, mask=pooled.mask)

        self.assertEqual(float(output.input_particles[0, 5].abs().sum().item()), 0.0)
        self.assertEqual(float(output.context_tokens[0, 5].abs().sum().item()), 0.0)

    def test_attention_dropout_is_separate_from_residual_dropout(self):
        config = LocalCompressionPartConfig(
            embed_dim=16,
            local_layers=1,
            local_heads=4,
            context_layers=1,
            context_heads=4,
            dropout=0.3,
            attention_dropout=0.0,
        )
        context = ParticleContextBlock(config)
        layer = context.layers[0]

        self.assertAlmostEqual(float(layer.self_attn.dropout), 0.0, places=7)
        self.assertAlmostEqual(float(layer.residual_dropout.p), 0.3, places=7)
        self.assertAlmostEqual(float(layer.ffn_dropout.p), 0.3, places=7)

    def test_gradients_flow_through_context_block(self):
        config, pooled = make_pooled(embed_dim=16)
        particles = pooled.local_particle_token.detach().clone().requires_grad_(True)
        context = ParticleContextBlock(config)

        output = context(particles, mask=pooled.mask)
        loss = output.context_tokens[output.mask].pow(2).mean()
        loss.backward()

        self.assertIsNotNone(particles.grad)
        self.assertGreater(float(particles.grad.abs().sum().item()), 0.0)
        grad_sums = [
            float(parameter.grad.detach().abs().sum().item())
            for parameter in context.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(any(value > 0.0 for value in grad_sums))


if __name__ == "__main__":
    unittest.main()
