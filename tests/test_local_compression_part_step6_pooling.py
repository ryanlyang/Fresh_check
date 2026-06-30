import math
import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_compression_part import (
    LOCAL_COMPRESSION_MODALITIES,
    LOCAL_COMPRESSION_POOL_LEARNED_QUERY,
    LOCAL_COMPRESSION_POOL_MEAN,
    LOCAL_COMPRESSION_POOLING_CONTRACT,
    LocalCompressionPartConfig,
    LocalCompressionProvisionalPooler,
    LocalCompressionSubtokenEncoder,
    LocalModalityCompressor,
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
            pt = 12.0 + 2.0 * particle_index + 0.5 * batch_index
            eta = -0.25 + 0.08 * particle_index
            phi = -math.pi + 0.21 * particle_index
            tokens[batch_index, particle_index, 0] = pt
            tokens[batch_index, particle_index, 1] = eta
            tokens[batch_index, particle_index, 2] = phi
            tokens[batch_index, particle_index, 3] = pt * math.cosh(eta) + 0.2
            tokens[batch_index, particle_index, 4] = 1.0 if particle_index % 2 == 0 else -1.0
            tokens[batch_index, particle_index, 5 + (particle_index % 5)] = 1.0
            tokens[batch_index, particle_index, 10] = 0.01 * particle_index
            tokens[batch_index, particle_index, 11] = 0.04 + 0.01 * particle_index
            tokens[batch_index, particle_index, 12] = -0.02 * particle_index
            tokens[batch_index, particle_index, 13] = 0.06 + 0.01 * batch_index
    return tokens, mask


def make_compressed(embed_dim: int = 16, *, include_empty_batch: bool = False, pool_mode: str = LOCAL_COMPRESSION_POOL_LEARNED_QUERY):
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
        context_heads=4,
        dropout=0.0,
        attention_dropout=0.0,
        pool_mode=pool_mode,
    )
    subtoken_output = LocalCompressionSubtokenEncoder(config)(canonical, modalities)
    compressed = LocalModalityCompressor(config)(subtoken_output)
    return config, compressed


class LocalCompressionStep6PoolingTests(unittest.TestCase):
    def test_learned_query_pooling_shapes_and_contract(self):
        config, compressed = make_compressed(embed_dim=16)
        pooler = LocalCompressionProvisionalPooler(config)

        output = pooler(compressed)

        self.assertEqual(tuple(output.local_particle_token.shape), (3, 6, 16))
        self.assertEqual(tuple(output.pool_weights.shape), (3, 6, len(LOCAL_COMPRESSION_MODALITIES)))
        self.assertTrue(torch.equal(output.mask, compressed.mask))
        self.assertTrue(torch.equal(output.modality_mask, compressed.modality_mask))
        self.assertEqual(output.modality_names, LOCAL_COMPRESSION_MODALITIES)
        self.assertEqual(output.summary()["contract"], LOCAL_COMPRESSION_POOLING_CONTRACT)
        self.assertEqual(output.pool_mode, LOCAL_COMPRESSION_POOL_LEARNED_QUERY)
        self.assertTrue(torch.isfinite(output.local_particle_token).all())

    def test_inactive_modalities_get_zero_weight_and_active_weights_sum_to_one(self):
        config, compressed = make_compressed(embed_dim=16)
        modality_mask = compressed.modality_mask.clone()
        modality_mask[:, :, 1] = False
        modality_mask[0, 0, 3] = False
        pooler = LocalCompressionProvisionalPooler(config)

        output = pooler(compressed.local_tokens, mask=compressed.mask, modality_mask=modality_mask)

        self.assertEqual(float(output.pool_weights[:, :, 1].abs().sum().item()), 0.0)
        self.assertEqual(float(output.pool_weights[0, 0, 3].abs().item()), 0.0)
        active_particle_mask = modality_mask.any(dim=-1)
        self.assertTrue(torch.allclose(output.pool_weights.sum(dim=-1), active_particle_mask.float(), atol=1e-6))

    def test_invalid_particles_and_empty_batch_rows_produce_zero_token(self):
        config, compressed = make_compressed(embed_dim=16, include_empty_batch=True)
        pooler = LocalCompressionProvisionalPooler(config)

        output = pooler(compressed)

        self.assertEqual(float(output.local_particle_token[0, 5].abs().sum().item()), 0.0)
        self.assertEqual(float(output.pool_weights[0, 5].abs().sum().item()), 0.0)
        self.assertEqual(float(output.local_particle_token[3].abs().sum().item()), 0.0)
        self.assertEqual(float(output.pool_weights[3].abs().sum().item()), 0.0)

    def test_mean_pooling_matches_manual_active_modality_average(self):
        config, compressed = make_compressed(embed_dim=16, pool_mode=LOCAL_COMPRESSION_POOL_MEAN)
        modality_mask = compressed.modality_mask.clone()
        modality_mask[:, :, 2] = False
        pooler = LocalCompressionProvisionalPooler(config)

        output = pooler(compressed.local_tokens, mask=compressed.mask, modality_mask=modality_mask)
        active_tokens = torch.where(modality_mask[:, :, :, None], compressed.local_tokens, torch.zeros_like(compressed.local_tokens))
        counts = modality_mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
        manual = active_tokens.sum(dim=2) / counts
        manual = torch.where(compressed.mask[:, :, None], manual, torch.zeros_like(manual))

        self.assertEqual(output.pool_mode, LOCAL_COMPRESSION_POOL_MEAN)
        self.assertTrue(torch.allclose(output.local_particle_token, manual, atol=1e-6))
        self.assertEqual(float(output.pool_weights[:, :, 2].abs().sum().item()), 0.0)

    def test_gradients_flow_through_learned_query_pooling(self):
        config, compressed = make_compressed(embed_dim=16)
        local_tokens = compressed.local_tokens.detach().clone().requires_grad_(True)
        pooler = LocalCompressionProvisionalPooler(config)

        output = pooler(local_tokens, mask=compressed.mask, modality_mask=compressed.modality_mask)
        loss = output.local_particle_token[output.mask].pow(2).mean()
        loss.backward()

        self.assertIsNotNone(local_tokens.grad)
        self.assertGreater(float(local_tokens.grad.abs().sum().item()), 0.0)
        self.assertIsNotNone(pooler.query.grad)


if __name__ == "__main__":
    unittest.main()
