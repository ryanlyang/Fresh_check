import math
import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_compression_part import (
    LOCAL_COMPRESSION_COMPRESSOR_CONTRACT,
    LOCAL_COMPRESSION_MODALITIES,
    LocalCompressionPartConfig,
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
            pt = 11.0 + 2.5 * particle_index + 0.7 * batch_index
            eta = -0.35 + 0.07 * particle_index + 0.02 * batch_index
            phi = -math.pi + 0.23 * particle_index
            tokens[batch_index, particle_index, 0] = pt
            tokens[batch_index, particle_index, 1] = eta
            tokens[batch_index, particle_index, 2] = phi
            tokens[batch_index, particle_index, 3] = pt * math.cosh(eta) + 0.3
            tokens[batch_index, particle_index, 4] = 1.0 if particle_index % 2 == 0 else -1.0
            tokens[batch_index, particle_index, 5 + (particle_index % 5)] = 1.0
            tokens[batch_index, particle_index, 10] = 0.01 * particle_index
            tokens[batch_index, particle_index, 11] = 0.05 + 0.01 * particle_index
            tokens[batch_index, particle_index, 12] = -0.02 * particle_index
            tokens[batch_index, particle_index, 13] = 0.07 + 0.01 * batch_index
    return tokens, mask


def make_subtokens(embed_dim: int = 16, *, include_empty_batch: bool = False):
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
    )
    encoder = LocalCompressionSubtokenEncoder(config)
    subtoken_output = encoder(canonical, modalities)
    return config, subtoken_output


class LocalCompressionStep5CompressorTests(unittest.TestCase):
    def test_compressor_preserves_shape_and_masks(self):
        config, subtoken_output = make_subtokens(embed_dim=16)
        compressor = LocalModalityCompressor(config)

        output = compressor(subtoken_output)

        self.assertEqual(tuple(output.local_tokens.shape), tuple(subtoken_output.subtokens.shape))
        self.assertTrue(torch.equal(output.mask, subtoken_output.mask))
        self.assertTrue(torch.equal(output.modality_mask, subtoken_output.modality_mask))
        self.assertEqual(output.modality_names, LOCAL_COMPRESSION_MODALITIES)
        self.assertEqual(output.summary()["contract"], LOCAL_COMPRESSION_COMPRESSOR_CONTRACT)
        self.assertTrue(torch.isfinite(output.local_tokens).all())
        self.assertEqual(float(output.local_tokens[0, 5].abs().sum().item()), 0.0)

    def test_compressor_is_finite_with_empty_particles_and_empty_batch_rows(self):
        config, subtoken_output = make_subtokens(embed_dim=16, include_empty_batch=True)
        compressor = LocalModalityCompressor(config)

        output = compressor(subtoken_output)

        self.assertTrue(torch.isfinite(output.local_tokens).all())
        self.assertEqual(float(output.local_tokens[3].abs().sum().item()), 0.0)
        self.assertFalse(bool(output.modality_mask[3].any().item()))

    def test_modality_mask_drops_inactive_modalities(self):
        config, subtoken_output = make_subtokens(embed_dim=16)
        compressor = LocalModalityCompressor(config)
        modality_mask = subtoken_output.modality_mask.clone()
        modality_mask[:, :, 2] = False

        output = compressor(
            subtoken_output.subtokens,
            mask=subtoken_output.mask,
            modality_mask=modality_mask,
        )

        self.assertEqual(float(output.local_tokens[:, :, 2].abs().sum().item()), 0.0)
        self.assertEqual(float(output.input_subtokens[:, :, 2].abs().sum().item()), 0.0)
        self.assertFalse(bool(output.modality_mask[:, :, 2].any().item()))

    def test_attention_dropout_is_separate_from_residual_dropout(self):
        config = LocalCompressionPartConfig(
            embed_dim=16,
            local_layers=1,
            local_heads=4,
            context_heads=4,
            dropout=0.2,
            attention_dropout=0.0,
        )
        compressor = LocalModalityCompressor(config)
        layer = compressor.layers[0]

        self.assertAlmostEqual(float(layer.self_attn.dropout), 0.0, places=7)
        self.assertAlmostEqual(float(layer.residual_dropout.p), 0.2, places=7)
        self.assertAlmostEqual(float(layer.ffn_dropout.p), 0.2, places=7)

    def test_compressor_is_modality_permutation_equivariant_without_positional_state(self):
        config, subtoken_output = make_subtokens(embed_dim=16)
        compressor = LocalModalityCompressor(config)
        compressor.eval()

        with torch.no_grad():
            base = compressor(
                subtoken_output.subtokens,
                mask=subtoken_output.mask,
                modality_mask=subtoken_output.modality_mask,
            ).local_tokens
            perm = torch.tensor([2, 0, 4, 1, 3], dtype=torch.long)
            inverse_perm = torch.argsort(perm)
            permuted = compressor(
                subtoken_output.subtokens[:, :, perm],
                mask=subtoken_output.mask,
                modality_mask=subtoken_output.modality_mask[:, :, perm],
            ).local_tokens[:, :, inverse_perm]

        self.assertTrue(torch.allclose(base, permuted, atol=1e-5))

    def test_gradients_flow_through_compressor_and_back_to_subtokens(self):
        config, subtoken_output = make_subtokens(embed_dim=16)
        subtokens = subtoken_output.subtokens.detach().clone().requires_grad_(True)
        compressor = LocalModalityCompressor(config)

        output = compressor(
            subtokens,
            mask=subtoken_output.mask,
            modality_mask=subtoken_output.modality_mask,
        )
        loss = output.local_tokens[output.modality_mask].pow(2).mean()
        loss.backward()

        self.assertIsNotNone(subtokens.grad)
        self.assertGreater(float(subtokens.grad.abs().sum().item()), 0.0)
        grad_sums = [
            float(parameter.grad.detach().abs().sum().item())
            for parameter in compressor.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(any(value > 0.0 for value in grad_sums))


if __name__ == "__main__":
    unittest.main()
