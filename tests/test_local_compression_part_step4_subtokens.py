import math
import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_compression_part import (
    LOCAL_COMPRESSION_MODALITIES,
    LOCAL_COMPRESSION_SUBTOKENS_CONTRACT,
    LocalCompressionPartConfig,
    LocalCompressionSubtokenEncoder,
    build_local_compression_modalities_from_tokens,
)


torch = require_torch()


def make_tokens(num_particles: int = 6):
    tokens = torch.zeros((3, num_particles, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.zeros((3, num_particles), dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :4] = True
    mask[2, :3] = True
    for batch in range(3):
        for idx in range(int(mask[batch].sum().item())):
            pt = 9.0 + 3.0 * idx + batch
            eta = -0.25 + 0.08 * idx + 0.03 * batch
            phi = -math.pi + 0.19 * idx
            tokens[batch, idx, 0] = pt
            tokens[batch, idx, 1] = eta
            tokens[batch, idx, 2] = phi
            tokens[batch, idx, 3] = pt * math.cosh(eta) + 0.2
            tokens[batch, idx, 4] = 1.0 if idx % 2 == 0 else -1.0
            tokens[batch, idx, 5 + (idx % 5)] = 1.0
            tokens[batch, idx, 10] = 0.01 * idx
            tokens[batch, idx, 11] = 0.04 + 0.01 * idx
            tokens[batch, idx, 12] = -0.02 * idx
            tokens[batch, idx, 13] = 0.06 + 0.01 * batch
    return tokens, mask


class LocalCompressionStep4SubtokenTests(unittest.TestCase):
    def test_subtoken_encoder_outputs_bound_shape_and_masks(self):
        tokens, mask = make_tokens()
        canonical, modalities = build_local_compression_modalities_from_tokens(tokens, mask, max_constits=6)
        cfg = LocalCompressionPartConfig(embed_dim=24, local_heads=4, context_heads=4, dropout=0.0)
        encoder = LocalCompressionSubtokenEncoder(cfg)
        output = encoder(canonical, modalities)

        self.assertEqual(tuple(output.subtokens.shape), (3, 6, len(LOCAL_COMPRESSION_MODALITIES), 24))
        self.assertEqual(tuple(output.anchor.shape), (3, 6, 24))
        self.assertTrue(torch.equal(output.mask, canonical.particle_mask))
        self.assertTrue(torch.equal(output.modality_mask, modalities.modality_mask))
        self.assertEqual(output.summary()["contract"], LOCAL_COMPRESSION_SUBTOKENS_CONTRACT)
        self.assertTrue(torch.isfinite(output.subtokens).all())
        self.assertEqual(float(output.subtokens[0, 5].abs().sum().item()), 0.0)
        self.assertEqual(float(output.subtokens[2, 3:].abs().sum().item()), 0.0)

    def test_modality_type_embeddings_change_bound_subtokens(self):
        tokens, mask = make_tokens()
        canonical, modalities = build_local_compression_modalities_from_tokens(tokens, mask, max_constits=6)
        cfg = LocalCompressionPartConfig(embed_dim=16, local_heads=4, context_heads=4, dropout=0.0)
        encoder = LocalCompressionSubtokenEncoder(cfg)
        with torch.no_grad():
            encoder.modality_type_embedding.weight.zero_()
            encoder.modality_type_embedding.weight[0].fill_(1.0)
            encoder.modality_type_embedding.weight[1].fill_(2.0)

        output = encoder(canonical, modalities)
        type0 = output.modality_type_embeddings[0, 0, 0]
        type1 = output.modality_type_embeddings[0, 0, 1]
        self.assertFalse(torch.allclose(type0, type1))
        without_anchor = output.component_without_anchor()
        self.assertFalse(torch.allclose(without_anchor[0, 0, 0], without_anchor[0, 0, 1]))

    def test_recovered_anchor_matches_particle_anchor_for_active_modalities(self):
        tokens, mask = make_tokens()
        canonical, modalities = build_local_compression_modalities_from_tokens(tokens, mask, max_constits=6)
        cfg = LocalCompressionPartConfig(embed_dim=20, local_heads=4, context_heads=4, dropout=0.0)
        encoder = LocalCompressionSubtokenEncoder(cfg)
        output = encoder(canonical, modalities)
        recovered = output.recovered_anchor()
        expected = output.anchor[:, :, None, :].expand_as(recovered)
        active = output.modality_mask[:, :, :, None].expand_as(recovered)
        self.assertTrue(torch.allclose(recovered[active], expected[active], atol=1e-5))
        self.assertEqual(float(recovered[~active].abs().sum().item()), 0.0)

    def test_pt_rank_embeddings_are_optional_and_masked(self):
        tokens, mask = make_tokens()
        canonical, modalities = build_local_compression_modalities_from_tokens(tokens, mask, max_constits=6)
        cfg_on = LocalCompressionPartConfig(embed_dim=16, local_heads=4, context_heads=4, dropout=0.0)
        cfg_off = LocalCompressionPartConfig(
            embed_dim=16,
            local_heads=4,
            context_heads=4,
            dropout=0.0,
            use_pt_rank_embedding=False,
        )
        out_on = LocalCompressionSubtokenEncoder(cfg_on)(canonical, modalities)
        out_off = LocalCompressionSubtokenEncoder(cfg_off)(canonical, modalities)
        self.assertIsNotNone(out_on.pt_rank_embeddings)
        self.assertIsNone(out_off.pt_rank_embeddings)
        self.assertEqual(float(out_on.pt_rank_embeddings[0, 5].abs().sum().item()), 0.0)

    def test_modality_mask_override_drops_full_bound_subtokens(self):
        tokens, mask = make_tokens()
        canonical, modalities = build_local_compression_modalities_from_tokens(tokens, mask, max_constits=6)
        cfg = LocalCompressionPartConfig(embed_dim=16, local_heads=4, context_heads=4, dropout=0.0)
        encoder = LocalCompressionSubtokenEncoder(cfg)
        override = modalities.modality_mask.clone()
        override[:, :, 1] = False
        output = encoder(canonical, modalities, modality_mask_override=override)
        self.assertEqual(float(output.subtokens[:, :, 1].abs().sum().item()), 0.0)
        self.assertEqual(float(output.modality_embeddings[:, :, 1].abs().sum().item()), 0.0)
        self.assertEqual(float(output.modality_type_embeddings[:, :, 1].abs().sum().item()), 0.0)
        self.assertFalse(bool(output.modality_mask[:, :, 1].any().item()))

    def test_gradients_flow_to_each_modality_encoder(self):
        tokens, mask = make_tokens()
        canonical, modalities = build_local_compression_modalities_from_tokens(tokens, mask, max_constits=6)
        cfg = LocalCompressionPartConfig(embed_dim=16, local_heads=4, context_heads=4, dropout=0.0)
        encoder = LocalCompressionSubtokenEncoder(cfg)
        output = encoder(canonical, modalities)
        loss = output.subtokens[output.modality_mask].pow(2).mean()
        loss.backward()

        for name, module in encoder.modality_encoders.items():
            grads = [
                param.grad.detach().abs().sum().item()
                for param in module.parameters()
                if param.grad is not None
            ]
            self.assertTrue(any(value > 0.0 for value in grads), msg=f"no gradient flowed to {name}")


if __name__ == "__main__":
    unittest.main()
