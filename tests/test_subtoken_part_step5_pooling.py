import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.subtoken_part import (
    SUBTOKEN_PART_LOCAL_POOL_CONTRACT,
    SUBTOKEN_PART_POOL_CLS_TOKEN,
    SUBTOKEN_PART_POOL_LEARNED_QUERY,
    SUBTOKEN_PART_POOL_MEAN,
    SubtokenAttentionPool,
    SubtokenPartConfig,
    SubtokenParticleEncoder,
    WithinParticleSubtokenTransformer,
    normalize_subtoken_pool_mode,
)


class SubtokenPartStep5PoolingTests(unittest.TestCase):
    def make_config(self, **kwargs):
        defaults = {
            "num_classes": 2,
            "embed_dim": 16,
            "local_layers": 1,
            "local_heads": 4,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        }
        defaults.update(kwargs)
        return SubtokenPartConfig(**defaults)

    def make_tokens(self):
        torch = require_torch()
        tokens = torch.zeros((2, 5, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.tensor(
            [
                [True, True, False, True, True],
                [True, False, True, False, True],
            ],
            dtype=torch.bool,
        )
        tokens[:, :, 0] = torch.tensor(
            [
                [50.0, 20.0, 900.0, 8.0, 4.0],
                [30.0, 800.0, 12.0, 700.0, 6.0],
            ]
        )
        tokens[:, :, 1] = torch.tensor(
            [
                [0.2, -0.5, 9.0, 1.0, -1.2],
                [-0.3, 8.0, 0.7, -7.0, 0.4],
            ]
        )
        tokens[:, :, 2] = torch.tensor(
            [
                [0.1, 2.9, -8.0, -2.7, 0.8],
                [1.5, 8.0, -2.1, 7.0, -0.6],
            ]
        )
        tokens[:, :, 3] = tokens[:, :, 0] + 10.0
        tokens[:, :, 4] = torch.tensor(
            [
                [1.0, -1.0, 3.0, 0.0, 1.0],
                [0.0, -2.0, 1.0, 5.0, -1.0],
            ]
        )
        tokens[:, :, 5:10] = torch.tensor(
            [
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [9, 9, 9, 9, 9],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                ],
                [
                    [0, 0, 0, 1, 0],
                    [8, 8, 8, 8, 8],
                    [0, 0, 0, 0, 1],
                    [7, 7, 7, 7, 7],
                    [1, 0, 0, 0, 0],
                ],
            ],
            dtype=torch.float32,
        )
        tokens[:, :, 10] = torch.tensor([[0.2, -0.3, 99.0, 0.4, -0.1], [0.5, 8.0, -0.2, 7.0, 0.2]])
        tokens[:, :, 11] = torch.tensor([[0.1, 0.2, 99.0, 0.3, 0.4], [0.4, 8.0, 0.5, 7.0, 0.6]])
        tokens[:, :, 12] = torch.tensor([[-0.1, 0.2, 99.0, -0.4, 0.3], [0.7, 8.0, 0.3, 7.0, -0.2]])
        tokens[:, :, 13] = torch.tensor([[0.6, 0.7, 99.0, 0.8, 0.9], [0.9, 8.0, 1.0, 7.0, 0.1]])
        return tokens, mask

    def make_local_tokens(self):
        torch = require_torch()
        local_tokens = torch.randn((2, 5, 3, 16), dtype=torch.float32)
        mask = torch.tensor(
            [
                [True, True, False, True, True],
                [True, False, True, False, True],
            ],
            dtype=torch.bool,
        )
        return local_tokens, mask

    def make_modality_mask(self, mask):
        torch = require_torch()
        modality_mask = mask[:, :, None].expand(*mask.shape, 3).clone()
        modality_mask[0, 0, 1] = False
        modality_mask[0, 1, 0] = False
        modality_mask[1, 0, 2] = False
        return modality_mask

    def test_pool_mode_aliases(self):
        self.assertEqual(normalize_subtoken_pool_mode("mean_pool"), SUBTOKEN_PART_POOL_MEAN)
        self.assertEqual(normalize_subtoken_pool_mode("attention"), SUBTOKEN_PART_POOL_LEARNED_QUERY)
        self.assertEqual(normalize_subtoken_pool_mode("cls"), SUBTOKEN_PART_POOL_CLS_TOKEN)
        with self.assertRaises(ValueError):
            normalize_subtoken_pool_mode("not_a_pool")

    def test_default_learned_query_pool_returns_particle_tokens_and_weights(self):
        torch = require_torch()
        local_tokens, mask = self.make_local_tokens()
        config = self.make_config()
        pool = SubtokenAttentionPool(config)

        output = pool(local_tokens, mask)

        self.assertEqual(output.pool_mode, SUBTOKEN_PART_POOL_LEARNED_QUERY)
        self.assertEqual(tuple(output.provisional_particles.shape), (2, 5, 16))
        self.assertEqual(tuple(output.pool_weights.shape), (2, 5, 3))
        self.assertEqual(tuple(output.modality_mask.shape), (2, 5, 3))
        self.assertEqual(output.summary()["contract"], SUBTOKEN_PART_LOCAL_POOL_CONTRACT)
        self.assertTrue(bool(torch.isfinite(output.provisional_particles).all()))
        self.assertTrue(bool(torch.isfinite(output.pool_weights).all()))
        self.assertTrue(bool((output.provisional_particles[~mask] == 0.0).all()))
        self.assertTrue(bool((output.pool_weights[~mask] == 0.0).all()))
        valid_sums = output.pool_weights[mask].sum(dim=-1)
        self.assertTrue(bool(torch.allclose(valid_sums, torch.ones_like(valid_sums), atol=1.0e-6)))

    def test_mean_pool_matches_plain_mean_for_valid_particles(self):
        torch = require_torch()
        local_tokens, mask = self.make_local_tokens()
        config = self.make_config(local_pool_mode=SUBTOKEN_PART_POOL_MEAN)
        pool = SubtokenAttentionPool(config)

        output = pool(local_tokens, mask)

        expected = local_tokens.mean(dim=2)
        self.assertTrue(bool(torch.allclose(output.provisional_particles[mask], expected[mask], atol=1.0e-6)))
        self.assertTrue(bool(torch.allclose(output.pool_weights[mask], torch.full_like(output.pool_weights[mask], 1.0 / 3.0))))
        self.assertTrue(bool((output.provisional_particles[~mask] == 0.0).all()))

    def test_pool_weights_respect_modality_mask_for_all_pool_modes(self):
        torch = require_torch()
        local_tokens, mask = self.make_local_tokens()
        modality_mask = self.make_modality_mask(mask)

        for pool_mode in (SUBTOKEN_PART_POOL_MEAN, SUBTOKEN_PART_POOL_LEARNED_QUERY, SUBTOKEN_PART_POOL_CLS_TOKEN):
            with self.subTest(pool_mode=pool_mode):
                config = self.make_config(local_pool_mode=pool_mode)
                pool = SubtokenAttentionPool(config)
                output = pool(local_tokens, mask, modality_mask=modality_mask)

                self.assertTrue(bool(torch.equal(output.modality_mask, modality_mask)))
                self.assertTrue(bool((output.pool_weights[~modality_mask] == 0.0).all()))
                active_sums = output.pool_weights[modality_mask].reshape(-1)
                self.assertTrue(bool(torch.isfinite(active_sums).all()))
                valid_sums = output.pool_weights[mask].sum(dim=-1)
                self.assertTrue(bool(torch.allclose(valid_sums, torch.ones_like(valid_sums), atol=1.0e-6)))
                self.assertTrue(bool((output.pool_weights[~mask] == 0.0).all()))

    def test_masked_mean_pool_matches_active_modality_mean(self):
        torch = require_torch()
        local_tokens, mask = self.make_local_tokens()
        modality_mask = self.make_modality_mask(mask)
        config = self.make_config(local_pool_mode=SUBTOKEN_PART_POOL_MEAN)
        pool = SubtokenAttentionPool(config)

        output = pool(local_tokens, mask, modality_mask=modality_mask)
        active_counts = modality_mask.sum(dim=2, keepdim=True).clamp(min=1).to(dtype=local_tokens.dtype)
        expected = (torch.where(modality_mask[:, :, :, None], local_tokens, torch.zeros_like(local_tokens))).sum(dim=2)
        expected = expected / active_counts

        self.assertTrue(bool(torch.allclose(output.provisional_particles[mask], expected[mask], atol=1.0e-6)))
        self.assertTrue(bool((output.pool_weights[~modality_mask] == 0.0).all()))

    def test_cls_token_pool_runs_and_respects_mask(self):
        torch = require_torch()
        local_tokens, mask = self.make_local_tokens()
        config = self.make_config(local_pool_mode=SUBTOKEN_PART_POOL_CLS_TOKEN)
        pool = SubtokenAttentionPool(config)

        output = pool(local_tokens, mask)

        self.assertEqual(tuple(output.provisional_particles.shape), (2, 5, 16))
        self.assertEqual(tuple(output.pool_weights.shape), (2, 5, 3))
        self.assertTrue(bool(torch.isfinite(output.provisional_particles).all()))
        valid_sums = output.pool_weights[mask].sum(dim=-1)
        self.assertTrue(bool(torch.allclose(valid_sums, torch.ones_like(valid_sums), atol=1.0e-6)))
        self.assertTrue(bool((output.provisional_particles[~mask] == 0.0).all()))
        self.assertTrue(bool((output.pool_weights[~mask] == 0.0).all()))

    def test_pool_accepts_mixer_output(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        config = self.make_config()
        encoder = SubtokenParticleEncoder(config)
        mixer = WithinParticleSubtokenTransformer(config)
        pool = SubtokenAttentionPool(config)

        mixed = mixer(encoder(tokens, mask))
        pooled = pool(mixed)

        self.assertEqual(tuple(pooled.provisional_particles.shape), (2, 5, 16))
        self.assertTrue(bool(torch.equal(pooled.modality_mask, mixed.modality_mask)))
        self.assertEqual(pooled.modality_names, mixed.modality_names)
        self.assertTrue(bool((pooled.pool_weights[~mixed.modality_mask] == 0.0).all()))
        self.assertTrue(bool(torch.isfinite(pooled.provisional_particles).all()))

    def test_gradients_flow_to_local_tokens_and_pool_parameters(self):
        local_tokens, mask = self.make_local_tokens()
        local_tokens.requires_grad_(True)
        config = self.make_config()
        pool = SubtokenAttentionPool(config)

        output = pool(local_tokens, mask)
        loss = output.provisional_particles[mask].square().mean() + output.pool_weights[mask].square().mean()
        loss.backward()

        self.assertIsNotNone(local_tokens.grad)
        self.assertGreater(float(local_tokens.grad[mask].abs().sum().detach().cpu().item()), 0.0)
        grad_sum = sum(
            float(param.grad.detach().abs().sum().cpu().item())
            for param in pool.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_sum, 0.0)

    def test_particle_permutation_equivariance(self):
        torch = require_torch()
        local_tokens, mask = self.make_local_tokens()
        config = self.make_config()
        pool = SubtokenAttentionPool(config)
        permutation = torch.tensor([3, 1, 0, 4, 2], dtype=torch.long)

        pooled = pool(local_tokens, mask)
        pooled_permuted = pool(local_tokens[:, permutation], mask[:, permutation])

        self.assertTrue(
            bool(
                torch.allclose(
                    pooled_permuted.provisional_particles,
                    pooled.provisional_particles[:, permutation],
                    atol=1.0e-6,
                    rtol=1.0e-6,
                )
            )
        )
        self.assertTrue(bool(torch.allclose(pooled_permuted.pool_weights, pooled.pool_weights[:, permutation], atol=1.0e-6)))

    def test_invalid_shapes_raise_clear_errors(self):
        torch = require_torch()
        config = self.make_config()
        pool = SubtokenAttentionPool(config)

        with self.assertRaises(ValueError):
            pool(torch.zeros((2, 3, 16)), torch.ones((2, 3), dtype=torch.bool))
        with self.assertRaises(ValueError):
            pool(torch.zeros((2, 3, 3, 15)), torch.ones((2, 3), dtype=torch.bool))
        with self.assertRaises(ValueError):
            pool(torch.zeros((2, 3, 3, 16)), torch.ones((2, 4), dtype=torch.bool))
        with self.assertRaises(ValueError):
            pool(
                torch.zeros((2, 3, 3, 16)),
                torch.ones((2, 3), dtype=torch.bool),
                modality_mask=torch.ones((2, 3, 4), dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
