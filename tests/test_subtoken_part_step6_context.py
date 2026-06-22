import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.subtoken_part import (
    SUBTOKEN_PART_CONTEXT_CONTRACT,
    ParticleContextTransformer,
    SubtokenAttentionPool,
    SubtokenPartConfig,
    SubtokenParticleEncoder,
    WithinParticleSubtokenTransformer,
)


class SubtokenPartStep6ContextTests(unittest.TestCase):
    def make_config(self, **kwargs):
        defaults = {
            "num_classes": 2,
            "embed_dim": 16,
            "local_layers": 1,
            "local_heads": 4,
            "context_layers": 1,
            "context_heads": 4,
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

    def make_particles(self):
        torch = require_torch()
        particles = torch.randn((2, 5, 16), dtype=torch.float32)
        mask = torch.tensor(
            [
                [True, True, False, True, True],
                [True, False, True, False, True],
            ],
            dtype=torch.bool,
        )
        return particles, mask

    def test_context_accepts_raw_particles_and_respects_mask(self):
        torch = require_torch()
        particles, mask = self.make_particles()
        context = ParticleContextTransformer(self.make_config())

        output = context(particles, mask)

        self.assertEqual(tuple(output.context_tokens.shape), (2, 5, 16))
        self.assertEqual(tuple(output.input_particles.shape), (2, 5, 16))
        self.assertEqual(tuple(output.mask.shape), tuple(mask.shape))
        self.assertEqual(output.summary()["contract"], SUBTOKEN_PART_CONTEXT_CONTRACT)
        self.assertTrue(bool(torch.isfinite(output.context_tokens).all()))
        self.assertTrue(bool((output.context_tokens[~mask] == 0.0).all()))
        self.assertTrue(bool((output.input_particles[~mask] == 0.0).all()))

    def test_context_accepts_pool_output(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        config = self.make_config()
        encoder = SubtokenParticleEncoder(config)
        mixer = WithinParticleSubtokenTransformer(config)
        pool = SubtokenAttentionPool(config)
        context = ParticleContextTransformer(config)

        pooled = pool(mixer(encoder(tokens, mask)))
        output = context(pooled)

        self.assertEqual(tuple(output.context_tokens.shape), tuple(pooled.provisional_particles.shape))
        self.assertTrue(bool(torch.equal(output.mask, pooled.mask)))
        self.assertTrue(bool(torch.isfinite(output.context_tokens).all()))

    def test_changing_neighboring_particles_can_change_context(self):
        torch = require_torch()
        particles, mask = self.make_particles()
        config = self.make_config()
        context = ParticleContextTransformer(config)
        context.eval()

        baseline = context(particles, mask).context_tokens
        changed_particles = particles.clone()
        changed_particles[0, 1] = changed_particles[0, 1] + 10.0
        changed = context(changed_particles, mask).context_tokens
        delta = (baseline[0, 0] - changed[0, 0]).abs().sum()

        self.assertGreater(float(delta.detach().cpu().item()), 1.0e-6)

    def test_gradients_flow_to_inputs_and_context_parameters(self):
        particles, mask = self.make_particles()
        particles.requires_grad_(True)
        context = ParticleContextTransformer(self.make_config())

        output = context(particles, mask)
        loss = output.context_tokens[mask].square().mean()
        loss.backward()

        self.assertIsNotNone(particles.grad)
        self.assertGreater(float(particles.grad[mask].abs().sum().detach().cpu().item()), 0.0)
        grad_sum = sum(
            float(param.grad.detach().abs().sum().cpu().item())
            for param in context.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_sum, 0.0)

    def test_invalid_shapes_raise_clear_errors(self):
        torch = require_torch()
        context = ParticleContextTransformer(self.make_config())

        with self.assertRaises(ValueError):
            context(torch.zeros((2, 5, 16)))
        with self.assertRaises(ValueError):
            context(torch.zeros((2, 5, 16)), torch.ones((2, 4), dtype=torch.bool))
        with self.assertRaises(ValueError):
            context(torch.zeros((2, 5, 15)), torch.ones((2, 5), dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
