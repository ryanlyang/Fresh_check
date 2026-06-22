import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.subtoken_part import (
    SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX,
    SUBTOKEN_PART_GATE_NONE,
    SUBTOKEN_PART_PARTICLE_TOKEN_CONTRACT,
    ParticleContextTransformer,
    ReliabilityAwareParticleTokenBuilder,
    ReliabilityGateHead,
    ReliabilityGateOutput,
    SubtokenAttentionPool,
    SubtokenPartConfig,
    SubtokenParticleEncoder,
    WithinParticleSubtokenTransformer,
)


class SubtokenPartStep8ParticleTokenTests(unittest.TestCase):
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

    def make_pipeline_outputs(self, config):
        tokens, mask = self.make_tokens()
        encoder = SubtokenParticleEncoder(config)
        mixer = WithinParticleSubtokenTransformer(config)
        pool = SubtokenAttentionPool(config)
        context = ParticleContextTransformer(config)

        encoded = encoder(tokens, mask)
        mixed = mixer(encoded)
        pooled = pool(mixed)
        contexted = context(pooled)
        return encoded, mixed, pooled, contexted

    def test_no_gate_equals_configured_pooling_behavior(self):
        torch = require_torch()
        config = self.make_config(gate_mode=SUBTOKEN_PART_GATE_NONE)
        encoded, mixed, pooled, _ = self.make_pipeline_outputs(config)
        builder = ReliabilityAwareParticleTokenBuilder(config)

        output = builder(mixed, pooled, None, encoded)

        self.assertEqual(output.summary()["contract"], SUBTOKEN_PART_PARTICLE_TOKEN_CONTRACT)
        self.assertFalse(output.used_reliability_gates)
        self.assertEqual(output.gate_mode, SUBTOKEN_PART_GATE_NONE)
        self.assertTrue(bool(torch.allclose(output.particle_tokens, pooled.provisional_particles, atol=1.0e-6)))
        self.assertTrue(bool(torch.allclose(output.weighted_modalities, pooled.provisional_particles, atol=1.0e-6)))
        self.assertTrue(bool(torch.allclose(output.gates, pooled.pool_weights, atol=1.0e-6)))
        self.assertTrue(bool((output.particle_tokens[~mixed.mask] == 0.0).all()))

    def test_context_gate_changes_particle_tokens_when_context_changes(self):
        config = self.make_config(gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX)
        encoded, mixed, pooled, contexted = self.make_pipeline_outputs(config)
        gate_head = ReliabilityGateHead(config)
        builder = ReliabilityAwareParticleTokenBuilder(config)
        gate_head.eval()
        builder.eval()

        baseline_gates = gate_head(mixed, pooled, contexted, encoded)
        baseline = builder(mixed, pooled, baseline_gates, encoded).particle_tokens
        changed_context = contexted.context_tokens.clone()
        changed_context[0, 0] = changed_context[0, 0] + require_torch().arange(
            16,
            dtype=changed_context.dtype,
            device=changed_context.device,
        )
        changed_gates = gate_head(mixed, pooled, changed_context, encoded)
        changed = builder(mixed, pooled, changed_gates, encoded).particle_tokens
        delta = (baseline[0, 0] - changed[0, 0]).abs().sum()

        self.assertGreater(float(delta.detach().cpu().item()), 1.0e-6)

    def test_gated_builder_matches_anchor_plus_weighted_modalities(self):
        torch = require_torch()
        config = self.make_config(gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX)
        encoded, mixed, pooled, contexted = self.make_pipeline_outputs(config)
        gate_output = ReliabilityGateHead(config)(mixed, pooled, contexted, encoded)
        builder = ReliabilityAwareParticleTokenBuilder(config)

        output = builder(mixed, pooled, gate_output, encoded)
        expected_weighted = (mixed.local_tokens * gate_output.gates[:, :, :, None]).sum(dim=2)
        expected_particle_tokens = encoded.anchor + expected_weighted
        expected_weighted = torch.where(mixed.mask[:, :, None], expected_weighted, torch.zeros_like(expected_weighted))
        expected_particle_tokens = torch.where(mixed.mask[:, :, None], expected_particle_tokens, torch.zeros_like(expected_particle_tokens))

        self.assertTrue(output.used_reliability_gates)
        self.assertTrue(bool(torch.allclose(output.weighted_modalities, expected_weighted, atol=1.0e-6)))
        self.assertTrue(bool(torch.allclose(output.particle_tokens, expected_particle_tokens, atol=1.0e-6)))
        self.assertTrue(bool(torch.isfinite(output.diagnostics()["mean_particle_token_norm"]).all()))

    def test_all_modality_paths_receive_gradients(self):
        torch = require_torch()
        config = self.make_config(gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX)
        builder = ReliabilityAwareParticleTokenBuilder(config)
        local_tokens = torch.randn((1, 2, 3, 16), dtype=torch.float32, requires_grad=True)
        pooled = torch.randn((1, 2, 16), dtype=torch.float32)
        anchor = torch.randn((1, 2, 16), dtype=torch.float32, requires_grad=True)
        mask = torch.ones((1, 2), dtype=torch.bool)
        modality_mask = torch.ones((1, 2, 3), dtype=torch.bool)
        gates = torch.tensor(
            [
                [
                    [0.2, 0.3, 0.5],
                    [0.4, 0.1, 0.5],
                ]
            ],
            dtype=torch.float32,
        )
        gate_output = ReliabilityGateOutput(
            gate_logits=torch.zeros_like(gates),
            gates=gates,
            gate_entropy=torch.zeros((1, 2), dtype=torch.float32),
            mask=mask,
            modality_mask=modality_mask,
            modality_names=("kinematics", "identity", "track"),
            gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX,
        )

        output = builder(local_tokens, pooled, gate_output, anchor, mask=mask, modality_mask=modality_mask)
        loss = output.particle_tokens.sum()
        loss.backward()

        self.assertIsNotNone(local_tokens.grad)
        self.assertIsNotNone(anchor.grad)
        for modality_index in range(3):
            grad_sum = local_tokens.grad[:, :, modality_index, :].abs().sum()
            self.assertGreater(float(grad_sum.detach().cpu().item()), 0.0)
        self.assertGreater(float(anchor.grad.abs().sum().detach().cpu().item()), 0.0)

    def test_invalid_particles_and_inactive_modalities_are_zeroed(self):
        torch = require_torch()
        config = self.make_config(gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX)
        local_tokens = torch.randn((1, 3, 3, 16), dtype=torch.float32)
        pooled = torch.randn((1, 3, 16), dtype=torch.float32)
        anchor = torch.randn((1, 3, 16), dtype=torch.float32)
        mask = torch.tensor([[True, False, True]], dtype=torch.bool)
        modality_mask = mask[:, :, None].expand(1, 3, 3).clone()
        modality_mask[0, 0, 1] = False
        gates = torch.full((1, 3, 3), 1.0 / 3.0, dtype=torch.float32)
        gate_output = ReliabilityGateOutput(
            gate_logits=torch.zeros_like(gates),
            gates=gates,
            gate_entropy=torch.zeros((1, 3), dtype=torch.float32),
            mask=mask,
            modality_mask=modality_mask,
            modality_names=("kinematics", "identity", "track"),
            gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX,
        )
        builder = ReliabilityAwareParticleTokenBuilder(config)

        output = builder(local_tokens, pooled, gate_output, anchor, mask=mask, modality_mask=modality_mask)

        self.assertTrue(bool((output.gates[~modality_mask] == 0.0).all()))
        self.assertTrue(bool((output.particle_tokens[~mask] == 0.0).all()))
        self.assertTrue(bool((output.weighted_modalities[~mask] == 0.0).all()))

    def test_context_gate_requires_gate_output(self):
        config = self.make_config(gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX)
        _, mixed, pooled, _ = self.make_pipeline_outputs(config)
        builder = ReliabilityAwareParticleTokenBuilder(config)

        with self.assertRaises(ValueError):
            builder(mixed, pooled, None)


if __name__ == "__main__":
    unittest.main()
