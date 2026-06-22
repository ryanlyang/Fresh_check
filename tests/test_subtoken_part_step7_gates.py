import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.subtoken_part import (
    SUBTOKEN_PART_GATE_CONTEXT_SIGMOID,
    SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX,
    SUBTOKEN_PART_GATE_CONTRACT,
    SUBTOKEN_PART_GATE_LOCAL_SOFTMAX,
    SUBTOKEN_PART_GATE_NONE,
    ParticleContextTransformer,
    ReliabilityGateHead,
    SubtokenAttentionPool,
    SubtokenPartConfig,
    SubtokenParticleEncoder,
    WithinParticleSubtokenTransformer,
)


class SubtokenPartStep7GateTests(unittest.TestCase):
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

    def test_context_softmax_gates_sum_to_one_and_respect_masks(self):
        torch = require_torch()
        config = self.make_config(gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX)
        encoded, mixed, pooled, contexted = self.make_pipeline_outputs(config)
        gate_head = ReliabilityGateHead(config)

        modality_mask = mixed.modality_mask.clone()
        modality_mask[0, 0, 1] = False
        local_tokens = torch.where(modality_mask[:, :, :, None], mixed.local_tokens, torch.zeros_like(mixed.local_tokens))
        output = gate_head(
            local_tokens,
            pooled,
            contexted,
            encoded,
            mask=mixed.mask,
            modality_mask=modality_mask,
        )

        self.assertEqual(output.gate_mode, SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX)
        self.assertEqual(output.summary()["contract"], SUBTOKEN_PART_GATE_CONTRACT)
        self.assertEqual(tuple(output.gates.shape), tuple(output.gate_logits.shape))
        self.assertEqual(tuple(output.gate_entropy.shape), tuple(mixed.mask.shape))
        self.assertTrue(bool(torch.isfinite(output.gates).all()))
        self.assertTrue(bool(torch.isfinite(output.gate_entropy).all()))
        self.assertTrue(bool((output.gates[~modality_mask] == 0.0).all()))
        self.assertTrue(bool((output.gates[~mixed.mask] == 0.0).all()))
        self.assertTrue(bool((output.gate_entropy[~mixed.mask] == 0.0).all()))
        valid_sums = output.gates[mixed.mask].sum(dim=-1)
        self.assertTrue(bool(torch.allclose(valid_sums, torch.ones_like(valid_sums), atol=1.0e-6)))

        diagnostics = output.diagnostics()
        self.assertIn("mean_gate_by_particle", diagnostics)
        self.assertIn("mean_gate_entropy", diagnostics)
        self.assertTrue(bool(torch.isfinite(diagnostics["mean_gate_by_particle"]).all()))
        self.assertTrue(bool(torch.isfinite(diagnostics["mean_gate_entropy"]).all()))

    def test_context_changes_can_change_gates(self):
        torch = require_torch()
        config = self.make_config(gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX)
        encoded, mixed, pooled, contexted = self.make_pipeline_outputs(config)
        gate_head = ReliabilityGateHead(config)
        gate_head.eval()

        baseline = gate_head(mixed, pooled, contexted, encoded).gates
        changed_context = contexted.context_tokens.clone()
        changed_context[0, 0] = changed_context[0, 0] + torch.arange(
            16,
            dtype=changed_context.dtype,
            device=changed_context.device,
        )
        changed = gate_head(mixed, pooled, changed_context, encoded).gates
        delta = (baseline[0, 0] - changed[0, 0]).abs().sum()

        self.assertGreater(float(delta.detach().cpu().item()), 1.0e-6)

    def test_local_softmax_gate_does_not_require_context(self):
        torch = require_torch()
        config = self.make_config(gate_mode=SUBTOKEN_PART_GATE_LOCAL_SOFTMAX)
        encoded, mixed, pooled, _ = self.make_pipeline_outputs(config)
        gate_head = ReliabilityGateHead(config)

        output = gate_head(mixed, pooled, None, encoded)

        self.assertEqual(output.gate_mode, SUBTOKEN_PART_GATE_LOCAL_SOFTMAX)
        valid_sums = output.gates[mixed.mask].sum(dim=-1)
        self.assertTrue(bool(torch.allclose(valid_sums, torch.ones_like(valid_sums), atol=1.0e-6)))
        self.assertTrue(bool((output.gates[~mixed.mask] == 0.0).all()))

    def test_context_sigmoid_gates_are_finite_and_masked(self):
        torch = require_torch()
        config = self.make_config(gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SIGMOID)
        encoded, mixed, pooled, contexted = self.make_pipeline_outputs(config)
        gate_head = ReliabilityGateHead(config)

        output = gate_head(mixed, pooled, contexted, encoded)

        self.assertEqual(output.gate_mode, SUBTOKEN_PART_GATE_CONTEXT_SIGMOID)
        self.assertTrue(bool(torch.isfinite(output.gates).all()))
        self.assertTrue(bool((output.gates >= 0.0).all()))
        self.assertTrue(bool((output.gates <= 1.0).all()))
        self.assertTrue(bool((output.gates[~mixed.modality_mask] == 0.0).all()))
        self.assertTrue(bool((output.gate_entropy[~mixed.mask] == 0.0).all()))

    def test_none_gate_returns_uniform_active_modality_weights(self):
        torch = require_torch()
        config = self.make_config(gate_mode=SUBTOKEN_PART_GATE_NONE)
        encoded, mixed, pooled, _ = self.make_pipeline_outputs(config)
        gate_head = ReliabilityGateHead(config)

        output = gate_head(mixed, pooled, None, encoded)

        expected = mixed.modality_mask.to(dtype=output.gates.dtype)
        expected = expected / expected.sum(dim=2, keepdim=True).clamp(min=1.0)
        expected = torch.where(mixed.mask[:, :, None], expected, torch.zeros_like(expected))
        self.assertTrue(bool(torch.allclose(output.gates, expected, atol=1.0e-6)))

    def test_gradients_flow_to_gate_inputs_and_parameters(self):
        torch = require_torch()
        config = self.make_config(gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX)
        gate_head = ReliabilityGateHead(config)
        local_tokens = torch.randn((2, 4, 3, 16), dtype=torch.float32, requires_grad=True)
        provisional = torch.randn((2, 4, 16), dtype=torch.float32, requires_grad=True)
        context = torch.randn((2, 4, 16), dtype=torch.float32, requires_grad=True)
        anchor = torch.randn((2, 4, 16), dtype=torch.float32, requires_grad=True)
        mask = torch.tensor([[True, True, False, True], [True, False, True, True]], dtype=torch.bool)
        modality_mask = mask[:, :, None].expand(2, 4, 3).clone()
        modality_mask[0, 0, 1] = False

        output = gate_head(
            local_tokens,
            provisional,
            context,
            anchor,
            mask=mask,
            modality_mask=modality_mask,
        )
        loss = output.gates[:, :, 0][mask].mean() + output.gate_logits[mask].square().mean()
        loss.backward()

        for tensor in (local_tokens, provisional, context, anchor):
            self.assertIsNotNone(tensor.grad)
            self.assertGreater(float(tensor.grad.abs().sum().detach().cpu().item()), 0.0)
        grad_sum = sum(
            float(param.grad.detach().abs().sum().cpu().item())
            for param in gate_head.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_sum, 0.0)

    def test_context_mode_requires_context_input(self):
        config = self.make_config(gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX)
        encoded, mixed, pooled, _ = self.make_pipeline_outputs(config)
        gate_head = ReliabilityGateHead(config)

        with self.assertRaises(ValueError):
            gate_head(mixed, pooled, None, encoded)


if __name__ == "__main__":
    unittest.main()
