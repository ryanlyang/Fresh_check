import math
import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_compression_part import (
    LOCAL_COMPRESSION_GATE_NONE,
    LOCAL_COMPRESSION_GATES_CONTRACT,
    LOCAL_COMPRESSION_MODALITIES,
    LocalCompressionContextGate,
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
            pt = 13.0 + 2.5 * particle_index + 0.6 * batch_index
            eta = -0.25 + 0.06 * particle_index + 0.01 * batch_index
            phi = -math.pi + 0.19 * particle_index
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


def make_gate_inputs(
    embed_dim: int = 16,
    *,
    include_empty_batch: bool = False,
    config: LocalCompressionPartConfig | None = None,
):
    tokens, mask = make_tokens(include_empty_batch=include_empty_batch)
    canonical, modalities = build_local_compression_modalities_from_tokens(
        tokens,
        mask,
        max_constits=tokens.shape[1],
    )
    config = config or LocalCompressionPartConfig(
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
    context = ParticleContextBlock(config)(pooled)
    return config, canonical, compressed, context


class LocalCompressionStep8GateTests(unittest.TestCase):
    def test_context_gate_shapes_range_and_contract(self):
        config, canonical, compressed, context = make_gate_inputs(embed_dim=16)
        gate = LocalCompressionContextGate(config)

        output = gate(compressed, context, canonical)

        self.assertEqual(tuple(output.gates.shape), (3, 6, len(LOCAL_COMPRESSION_MODALITIES)))
        self.assertEqual(tuple(output.gate_logits.shape), tuple(output.gates.shape))
        self.assertEqual(tuple(output.diagnostic_weights.shape), tuple(output.gates.shape))
        self.assertTrue(torch.equal(output.mask, compressed.mask))
        self.assertTrue(torch.equal(output.modality_mask, compressed.modality_mask))
        self.assertEqual(output.modality_names, LOCAL_COMPRESSION_MODALITIES)
        self.assertEqual(output.summary()["contract"], LOCAL_COMPRESSION_GATES_CONTRACT)
        self.assertGreaterEqual(float(output.gates.min().item()), 0.0)
        self.assertLessEqual(float(output.gates.max().item()), 1.0)

    def test_inactive_modalities_have_zero_gate_and_diagnostic_weight(self):
        config, canonical, compressed, context = make_gate_inputs(embed_dim=16)
        modality_mask = compressed.modality_mask.clone()
        modality_mask[:, :, 2] = False
        modality_mask[0, 0, 4] = False
        gate = LocalCompressionContextGate(config)

        output = gate(
            compressed.local_tokens,
            context.context_tokens,
            canonical,
            mask=compressed.mask,
            modality_mask=modality_mask,
        )

        self.assertEqual(float(output.gates[:, :, 2].abs().sum().item()), 0.0)
        self.assertEqual(float(output.diagnostic_weights[:, :, 2].abs().sum().item()), 0.0)
        self.assertEqual(float(output.gates[0, 0, 4].abs().item()), 0.0)
        self.assertEqual(float(output.diagnostic_weights[0, 0, 4].abs().item()), 0.0)

    def test_invalid_particles_and_empty_batch_rows_have_zero_gates(self):
        config, canonical, compressed, context = make_gate_inputs(embed_dim=16, include_empty_batch=True)
        gate = LocalCompressionContextGate(config)

        output = gate(compressed, context, canonical)

        self.assertEqual(float(output.gates[0, 5].abs().sum().item()), 0.0)
        self.assertEqual(float(output.diagnostic_weights[0, 5].abs().sum().item()), 0.0)
        self.assertEqual(float(output.gates[3].abs().sum().item()), 0.0)
        self.assertEqual(float(output.diagnostic_weights[3].abs().sum().item()), 0.0)

    def test_changing_context_can_change_gates(self):
        config, canonical, compressed, context = make_gate_inputs(embed_dim=16)
        gate = LocalCompressionContextGate(config)
        gate.eval()
        changed_context = context.context_tokens.detach().clone()
        changed_context[0, 0, 0] = changed_context[0, 0, 0] + 25.0

        with torch.no_grad():
            base = gate(compressed, context, canonical).gates
            changed = gate(compressed, changed_context, canonical, mask=compressed.mask).gates

        self.assertGreater(float((base[0, 0] - changed[0, 0]).abs().sum().item()), 1.0e-5)

    def test_gates_are_not_saturated_at_initialization(self):
        config, canonical, compressed, context = make_gate_inputs(embed_dim=16)
        gate = LocalCompressionContextGate(config)

        output = gate(compressed, context, canonical)
        active_gates = output.gates[output.modality_mask]

        self.assertGreater(float(active_gates.min().item()), 0.05)
        self.assertLess(float(active_gates.max().item()), 0.95)
        self.assertGreater(float(active_gates.mean().item()), 0.35)
        self.assertLess(float(active_gates.mean().item()), 0.65)

    def test_none_gate_mode_returns_active_modality_ones(self):
        config = LocalCompressionPartConfig(
            variant="mlp_delta",
            gate_mode=LOCAL_COMPRESSION_GATE_NONE,
            embed_dim=16,
            local_layers=1,
            local_heads=4,
            context_layers=1,
            context_heads=4,
            dropout=0.0,
            attention_dropout=0.0,
        )
        config, canonical, compressed, context = make_gate_inputs(embed_dim=16, config=config)
        gate = LocalCompressionContextGate(config)

        output = gate(compressed, context, canonical)

        self.assertTrue(torch.equal(output.gates.bool(), output.modality_mask))
        self.assertTrue(torch.equal(output.diagnostic_weights.bool(), output.modality_mask))
        self.assertEqual(float(output.gate_logits.abs().sum().item()), 0.0)

    def test_gradients_flow_through_context_gate(self):
        config, canonical, compressed, context = make_gate_inputs(embed_dim=16)
        local_tokens = compressed.local_tokens.detach().clone().requires_grad_(True)
        context_tokens = context.context_tokens.detach().clone().requires_grad_(True)
        gate = LocalCompressionContextGate(config)

        output = gate(local_tokens, context_tokens, canonical, mask=compressed.mask, modality_mask=compressed.modality_mask)
        loss = output.gates[output.modality_mask].mean()
        loss.backward()

        self.assertIsNotNone(local_tokens.grad)
        self.assertIsNotNone(context_tokens.grad)
        self.assertGreater(float(local_tokens.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(context_tokens.grad.abs().sum().item()), 0.0)
        grad_sums = [
            float(parameter.grad.detach().abs().sum().item())
            for parameter in gate.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(any(value > 0.0 for value in grad_sums))


if __name__ == "__main__":
    unittest.main()
