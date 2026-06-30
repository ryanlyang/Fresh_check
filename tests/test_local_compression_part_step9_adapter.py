import math
import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_compression_part import (
    LOCAL_COMPRESSION_ADAPTER_CONTRACT,
    LOCAL_COMPRESSION_GEOMETRY_FEATURES,
    LOCAL_COMPRESSION_PID_FEATURES,
    LocalCompressionContextGate,
    LocalCompressionDeltaFAdapter,
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
            pt = 14.0 + 2.25 * particle_index + 0.6 * batch_index
            eta = -0.22 + 0.07 * particle_index + 0.01 * batch_index
            phi = -math.pi + 0.18 * particle_index
            tokens[batch_index, particle_index, 0] = pt
            tokens[batch_index, particle_index, 1] = eta
            tokens[batch_index, particle_index, 2] = phi
            tokens[batch_index, particle_index, 3] = pt * math.cosh(eta) + 0.4
            tokens[batch_index, particle_index, 4] = 1.0 if particle_index % 2 == 0 else -1.0
            tokens[batch_index, particle_index, 5 + (particle_index % 5)] = 1.0
            tokens[batch_index, particle_index, 10] = 0.015 * particle_index
            tokens[batch_index, particle_index, 11] = 0.05 + 0.01 * particle_index
            tokens[batch_index, particle_index, 12] = -0.025 * particle_index
            tokens[batch_index, particle_index, 13] = 0.08 + 0.01 * batch_index
    return tokens, mask


def make_adapter_inputs(
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
    gates = LocalCompressionContextGate(config)(compressed, context, canonical)
    return config, canonical, pooled, context, gates


def perturb_final_layer(adapter: LocalCompressionDeltaFAdapter, *, value: float = 0.5) -> None:
    with torch.no_grad():
        final_linear = adapter.projector[-1]
        final_linear.bias.fill_(float(value))


class LocalCompressionStep9AdapterTests(unittest.TestCase):
    def test_zero_initialized_adapter_outputs_exact_zero_delta(self):
        config, canonical, pooled, context, gates = make_adapter_inputs(embed_dim=16)
        adapter = LocalCompressionDeltaFAdapter(config)

        output = adapter(canonical, pooled, context, gates)

        self.assertEqual(output.summary()["contract"], LOCAL_COMPRESSION_ADAPTER_CONTRACT)
        self.assertEqual(tuple(output.delta_F_rows.shape), tuple(canonical.feature_rows().shape))
        self.assertEqual(float(output.delta_F_rows.abs().sum().item()), 0.0)
        self.assertEqual(float(output.raw_delta_rows.abs().sum().item()), 0.0)
        self.assertTrue(torch.allclose(output.adapted_feature_rows, canonical.feature_rows()))

    def test_delta_becomes_nonzero_after_perturbing_final_layer(self):
        config, canonical, pooled, context, gates = make_adapter_inputs(embed_dim=16)
        adapter = LocalCompressionDeltaFAdapter(config)
        perturb_final_layer(adapter, value=0.35)

        output = adapter(canonical, pooled, context, gates)

        self.assertGreater(float(output.delta_F_rows[output.mask].abs().sum().item()), 0.0)
        self.assertGreater(float((output.adapted_feature_rows - output.feature_rows)[output.mask].abs().sum().item()), 0.0)

    def test_invalid_particles_get_zero_delta(self):
        config, canonical, pooled, context, gates = make_adapter_inputs(embed_dim=16, include_empty_batch=True)
        adapter = LocalCompressionDeltaFAdapter(config)
        perturb_final_layer(adapter, value=0.5)

        output = adapter(canonical, pooled, context, gates)

        self.assertEqual(float(output.delta_F_rows[0, 5].abs().sum().item()), 0.0)
        self.assertEqual(float(output.delta_F_rows[3].abs().sum().item()), 0.0)
        self.assertEqual(float(output.adapted_feature_rows[3].abs().sum().item()), 0.0)

    def test_delta_is_bounded_by_tanh_and_feature_scales(self):
        config, canonical, pooled, context, gates = make_adapter_inputs(embed_dim=16)
        adapter = LocalCompressionDeltaFAdapter(config)
        perturb_final_layer(adapter, value=100.0)

        output = adapter(canonical, pooled, context, gates)
        scales = output.feature_delta_scales.view(1, 1, -1) * output.feature_active_mask.view(1, 1, -1)

        self.assertTrue(bool(torch.all(output.delta_F_rows.abs() <= scales + 1.0e-6)))
        self.assertLessEqual(float(output.bounded_delta_rows.abs().max().item()), 1.0)
        self.assertGreater(float(output.delta_F_rows[output.mask].abs().max().item()), 0.0)

    def test_freeze_pid_and_geometry_deltas_zeroes_those_features(self):
        config = LocalCompressionPartConfig(
            embed_dim=16,
            local_layers=1,
            local_heads=4,
            context_layers=1,
            context_heads=4,
            dropout=0.0,
            attention_dropout=0.0,
            freeze_pid_deltas=True,
            freeze_geometry_deltas=True,
        )
        config, canonical, pooled, context, gates = make_adapter_inputs(embed_dim=16, config=config)
        adapter = LocalCompressionDeltaFAdapter(config)
        perturb_final_layer(adapter, value=0.5)

        output = adapter(canonical, pooled, context, gates)
        frozen_names = set(LOCAL_COMPRESSION_PID_FEATURES) | set(LOCAL_COMPRESSION_GEOMETRY_FEATURES)
        frozen_indices = [index for index, name in enumerate(output.feature_names) if name in frozen_names]

        self.assertTrue(frozen_indices)
        self.assertEqual(float(output.delta_F_rows[:, :, frozen_indices].abs().sum().item()), 0.0)

    def test_gradients_reach_final_projection_at_zero_init(self):
        config, canonical, pooled, context, gates = make_adapter_inputs(embed_dim=16)
        adapter = LocalCompressionDeltaFAdapter(config)

        output = adapter(canonical, pooled, context, gates)
        loss = output.delta_F_rows[output.mask].sum()
        loss.backward()

        self.assertIsNotNone(adapter.projector[-1].weight.grad)
        self.assertGreater(float(adapter.projector[-1].weight.grad.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
