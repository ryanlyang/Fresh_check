import math
import unittest

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_compression_part import (
    LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES,
    LOCAL_COMPRESSION_MODEL_CONTRACT,
    LocalCompressionFeatureAdapterParT,
    LocalCompressionPartConfig,
    build_local_compression_feature_adapter_part,
)


torch = require_torch()


class DummyReferencePart(ParticleTransformerHLTClassifier):
    """Tiny local stand-in that still is-a ParticleTransformerHLTClassifier."""

    def __init__(self, num_classes: int = 2):
        torch.nn.Module.__init__(self)
        self.config = {"dummy_reference_part": True, "num_classes": int(num_classes)}
        self.linear = torch.nn.Linear(len(LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES), int(num_classes))
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
            self.linear.weight[0, 0] = 0.7
            self.linear.weight[0, 5] = -0.2
            self.linear.weight[1, 1] = -0.4
            self.linear.weight[1, 7] = 0.3

    def no_weight_decay(self):
        return {"mod.cls_token"}

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        rows = features.transpose(1, 2).contiguous()
        particle_mask = mask.squeeze(1).to(dtype=rows.dtype)
        pooled = (rows * particle_mask[:, :, None]).sum(dim=1) / particle_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.linear(pooled)


class NotReferencePart(torch.nn.Module):
    def forward(self, points, features, lorentz_vectors, mask):
        del points, features, lorentz_vectors, mask
        return torch.zeros((2, 2), dtype=torch.float32)


def make_tokens(num_particles: int = 6):
    tokens = torch.zeros((2, num_particles, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.zeros((2, num_particles), dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :4] = True
    for batch_index in range(2):
        for particle_index in range(int(mask[batch_index].sum().item())):
            pt = 16.0 + 3.0 * particle_index + 0.8 * batch_index
            eta = -0.2 + 0.08 * particle_index
            phi = -math.pi + 0.16 * particle_index
            tokens[batch_index, particle_index, 0] = pt
            tokens[batch_index, particle_index, 1] = eta
            tokens[batch_index, particle_index, 2] = phi
            tokens[batch_index, particle_index, 3] = pt * math.cosh(eta) + 0.5
            tokens[batch_index, particle_index, 4] = 1.0 if particle_index % 2 == 0 else -1.0
            tokens[batch_index, particle_index, 5 + (particle_index % 5)] = 1.0
            tokens[batch_index, particle_index, 10] = 0.015 * particle_index
            tokens[batch_index, particle_index, 11] = 0.05 + 0.01 * particle_index
            tokens[batch_index, particle_index, 12] = -0.025 * particle_index
            tokens[batch_index, particle_index, 13] = 0.08 + 0.01 * batch_index
    return tokens, mask


def small_config(**kwargs):
    payload = {
        "embed_dim": 16,
        "local_layers": 1,
        "local_heads": 4,
        "context_layers": 1,
        "context_heads": 4,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    payload.update(kwargs)
    return LocalCompressionPartConfig(**payload)


class LocalCompressionStep10ModelTests(unittest.TestCase):
    def test_zero_delta_logits_match_reference_part_baseline(self):
        tokens, mask = make_tokens()
        part_model = DummyReferencePart(num_classes=2)
        model = LocalCompressionFeatureAdapterParT(small_config(), part_model=part_model)
        model.eval()
        canonical = model.build_canonical_inputs(tokens, mask, max_constits=tokens.shape[1])

        with torch.no_grad():
            baseline_logits = part_model(
                canonical.points,
                canonical.features,
                canonical.lorentz_vectors,
                canonical.mask,
            )
            output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])

        self.assertEqual(output.output_contract, LOCAL_COMPRESSION_MODEL_CONTRACT)
        self.assertTrue(output.baseline_recoverable_at_zero_delta)
        self.assertTrue(output.uses_reference_part_backbone)
        self.assertEqual(output.part_model_class, "DummyReferencePart")
        self.assertEqual(float(output.delta_output.delta_F_rows.abs().sum().item()), 0.0)
        self.assertTrue(torch.allclose(output.adapted_inputs.feature_rows(), output.canonical_inputs.feature_rows()))
        self.assertTrue(torch.allclose(output.logits, baseline_logits, atol=1.0e-6))

    def test_changing_adapter_changes_logits_and_adapted_features(self):
        tokens, mask = make_tokens()
        model = LocalCompressionFeatureAdapterParT(small_config(), part_model=DummyReferencePart(num_classes=2))
        model.eval()

        with torch.no_grad():
            zero_output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])
            model.adapter.projector[-1].bias.fill_(0.5)
            changed_output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])

        self.assertGreater(float(changed_output.delta_output.delta_F_rows.abs().sum().item()), 0.0)
        self.assertGreater(
            float((changed_output.adapted_inputs.feature_rows() - zero_output.canonical_inputs.feature_rows()).abs().sum().item()),
            0.0,
        )
        self.assertGreater(float((changed_output.logits - zero_output.logits).abs().sum().item()), 0.0)

    def test_forward_mapping_and_diagnostics(self):
        tokens, mask = make_tokens()
        model = build_local_compression_feature_adapter_part(small_config(), part_model=DummyReferencePart(num_classes=2))

        logits, diagnostics = model(
            {"hlt_tokens": tokens, "hlt_mask": mask},
            return_diagnostics=True,
            max_constits=tokens.shape[1],
        )

        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(diagnostics["contract"], LOCAL_COMPRESSION_MODEL_CONTRACT)
        self.assertTrue(diagnostics["baseline_recoverable_at_zero_delta"])
        self.assertTrue(diagnostics["uses_reference_part_backbone"])
        self.assertEqual(diagnostics["part_model_class"], "DummyReferencePart")
        self.assertEqual(diagnostics["delta_abs_max"], 0.0)
        self.assertIn("gate_summary", diagnostics)
        self.assertIn("delta_summary", diagnostics)

    def test_model_reports_baseline_recoverability_and_no_weight_decay(self):
        model = LocalCompressionFeatureAdapterParT(small_config(), part_model=DummyReferencePart(num_classes=2))
        config_dict = model.to_config_dict()

        self.assertTrue(model.baseline_recoverable_at_zero_delta)
        self.assertTrue(config_dict["baseline_recoverable_at_zero_delta"])
        self.assertTrue(config_dict["uses_reference_part_backbone"])
        self.assertIn("part_model.mod.cls_token", model.no_weight_decay())

    def test_rejects_non_reference_part_backbone(self):
        with self.assertRaisesRegex(ValueError, "ParticleTransformerHLTClassifier"):
            LocalCompressionFeatureAdapterParT(small_config(), part_model=NotReferencePart())


if __name__ == "__main__":
    unittest.main()
