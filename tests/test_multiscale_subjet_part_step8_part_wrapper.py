import unittest

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.multiscale_subjet_part import (
    CANONICAL_PART_FEATURE_NAMES,
    MULTISCALE_SUBJET_BASELINE_VARIANT,
    MULTISCALE_SUBJET_HLT_PART_BASELINE_CONTRACT,
    MULTISCALE_SUBJET_HLT_PART_BASELINE_STEP,
    MULTISCALE_SUBJET_REFERENCE_PART_BACKBONE_CONTRACT,
    MultiScaleSubjetHLTPartBaselineClassifier,
    MultiScaleSubjetReferencePartBackbone,
    MultiScaleSubjetReferencePartConfig,
    MultiScaleSubjetTokenBuilder,
    MultiScaleSubjetTokenBuilderConfig,
    ParticleSubjetCrossAttentionConfig,
    ParticleSubjetCrossAttentionReadback,
    SoftSubjetAssignmentConfig,
    SubjetScaleSpec,
    SubjetSubjetTransformer,
    SubjetTransformerConfig,
)

torch = require_torch()


class DummyReferencePart(ParticleTransformerHLTClassifier):
    """Tiny local stand-in that still is-a ParticleTransformerHLTClassifier."""

    def __init__(self, num_classes=2):
        torch.nn.Module.__init__(self)
        self.config = {"dummy_reference_part": True, "num_classes": int(num_classes)}
        self.linear = torch.nn.Linear(len(CANONICAL_PART_FEATURE_NAMES), int(num_classes))

    def no_weight_decay(self):
        return {"mod.cls_token"}

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        rows = features.transpose(1, 2).contiguous()
        particle_mask = mask.squeeze(1).to(dtype=rows.dtype)
        pooled = (rows * particle_mask[:, :, None]).sum(dim=1) / particle_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.linear(pooled)


def make_tokens():
    tokens = torch.zeros((2, 5, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, False, False, False],
        ],
        dtype=torch.bool,
    )
    rows = [
        [(50.0, 0.00, 0.00, 55.0), (20.0, 0.05, 0.01, 22.0), (10.0, 1.00, 1.00, 12.0)],
        [(30.0, -0.10, 0.20, 35.0), (5.0, 2.00, -2.00, 6.0)],
    ]
    for batch_index, batch_rows in enumerate(rows):
        for particle_index, (pt, eta, phi, energy) in enumerate(batch_rows):
            tokens[batch_index, particle_index, 0] = pt
            tokens[batch_index, particle_index, 1] = eta
            tokens[batch_index, particle_index, 2] = phi
            tokens[batch_index, particle_index, 3] = energy
            tokens[batch_index, particle_index, 5] = 1.0
    return tokens, mask


def make_readback(tokens, mask):
    assignment = SoftSubjetAssignmentConfig(
        scale_specs=(
            SubjetScaleSpec("small", 2, 0.05, 0.12, "tight"),
            SubjetScaleSpec("medium", 1, 0.12, 0.25, "medium"),
            SubjetScaleSpec("large", 1, 0.25, 0.50, "wide"),
        ),
        embed_dim=16,
        hidden_dim=32,
    )
    token_output = MultiScaleSubjetTokenBuilder(
        MultiScaleSubjetTokenBuilderConfig(
            assignment_config=assignment,
            token_dim=24,
            hidden_dim=48,
            dropout=0.0,
        )
    )(tokens, mask)
    subjet_output = SubjetSubjetTransformer(
        SubjetTransformerConfig(
            token_dim=24,
            num_layers=1,
            num_heads=4,
            ffn_dim=48,
            dropout=0.0,
            attention_dropout=0.0,
            num_scales=3,
        )
    )(token_output)
    return ParticleSubjetCrossAttentionReadback(
        ParticleSubjetCrossAttentionConfig(
            feature_dim=len(CANONICAL_PART_FEATURE_NAMES),
            subjet_token_dim=24,
            hidden_dim=32,
            num_heads=4,
            delta_hidden_dim=64,
            dropout=0.0,
            attention_dropout=0.0,
            residual_gamma_init=0.0,
        )
    )(tokens, mask, subjet_output)


class MultiscaleSubjetPartStep8PartWrapperTests(unittest.TestCase):
    def test_hlt_part_baseline_uses_reference_backbone_and_raw_token_contract(self):
        tokens, mask = make_tokens()
        classifier = MultiScaleSubjetHLTPartBaselineClassifier(
            MultiScaleSubjetReferencePartConfig(num_classes=2, model_size="tiny"),
            part_model=DummyReferencePart(num_classes=2),
        )
        output = classifier.forward_outputs(tokens, mask)
        summary = output.summary()

        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertEqual(summary["contract"], MULTISCALE_SUBJET_HLT_PART_BASELINE_CONTRACT)
        self.assertEqual(summary["step"], MULTISCALE_SUBJET_HLT_PART_BASELINE_STEP)
        self.assertEqual(summary["variant"], MULTISCALE_SUBJET_BASELINE_VARIANT)
        self.assertEqual(summary["baseline_variant"], MULTISCALE_SUBJET_BASELINE_VARIANT)
        self.assertTrue(summary["uses_reference_part_backbone"])
        self.assertFalse(summary["uses_custom_raw_token_transformer"])
        self.assertTrue(summary["serious_comparison_ready"])
        self.assertEqual(output.part_inputs["features"].shape[1], len(CANONICAL_PART_FEATURE_NAMES))
        logits = classifier(tokens, mask)
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_reference_backbone_rejects_non_reference_part_model_by_default(self):
        with self.assertRaisesRegex(ValueError, "real ParticleTransformerHLTClassifier"):
            MultiScaleSubjetHLTPartBaselineClassifier(
                MultiScaleSubjetReferencePartConfig(num_classes=2, model_size="tiny"),
                part_model=torch.nn.Linear(2, 2),
            )

    def test_reference_backbone_can_consume_step7_readback_outputs(self):
        tokens, mask = make_tokens()
        readback = make_readback(tokens, mask)
        backbone = MultiScaleSubjetReferencePartBackbone(
            MultiScaleSubjetReferencePartConfig(num_classes=2, model_size="tiny"),
            part_model=DummyReferencePart(num_classes=2),
        )
        output = backbone.forward_readback(readback)
        summary = output.summary()

        self.assertEqual(summary["contract"], MULTISCALE_SUBJET_REFERENCE_PART_BACKBONE_CONTRACT)
        self.assertTrue(summary["adapted_features"])
        self.assertTrue(summary["uses_reference_part_backbone"])
        self.assertTrue(bool(torch.allclose(output.part_inputs["features"], readback.part_features)))
        self.assertTrue(bool(torch.equal(output.part_inputs["lorentz_vectors"], readback.part_lorentz_vectors)))
        self.assertTrue(bool(torch.equal(output.part_inputs["mask"], readback.part_mask)))
        self.assertEqual(tuple(output.logits.shape), (2, 2))

    def test_no_weight_decay_is_prefixed_from_part_backbone(self):
        classifier = MultiScaleSubjetHLTPartBaselineClassifier(
            MultiScaleSubjetReferencePartConfig(num_classes=2, model_size="tiny"),
            part_model=DummyReferencePart(num_classes=2),
        )

        self.assertIn("part_model.mod.cls_token", classifier.no_weight_decay())

    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, "baseline_variant"):
            MultiScaleSubjetReferencePartConfig(baseline_variant="not_the_baseline")
        with self.assertRaisesRegex(ValueError, "model_size"):
            MultiScaleSubjetReferencePartConfig(model_size="medium")


if __name__ == "__main__":
    unittest.main()
