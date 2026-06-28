import unittest

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.multiscale_subjet_part import (
    CANONICAL_PART_FEATURE_NAMES,
    MULTISCALE_SUBJET_CLASSIFIER_CONTRACT,
    MULTISCALE_SUBJET_HLT_PART_BASELINE_CONTRACT,
    MULTISCALE_SUBJET_VARIANT_CLS_FUSION,
    MULTISCALE_SUBJET_VARIANT_CROSS_ATTENTION_FUSION,
    MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE,
    MULTISCALE_SUBJET_VARIANT_LATE_FUSION,
    MULTISCALE_SUBJET_VARIANT_PURE_PERCEIVER_LATENT_CONTROL,
    MULTISCALE_SUBJET_VARIANT_RANDOM_SUBJET_CONTROL,
    MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER,
    MULTISCALE_SUBJET_VARIANT_SUBJET_BRANCH_ONLY,
    MULTISCALE_SUBJET_VARIANT_TWO_HLT_PART_ENSEMBLE,
    MultiScaleSubjetClassifierConfig,
    MultiScaleSubjetPartClassifier,
    build_multiscale_subjet_comparison_classifier,
    normalize_multiscale_subjet_variant,
)

torch = require_torch()


class DummyReferencePart(ParticleTransformerHLTClassifier):
    """Tiny local stand-in that still satisfies the reference ParT type check."""

    def __init__(self, num_classes=2):
        torch.nn.Module.__init__(self)
        self.config = {"dummy_reference_part": True, "num_classes": int(num_classes)}
        self.linear = torch.nn.Linear(len(CANONICAL_PART_FEATURE_NAMES) + 4 + 1, int(num_classes))

    def no_weight_decay(self):
        return {"mod.cls_token"}

    def forward(self, points, features, lorentz_vectors, mask):
        del points
        rows = torch.cat([features, lorentz_vectors, mask.float()], dim=1).transpose(1, 2).contiguous()
        particle_mask = mask.squeeze(1).to(dtype=rows.dtype)
        pooled = (rows * particle_mask[:, :, None]).sum(dim=1) / particle_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.linear(pooled)


class NotAReferencePart(torch.nn.Module):
    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors, mask
        return torch.zeros((features.shape[0], 2), dtype=features.dtype, device=features.device)


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
            tokens[batch_index, particle_index, 10] = 0.03 * particle_index
            tokens[batch_index, particle_index, 11] = 0.04
            tokens[batch_index, particle_index, 12] = -0.07 * particle_index
            tokens[batch_index, particle_index, 13] = 0.08
    return tokens, mask


def small_config(variant, **kwargs):
    payload = {
        "variant": variant,
        "num_classes": 2,
        "model_size": "tiny",
        "max_constits": 5,
        "token_dim": 24,
        "token_hidden_dim": 48,
        "assignment_embed_dim": 16,
        "assignment_hidden_dim": 32,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "transformer_ffn_dim": 48,
        "transformer_pair_bias_hidden_dim": 16,
        "readback_hidden_dim": 32,
        "readback_heads": 4,
        "readback_delta_hidden_dim": 64,
        "branch_hidden_dim": 48,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    payload.update(kwargs)
    return MultiScaleSubjetClassifierConfig(**payload)


class MultiscaleSubjetPartStep9ClassifierTests(unittest.TestCase):
    def test_comparison_builder_returns_exact_step8_baseline_for_baseline_variant(self):
        tokens, mask = make_tokens()
        model = build_multiscale_subjet_comparison_classifier(
            MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE,
            model_size="tiny",
            max_constits=5,
            part_model=DummyReferencePart(num_classes=2),
        )
        output = model(tokens, mask, return_outputs=True)

        self.assertEqual(output.summary()["contract"], MULTISCALE_SUBJET_HLT_PART_BASELINE_CONTRACT)
        self.assertEqual(output.summary()["variant"], MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE)
        self.assertTrue(output.summary()["uses_reference_part_backbone"])
        self.assertTrue(output.summary()["serious_comparison_ready"])
        self.assertEqual(tuple(output.logits.shape), (2, 2))

    def test_primary_residual_adapter_zero_gamma_preserves_part_inputs(self):
        tokens, mask = make_tokens()
        part_model = DummyReferencePart(num_classes=2)
        baseline = build_multiscale_subjet_comparison_classifier(
            MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE,
            model_size="tiny",
            max_constits=5,
            part_model=part_model,
        )
        adapter = MultiScaleSubjetPartClassifier(
            small_config(MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER, residual_gamma_init=0.0),
            part_model=part_model,
        )

        baseline_output = baseline(tokens, mask, return_outputs=True)
        adapter_output = adapter(tokens, mask, return_outputs=True)
        summary = adapter_output.summary()
        config_summary = adapter.to_config_dict()

        self.assertEqual(summary["contract"], MULTISCALE_SUBJET_CLASSIFIER_CONTRACT)
        self.assertEqual(summary["variant"], MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER)
        self.assertTrue(summary["uses_reference_part_backbone"])
        self.assertTrue(summary["serious_comparison_ready"])
        self.assertTrue(summary["baseline_recoverable_at_zero_gamma"])
        self.assertTrue(config_summary["baseline_recoverable_at_zero_gamma"])
        self.assertIsNotNone(adapter_output.token_output)
        self.assertIsNotNone(adapter_output.transformer_output)
        self.assertIsNotNone(adapter_output.readback_output)
        self.assertTrue(
            bool(torch.allclose(adapter_output.readback_output.adapted_features, adapter_output.readback_output.canonical_inputs.feature_rows()))
        )
        self.assertTrue(bool(torch.allclose(adapter_output.logits, baseline_output.logits, atol=1.0e-6)))

    def test_primary_residual_adapter_active_gamma_changes_features_and_gets_gradients(self):
        tokens, mask = make_tokens()
        model = MultiScaleSubjetPartClassifier(
            small_config(MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER, residual_gamma_init=0.2),
            part_model=DummyReferencePart(num_classes=2),
        )
        output = model(tokens, mask, return_outputs=True)
        delta = (output.readback_output.adapted_features - output.readback_output.canonical_inputs.feature_rows()).abs().sum()
        loss = output.logits.square().mean()
        loss.backward()

        self.assertGreater(float(delta.detach().item()), 1.0e-6)
        self.assertIsNotNone(model.readback.gamma_F.grad)
        self.assertGreater(float(model.readback.gamma_F.grad.abs().item()), 0.0)
        self.assertIsNotNone(model.token_builder.token_projection[1].weight.grad)

    def test_subjet_branch_only_does_not_use_part_anchor(self):
        tokens, mask = make_tokens()
        model = MultiScaleSubjetPartClassifier(small_config(MULTISCALE_SUBJET_VARIANT_SUBJET_BRANCH_ONLY))
        output = model({"hlt_tokens": tokens, "hlt_mask": mask}, return_outputs=True)

        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertFalse(output.summary()["uses_reference_part_backbone"])
        self.assertFalse(output.summary()["serious_comparison_ready"])
        self.assertTrue(output.summary()["subjet_branch_only"])
        self.assertIsNone(output.readback_output)
        self.assertIsNotNone(output.subjet_branch_attention)

    def test_pure_perceiver_variant_uses_learned_queries(self):
        tokens, mask = make_tokens()
        model = MultiScaleSubjetPartClassifier(
            small_config(MULTISCALE_SUBJET_VARIANT_PURE_PERCEIVER_LATENT_CONTROL),
            part_model=DummyReferencePart(num_classes=2),
        )
        output = model(tokens, mask, return_outputs=True)

        self.assertEqual(output.token_output.assignment_output.query_mode, "learned")
        self.assertTrue(output.summary()["pure_perceiver_latent_control"])
        self.assertTrue(output.summary()["uses_reference_part_backbone"])

    def test_random_subjet_control_replaces_token_content_but_keeps_shapes(self):
        tokens, mask = make_tokens()
        model = MultiScaleSubjetPartClassifier(
            small_config(MULTISCALE_SUBJET_VARIANT_RANDOM_SUBJET_CONTROL),
            part_model=DummyReferencePart(num_classes=2),
        )
        output = model(tokens, mask, return_outputs=True)

        self.assertTrue(output.summary()["random_subjet_token_control"])
        self.assertTrue(output.summary()["random_subjet_pair_bias_disabled"])
        self.assertEqual(tuple(output.token_output.subjet_tokens.shape), (2, 20, 24))
        self.assertFalse(output.transformer_output.diagnostics["use_pairwise_bias"])
        self.assertIsNone(output.transformer_output.pair_bias)
        self.assertTrue(model.to_config_dict()["random_subjet_pair_bias_disabled"])
        expanded = model.random_control_tokens.expand(2, -1, -1) * output.token_output.subjet_mask[:, :, None].float()
        self.assertTrue(bool(torch.allclose(output.token_output.subjet_tokens, expanded)))

    def test_part_anchored_variants_reject_non_reference_part_backbone(self):
        with self.assertRaisesRegex(ValueError, "ParticleTransformerHLTClassifier"):
            MultiScaleSubjetPartClassifier(
                small_config(MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER),
                part_model=NotAReferencePart(),
            )

    def test_variant_normalization_and_config_validation(self):
        self.assertEqual(normalize_multiscale_subjet_variant("primary"), MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER)
        self.assertEqual(normalize_multiscale_subjet_variant("late_fusion"), MULTISCALE_SUBJET_VARIANT_LATE_FUSION)
        self.assertEqual(normalize_multiscale_subjet_variant("cls_fusion"), MULTISCALE_SUBJET_VARIANT_CLS_FUSION)
        self.assertEqual(
            normalize_multiscale_subjet_variant("cross_attention_fusion"),
            MULTISCALE_SUBJET_VARIANT_CROSS_ATTENTION_FUSION,
        )
        self.assertEqual(normalize_multiscale_subjet_variant("two_hlt_part"), MULTISCALE_SUBJET_VARIANT_TWO_HLT_PART_ENSEMBLE)
        with self.assertRaisesRegex(ValueError, "variant"):
            normalize_multiscale_subjet_variant("mystery")
        with self.assertRaisesRegex(ValueError, "divisible"):
            small_config(MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER, token_dim=25, transformer_heads=4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            small_config(MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER, readback_hidden_dim=30, readback_heads=4)

    def test_step14_scale_and_zero_transformer_ablation_knobs_are_real(self):
        tokens, mask = make_tokens()
        model = MultiScaleSubjetPartClassifier(
            small_config(
                MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER,
                scale_profile="few_subjets",
                use_assignment_scale_embedding=False,
                use_token_scale_embedding=False,
                use_scale_pair_embedding=False,
                transformer_layers=0,
            ),
            part_model=DummyReferencePart(num_classes=2),
        )
        output = model(tokens, mask, return_outputs=True)

        self.assertEqual(tuple(output.token_output.subjet_tokens.shape[:2]), (2, 10))
        self.assertEqual(output.token_output.assignment_output.diagnostics["use_scale_embedding"], False)
        self.assertEqual(output.token_output.diagnostics["use_token_scale_embedding"], False)
        self.assertEqual(output.transformer_output.diagnostics["num_layers"], 0)
        self.assertEqual(output.transformer_output.diagnostics["use_scale_pair_embedding"], False)

    def test_step14_fusion_and_ensemble_variants_are_queueable_models(self):
        tokens, mask = make_tokens()
        for variant in (
            MULTISCALE_SUBJET_VARIANT_LATE_FUSION,
            MULTISCALE_SUBJET_VARIANT_CLS_FUSION,
            MULTISCALE_SUBJET_VARIANT_CROSS_ATTENTION_FUSION,
        ):
            model = MultiScaleSubjetPartClassifier(
                small_config(variant, residual_gamma_init=0.0),
                part_model=DummyReferencePart(num_classes=2),
            )
            output = model(tokens, mask, return_outputs=True)
            self.assertEqual(tuple(output.logits.shape), (2, 2))
            self.assertTrue(output.summary()["uses_reference_part_backbone"])
            self.assertTrue(output.summary()["uses_subjet_branch_logits"])

        ensemble = MultiScaleSubjetPartClassifier(
            small_config(MULTISCALE_SUBJET_VARIANT_TWO_HLT_PART_ENSEMBLE),
            part_model=DummyReferencePart(num_classes=2),
        )
        output = ensemble(tokens, mask, return_outputs=True)
        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertTrue(output.summary()["uses_two_part_ensemble"])
        self.assertIsNone(output.token_output)


if __name__ == "__main__":
    unittest.main()
