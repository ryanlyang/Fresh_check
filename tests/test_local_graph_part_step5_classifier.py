import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_ADAPTER_EDGECONV,
    LOCAL_GRAPH_ADAPTER_NONE,
    LOCAL_GRAPH_ADAPTER_POINT_ATTENTION,
    LOCAL_GRAPH_AUGMENTED_PART_CONTRACT,
    LOCAL_GRAPH_MODEL_VARIANT_EDGECONV,
    LOCAL_GRAPH_MODEL_VARIANT_NO_LOCAL,
    LOCAL_GRAPH_MODEL_VARIANT_POINT_ATTENTION,
    LOCAL_GRAPH_PART_CLASSIFIER_CONTRACT,
    LocalGraphAugmentedPartConfig,
    LocalGraphParticleTransformerConfig,
    build_local_graph_augmented_part_classifier,
    build_local_graph_particle_transformer_classifier,
    build_local_knn_graph,
    LocalKnnConfig,
)


class DummyPartBackbone:
    def __init__(self, num_classes=2):
        torch = require_torch()
        self.config = {"architecture": "dummy_part_backbone", "num_classes": int(num_classes)}
        self.proj = torch.nn.Linear(17 + 4 + 1, int(num_classes))

    def __call__(self, points, features, lorentz_vectors, mask):
        torch = require_torch()
        del points
        particle_mask = mask.float()
        packed = torch.cat([features, lorentz_vectors, particle_mask], dim=1).transpose(1, 2)
        denom = torch.clamp(particle_mask.squeeze(1).sum(dim=1, keepdim=True), min=1.0)
        pooled = (packed * particle_mask.transpose(1, 2)).sum(dim=1) / denom
        return self.proj(pooled)

    def no_weight_decay(self):
        return set()


class LocalGraphPartStep5ClassifierTests(unittest.TestCase):
    def make_tokens(self):
        torch = require_torch()
        tokens = torch.zeros((2, 6, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.zeros((2, 6), dtype=torch.bool)
        mask[0, [0, 1, 2, 4]] = True
        mask[1, [0, 2, 3, 5]] = True
        for batch in range(2):
            for index in range(6):
                tokens[batch, index, 0] = 15.0 + batch + 2.5 * index
                tokens[batch, index, 1] = -0.7 + 0.22 * index
                tokens[batch, index, 2] = -0.4 + 0.16 * index
                tokens[batch, index, 3] = tokens[batch, index, 0] * torch.cosh(tokens[batch, index, 1]) + 0.5
                tokens[batch, index, 4] = -1.0 + (index % 3)
                tokens[batch, index, 5 + (index % 5)] = 1.0
                tokens[batch, index, 10] = 0.06 * index
                tokens[batch, index, 11] = 0.04
                tokens[batch, index, 12] = -0.13 * index
                tokens[batch, index, 13] = 0.08
        return tokens, mask

    def make_config(self, adapter: str, **kwargs):
        payload = {
            "num_classes": 2,
            "embed_dim": 24,
            "global_layers": 2,
            "global_heads": 4,
            "local_adapter": adapter,
            "local_heads": 3,
            "k": 3,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "pairwise_hidden_dim": 16,
        }
        payload.update(kwargs)
        return LocalGraphParticleTransformerConfig(**payload)

    def test_classifier_forward_for_each_adapter_variant(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        cases = [
            (LOCAL_GRAPH_ADAPTER_NONE, LOCAL_GRAPH_MODEL_VARIANT_NO_LOCAL),
            (LOCAL_GRAPH_ADAPTER_EDGECONV, LOCAL_GRAPH_MODEL_VARIANT_EDGECONV),
            (LOCAL_GRAPH_ADAPTER_POINT_ATTENTION, LOCAL_GRAPH_MODEL_VARIANT_POINT_ATTENTION),
        ]
        for adapter, variant in cases:
            with self.subTest(adapter=adapter):
                torch.manual_seed(101)
                model = build_local_graph_particle_transformer_classifier(self.make_config(adapter))
                output = model(tokens, mask, return_outputs=True)

                self.assertEqual(tuple(output.logits.shape), (2, 2))
                self.assertEqual(tuple(output.sequence_tokens.shape), (2, 7, 24))
                self.assertEqual(tuple(output.sequence_mask.shape), (2, 7))
                self.assertTrue(bool(torch.isfinite(output.logits).all()))
                self.assertEqual(output.summary()["contract"], LOCAL_GRAPH_PART_CLASSIFIER_CONTRACT)
                self.assertEqual(output.summary()["variant"], variant)
                self.assertTrue(output.summary()["prototype_only"])
                self.assertFalse(output.summary()["uses_reference_part_backbone"])
                self.assertFalse(output.summary()["serious_comparison_ready"])
                self.assertIsNotNone(output.pairwise_features)
                self.assertIsNotNone(output.attention_bias)
                self.assertEqual(tuple(output.attention_bias.shape), (2, 4, 7, 7))
                self.assertTrue(bool((output.sequence_tokens[~output.sequence_mask] == 0.0).all()))
                if adapter == LOCAL_GRAPH_ADAPTER_NONE:
                    self.assertIsNone(output.local_adapter_output)
                    self.assertIsNone(output.knn)
                else:
                    self.assertIsNotNone(output.local_adapter_output)
                    self.assertIsNotNone(output.knn)

    def test_zero_gamma_local_adapter_matches_embedded_particles_before_global_stage(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        for adapter in (LOCAL_GRAPH_ADAPTER_EDGECONV, LOCAL_GRAPH_ADAPTER_POINT_ATTENTION):
            with self.subTest(adapter=adapter):
                torch.manual_seed(103)
                model = build_local_graph_particle_transformer_classifier(self.make_config(adapter, residual_gamma_init=0.0))
                output = model(tokens, mask, return_outputs=True)

                self.assertTrue(bool(torch.allclose(output.adapted_particles[mask], output.particle_embeddings[mask], atol=1.0e-6)))
                self.assertTrue(bool((output.adapted_particles[~mask] == 0.0).all()))

    def test_precomputed_knn_is_used_by_classifier_local_adapter(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        graph = build_local_knn_graph(tokens, mask, LocalKnnConfig(k=2, include_self=False))
        model = build_local_graph_particle_transformer_classifier(self.make_config(LOCAL_GRAPH_ADAPTER_POINT_ATTENTION, k=2))
        output = model(tokens, mask, return_outputs=True, knn=graph)

        self.assertIsNotNone(output.local_adapter_output)
        self.assertTrue(bool(torch.equal(output.local_adapter_output.edge_features.knn.indices, graph.indices)))
        self.assertTrue(bool(torch.equal(output.local_adapter_output.edge_features.edge_mask, graph.neighbor_mask)))

    def test_forward_diagnostics_and_mapping_inputs(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        model = build_local_graph_particle_transformer_classifier(self.make_config(LOCAL_GRAPH_ADAPTER_EDGECONV))
        logits, diagnostics = model({"hlt_tokens": tokens, "hlt_mask": mask}, return_diagnostics=True)

        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertIn("pairwise_attention_bias_abs_mean", diagnostics)
        self.assertIn("local_gamma", diagnostics)
        self.assertEqual(float(diagnostics["serious_comparison_ready"].detach().cpu().item()), 0.0)
        self.assertTrue(bool(torch.isfinite(diagnostics["valid_particle_count_mean"])))

    def test_config_validation_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            LocalGraphParticleTransformerConfig(embed_dim=22, global_heads=4)
        with self.assertRaises(ValueError):
            LocalGraphParticleTransformerConfig(embed_dim=24, local_adapter=LOCAL_GRAPH_ADAPTER_POINT_ATTENTION, local_heads=5)
        with self.assertRaises(ValueError):
            LocalGraphParticleTransformerConfig(local_adapter="mystery")
        with self.assertRaises(ValueError):
            LocalGraphParticleTransformerConfig(local_adapter="baseline")

    def test_augmented_part_wrapper_feeds_adapted_pf_features_to_part_backbone(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        dummy = DummyPartBackbone(num_classes=2)
        model = build_local_graph_augmented_part_classifier(
            LocalGraphAugmentedPartConfig(
                num_classes=2,
                local_adapter=LOCAL_GRAPH_ADAPTER_POINT_ATTENTION,
                local_heads=1,
                k=3,
                dropout=0.0,
                attention_dropout=0.0,
                residual_gamma_init=0.0,
            ),
            part_model=dummy,
        )
        output = model(tokens, mask, return_outputs=True)

        self.assertEqual(output.summary()["contract"], LOCAL_GRAPH_AUGMENTED_PART_CONTRACT)
        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertEqual(tuple(output.canonical_features.shape), (2, 6, 17))
        self.assertTrue(bool(torch.allclose(output.adapted_features[mask], output.canonical_features[mask], atol=1.0e-6)))
        self.assertFalse(output.summary()["uses_reference_part_backbone"])
        self.assertFalse(output.summary()["serious_comparison_ready"])
        self.assertIsNotNone(output.local_adapter_output)

    def test_augmented_part_wrapper_active_gamma_changes_pf_features_and_gets_gradients(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        dummy = DummyPartBackbone(num_classes=2)
        model = build_local_graph_augmented_part_classifier(
            LocalGraphAugmentedPartConfig(
                num_classes=2,
                local_adapter=LOCAL_GRAPH_ADAPTER_EDGECONV,
                k=3,
                dropout=0.0,
                attention_dropout=0.0,
                residual_gamma_init=1.0,
            ),
            part_model=dummy,
        )
        output = model(tokens, mask, return_outputs=True)
        delta = (output.adapted_features[mask] - output.canonical_features[mask]).abs().sum()
        loss = output.logits.square().mean()
        loss.backward()

        self.assertGreater(float(delta.detach().item()), 1.0e-6)
        self.assertIsNotNone(model.local_adapter.gamma.grad)
        self.assertGreater(float(model.local_adapter.gamma.grad.detach().abs().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
