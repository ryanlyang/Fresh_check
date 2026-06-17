import importlib.util
import tempfile
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from teacher_logit_reco.set_matching import (
    FIVE_VIEW_TAGGER_CONTRACT,
    GEOMETRY_AWARE_FIVE_VIEW_TAGGER_CONTRACT,
    PAIRWISE_GEOMETRY_FEATURE_NAMES,
    RELATION_GLOBAL,
    RELATION_PARTICLE_DIFFERENT_VIEW,
    RELATION_PARTICLE_SAME_VIEW,
    RELATION_SUMMARY_PARTICLE,
    RELATION_VIEW_SUMMARY,
    SET_MATCHING_FIVE_VIEW_TAGGER_STEP,
    TOKEN_KIND_GLOBAL,
    TOKEN_KIND_PARTICLE,
    TOKEN_KIND_VIEW_SUMMARY,
    FiveViewParticleTransformerConfig,
    FiveViewParticleTransformerTagger,
    build_pairwise_geometry_features,
    build_pairwise_relation_type_ids,
    build_five_view_tagger,
    load_five_view_tagger_checkpoint,
    save_five_view_tagger_checkpoint,
    wrapped_five_view_delta_phi,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


def tiny_config(**overrides):
    payload = {
        "particle_feature_dim": RAW_TOKEN_DIM,
        "num_classes": 10,
        "num_views": 5,
        "embed_dim": 32,
        "stage1_layers": 1,
        "stage1_heads": 4,
        "stage2_layers": 1,
        "stage2_heads": 4,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    payload.update(overrides)
    return FiveViewParticleTransformerConfig(**payload)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for five-view model tests")
class FiveViewParticleTransformerTaggerTests(unittest.TestCase):
    def make_batch(self, *, batch_size=2, tokens=4):
        torch.manual_seed(7)
        view_features = torch.randn(batch_size, 5, tokens, RAW_TOKEN_DIM)
        view_masks = torch.ones(batch_size, 5, tokens, dtype=torch.bool)
        view_masks[:, :, -1] = False
        view_masks[:, 3, :] = False
        view_confidence = torch.rand(batch_size, 5, tokens)
        view_confidence = torch.where(view_masks, view_confidence, torch.zeros_like(view_confidence))
        view_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        source_type_ids = torch.tensor([0, 1, 1, 1, 1], dtype=torch.long)
        return {
            "view_features": view_features,
            "view_masks": view_masks,
            "view_confidence": view_confidence,
            "view_ids": view_ids,
            "source_type_ids": source_type_ids,
        }

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            FiveViewParticleTransformerConfig(embed_dim=30, stage1_heads=8)

    def test_forward_returns_logits_and_diagnostics(self):
        model = build_five_view_tagger(tiny_config())
        model.eval()
        batch = self.make_batch()
        with torch.no_grad():
            logits, diagnostics = model(**batch, return_diagnostics=True)

        self.assertEqual(logits.shape, (2, 10))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(diagnostics["contract"], FIVE_VIEW_TAGGER_CONTRACT)
        self.assertEqual(diagnostics["num_views"], 5)
        self.assertEqual(diagnostics["stage1_sequence_length"], 5)
        self.assertFalse(diagnostics["geometry_attention_enabled"])

    def test_geometry_enabled_forward_returns_logits_and_diagnostics(self):
        model = build_five_view_tagger(tiny_config(use_geometry_attention=True, geometry_hidden_dim=16))
        model.eval()
        batch = self.make_batch()
        with torch.no_grad():
            logits, diagnostics = model(**batch, return_diagnostics=True)

        self.assertEqual(logits.shape, (2, 10))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(diagnostics["contract"], GEOMETRY_AWARE_FIVE_VIEW_TAGGER_CONTRACT)
        self.assertTrue(diagnostics["geometry_attention_enabled"])

    def test_masked_particle_values_do_not_leak_into_logits(self):
        torch.manual_seed(19)
        model = FiveViewParticleTransformerTagger(tiny_config())
        model.eval()
        batch_a = self.make_batch(tokens=5)
        batch_b = {key: value.clone() for key, value in batch_a.items()}
        masked = ~batch_b["view_masks"]
        batch_b["view_features"][masked] = 100000.0
        batch_b["view_confidence"][masked] = 1.0

        with torch.no_grad():
            logits_a = model(**batch_a)
            logits_b = model(**batch_b)

        self.assertTrue(torch.allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5))

    def test_view_embedding_changes_output_and_can_be_disabled(self):
        torch.manual_seed(23)
        batch = self.make_batch()
        shuffled_view_ids = torch.tensor([0, 4, 3, 2, 1], dtype=torch.long)
        model = FiveViewParticleTransformerTagger(tiny_config(use_view_embedding=True))
        model.eval()
        with torch.no_grad():
            model.view_embedding.weight.zero_()
            model.view_embedding.weight[:5] = torch.arange(5).float().unsqueeze(1) * 0.25
            logits_a = model(**batch)
            logits_b = model(**{**batch, "view_ids": shuffled_view_ids})
        self.assertFalse(torch.allclose(logits_a, logits_b, atol=1e-6, rtol=1e-6))

        no_view_model = FiveViewParticleTransformerTagger(tiny_config(use_view_embedding=False))
        no_view_model.eval()
        with torch.no_grad():
            logits_c = no_view_model(**batch)
            logits_d = no_view_model(**{**batch, "view_ids": shuffled_view_ids})
        self.assertTrue(torch.allclose(logits_c, logits_d, atol=1e-6, rtol=1e-6))

    def test_checkpoint_roundtrip_preserves_config_and_logits(self):
        torch.manual_seed(31)
        model = FiveViewParticleTransformerTagger(tiny_config())
        model.eval()
        batch = self.make_batch()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_five_view_tagger_checkpoint(
                f"{tmpdir}/best_model_val.pt",
                model,
                extra_payload={"epoch": 3, "metrics": {"stack_val_accuracy": 0.5}},
            )
            loaded, payload = load_five_view_tagger_checkpoint(path)

        self.assertEqual(payload["experiment_step"], SET_MATCHING_FIVE_VIEW_TAGGER_STEP)
        self.assertEqual(payload["output_contract"], FIVE_VIEW_TAGGER_CONTRACT)
        self.assertEqual(loaded.to_config_dict(), model.to_config_dict())
        with torch.no_grad():
            logits_a = model(**batch)
            logits_b = loaded(**batch)
        self.assertTrue(torch.allclose(logits_a, logits_b, atol=1e-6, rtol=1e-6))

    def test_geometry_checkpoint_roundtrip_preserves_contract(self):
        model = FiveViewParticleTransformerTagger(tiny_config(use_geometry_attention=True, geometry_hidden_dim=16))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_five_view_tagger_checkpoint(f"{tmpdir}/best_model_val.pt", model)
            loaded, payload = load_five_view_tagger_checkpoint(path)

        self.assertEqual(payload["output_contract"], GEOMETRY_AWARE_FIVE_VIEW_TAGGER_CONTRACT)
        self.assertTrue(loaded.config.use_geometry_attention)
        self.assertEqual(loaded.output_contract, GEOMETRY_AWARE_FIVE_VIEW_TAGGER_CONTRACT)

    def test_pairwise_geometry_features_wrap_delta_phi(self):
        tokens = torch.zeros(1, 2, RAW_TOKEN_DIM)
        tokens[0, :, 0] = 10.0
        tokens[0, :, 1] = 0.25
        tokens[0, 0, 2] = torch.pi - 0.01
        tokens[0, 1, 2] = -torch.pi + 0.02
        delta = wrapped_five_view_delta_phi(tokens[0, 0, 2] - tokens[0, 1, 2])
        self.assertAlmostEqual(float(delta), -0.03, places=4)

        features = build_pairwise_geometry_features(tokens, token_is_particle=torch.ones(1, 2, dtype=torch.bool))
        names = list(PAIRWISE_GEOMETRY_FEATURE_NAMES)
        sin_index = names.index("sin_delta_phi")
        abs_index = names.index("abs_delta_phi")
        self.assertAlmostEqual(float(features[0, 0, 1, sin_index]), float(torch.sin(delta)), places=5)
        self.assertAlmostEqual(float(features[0, 0, 1, abs_index]), float(delta.abs() / torch.pi), places=5)

    def test_relation_type_ids_mark_same_and_cross_view_pairs(self):
        token_kinds = torch.tensor(
            [[TOKEN_KIND_GLOBAL, TOKEN_KIND_VIEW_SUMMARY, TOKEN_KIND_PARTICLE, TOKEN_KIND_PARTICLE, TOKEN_KIND_PARTICLE]]
        )
        token_view_ids = torch.tensor([[0, 0, 0, 1, 1]])
        relation_ids = build_pairwise_relation_type_ids(token_kinds, token_view_ids)

        self.assertEqual(int(relation_ids[0, 0, 4]), RELATION_GLOBAL)
        self.assertEqual(int(relation_ids[0, 1, 1]), RELATION_VIEW_SUMMARY)
        self.assertEqual(int(relation_ids[0, 1, 2]), RELATION_SUMMARY_PARTICLE)
        self.assertEqual(int(relation_ids[0, 2, 2]), RELATION_PARTICLE_SAME_VIEW)
        self.assertEqual(int(relation_ids[0, 3, 4]), RELATION_PARTICLE_SAME_VIEW)
        self.assertEqual(int(relation_ids[0, 2, 3]), RELATION_PARTICLE_DIFFERENT_VIEW)


if __name__ == "__main__":
    unittest.main()
