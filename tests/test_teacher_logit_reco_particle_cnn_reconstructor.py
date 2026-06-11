import importlib.util
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from teacher_logit_reco.particle_cnn_reconstructor import (
    PARTICLE_CNN_INPUT_FEATURE_DIM,
    PARTICLE_CNN_ORDERING_ASSUMPTION,
    PARTICLE_CNN_RANK_FEATURE_NAMES,
    PARTICLE_CNN_RECONSTRUCTOR_ARCHITECTURE,
    ParticleCnnReconstructorConfig,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.particle_cnn_reconstructor import (
        ParticleCnnBlock,
        ParticleCnnContextBuilder,
        ParticleCnnEncoder,
        ParticleCnnEncoderOutput,
        ParticleCnnReconstructor,
        apply_particle_mask_channels,
        audit_particle_cnn_cache_order,
        build_particle_cnn_features,
        build_rank_features,
        masked_max_pool,
        masked_mean_pool,
        masked_sum_pool,
    )
    from teacher_logit_reco.global_transformer import physical_energy_floor
    from teacher_logit_reco.views import SoftReconstructedView


class ParticleCnnReconstructorConfigTests(unittest.TestCase):
    def test_defaults_are_step1_checkpoint_ready(self):
        config = ParticleCnnReconstructorConfig()
        payload = config.to_dict()
        self.assertEqual(config.input_dim, RAW_TOKEN_DIM)
        self.assertEqual(config.num_blocks, len(config.kernel_sizes))
        self.assertEqual(config.num_blocks, len(config.dilations))
        self.assertEqual(payload["reconstructor_architecture"], PARTICLE_CNN_RECONSTRUCTOR_ARCHITECTURE)
        self.assertEqual(payload["kernel_sizes"], [5, 5, 3, 3, 3, 3])
        self.assertGreater(PARTICLE_CNN_INPUT_FEATURE_DIM, RAW_TOKEN_DIM)
        self.assertIn("rank_fraction", PARTICLE_CNN_RANK_FEATURE_NAMES)

    def test_from_mapping_normalizes_tuple_fields_and_strips_architecture_keys(self):
        config = ParticleCnnReconstructorConfig.from_mapping(
            {
                "reconstructor_architecture": "particle_cnn",
                "architecture": "pcnn",
                "num_blocks": 2,
                "kernel_sizes": [5, 3],
                "dilations": [1, 2],
                "context_dims": [64],
                "decoder_dims": [32],
                "dropout": 0.0,
            }
        )
        self.assertEqual(config.kernel_sizes, (5, 3))
        self.assertEqual(config.dilations, (1, 2))
        self.assertEqual(config.context_mlp_dims, (64,))
        self.assertEqual(config.decoder_dims, (32,))

    def test_rejects_even_kernel_or_length_mismatch(self):
        with self.assertRaises(ValueError):
            ParticleCnnReconstructorConfig(num_blocks=1, kernel_sizes=(4,), dilations=(1,))
        with self.assertRaises(ValueError):
            ParticleCnnReconstructorConfig(num_blocks=2, kernel_sizes=(3,), dilations=(1, 2))
        with self.assertRaises(ValueError):
            ParticleCnnReconstructorConfig(num_blocks=2, kernel_sizes=(3, 3), dilations=(1,))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ParticleCnnReconstructorModuleTests(unittest.TestCase):
    def make_tokens(self):
        tokens = torch.zeros(2, 4, RAW_TOKEN_DIM)
        mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
        tokens[:, :, 0] = torch.tensor([[100.0, 25.0, float("nan"), 999.0], [80.0, 777.0, 10.0, 0.0]])
        tokens[:, :, 1] = torch.tensor([[0.2, -0.3, float("inf"), 9.0], [0.1, 9.0, -0.4, 0.0]])
        tokens[:, :, 2] = torch.tensor([[0.0, 3.5, float("nan"), 1.0], [0.5, 1.0, -4.0, 0.0]])
        tokens[:, :, 3] = torch.tensor([[120.0, 30.0, float("nan"), 999.0], [95.0, 777.0, 12.0, 0.0]])
        tokens[:, :, 4] = torch.tensor([[1.0, -1.0, 5.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        tokens[:, :, 5] = torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
        tokens[:, :, 6] = 1.0 - tokens[:, :, 5]
        tokens[:, :, 10] = torch.tensor([[0.1, -0.2, 100.0, 0.0], [0.0, 100.0, 0.3, 0.0]])
        tokens[:, :, 11] = torch.tensor([[0.01, 0.02, 100.0, 0.0], [0.0, 100.0, 0.03, 0.0]])
        tokens[:, :, 12] = torch.tensor([[0.2, -0.1, 100.0, 0.0], [0.0, 100.0, -0.2, 0.0]])
        tokens[:, :, 13] = torch.tensor([[0.02, 0.04, 100.0, 0.0], [0.0, 100.0, 0.05, 0.0]])
        return tokens, mask

    def test_module_shell_exposes_architecture_and_config(self):
        model = ParticleCnnReconstructor(
            {
                "hidden_channels": 16,
                "num_blocks": 1,
                "kernel_sizes": [3],
                "dilations": [1],
                "context_dim": 32,
                "context_mlp_dims": [32],
                "decoder_dims": [16],
                "dropout": 0.0,
            }
        )
        self.assertEqual(model.reconstructor_architecture, PARTICLE_CNN_RECONSTRUCTOR_ARCHITECTURE)
        self.assertEqual(model.config.hidden_channels, 16)
        self.assertEqual(model.encoder.output_dim, 16)

    def test_rank_features_are_masked_and_finite(self):
        mask = torch.tensor([[True, True, False, True]])
        features = build_rank_features(mask)
        self.assertEqual(tuple(features.shape), (1, 4, len(PARTICLE_CNN_RANK_FEATURE_NAMES)))
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.allclose(features[0, 2], torch.zeros_like(features[0, 2])))
        self.assertGreater(float(features[0, 0, 3]), 0.0)
        self.assertGreater(float(features[0, 3, 0]), float(features[0, 1, 0]))

    def test_particle_cnn_features_are_finite_and_zero_padded(self):
        tokens, mask = self.make_tokens()
        features = build_particle_cnn_features(tokens, mask)
        self.assertEqual(tuple(features.shape), (2, 4, PARTICLE_CNN_INPUT_FEATURE_DIM))
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.allclose(features[0, 2], torch.zeros_like(features[0, 2])))
        self.assertTrue(torch.allclose(features[1, 1], torch.zeros_like(features[1, 1])))
        self.assertFalse(torch.allclose(features[0, 0], features[0, 1]))

    def test_apply_particle_mask_channels_zeroes_padded_conv_positions(self):
        values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        mask = torch.tensor([[True, False, True, False], [False, True, True, False]])
        masked = apply_particle_mask_channels(values, mask)
        self.assertTrue(torch.allclose(masked[0, :, 1], torch.zeros(3)))
        self.assertTrue(torch.allclose(masked[1, :, 0], torch.zeros(3)))
        self.assertTrue(torch.allclose(masked[0, :, 2], values[0, :, 2]))

    def test_masked_pooling_ignores_padded_particles(self):
        values = torch.tensor([[[1.0, 2.0], [1000.0, 1000.0], [3.0, -1.0]]])
        mask = torch.tensor([[True, False, True]])
        self.assertTrue(torch.allclose(masked_sum_pool(values, mask), torch.tensor([[4.0, 1.0]])))
        self.assertTrue(torch.allclose(masked_mean_pool(values, mask), torch.tensor([[2.0, 0.5]])))
        self.assertTrue(torch.allclose(masked_max_pool(values, mask), torch.tensor([[3.0, 2.0]])))

    def test_step2_helpers_validate_shapes(self):
        tokens, mask = self.make_tokens()
        with self.assertRaises(ValueError):
            build_particle_cnn_features(tokens[:, :, :3], mask)
        with self.assertRaises(ValueError):
            apply_particle_mask_channels(torch.zeros(2, 4, 3), mask)
        with self.assertRaises(ValueError):
            masked_sum_pool(torch.zeros(2, 4), mask)

    def test_particle_cnn_block_preserves_shape_and_masks_tail(self):
        torch.manual_seed(7)
        block = ParticleCnnBlock(5, kernel_size=3, dilation=2, dropout=0.0)
        values = torch.randn(2, 5, 6)
        mask = torch.tensor([[True, True, True, False, False, False], [True, False, True, True, False, False]])
        output = block(values, mask)
        self.assertEqual(tuple(output.shape), tuple(values.shape))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(output[0, :, 3:], torch.zeros_like(output[0, :, 3:])))
        self.assertTrue(torch.allclose(output[1, :, 1], torch.zeros_like(output[1, :, 1])))
        with self.assertRaises(ValueError):
            ParticleCnnBlock(5, kernel_size=4)
        with self.assertRaises(ValueError):
            block(torch.randn(2, 4, 6), mask)

    def test_particle_cnn_context_builder_pools_and_zeros_empty_jets(self):
        torch.manual_seed(9)
        builder = ParticleCnnContextBuilder(
            embedding_dim=4,
            context_dim=6,
            context_mlp_dims=(8,),
            summary_dim=2,
            dropout=0.0,
        )
        embeddings = torch.randn(2, 5, 4)
        mask = torch.tensor([[True, False, True, False, False], [False, False, False, False, False]])
        summary = torch.randn(2, 2)
        context, report = builder(embeddings, mask, summary_features=summary)
        self.assertEqual(tuple(context.shape), (2, 6))
        self.assertTrue(torch.isfinite(context).all())
        self.assertTrue(torch.allclose(context[1], torch.zeros_like(context[1])))
        self.assertTrue(torch.allclose(report["valid_count"], torch.tensor([2.0, 0.0])))
        self.assertEqual(tuple(report["pooled_context_input"].shape), (2, 15))
        with self.assertRaises(ValueError):
            builder(embeddings[:, :, :3], mask, summary_features=summary)
        with self.assertRaises(ValueError):
            builder(embeddings, mask, summary_features=torch.randn(2, 3))

    def test_particle_cnn_encoder_outputs_embeddings_context_and_diagnostics(self):
        torch.manual_seed(13)
        tokens, mask = self.make_tokens()
        features = build_particle_cnn_features(tokens, mask)
        encoder = ParticleCnnEncoder(
            input_dim=PARTICLE_CNN_INPUT_FEATURE_DIM,
            hidden_channels=12,
            kernel_sizes=(5, 3),
            dilations=(1, 2),
            context_dim=10,
            context_mlp_dims=(16,),
            dropout=0.0,
        )
        output = encoder(features, mask)
        self.assertIsInstance(output, ParticleCnnEncoderOutput)
        self.assertEqual(tuple(output.particle_embeddings.shape), (2, 4, 12))
        self.assertEqual(tuple(output.jet_context.shape), (2, 10))
        self.assertEqual(tuple(output.rank_features.shape), (2, 4, len(PARTICLE_CNN_RANK_FEATURE_NAMES)))
        self.assertTrue(torch.isfinite(output.particle_embeddings).all())
        self.assertTrue(torch.isfinite(output.jet_context).all())
        self.assertTrue(torch.allclose(output.particle_embeddings[0, 2], torch.zeros_like(output.particle_embeddings[0, 2])))
        self.assertIn("pooled_context_input", output.pooling_report)
        with self.assertRaises(ValueError):
            encoder(features[:, :, :-1], mask)

    def test_particle_cnn_encoder_is_rank_order_sensitive(self):
        torch.manual_seed(17)
        tokens, mask = self.make_tokens()
        features = build_particle_cnn_features(tokens, mask)
        encoder = ParticleCnnEncoder(
            input_dim=PARTICLE_CNN_INPUT_FEATURE_DIM,
            hidden_channels=8,
            kernel_sizes=(3,),
            dilations=(1,),
            context_dim=8,
            context_mlp_dims=(8,),
            dropout=0.0,
        )
        encoder.eval()
        original = encoder(features, mask).particle_embeddings
        permuted_features = features[:, [1, 0, 2, 3], :]
        permuted_mask = mask[:, [1, 0, 2, 3]]
        permuted = encoder(permuted_features, permuted_mask).particle_embeddings
        self.assertFalse(torch.allclose(original[:, :2], permuted[:, :2]))

    def make_reconstructor(self, *, num_extra_candidates=3):
        torch.manual_seed(23)
        return ParticleCnnReconstructor(
            ParticleCnnReconstructorConfig(
                hidden_channels=12,
                num_blocks=2,
                kernel_sizes=(5, 3),
                dilations=(1, 2),
                context_dim=10,
                context_mlp_dims=(14,),
                decoder_dims=(13,),
                dropout=0.0,
                num_extra_candidates=num_extra_candidates,
                max_delta_logpt=0.25,
                max_delta_eta=0.15,
                max_delta_phi=0.15,
                max_delta_loge=0.25,
            )
        )

    def test_cache_order_audit_reports_descending_pt_summary(self):
        tokens = torch.zeros(2, 4, RAW_TOKEN_DIM)
        mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        tokens[:, :, 0] = torch.tensor([[10.0, 9.0, 1.0, 0.0], [5.0, 8.0, 0.0, 0.0]])
        audit = audit_particle_cnn_cache_order(tokens, mask)
        self.assertEqual(audit["ordering_assumption"], PARTICLE_CNN_ORDERING_ASSUMPTION)
        self.assertEqual(audit["checked_adjacent_valid_pairs"], 3)
        self.assertEqual(audit["non_descending_adjacent_valid_pairs"], 1)
        self.assertEqual(audit["jets_with_any_non_descending_adjacent_pair"], 1)
        self.assertAlmostEqual(audit["descending_adjacent_valid_pair_fraction"], 2.0 / 3.0)

    def test_particle_cnn_reconstructor_returns_soft_view_with_parents_and_extras(self):
        model = self.make_reconstructor(num_extra_candidates=3)
        tokens, mask = self.make_tokens()
        labels = torch.tensor([1, 2])
        view = model(tokens, mask, labels=labels, split="model_train")
        self.assertIsInstance(view, SoftReconstructedView)
        self.assertEqual(tuple(view.tokens.shape), (2, 7, RAW_TOKEN_DIM))
        self.assertEqual(tuple(view.mask.shape), (2, 7))
        self.assertEqual(tuple(view.weights.shape), (2, 7))
        self.assertTrue(torch.isfinite(view.tokens).all())
        self.assertTrue(torch.isfinite(view.weights).all())
        self.assertTrue(bool(((view.weights >= 0.0) & (view.weights <= 1.0)).all()))
        self.assertTrue(torch.equal(view.mask[:, :4], view.aux["sanitized_hlt_mask"]))
        self.assertTrue(torch.all(view.mask[:, 4:]))
        self.assertTrue(torch.allclose(view.tokens[0, 2], torch.zeros_like(view.tokens[0, 2])))
        self.assertTrue(torch.allclose(view.weights[0, 2], torch.zeros_like(view.weights[0, 2])))
        self.assertEqual(view.metadata["construction"], "particle_cnn_parents_plus_extras")
        self.assertEqual(view.metadata["reconstructor_architecture"], PARTICLE_CNN_RECONSTRUCTOR_ARCHITECTURE)
        self.assertEqual(view.metadata["ordering_assumption"], PARTICLE_CNN_ORDERING_ASSUMPTION)
        self.assertIn("cache_order_audit", view.metadata)
        self.assertEqual(view.aux["cache_order_assumption"], PARTICLE_CNN_ORDERING_ASSUMPTION)
        self.assertIn("cache_order_audit", view.aux)
        self.assertIn("particle_cnn_features", view.aux)
        self.assertIn("rank_features", view.aux)

    def test_parent_corrections_are_bounded_and_physical(self):
        model = self.make_reconstructor(num_extra_candidates=2)
        tokens, mask = self.make_tokens()
        view = model(tokens, mask)
        delta = view.aux["parent_delta"]
        self.assertLessEqual(float(delta[:, :, 0].abs().max()), model.config.max_delta_logpt + 1.0e-6)
        self.assertLessEqual(float(delta[:, :, 1].abs().max()), model.config.max_delta_eta + 1.0e-6)
        self.assertLessEqual(float(delta[:, :, 2].abs().max()), model.config.max_delta_phi + 1.0e-6)
        self.assertLessEqual(float(delta[:, :, 3].abs().max()), model.config.max_delta_loge + 1.0e-6)
        valid_tokens = view.tokens[view.mask]
        floor = physical_energy_floor(
            valid_tokens[:, 0],
            valid_tokens[:, 1],
            eps=float(model.config.energy_eps),
        )
        self.assertTrue(bool((valid_tokens[:, 0] >= model.config.min_pt).all()))
        self.assertTrue(bool((valid_tokens[:, 3] + 1.0e-6 >= floor).all()))

    def test_particle_cnn_reconstructor_handles_no_extra_candidates(self):
        model = self.make_reconstructor(num_extra_candidates=0)
        tokens, mask = self.make_tokens()
        view = model(tokens, mask)
        self.assertEqual(tuple(view.tokens.shape), (2, 4, RAW_TOKEN_DIM))
        self.assertEqual(view.metadata["n_extra_candidates"], 0)
        self.assertEqual(tuple(view.aux["extra_tokens"].shape), (2, 0, RAW_TOKEN_DIM))
        self.assertEqual(tuple(view.aux["extra_mask"].shape), (2, 0))

    def test_particle_cnn_reconstructor_sanitizes_empty_input_jets(self):
        model = self.make_reconstructor(num_extra_candidates=1)
        tokens = torch.zeros(1, 4, RAW_TOKEN_DIM)
        mask = torch.zeros(1, 4, dtype=torch.bool)
        view = model(tokens, mask)
        self.assertTrue(bool(view.aux["sanitized_hlt_mask"][0, 0]))
        self.assertEqual(view.aux["diagnostics"]["empty_input_jet_count"], 1)
        self.assertEqual(view.metadata["diagnostics"]["cache_order_assumption"], PARTICLE_CNN_ORDERING_ASSUMPTION)


if __name__ == "__main__":
    unittest.main()
