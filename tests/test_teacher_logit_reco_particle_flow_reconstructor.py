import importlib.util
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from teacher_logit_reco.global_transformer import physical_energy_floor
from teacher_logit_reco.particle_flow_reconstructor import (
    PARTICLE_FLOW_FEATURE_NAMES,
    PARTICLE_FLOW_INPUT_FEATURE_DIM,
    PARTICLE_FLOW_SUMMARY_FEATURE_DIM,
    ParticleFlowContextBuilder,
    ParticleFlowEncoder,
    ParticleFlowReconstructor,
    ParticleFlowReconstructorConfig,
    build_particle_flow_reconstructor,
    build_particle_flow_features,
    build_particle_flow_summary_features,
    masked_max_pool,
    masked_mean_pool,
    masked_sum_pool,
)
from teacher_logit_reco.views import SoftReconstructedView

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ParticleFlowFeatureUtilityTests(unittest.TestCase):
    def make_tokens(self):
        tokens = torch.zeros((3, 5, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.tensor(
            [
                [True, False, True, False, True],
                [False, True, False, False, False],
                [False, False, False, False, False],
            ],
            dtype=torch.bool,
        )
        for batch in range(tokens.shape[0]):
            for index in range(tokens.shape[1]):
                pt = 4.0 + 2.0 * batch + float(index)
                eta = -0.6 + 0.2 * index
                phi = -0.7 + 0.3 * index
                tokens[batch, index, 0] = pt
                tokens[batch, index, 1] = eta
                tokens[batch, index, 2] = phi
                tokens[batch, index, 3] = pt * torch.cosh(torch.tensor(eta)) + 0.25
                tokens[batch, index, 4] = -1.0 + 0.5 * index
                tokens[batch, index, 5 + (index % 5)] = 1.0
                tokens[batch, index, 10:14] = torch.tensor([0.2, 0.1, -0.3, 0.2])
        tokens[~mask] = 12345.0
        return tokens, mask

    def feature_index(self, name):
        return PARTICLE_FLOW_FEATURE_NAMES.index(name)

    def test_features_are_finite_shaped_and_masked(self):
        tokens, mask = self.make_tokens()
        bad_padding = tokens.clone()
        bad_padding[~mask] = float("nan")
        bad_padding[0, 3, 0] = float("inf")

        features = build_particle_flow_features(bad_padding, mask)
        self.assertEqual(tuple(features.shape), (3, 5, PARTICLE_FLOW_INPUT_FEATURE_DIM))
        self.assertEqual(PARTICLE_FLOW_INPUT_FEATURE_DIM, len(PARTICLE_FLOW_FEATURE_NAMES))
        self.assertTrue(bool(torch.isfinite(features).all()))
        self.assertTrue(bool((features[~mask] == 0.0).all()))
        self.assertTrue(torch.equal(features[:, :, self.feature_index("valid_mask")], mask.float()))
        self.assertTrue(bool((features[2] == 0.0).all()))

    def test_relative_flow_features_match_manual_values(self):
        tokens, mask = self.make_tokens()
        features = build_particle_flow_features(tokens, mask)

        batch = 0
        particle = 2
        valid = mask[batch]
        sum_pt = tokens[batch, valid, 0].sum()
        sum_energy = tokens[batch, valid, 3].sum()
        expected_pt_fraction = torch.log(tokens[batch, particle, 0] / sum_pt)
        expected_energy_fraction = torch.log(tokens[batch, particle, 3] / sum_energy)

        self.assertAlmostEqual(
            float(features[batch, particle, self.feature_index("log_pt_fraction")]),
            float(expected_pt_fraction),
            places=6,
        )
        self.assertAlmostEqual(
            float(features[batch, particle, self.feature_index("log_energy_fraction")]),
            float(expected_energy_fraction),
            places=6,
        )

    def test_summary_features_are_finite_permutation_invariant_and_padding_safe(self):
        tokens, mask = self.make_tokens()
        summary = build_particle_flow_summary_features(tokens, mask)
        self.assertEqual(tuple(summary.shape), (3, PARTICLE_FLOW_SUMMARY_FEATURE_DIM))
        self.assertTrue(bool(torch.isfinite(summary).all()))
        self.assertTrue(bool((summary[2] == 0.0).all()))

        changed_padding = tokens.clone()
        changed_padding[~mask] = -123456.0
        self.assertTrue(torch.equal(summary, build_particle_flow_summary_features(changed_padding, mask)))

        perm = torch.tensor([2, 0, 4, 1, 3], dtype=torch.long)
        permuted_summary = build_particle_flow_summary_features(tokens[:, perm], mask[:, perm])
        self.assertTrue(torch.allclose(summary, permuted_summary, atol=1.0e-6, rtol=1.0e-6))

    def test_features_do_not_depend_on_padding_values(self):
        tokens, mask = self.make_tokens()
        baseline = build_particle_flow_features(tokens, mask)

        changed_padding = tokens.clone()
        changed_padding[~mask] = -999999.0
        changed_padding[1, 0, :] = float("inf")
        changed = build_particle_flow_features(changed_padding, mask)

        self.assertTrue(torch.equal(baseline, changed))

    def test_masked_pooling_ignores_padding_and_handles_empty_rows(self):
        values = torch.tensor(
            [
                [[1.0, 4.0], [99.0, 99.0], [3.0, -2.0], [88.0, 88.0]],
                [[77.0, 77.0], [2.0, 5.0], [66.0, 66.0], [55.0, 55.0]],
                [[9.0, 9.0], [8.0, 8.0], [7.0, 7.0], [6.0, 6.0]],
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor(
            [
                [True, False, True, False],
                [False, True, False, False],
                [False, False, False, False],
            ],
            dtype=torch.bool,
        )

        expected_sum = torch.tensor([[4.0, 2.0], [2.0, 5.0], [0.0, 0.0]], dtype=torch.float32)
        expected_mean = torch.tensor([[2.0, 1.0], [2.0, 5.0], [0.0, 0.0]], dtype=torch.float32)
        expected_max = torch.tensor([[3.0, 4.0], [2.0, 5.0], [0.0, 0.0]], dtype=torch.float32)

        self.assertTrue(torch.allclose(masked_sum_pool(values, mask), expected_sum))
        self.assertTrue(torch.allclose(masked_mean_pool(values, mask), expected_mean))
        self.assertTrue(torch.allclose(masked_max_pool(values, mask), expected_max))

    def test_zero_particle_axis_is_supported(self):
        values = torch.zeros((2, 0, 3), dtype=torch.float32)
        mask = torch.zeros((2, 0), dtype=torch.bool)
        self.assertEqual(tuple(masked_sum_pool(values, mask).shape), (2, 3))
        self.assertEqual(tuple(masked_mean_pool(values, mask).shape), (2, 3))
        self.assertEqual(tuple(masked_max_pool(values, mask).shape), (2, 3))
        self.assertTrue(bool((masked_sum_pool(values, mask) == 0.0).all()))
        self.assertTrue(bool((masked_mean_pool(values, mask) == 0.0).all()))
        self.assertTrue(bool((masked_max_pool(values, mask) == 0.0).all()))

        tokens = torch.zeros((2, 0, RAW_TOKEN_DIM), dtype=torch.float32)
        features = build_particle_flow_features(tokens, mask)
        self.assertEqual(tuple(features.shape), (2, 0, PARTICLE_FLOW_INPUT_FEATURE_DIM))

    def test_invalid_shapes_raise(self):
        tokens, mask = self.make_tokens()
        with self.assertRaises(ValueError):
            build_particle_flow_features(tokens[:, :, : RAW_TOKEN_DIM - 1], mask)
        with self.assertRaises(ValueError):
            build_particle_flow_features(tokens, mask[:, :3])
        with self.assertRaises(ValueError):
            masked_sum_pool(torch.zeros((2, 3), dtype=torch.float32), torch.zeros((2, 3), dtype=torch.bool))
        with self.assertRaises(ValueError):
            masked_mean_pool(torch.zeros((2, 3, 4), dtype=torch.float32), torch.zeros((2, 2), dtype=torch.bool))
        with self.assertRaises(ValueError):
            masked_max_pool(torch.zeros((2, 3, 4), dtype=torch.float32), torch.zeros((2, 2), dtype=torch.bool))

    def test_context_builder_shapes_reports_and_empty_context(self):
        torch.manual_seed(101)
        embeddings = torch.randn((3, 5, 8), dtype=torch.float32)
        mask = torch.tensor(
            [
                [True, False, True, False, True],
                [False, True, False, False, False],
                [False, False, False, False, False],
            ],
            dtype=torch.bool,
        )
        summary = torch.randn((3, PARTICLE_FLOW_SUMMARY_FEATURE_DIM), dtype=torch.float32)
        builder = ParticleFlowContextBuilder(
            embedding_dim=8,
            context_dim=12,
            context_mlp_dims=(16,),
            dropout=0.0,
        )
        context, report = builder(embeddings, mask, summary_features=summary)
        self.assertEqual(tuple(context.shape), (3, 12))
        self.assertTrue(bool(torch.isfinite(context).all()))
        self.assertTrue(bool((context[2] == 0.0).all()))
        self.assertEqual(tuple(report["sum_pool"].shape), (3, 8))
        self.assertEqual(tuple(report["mean_pool"].shape), (3, 8))
        self.assertEqual(tuple(report["max_pool"].shape), (3, 8))
        self.assertTrue(torch.equal(report["valid_count"], mask.sum(dim=1).float()))

        with self.assertRaises(ValueError):
            builder(embeddings[:, :, :7], mask, summary_features=summary)
        with self.assertRaises(ValueError):
            builder(embeddings, mask, summary_features=summary[:, :3])

    def test_encoder_forward_shape_and_mask_invariance(self):
        torch.manual_seed(103)
        tokens, mask = self.make_tokens()
        features = build_particle_flow_features(tokens, mask)
        summary = build_particle_flow_summary_features(tokens, mask)
        encoder = ParticleFlowEncoder(
            input_dim=PARTICLE_FLOW_INPUT_FEATURE_DIM,
            phi_dims=(12, 10),
            context_dim=9,
            context_mlp_dims=(11,),
            dropout=0.0,
        )
        output = encoder(features, mask, summary_features=summary)
        self.assertEqual(tuple(output.particle_embeddings.shape), (3, 5, 10))
        self.assertEqual(tuple(output.jet_context.shape), (3, 9))
        self.assertTrue(bool(torch.isfinite(output.particle_embeddings).all()))
        self.assertTrue(bool(torch.isfinite(output.jet_context).all()))
        self.assertTrue(bool((output.particle_embeddings[~mask] == 0.0).all()))
        self.assertTrue(bool((output.jet_context[2] == 0.0).all()))

        changed_features = features.clone()
        changed_features[~mask] = 1.0e9
        changed_output = encoder(changed_features, mask, summary_features=summary)
        self.assertTrue(torch.allclose(output.particle_embeddings, changed_output.particle_embeddings))
        self.assertTrue(torch.allclose(output.jet_context, changed_output.jet_context))

    def test_encoder_is_particle_equivariant_and_context_invariant(self):
        torch.manual_seed(107)
        tokens, mask = self.make_tokens()
        features = build_particle_flow_features(tokens, mask)
        summary = build_particle_flow_summary_features(tokens, mask)
        encoder = ParticleFlowEncoder(
            input_dim=PARTICLE_FLOW_INPUT_FEATURE_DIM,
            phi_dims=(16, 12),
            context_dim=10,
            context_mlp_dims=(14,),
            dropout=0.0,
        )
        encoder.eval()

        output = encoder(features, mask, summary_features=summary)
        perm = torch.tensor([2, 0, 4, 1, 3], dtype=torch.long)
        inverse_perm = torch.argsort(perm)
        permuted_tokens = tokens[:, perm]
        permuted_output = encoder(
            build_particle_flow_features(permuted_tokens, mask[:, perm]),
            mask[:, perm],
            summary_features=build_particle_flow_summary_features(permuted_tokens, mask[:, perm]),
        )

        self.assertTrue(
            torch.allclose(
                output.particle_embeddings,
                permuted_output.particle_embeddings[:, inverse_perm],
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        )
        self.assertTrue(torch.allclose(output.jet_context, permuted_output.jet_context, atol=1.0e-6, rtol=1.0e-6))

    def test_encoder_validation_rejects_bad_feature_dimension(self):
        tokens, mask = self.make_tokens()
        features = build_particle_flow_features(tokens, mask)
        encoder = ParticleFlowEncoder(
            input_dim=PARTICLE_FLOW_INPUT_FEATURE_DIM,
            phi_dims=(8,),
            context_dim=6,
            context_mlp_dims=(7,),
            dropout=0.0,
        )
        with self.assertRaises(ValueError):
            encoder(features[:, :, : PARTICLE_FLOW_INPUT_FEATURE_DIM - 1], mask)

    def make_reconstructor(self, *, num_extra_candidates=3):
        torch.manual_seed(109)
        return ParticleFlowReconstructor(
            ParticleFlowReconstructorConfig(
                phi_dims=(16, 12),
                context_dim=10,
                context_mlp_dims=(14,),
                decoder_dims=(13,),
                dropout=0.0,
                num_extra_candidates=num_extra_candidates,
            )
        )

    def test_particle_flow_reconstructor_forward_returns_soft_view(self):
        model = self.make_reconstructor(num_extra_candidates=3)
        tokens, mask = self.make_tokens()
        labels = torch.tensor([1, 4, 8], dtype=torch.long)
        view = model(tokens, mask, labels=labels, split="model_train")

        self.assertIsInstance(view, SoftReconstructedView)
        self.assertEqual(tuple(view.tokens.shape), (3, 8, RAW_TOKEN_DIM))
        self.assertEqual(tuple(view.weights.shape), (3, 8))
        self.assertTrue(torch.equal(view.mask[:, :5], view.aux["sanitized_hlt_mask"]))
        self.assertTrue(bool(view.mask[:, 5:].all()))
        self.assertTrue(bool(torch.isfinite(view.tokens).all()))
        self.assertTrue(bool(torch.isfinite(view.weights).all()))
        self.assertTrue(bool((view.weights >= 0.0).all()))
        self.assertTrue(bool((view.weights <= 1.0).all()))
        self.assertTrue(bool((view.weights[:, :5][~view.aux["sanitized_hlt_mask"]] == 0.0).all()))
        self.assertTrue(bool((view.tokens[:, :5][~view.aux["sanitized_hlt_mask"]] == 0.0).all()))
        self.assertEqual(view.metadata["reconstructor_architecture"], "particle_flow")
        self.assertEqual(view.metadata["construction"], "particle_flow_parents_plus_extras")
        self.assertIn("particle_embeddings", view.aux)
        self.assertIn("jet_context", view.aux)
        self.assertIn("pooling_report", view.aux)

    def test_particle_flow_reconstructor_outputs_are_physical_and_bounded(self):
        model = self.make_reconstructor(num_extra_candidates=2)
        tokens, mask = self.make_tokens()
        view = model(tokens, mask)
        cfg = model.config
        delta = view.aux["parent_delta"]

        self.assertLessEqual(float(delta[:, :, 0].abs().max()), cfg.max_delta_logpt + 1.0e-6)
        self.assertLessEqual(float(delta[:, :, 1].abs().max()), cfg.max_delta_eta + 1.0e-6)
        self.assertLessEqual(float(delta[:, :, 2].abs().max()), cfg.max_delta_phi + 1.0e-6)
        self.assertLessEqual(float(delta[:, :, 3].abs().max()), cfg.max_delta_loge + 1.0e-6)

        valid = view.mask
        pt = view.tokens[:, :, 0][valid]
        eta = view.tokens[:, :, 1][valid]
        phi = view.tokens[:, :, 2][valid]
        energy = view.tokens[:, :, 3][valid]
        self.assertTrue(bool((pt >= cfg.min_pt).all()))
        self.assertTrue(bool((eta.abs() <= cfg.eta_limit + 1.0e-6).all()))
        self.assertTrue(bool((phi >= -torch.pi - 1.0e-6).all()))
        self.assertTrue(bool((phi < torch.pi + 1.0e-6).all()))
        floor = physical_energy_floor(pt, eta, eps=cfg.energy_eps)
        self.assertTrue(bool((energy + 1.0e-5 >= floor).all()))

    def test_particle_flow_reconstructor_handles_empty_input_and_aux(self):
        model = self.make_reconstructor(num_extra_candidates=2)
        tokens = torch.zeros((1, 4, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.zeros((1, 4), dtype=torch.bool)
        view = model(tokens, mask)

        self.assertEqual(view.aux["diagnostics"]["empty_input_jet_count"], 1)
        self.assertTrue(bool(view.mask[0, 0]))
        required_aux_keys = {
            "sanitized_hlt_tokens",
            "sanitized_hlt_mask",
            "particle_flow_features",
            "particle_flow_summary_features",
            "particle_embeddings",
            "jet_context",
            "pooling_report",
            "parent_decoder_input",
            "parent_tokens",
            "parent_delta",
            "parent_weights",
            "extra_tokens",
            "extra_weights",
            "extra_mask",
            "jet_axes",
            "diagnostics",
        }
        self.assertTrue(required_aux_keys.issubset(view.aux.keys()))

    def test_particle_flow_reconstructor_supports_no_extra_candidates(self):
        model = self.make_reconstructor(num_extra_candidates=0)
        tokens, mask = self.make_tokens()
        view = model(tokens, mask)

        self.assertEqual(tuple(view.tokens.shape), (3, 5, RAW_TOKEN_DIM))
        self.assertEqual(tuple(view.aux["extra_tokens"].shape), (3, 0, RAW_TOKEN_DIM))
        self.assertEqual(tuple(view.aux["extra_weights"].shape), (3, 0))
        self.assertEqual(tuple(view.aux["extra_mask"].shape), (3, 0))
        self.assertEqual(view.metadata["n_extra_candidates"], 0)

    def test_build_particle_flow_reconstructor_constructs_full_step4_model(self):
        model = build_particle_flow_reconstructor(
            {
                "phi_dims": [8],
                "context_dim": 6,
                "context_mlp_dims": [7],
                "decoder_dims": [9],
                "num_extra_candidates": 1,
                "dropout": 0.0,
            }
        )
        self.assertIsInstance(model, ParticleFlowReconstructor)
        self.assertTrue(hasattr(model, "parent_head"))
        self.assertTrue(hasattr(model, "extra_head"))


if __name__ == "__main__":
    unittest.main()
