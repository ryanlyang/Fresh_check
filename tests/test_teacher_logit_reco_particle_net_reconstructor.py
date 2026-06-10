import importlib.util
import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from teacher_logit_reco.global_transformer import physical_energy_floor
from teacher_logit_reco.losses import correction_budget_loss
from teacher_logit_reco.particle_net_reconstructor import (
    EdgeConvBlock,
    PARTICLE_NET_INPUT_FEATURE_DIM,
    PARTICLE_NET_KNN_COORD_DIM,
    ParticleNetReconstructor,
    ParticleNetReconstructorConfig,
    ParticleNetEncoder,
    build_particle_net_reconstructor,
    gather_neighbor_features,
    masked_knn_indices,
    particle_net_input_features,
    particle_net_knn_coordinates,
)
from teacher_logit_reco.views import SoftReconstructedView

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


class ParticleNetReconstructorConfigTests(unittest.TestCase):
    def test_config_roundtrip_and_validation(self):
        cfg = ParticleNetReconstructorConfig(edgeconv_dims=(16, 32), k=8, num_extra_candidates=4)
        self.assertEqual(cfg.to_dict()["edgeconv_dims"], [16, 32])
        self.assertEqual(ParticleNetReconstructorConfig.from_mapping(cfg.to_dict()).edgeconv_dims, (16, 32))
        self.assertEqual(ParticleNetReconstructorConfig.from_mapping(cfg).k, 8)
        with self.assertRaises(ValueError):
            ParticleNetReconstructorConfig(edgeconv_dims=())
        with self.assertRaises(ValueError):
            ParticleNetReconstructorConfig(k=0)
        with self.assertRaises(ValueError):
            ParticleNetReconstructorConfig(dropout=1.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ParticleNetGraphUtilityTests(unittest.TestCase):
    def make_tokens(self):
        tokens = torch.zeros((2, 5, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.zeros((2, 5), dtype=torch.bool)
        mask[0, [0, 2, 4]] = True
        mask[1, [1, 3]] = True
        for batch in range(2):
            for index in range(5):
                tokens[batch, index, 0] = 5.0 + batch + index
                tokens[batch, index, 1] = -0.4 + 0.2 * index
                tokens[batch, index, 2] = -0.3 + 0.15 * index
                tokens[batch, index, 3] = tokens[batch, index, 0] * torch.cosh(tokens[batch, index, 1]) + 0.2
                tokens[batch, index, 4] = -1.0 + 0.5 * index
                tokens[batch, index, 5 + (index % 5)] = 1.0
                tokens[batch, index, 10:14] = torch.tensor([0.2, 0.1, -0.3, 0.2])
        tokens[:, ~mask.any(dim=0), 0] = 99.0
        return tokens, mask

    def test_input_features_and_coordinates_are_finite_and_masked(self):
        tokens, mask = self.make_tokens()
        tokens[0, 1, 0] = float("nan")
        features = particle_net_input_features(tokens, mask)
        coords = particle_net_knn_coordinates(tokens, mask)
        self.assertEqual(tuple(features.shape), (2, 5, PARTICLE_NET_INPUT_FEATURE_DIM))
        self.assertEqual(tuple(coords.shape), (2, 5, PARTICLE_NET_KNN_COORD_DIM))
        self.assertTrue(bool(torch.isfinite(features).all()))
        self.assertTrue(bool(torch.isfinite(coords).all()))
        self.assertTrue(bool((features[~mask] == 0.0).all()))
        self.assertTrue(bool((coords[~mask] == 0.0).all()))

    def test_masked_knn_never_selects_padding_when_valid_particles_exist(self):
        tokens, mask = self.make_tokens()
        coords = particle_net_knn_coordinates(tokens, mask)
        indices = masked_knn_indices(coords, mask, k=4)
        self.assertEqual(tuple(indices.shape), (2, 5, 4))
        selected_valid = torch.gather(
            mask[:, None, :].expand(-1, mask.shape[1], -1),
            dim=2,
            index=indices,
        )
        self.assertTrue(bool(selected_valid.all()))

    def test_masked_knn_repeats_valid_neighbors_when_k_exceeds_valid_count(self):
        tokens, mask = self.make_tokens()
        coords = particle_net_knn_coordinates(tokens, mask)
        indices = masked_knn_indices(coords, mask, k=8)
        self.assertEqual(tuple(indices.shape), (2, 5, 8))
        selected_valid = torch.gather(
            mask[:, None, :].expand(-1, mask.shape[1], -1),
            dim=2,
            index=indices,
        )
        self.assertTrue(bool(selected_valid.all()))
        self.assertLessEqual(len(torch.unique(indices[1]).tolist()), int(mask[1].sum().item()))

    def test_masked_knn_handles_empty_jets_and_zero_particle_axis(self):
        coords = torch.zeros((1, 4, PARTICLE_NET_KNN_COORD_DIM), dtype=torch.float32)
        mask = torch.zeros((1, 4), dtype=torch.bool)
        indices = masked_knn_indices(coords, mask, k=3)
        self.assertEqual(tuple(indices.shape), (1, 4, 3))
        self.assertTrue(bool((indices == 0).all()))

        empty_coords = torch.zeros((2, 0, PARTICLE_NET_KNN_COORD_DIM), dtype=torch.float32)
        empty_mask = torch.zeros((2, 0), dtype=torch.bool)
        empty_indices = masked_knn_indices(empty_coords, empty_mask, k=3)
        self.assertEqual(tuple(empty_indices.shape), (2, 0, 3))

    def test_gather_neighbor_features_matches_manual_indexing(self):
        features = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        indices = torch.tensor(
            [
                [[0, 2], [1, 3], [2, 0], [3, 1]],
                [[3, 1], [2, 0], [1, 3], [0, 2]],
            ],
            dtype=torch.long,
        )
        gathered = gather_neighbor_features(features, indices)
        self.assertEqual(tuple(gathered.shape), (2, 4, 2, 3))
        self.assertTrue(torch.equal(gathered[0, 1, 1], features[0, 3]))
        self.assertTrue(torch.equal(gathered[1, 0, 0], features[1, 3]))

    def test_invalid_shapes_and_indices_raise(self):
        tokens, mask = self.make_tokens()
        coords = particle_net_knn_coordinates(tokens, mask)
        with self.assertRaises(ValueError):
            masked_knn_indices(coords, mask, k=0)
        with self.assertRaises(IndexError):
            gather_neighbor_features(torch.zeros((1, 2, 3)), torch.tensor([[[0, 2]]]))

    def test_edgeconv_forward_is_finite_and_zeroes_invalid_particles(self):
        torch.manual_seed(17)
        tokens, mask = self.make_tokens()
        features = particle_net_input_features(tokens, mask)
        coords = particle_net_knn_coordinates(tokens, mask)
        block = EdgeConvBlock(PARTICLE_NET_INPUT_FEATURE_DIM, 24, k=8, dropout=0.0)
        out = block(features, coords, mask)
        self.assertEqual(tuple(out.shape), (2, 5, 24))
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(bool((out[~mask] == 0.0).all()))

    def test_particle_net_encoder_forward_shape_and_empty_jet_behavior(self):
        torch.manual_seed(23)
        tokens, mask = self.make_tokens()
        features = particle_net_input_features(tokens, mask)
        coords = particle_net_knn_coordinates(tokens, mask)
        encoder = ParticleNetEncoder(
            input_dim=PARTICLE_NET_INPUT_FEATURE_DIM,
            hidden_dims=(16, 20, 12),
            k=6,
            dropout=0.0,
        )
        out = encoder(features, coords, mask)
        self.assertEqual(tuple(out.shape), (2, 5, 12))
        self.assertEqual(encoder.output_dim, 12)
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(bool((out[~mask] == 0.0).all()))

        empty_features = torch.zeros((1, 4, PARTICLE_NET_INPUT_FEATURE_DIM), dtype=torch.float32)
        empty_coords = torch.zeros((1, 4, PARTICLE_NET_KNN_COORD_DIM), dtype=torch.float32)
        empty_mask = torch.zeros((1, 4), dtype=torch.bool)
        empty_out = encoder(empty_features, empty_coords, empty_mask)
        self.assertEqual(tuple(empty_out.shape), (1, 4, 12))
        self.assertTrue(bool((empty_out == 0.0).all()))

    def test_edgeconv_gradients_flow_through_mlp_parameters(self):
        torch.manual_seed(31)
        tokens, mask = self.make_tokens()
        features = particle_net_input_features(tokens, mask).detach().requires_grad_(True)
        coords = particle_net_knn_coordinates(tokens, mask)
        block = EdgeConvBlock(PARTICLE_NET_INPUT_FEATURE_DIM, 16, k=4, dropout=0.0)
        out = block(features, coords, mask)
        loss = out[mask].pow(2).mean()
        loss.backward()
        grads = [
            parameter.grad
            for name, parameter in block.named_parameters()
            if "edge_mlp" in name and parameter.requires_grad
        ]
        self.assertTrue(grads)
        self.assertTrue(any(grad is not None and bool(torch.isfinite(grad).all()) for grad in grads))
        self.assertIsNotNone(features.grad)
        self.assertTrue(bool(torch.isfinite(features.grad).all()))

    def test_encoder_validation_rejects_bad_configuration_and_shapes(self):
        with self.assertRaises(ValueError):
            ParticleNetEncoder(hidden_dims=())
        with self.assertRaises(ValueError):
            EdgeConvBlock(PARTICLE_NET_INPUT_FEATURE_DIM, 16, k=0)

        encoder = ParticleNetEncoder(input_dim=PARTICLE_NET_INPUT_FEATURE_DIM, hidden_dims=(8,), k=2)
        tokens, mask = self.make_tokens()
        features = particle_net_input_features(tokens, mask)
        coords = particle_net_knn_coordinates(tokens, mask)
        with self.assertRaises(ValueError):
            encoder(features[:, :, :3], coords, mask)

    def make_reconstructor(self):
        torch.manual_seed(41)
        return ParticleNetReconstructor(
            ParticleNetReconstructorConfig(
                edgeconv_dims=(16, 20),
                k=6,
                dropout=0.0,
                num_extra_candidates=3,
            )
        )

    def test_particle_net_reconstructor_forward_returns_soft_view(self):
        model = self.make_reconstructor()
        tokens, mask = self.make_tokens()
        labels = torch.tensor([1, 8], dtype=torch.long)
        view = model(tokens, mask, labels=labels, split="model_train")
        self.assertIsInstance(view, SoftReconstructedView)
        self.assertEqual(tuple(view.tokens.shape), (2, 8, RAW_TOKEN_DIM))
        self.assertEqual(tuple(view.weights.shape), (2, 8))
        self.assertTrue(torch.equal(view.mask[:, :5], mask))
        self.assertTrue(bool(view.mask[:, 5:].all()))
        self.assertTrue(bool(torch.isfinite(view.tokens).all()))
        self.assertTrue(bool(torch.isfinite(view.weights).all()))
        self.assertTrue(bool((view.weights >= 0.0).all()))
        self.assertTrue(bool((view.weights <= 1.0).all()))
        self.assertTrue(bool((view.weights[:, :5][~mask] == 0.0).all()))
        self.assertTrue(bool((view.tokens[:, :5][~mask] == 0.0).all()))
        self.assertEqual(view.metadata["reconstructor_architecture"], "particle_net")
        self.assertIn("global_context", view.aux)

    def test_particle_net_reconstructor_outputs_are_physical_and_bounded(self):
        model = self.make_reconstructor()
        tokens, mask = self.make_tokens()
        view = model(tokens, mask)
        delta = view.aux["parent_delta"]
        cfg = model.config
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

    def test_particle_net_reconstructor_handles_empty_input_and_budget_loss_aux(self):
        model = self.make_reconstructor()
        tokens = torch.zeros((1, 4, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.zeros((1, 4), dtype=torch.bool)
        view = model(tokens, mask)
        self.assertEqual(view.aux["diagnostics"]["empty_input_jet_count"], 1)
        self.assertTrue(bool(view.mask[0, 0]))
        required_aux_keys = {
            "sanitized_hlt_tokens",
            "sanitized_hlt_mask",
            "parent_tokens",
            "parent_delta",
            "parent_weights",
            "extra_tokens",
            "extra_weights",
            "extra_mask",
            "jet_axes",
            "diagnostics",
        }
        self.assertTrue(required_aux_keys.issubset(set(view.aux)))
        budget, components = correction_budget_loss(view)
        self.assertTrue(bool(torch.isfinite(budget)))
        self.assertEqual(set(components), {"parent_delta", "parent_weight", "extra_weight", "extra_pt_fraction"})
        self.assertTrue(all(bool(torch.isfinite(value)) for value in components.values()))

    def test_build_particle_net_reconstructor_helper(self):
        model = build_particle_net_reconstructor({"edgeconv_dims": [8], "k": 2, "num_extra_candidates": 0})
        self.assertIsInstance(model, ParticleNetReconstructor)
        self.assertEqual(model.config.edgeconv_dims, (8,))


if __name__ == "__main__":
    unittest.main()
