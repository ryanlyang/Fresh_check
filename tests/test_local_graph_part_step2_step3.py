import math
import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_EDGECONV_ADAPTER_CONTRACT,
    LOCAL_GRAPH_EDGE_FEATURE_DIM,
    LOCAL_GRAPH_EDGE_FEATURE_NAMES,
    LOCAL_GRAPH_KNN_CONTRACT,
    EdgeConvLocalAdapter,
    EdgeConvLocalAdapterConfig,
    LocalEdgeFeatureBuilder,
    LocalEdgeFeatureConfig,
    LocalKnnConfig,
    build_local_knn_graph,
    gather_local_neighbors,
    local_eta_phi_coordinates,
    pairwise_eta_phi_distance,
    wrap_local_delta_phi,
)


class LocalGraphPartStep2Step3Tests(unittest.TestCase):
    def make_tokens(self):
        torch = require_torch()
        tokens = torch.zeros((2, 5, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.zeros((2, 5), dtype=torch.bool)
        mask[0, [0, 1, 3]] = True
        mask[1, [0, 2, 4]] = True
        for batch in range(2):
            for index in range(5):
                tokens[batch, index, 0] = 10.0 + batch + 2.0 * index
                tokens[batch, index, 1] = -0.4 + 0.25 * index
                tokens[batch, index, 2] = -0.2 + 0.17 * index
                tokens[batch, index, 3] = tokens[batch, index, 0] * torch.cosh(tokens[batch, index, 1]) + 0.3
                tokens[batch, index, 4] = -1.0 + (index % 3)
                tokens[batch, index, 5 + (index % 5)] = 1.0
                tokens[batch, index, 10] = 0.1 * index
                tokens[batch, index, 11] = 0.05
                tokens[batch, index, 12] = -0.2 * index
                tokens[batch, index, 13] = 0.10
        return tokens, mask

    def test_eta_phi_coordinates_are_finite_and_masked(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        tokens[0, 2, 1] = float("nan")
        coords = local_eta_phi_coordinates(tokens, mask)

        self.assertEqual(tuple(coords.shape), (2, 5, 2))
        self.assertTrue(bool(torch.isfinite(coords).all()))
        self.assertTrue(bool((coords[~mask] == 0.0).all()))

    def test_phi_wraparound_distance(self):
        torch = require_torch()
        coords = torch.tensor([[[0.0, math.pi - 0.02], [0.0, -math.pi + 0.03]]], dtype=torch.float32)
        distance = pairwise_eta_phi_distance(coords)
        delta = wrap_local_delta_phi(coords[0, 0, 1] - coords[0, 1, 1])

        self.assertLess(float(delta.abs().item()), 0.06)
        self.assertAlmostEqual(float(distance[0, 0, 1].item()), float(delta.abs().item()), places=6)

    def test_knn_never_selects_padding_and_marks_underfilled_slots_invalid(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        graph = build_local_knn_graph(tokens, mask, LocalKnnConfig(k=6))
        diagnostics = graph.diagnostics()
        summary = graph.summary()

        self.assertEqual(tuple(graph.indices.shape), (2, 5, 6))
        self.assertEqual(graph.summary()["contract"], LOCAL_GRAPH_KNN_CONTRACT)
        selected_valid = torch.gather(mask[:, None, :].expand(-1, mask.shape[1], -1), dim=2, index=graph.indices)
        self.assertTrue(bool((selected_valid[graph.neighbor_mask]).all()))
        for batch in range(mask.shape[0]):
            expected_valid_neighbors = int(mask[batch].sum().item())
            for particle in range(mask.shape[1]):
                if bool(mask[batch, particle]):
                    self.assertEqual(int(graph.neighbor_mask[batch, particle].sum().item()), expected_valid_neighbors)
        self.assertTrue(bool((graph.neighbor_mask[~mask] == 0).all()))
        self.assertTrue(bool((graph.distances[~graph.neighbor_mask] == 0.0).all()))
        self.assertGreater(float(diagnostics["underfilled_particle_fraction"].detach().cpu().item()), 0.0)
        self.assertGreater(float(diagnostics["masked_placeholder_fraction"].detach().cpu().item()), 0.0)
        self.assertGreater(summary["masked_placeholder_fraction"], 0.0)

    def test_knn_can_exclude_self_when_other_neighbors_exist(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        graph = build_local_knn_graph(tokens, mask, LocalKnnConfig(k=2, include_self=False))
        center = torch.arange(mask.shape[1], dtype=graph.indices.dtype)[None, :, None]
        self_edges = graph.indices == center

        self.assertFalse(bool((self_edges & graph.neighbor_mask).any()))

    def test_gather_local_neighbors_matches_manual_indexing(self):
        torch = require_torch()
        features = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        indices = torch.tensor(
            [
                [[0, 2], [1, 3], [2, 0], [3, 1]],
                [[3, 1], [2, 0], [1, 3], [0, 2]],
            ],
            dtype=torch.long,
        )
        gathered = gather_local_neighbors(features, indices)

        self.assertEqual(tuple(gathered.shape), (2, 4, 2, 3))
        self.assertTrue(bool(torch.equal(gathered[0, 1, 1], features[0, 3])))
        self.assertTrue(bool(torch.equal(gathered[1, 0, 0], features[1, 3])))

    def test_edge_features_are_finite_masked_and_physics_aware(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        tokens[0, 0, 2] = math.pi - 0.02
        tokens[0, 1, 2] = -math.pi + 0.03
        graph = build_local_knn_graph(tokens, mask, LocalKnnConfig(k=3))
        output = LocalEdgeFeatureBuilder(LocalEdgeFeatureConfig(knn=LocalKnnConfig(k=3)))(tokens, mask, graph)

        self.assertEqual(tuple(output.edge_features.shape), (2, 5, 3, LOCAL_GRAPH_EDGE_FEATURE_DIM))
        self.assertEqual(len(LOCAL_GRAPH_EDGE_FEATURE_NAMES), LOCAL_GRAPH_EDGE_FEATURE_DIM)
        self.assertTrue(bool(torch.isfinite(output.edge_features).all()))
        self.assertTrue(bool((output.edge_features[~output.edge_mask] == 0.0).all()))
        self.assertIn("delta_phi", output.feature_names)
        self.assertIn("log_pair_mass", output.feature_names)
        self.assertIn("relative_kt", output.feature_names)
        self.assertIn("neighbor_pt_fraction", output.feature_names)

        names = list(output.feature_names)
        delta_phi_index = names.index("delta_phi")
        expected_delta = wrap_local_delta_phi(tokens[0, 0, 2] - tokens[0, 1, 2])
        edge_to_one = output.knn.indices[0, 0] == 1
        self.assertTrue(bool(edge_to_one.any()))
        selected_edge = int(edge_to_one.nonzero(as_tuple=False)[0, 0].item())
        self.assertAlmostEqual(
            float(output.edge_features[0, 0, selected_edge, delta_phi_index].detach().cpu().item()),
            float(expected_delta.detach().cpu().item()),
            places=6,
        )

    def test_edgeconv_adapter_gamma_zero_is_identity_on_valid_particles(self):
        torch = require_torch()
        torch.manual_seed(11)
        tokens, mask = self.make_tokens()
        embeddings = torch.randn((2, 5, 16), dtype=torch.float32)
        adapter = EdgeConvLocalAdapter(EdgeConvLocalAdapterConfig(input_dim=16, k=3, dropout=0.0))
        output = adapter(embeddings, tokens, mask)

        self.assertEqual(tuple(output.tokens.shape), tuple(embeddings.shape))
        self.assertEqual(output.summary()["contract"], LOCAL_GRAPH_EDGECONV_ADAPTER_CONTRACT)
        self.assertAlmostEqual(float(output.gamma.detach().cpu().item()), 0.0, places=7)
        self.assertTrue(bool(torch.allclose(output.tokens[mask], embeddings[mask], atol=1.0e-6)))
        self.assertTrue(bool((output.tokens[~mask] == 0.0).all()))
        self.assertTrue(bool(torch.isfinite(output.local_update).all()))
        self.assertTrue(bool((output.local_update[~mask] == 0.0).all()))
        self.assertIn("mean_valid_neighbors", output.diagnostics)

    def test_edgeconv_adapter_uses_precomputed_knn(self):
        torch = require_torch()
        torch.manual_seed(13)
        tokens, mask = self.make_tokens()
        embeddings = torch.randn((2, 5, 10), dtype=torch.float32)
        graph = build_local_knn_graph(tokens, mask, LocalKnnConfig(k=2, include_self=False))
        adapter = EdgeConvLocalAdapter(EdgeConvLocalAdapterConfig(input_dim=10, k=2, dropout=0.0))
        output = adapter(embeddings, tokens, mask, knn=graph)

        self.assertTrue(bool(torch.equal(output.edge_features.knn.indices, graph.indices)))
        self.assertTrue(bool(torch.equal(output.edge_features.edge_mask, graph.neighbor_mask)))
        self.assertEqual(tuple(output.edge_features.edge_features.shape[:3]), tuple(graph.indices.shape))

    def test_edgeconv_adapter_updates_and_receives_gradients_when_gamma_active(self):
        torch = require_torch()
        torch.manual_seed(17)
        tokens, mask = self.make_tokens()
        embeddings = torch.randn((2, 5, 12), dtype=torch.float32, requires_grad=True)
        adapter = EdgeConvLocalAdapter(EdgeConvLocalAdapterConfig(input_dim=12, k=3, dropout=0.0, residual_gamma_init=1.0))
        output = adapter(embeddings, tokens, mask)
        delta = (output.tokens[mask] - embeddings[mask]).abs().sum()
        loss = output.tokens[mask].square().mean()
        loss.backward()

        self.assertGreater(float(delta.detach().item()), 1.0e-6)
        self.assertIsNotNone(adapter.gamma.grad)
        self.assertGreater(float(adapter.gamma.grad.detach().abs().item()), 0.0)
        self.assertIsNotNone(embeddings.grad)
        self.assertTrue(bool(torch.isfinite(embeddings.grad).all()))
        grad_sum = sum(
            float(param.grad.detach().abs().sum().item())
            for param in adapter.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_sum, 0.0)

    def test_invalid_configuration_and_shapes_raise(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        with self.assertRaises(ValueError):
            LocalKnnConfig(k=0)
        with self.assertRaises(ValueError):
            EdgeConvLocalAdapterConfig(input_dim=0)
        with self.assertRaises(ValueError):
            build_local_knn_graph(tokens[:, :, :2], mask, LocalKnnConfig(raw_feature_dim=RAW_TOKEN_DIM))
        adapter = EdgeConvLocalAdapter(EdgeConvLocalAdapterConfig(input_dim=8, k=2))
        with self.assertRaises(ValueError):
            adapter(torch.zeros((2, 5, 7)), tokens, mask)


if __name__ == "__main__":
    unittest.main()
