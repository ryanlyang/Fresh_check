import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_POINT_ATTENTION_ADAPTER_CONTRACT,
    LocalKnnConfig,
    PointAttentionLocalAdapter,
    PointAttentionLocalAdapterConfig,
    build_local_knn_graph,
)


class LocalGraphPartStep4PointAttentionTests(unittest.TestCase):
    def make_tokens(self):
        torch = require_torch()
        tokens = torch.zeros((2, 5, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.zeros((2, 5), dtype=torch.bool)
        mask[0, [0, 1, 3]] = True
        mask[1, [0, 2, 4]] = True
        for batch in range(2):
            for index in range(5):
                tokens[batch, index, 0] = 12.0 + batch + 1.5 * index
                tokens[batch, index, 1] = -0.5 + 0.30 * index
                tokens[batch, index, 2] = -0.35 + 0.19 * index
                tokens[batch, index, 3] = tokens[batch, index, 0] * torch.cosh(tokens[batch, index, 1]) + 0.4
                tokens[batch, index, 4] = -1.0 + (index % 3)
                tokens[batch, index, 5 + (index % 5)] = 1.0
                tokens[batch, index, 10] = 0.08 * index
                tokens[batch, index, 11] = 0.04
                tokens[batch, index, 12] = -0.16 * index
                tokens[batch, index, 13] = 0.09
        return tokens, mask

    def test_point_attention_gamma_zero_is_identity_and_masks_attention(self):
        torch = require_torch()
        torch.manual_seed(41)
        tokens, mask = self.make_tokens()
        embeddings = torch.randn((2, 5, 16), dtype=torch.float32)
        adapter = PointAttentionLocalAdapter(
            PointAttentionLocalAdapterConfig(input_dim=16, num_heads=4, k=3, dropout=0.0, attention_dropout=0.0)
        )
        output = adapter(embeddings, tokens, mask)

        self.assertEqual(output.summary()["contract"], LOCAL_GRAPH_POINT_ATTENTION_ADAPTER_CONTRACT)
        self.assertEqual(tuple(output.tokens.shape), tuple(embeddings.shape))
        self.assertEqual(tuple(output.attention_weights.shape), (2, 5, 4, 3))
        self.assertAlmostEqual(float(output.gamma.detach().cpu().item()), 0.0, places=7)
        self.assertTrue(bool(torch.allclose(output.tokens[mask], embeddings[mask], atol=1.0e-6)))
        self.assertTrue(bool((output.tokens[~mask] == 0.0).all()))
        self.assertTrue(bool(torch.isfinite(output.attention_weights).all()))
        invalid_edges = ~output.edge_features.edge_mask
        self.assertTrue(bool((output.attention_weights.permute(0, 1, 3, 2)[invalid_edges] == 0.0).all()))
        sums = output.attention_weights.sum(dim=-1)
        active = output.edge_features.edge_mask.any(dim=-1)[:, :, None].expand_as(sums)
        self.assertTrue(bool(torch.allclose(sums[active], torch.ones_like(sums[active]), atol=1.0e-6)))
        self.assertIn("attention_entropy_mean", output.diagnostics)

    def test_point_attention_uses_precomputed_knn(self):
        torch = require_torch()
        torch.manual_seed(43)
        tokens, mask = self.make_tokens()
        embeddings = torch.randn((2, 5, 12), dtype=torch.float32)
        graph = build_local_knn_graph(tokens, mask, LocalKnnConfig(k=2, include_self=False))
        adapter = PointAttentionLocalAdapter(
            PointAttentionLocalAdapterConfig(input_dim=12, num_heads=3, k=2, dropout=0.0, attention_dropout=0.0)
        )
        output = adapter(embeddings, tokens, mask, knn=graph)

        self.assertTrue(bool(torch.equal(output.edge_features.knn.indices, graph.indices)))
        self.assertTrue(bool(torch.equal(output.edge_features.edge_mask, graph.neighbor_mask)))
        self.assertEqual(tuple(output.attention_weights.shape), (2, 5, 3, 2))

    def test_point_attention_updates_and_receives_gradients_when_gamma_active(self):
        torch = require_torch()
        torch.manual_seed(47)
        tokens, mask = self.make_tokens()
        embeddings = torch.randn((2, 5, 12), dtype=torch.float32, requires_grad=True)
        adapter = PointAttentionLocalAdapter(
            PointAttentionLocalAdapterConfig(
                input_dim=12,
                num_heads=3,
                k=3,
                dropout=0.0,
                attention_dropout=0.0,
                residual_gamma_init=1.0,
            )
        )
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

    def test_invalid_point_attention_config_raises(self):
        with self.assertRaises(ValueError):
            PointAttentionLocalAdapterConfig(input_dim=10, num_heads=4)
        with self.assertRaises(ValueError):
            PointAttentionLocalAdapterConfig(input_dim=12, num_heads=0)
        with self.assertRaises(ValueError):
            PointAttentionLocalAdapterConfig(input_dim=12, k=0)


if __name__ == "__main__":
    unittest.main()
