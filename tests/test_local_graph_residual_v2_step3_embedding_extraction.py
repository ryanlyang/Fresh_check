import unittest

import torch

from teacher_logit_reco.local_graph_part import (
    HLTPartEmbeddingAnchor,
    HLTPartEmbeddingAnchorConfig,
    LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
)


class ExactEmbeddingPart(torch.nn.Module):
    def __init__(self, *, in_dim: int = 5, hidden_dim: int = 7):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.LayerNorm(hidden_dim),
        )
        self.classifier = torch.nn.Linear(hidden_dim, 2)

    def embedding(self, tokens, mask):
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.encoder(pooled)

    def forward(self, tokens, mask):
        return self.classifier(self.embedding(tokens, mask))


class ClassifierNotUsedPart(ExactEmbeddingPart):
    def forward(self, tokens, mask):
        embedding = self.embedding(tokens, mask)
        return torch.stack((embedding[:, 0], embedding[:, 1]), dim=1)


class NoTwoClassHeadPart(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(5, 7)
        self.classifier = torch.nn.Linear(7, 4)

    def forward(self, tokens, mask):
        del mask
        return self.classifier(self.encoder(tokens.mean(dim=1)))


class LocalGraphResidualV2Step3EmbeddingExtractionTest(unittest.TestCase):
    def _batch(self):
        generator = torch.Generator().manual_seed(123)
        tokens = torch.randn(4, 6, 5, generator=generator)
        mask = torch.tensor(
            [
                [True, True, True, False, False, False],
                [True, True, True, True, False, False],
                [True, False, False, False, False, False],
                [True, True, True, True, True, True],
            ],
            dtype=torch.bool,
        )
        return tokens, mask

    def test_embedding_forward_matches_normal_forward_logits(self):
        model = ExactEmbeddingPart()
        anchor = HLTPartEmbeddingAnchor(
            model,
            config=HLTPartEmbeddingAnchorConfig(final_head_name="classifier"),
        )
        tokens, mask = self._batch()

        normal_logits = model(tokens, mask)
        output = anchor.forward_outputs(tokens, mask)
        tuple_logits, tuple_embedding = anchor(tokens, mask)

        self.assertTrue(torch.allclose(output.logits, normal_logits))
        self.assertTrue(torch.allclose(tuple_logits, normal_logits))
        self.assertTrue(torch.allclose(tuple_embedding, output.embedding))
        self.assertTrue(torch.allclose(output.final_head_output, normal_logits))
        self.assertEqual(output.embedding.ndim, 2)
        self.assertEqual(output.embedding.shape[0], tokens.shape[0])
        self.assertEqual(output.embedding.shape[1], 7)

    def test_anchor_is_frozen_and_reports_embedding_contract(self):
        anchor = HLTPartEmbeddingAnchor(
            ExactEmbeddingPart(),
            config=HLTPartEmbeddingAnchorConfig(final_head_name="classifier"),
            checkpoint_path="dummy/best_model_val.pt",
        )
        tokens, mask = self._batch()
        output = anchor.forward_outputs(tokens, mask)
        metadata = anchor.metadata()
        summary = output.summary()

        self.assertTrue(anchor.anchor_parameters_frozen())
        self.assertEqual(anchor.trainable_parameter_count(), 0)
        self.assertTrue(all(not parameter.requires_grad for parameter in anchor.model.parameters()))
        self.assertEqual(metadata["output_contract"], LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_CONTRACT)
        self.assertEqual(metadata["required_embedding_role"], LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE)
        self.assertEqual(metadata["final_head_name"], "classifier")
        self.assertEqual(summary["required_embedding_role"], LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE)
        self.assertEqual(summary["embedding_shape"], [4, 7])

    def test_missing_two_class_head_failure_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Could not locate a final 2-class Linear head"):
            HLTPartEmbeddingAnchor(NoTwoClassHeadPart())

    def test_selected_head_must_be_used_by_forward(self):
        anchor = HLTPartEmbeddingAnchor(
            ClassifierNotUsedPart(),
            config=HLTPartEmbeddingAnchorConfig(final_head_name="classifier"),
        )
        tokens, mask = self._batch()

        with self.assertRaisesRegex(RuntimeError, "embedding hook did not fire"):
            anchor.forward_outputs(tokens, mask)

    def test_disallowed_proxy_embedding_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disallowed V2 embedding fallback"):
            HLTPartEmbeddingAnchorConfig(embedding_source="widened_classifier_logits")


if __name__ == "__main__":
    unittest.main()
