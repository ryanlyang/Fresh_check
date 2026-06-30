import unittest

import torch

from teacher_logit_reco.local_graph_part import (
    HLT_PART_EMBEDDING_SOURCE_FINAL_HEAD_HOOK,
    HLTPartEmbeddingAnchor,
    HLTPartEmbeddingAnchorConfig,
    LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_CONTRACT,
    build_hlt_part_embedding_anchor,
)


class TinyExactHLTPart(torch.nn.Module):
    def __init__(self, *, in_dim: int = 3, embed_dim: int = 4, num_classes: int = 2):
        super().__init__()
        self.embed = torch.nn.Linear(in_dim, embed_dim)
        self.final = torch.nn.Linear(embed_dim, num_classes)

    def penultimate(self, tokens, mask):
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return torch.tanh(self.embed(pooled))

    def forward(self, tokens, mask):
        return self.final(self.penultimate(tokens, mask))


class AmbiguousTinyHLTPart(TinyExactHLTPart):
    def __init__(self):
        super().__init__()
        self.aux_final = torch.nn.Linear(4, 2)


class NoFinalHeadTinyPart(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Linear(3, 4)
        self.final = torch.nn.Linear(4, 3)

    def forward(self, tokens, mask):
        del mask
        return self.final(torch.tanh(self.embed(tokens.mean(dim=1))))


class MismatchedLogitTinyPart(TinyExactHLTPart):
    def forward(self, tokens, mask):
        return super().forward(tokens, mask) + 0.25


class LocalGraphResidualV2Step2AnchorTest(unittest.TestCase):
    def _inputs(self):
        tokens = torch.tensor(
            [
                [[1.0, 0.5, -0.5], [2.0, -1.0, 0.25], [0.0, 0.0, 0.0]],
                [[-0.5, 1.5, 0.75], [0.25, -0.25, 0.5], [1.0, 1.0, -1.0]],
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor([[True, True, False], [True, True, True]])
        return tokens, mask

    def test_anchor_returns_true_final_head_input_embedding(self):
        model = TinyExactHLTPart()
        anchor = build_hlt_part_embedding_anchor(
            model,
            config=HLTPartEmbeddingAnchorConfig(final_head_name="final"),
        )
        tokens, mask = self._inputs()

        output = anchor.forward_outputs(tokens, mask)
        expected_embedding = model.penultimate(tokens, mask)
        expected_logits = model(tokens, mask)

        self.assertEqual(anchor.output_contract, LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_CONTRACT)
        self.assertEqual(output.embedding_source, HLT_PART_EMBEDDING_SOURCE_FINAL_HEAD_HOOK)
        self.assertEqual(output.final_head_name, "final")
        self.assertEqual(tuple(output.embedding.shape), tuple(expected_embedding.shape))
        self.assertTrue(torch.allclose(output.embedding, expected_embedding))
        self.assertTrue(torch.allclose(output.logits, expected_logits))
        self.assertTrue(torch.allclose(output.final_head_output, expected_logits))
        self.assertTrue(anchor.anchor_parameters_frozen())
        self.assertFalse(output.embedding.requires_grad)
        self.assertEqual(output.summary()["embedding_shape"], [2, 4])
        self.assertTrue(bool(output.diagnostics["embedding_reproduces_logits"]))

    def test_anchor_auto_finds_unique_two_class_final_head(self):
        model = TinyExactHLTPart()
        anchor = HLTPartEmbeddingAnchor(model)
        tokens, mask = self._inputs()

        output = anchor.forward_outputs(tokens, mask)

        self.assertEqual(output.final_head_name, "final")
        self.assertTrue(torch.allclose(output.logits, model(tokens, mask)))

    def test_ambiguous_final_head_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous final 2-class Linear heads"):
            HLTPartEmbeddingAnchor(AmbiguousTinyHLTPart())

    def test_missing_final_head_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "Could not locate a final 2-class Linear head"):
            HLTPartEmbeddingAnchor(NoFinalHeadTinyPart())

    def test_embedding_must_reproduce_logits(self):
        anchor = HLTPartEmbeddingAnchor(
            MismatchedLogitTinyPart(),
            config=HLTPartEmbeddingAnchorConfig(final_head_name="final"),
        )
        tokens, mask = self._inputs()

        with self.assertRaisesRegex(ValueError, "does not reproduce the HLT ParT logits"):
            anchor.forward_outputs(tokens, mask)


if __name__ == "__main__":
    unittest.main()
