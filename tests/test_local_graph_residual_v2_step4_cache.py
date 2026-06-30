import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from teacher_logit_reco.local_graph_part import (
    HLTPartEmbeddingAnchor,
    HLTPartEmbeddingAnchorConfig,
    LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_CONDITION_REFERENCE_SPLIT,
    LocalGraphResidualV2EmbeddingCacheConfig,
    cache_local_graph_residual_v2_baseline_embeddings,
    load_residual_v2_embedding_block,
)


class TinyPart(torch.nn.Module):
    def __init__(self, *, in_dim: int = 14, hidden_dim: int = 5):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.Tanh(),
        )
        self.classifier = torch.nn.Linear(hidden_dim, 2)

    def embedding(self, tokens, mask):
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.encoder(pooled)

    def forward(self, tokens, mask):
        return self.classifier(self.embedding(tokens, mask))


def _fake_view(split: str, *, n_jets: int, offset: int = 0):
    rng = np.random.default_rng(1000 + offset)
    tokens = rng.normal(size=(n_jets, 8, 14)).astype(np.float32)
    mask = np.ones((n_jets, 8), dtype=bool)
    mask[:, 5:] = False
    labels = np.asarray(([0, 1] * ((n_jets + 1) // 2))[:n_jets], dtype=np.int64)
    return SimpleNamespace(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=[(split, int(index), int(labels[index])) for index in range(n_jets)],
        split=split,
        metadata={
            "view": "fixed_hlt",
            "hlt_content_hash": f"hlt_hash_{split}",
            "jet_identity_hash": f"jet_hash_{split}",
            "source_manifest_hash": "manifest_hash_shared",
            "hlt_params": {"test_strength": 0.6},
            "seed": 123 + offset,
        },
    )


class LocalGraphResidualV2Step4CacheTest(unittest.TestCase):
    def test_cache_writes_embeddings_and_suppresses_final_test_metrics(self):
        views = {
            "model_train": _fake_view("model_train", n_jets=8, offset=1),
            "final_test": _fake_view("final_test", n_jets=6, offset=2),
        }

        def fake_load_cached_hlt_view(_cache_dir, split, *, verify_hash=True):
            del verify_hash
            return views[str(split)]

        anchor = HLTPartEmbeddingAnchor(
            TinyPart(),
            config=HLTPartEmbeddingAnchorConfig(final_head_name="classifier"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            config = LocalGraphResidualV2EmbeddingCacheConfig(
                output_dir=tmp,
                hlt_cache_dir="fake_hlt_cache",
                checkpoint_path=None,
                splits=("model_train", "final_test"),
                metric_splits=("model_train",),
                batch_size=3,
                verify_hlt_hash=False,
                verify_hlt_params=False,
            )
            with patch(
                "teacher_logit_reco.local_graph_part.residual_v2_cache.load_cached_hlt_view",
                side_effect=fake_load_cached_hlt_view,
            ):
                report = cache_local_graph_residual_v2_baseline_embeddings(config, anchor=anchor)

            self.assertEqual(report["contract"], LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT)
            self.assertEqual(
                report["condition_reference"]["source_split"],
                LOCAL_GRAPH_RESIDUAL_V2_CONDITION_REFERENCE_SPLIT,
            )
            self.assertIn("fpr_at_signal_eff_0p50", report["manifest_rows"][0])

            train_block = load_residual_v2_embedding_block(tmp, "model_train")
            final_block = load_residual_v2_embedding_block(tmp, "final_test")

            self.assertEqual(train_block.logits.shape, (8, 2))
            self.assertEqual(train_block.embedding.shape, (8, 5))
            self.assertEqual(final_block.logits.shape, (6, 2))
            self.assertEqual(final_block.embedding.shape, (6, 5))
            self.assertEqual(train_block.condition_features_array.shape, (8, 6))
            self.assertEqual(final_block.condition_features_array.shape, (6, 6))
            self.assertEqual(train_block.metadata["contract"], LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT)
            self.assertEqual(final_block.metadata["contract"], LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT)
            self.assertTrue(train_block.metadata["metrics_computed"])
            self.assertFalse(final_block.metadata["metrics_computed"])
            self.assertIsNone(final_block.metadata["metrics"])
            self.assertIsNone(final_block.metadata["operating_points"])
            self.assertEqual(final_block.metadata["condition_reference"]["source_split"], "model_train")
            self.assertEqual(final_block.metadata["hlt_content_hash"], "hlt_hash_final_test")
            self.assertEqual(final_block.metadata["jet_identity_hash"], "jet_hash_final_test")
            self.assertEqual(final_block.metadata["split_manifest_hash"], "manifest_hash_shared")

    def test_final_test_metric_split_is_rejected_for_cache_writer(self):
        with self.assertRaisesRegex(ValueError, "final_test metrics"):
            LocalGraphResidualV2EmbeddingCacheConfig(
                output_dir="unused",
                hlt_cache_dir="unused",
                splits=("model_train", "final_test"),
                metric_splits=("model_train", "final_test"),
                verify_hlt_params=False,
            )


if __name__ == "__main__":
    unittest.main()
