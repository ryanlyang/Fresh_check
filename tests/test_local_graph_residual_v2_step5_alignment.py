import unittest

import numpy as np

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
    LocalGraphResidualV2BaselineEmbeddingBlock,
    residual_v2_checkpoint_identity,
    verify_residual_v2_embedding_block_alignment,
    verify_residual_v2_embedding_cache_family,
)


def _condition_reference(**overrides):
    payload = {
        "source_split": "model_train",
        "feature_names": [
            "z_base",
            "p_base",
            "delta_tau50",
            "abs_delta_tau50",
            "delta_tau30",
            "near_tau50_weight",
        ],
        "tau50": 0.5,
        "tau30": 1.0,
        "near_tau50_scale": 0.25,
        "n_jets": 6,
        "label_dependent": True,
    }
    payload.update(overrides)
    return payload


def _checkpoint_identity(**overrides):
    payload = {
        "checkpoint_path": "/tmp/best_model_val.pt",
        "checkpoint_sha256": "abc123",
        "checkpoint_variant": "hlt_part_baseline",
        "checkpoint_epoch": 4,
        "checkpoint_output_contract": "local_graph_hlt_part_baseline_v1",
        "final_head_name": "mod.fc.0",
        "embedding_source": "final_head_forward_hook",
        "required_embedding_role": LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
        "checkpoint_identity_hash": "identity_hash_1",
    }
    payload.update(overrides)
    return payload


def _block(
    split: str = "model_val",
    *,
    n: int = 6,
    embedding_dim: int = 5,
    labels: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    metadata_overrides: dict | None = None,
    checkpoint_identity: dict | None = None,
    condition_reference: dict | None = None,
):
    labels = np.asarray(labels if labels is not None else ([0, 1] * ((n + 1) // 2))[:n], dtype=np.int64)
    indices = np.asarray(indices if indices is not None else np.arange(n, dtype=np.int64), dtype=np.int64)
    logits = np.stack((-np.linspace(-1.0, 1.0, n), np.linspace(-1.0, 1.0, n)), axis=1).astype(np.float32)
    embedding = np.arange(n * embedding_dim, dtype=np.float32).reshape(n, embedding_dim) / 10.0
    identity = checkpoint_identity or _checkpoint_identity()
    reference = condition_reference or _condition_reference()
    metadata = {
        "contract": LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
        "label_names": ["QCD", "Hgg"],
        "positive_class_name": "Hgg",
        "positive_class_index": 1,
        "required_embedding_role": LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
        "embedding_dim": embedding_dim,
        "checkpoint_identity": identity,
        "checkpoint_identity_hash": identity["checkpoint_identity_hash"],
        "condition_reference": reference,
        "hlt_content_hash": f"hlt_hash_{split}",
        "jet_identity_hash": f"jet_hash_{split}",
        "split_manifest_hash": "manifest_hash",
        "dataset": {
            "split": split,
            "hlt_content_hash": f"hlt_hash_{split}",
            "jet_identity_hash": f"jet_hash_{split}",
            "split_manifest_hash": "manifest_hash",
            "n_jets": n,
        },
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    return LocalGraphResidualV2BaselineEmbeddingBlock(
        split=split,
        logits=logits,
        embedding=embedding,
        labels=labels,
        indices=indices,
        metadata=metadata,
        condition_features_array=np.ones((n, 6), dtype=np.float32),
    )


class LocalGraphResidualV2Step5AlignmentTest(unittest.TestCase):
    def test_block_alignment_accepts_exact_dataset_and_supports_index_lookup(self):
        block = _block()
        dataset_metadata = {
            "split": "model_val",
            "hlt_content_hash": "hlt_hash_model_val",
            "jet_identity_hash": "jet_hash_model_val",
            "source_manifest_hash": "manifest_hash",
            "n_jets": 6,
        }

        report = verify_residual_v2_embedding_block_alignment(
            block,
            dataset_metadata,
            split="model_val",
            dataset_length=6,
            expected_indices=np.arange(6, dtype=np.int64),
            expected_labels=block.labels,
            expected_checkpoint_identity=residual_v2_checkpoint_identity(block),
            expected_condition_reference=block.condition_reference(require=True),
            expected_embedding_dim=5,
        )
        arrays = block.arrays_for_indices(np.asarray([4, 1, 3], dtype=np.int64))

        self.assertTrue(report["ok"])
        self.assertEqual(report["embedding_dim"], 5)
        np.testing.assert_array_equal(arrays["indices"], np.asarray([4, 1, 3], dtype=np.int64))
        self.assertEqual(arrays["embedding"].shape, (3, 5))
        self.assertEqual(arrays["condition_features"].shape, (3, 6))

    def test_block_alignment_rejects_stale_hlt_hash(self):
        block = _block()

        with self.assertRaisesRegex(ValueError, "hlt_content_hash mismatch"):
            verify_residual_v2_embedding_block_alignment(
                block,
                {
                    "split": "model_val",
                    "hlt_content_hash": "other_hash",
                    "jet_identity_hash": "jet_hash_model_val",
                    "source_manifest_hash": "manifest_hash",
                },
                split="model_val",
            )

    def test_block_alignment_rejects_indices_and_labels_that_only_match_length(self):
        block = _block()
        dataset_metadata = {
            "split": "model_val",
            "hlt_content_hash": "hlt_hash_model_val",
            "jet_identity_hash": "jet_hash_model_val",
            "source_manifest_hash": "manifest_hash",
        }

        with self.assertRaisesRegex(ValueError, "split indices/order mismatch"):
            verify_residual_v2_embedding_block_alignment(
                block,
                dataset_metadata,
                split="model_val",
                expected_indices=np.asarray([0, 2, 1, 3, 4, 5], dtype=np.int64),
                expected_labels=block.labels,
            )
        with self.assertRaisesRegex(ValueError, "labels mismatch"):
            verify_residual_v2_embedding_block_alignment(
                block,
                dataset_metadata,
                split="model_val",
                expected_indices=block.indices,
                expected_labels=1 - block.labels,
            )

    def test_cache_family_rejects_checkpoint_embedding_and_condition_mismatches(self):
        block_a = _block("model_train")
        block_b = _block("model_val")

        self.assertTrue(verify_residual_v2_embedding_cache_family([block_a, block_b])["ok"])

        bad_checkpoint = _block(
            "stack_val",
            checkpoint_identity=_checkpoint_identity(checkpoint_identity_hash="different_identity"),
        )
        with self.assertRaisesRegex(ValueError, "checkpoint_identity_hash differs"):
            verify_residual_v2_embedding_cache_family([block_a, bad_checkpoint])

        bad_embedding = _block("stack_val", embedding_dim=4)
        with self.assertRaisesRegex(ValueError, "embedding_dim differs"):
            verify_residual_v2_embedding_cache_family([block_a, bad_embedding])

        bad_reference = _block("stack_val", condition_reference=_condition_reference(tau50=9.0))
        with self.assertRaisesRegex(ValueError, "condition_reference differs"):
            verify_residual_v2_embedding_cache_family([block_a, bad_reference])

    def test_condition_reference_must_come_from_model_train(self):
        block = _block(condition_reference=_condition_reference(source_split="final_test"))

        with self.assertRaisesRegex(ValueError, "source_split must be 'model_train'"):
            block.condition_reference(require=True)


if __name__ == "__main__":
    unittest.main()
