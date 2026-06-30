import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
    LocalGraphResidualExpertV2ReportConfig,
    LocalGraphResidualExpertV2TrainConfig,
    LocalGraphResidualV2BaselineEmbeddingBlock,
    build_local_graph_residual_expert_v2_report,
    save_residual_v2_embedding_block,
    train_local_graph_residual_expert_v2,
)
from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset


def _fake_view(split: str, *, n_jets: int, seed: int = 0):
    rng = np.random.default_rng(6100 + seed)
    tokens = rng.normal(0.0, 0.08, size=(n_jets, 7, 14)).astype(np.float32)
    mask = np.ones((n_jets, 7), dtype=bool)
    mask[:, 5:] = False
    labels = np.asarray(([0, 1] * ((n_jets + 1) // 2))[:n_jets], dtype=np.int64)
    for index, label in enumerate(labels.tolist()):
        valid = int(mask[index].sum())
        tokens[index, :valid, 0] = 20.0 + index + 2.0 * int(label)
        tokens[index, :valid, 1] = -0.3 + 0.05 * np.arange(valid)
        tokens[index, :valid, 2] = 0.2 * int(label) + 0.04 * np.arange(valid)
        tokens[index, :valid, 3] = tokens[index, :valid, 0] * np.cosh(tokens[index, :valid, 1]) + 0.1
        tokens[index, :valid, 5 + int(label)] = 1.0
    return SimpleNamespace(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=[(split, int(index), int(labels[index])) for index in range(n_jets)],
        split=split,
        metadata={
            "split": split,
            "view": "fixed_hlt",
            "hlt_content_hash": f"hlt_hash_{split}",
            "jet_identity_hash": f"jet_hash_{split}",
            "source_manifest_hash": "manifest_hash_shared",
            "hlt_params": {"test_strength": 0.6},
            "seed": 7400 + seed,
        },
    )


def _dataset(split: str, *, n_jets: int, seed: int = 0) -> SubtokenHLTJetDataset:
    dataset = SubtokenHLTJetDataset(
        _fake_view(split, n_jets=n_jets, seed=seed),
        label_filter=(0, 1),
        label_names=("QCD", "Hgg"),
    )
    dataset.metadata["split"] = split
    dataset.metadata["split_manifest_hash"] = "manifest_hash_shared"
    return dataset


def _condition_reference(n_jets: int) -> dict:
    return {
        "source_split": "model_train",
        "feature_names": [
            "z_base",
            "p_base",
            "delta_tau50",
            "abs_delta_tau50",
            "delta_tau30",
            "near_tau50_weight",
        ],
        "tau50": 0.0,
        "tau30": 0.75,
        "near_tau50_scale": 0.25,
        "n_jets": int(n_jets),
        "label_dependent": True,
    }


def _checkpoint_identity() -> dict:
    return {
        "checkpoint_path": "/tmp/hlt_part_baseline/best_model_val.pt",
        "checkpoint_sha256": "sha256_anchor",
        "checkpoint_variant": "hlt_part_baseline",
        "checkpoint_epoch": 3,
        "checkpoint_output_contract": "local_graph_hlt_part_baseline_v1",
        "final_head_name": "mod.fc.0",
        "embedding_source": "final_head_forward_hook",
        "required_embedding_role": LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
        "checkpoint_identity_hash": "identity_hash_shared",
    }


def _block(split: str, dataset: SubtokenHLTJetDataset, *, embedding_dim: int = 8) -> LocalGraphResidualV2BaselineEmbeddingBlock:
    labels = np.asarray(dataset.labels, dtype=np.int64)
    margin = np.where(labels == 1, 0.35, -0.35).astype(np.float32)
    margin += np.linspace(-0.05, 0.05, labels.shape[0], dtype=np.float32)
    logits = np.stack((-0.5 * margin, 0.5 * margin), axis=1).astype(np.float32)
    embedding = np.zeros((labels.shape[0], embedding_dim), dtype=np.float32)
    embedding[:, 0] = margin
    embedding[:, 1] = labels.astype(np.float32)
    if embedding_dim > 2:
        embedding[:, 2:] = np.arange(labels.shape[0], dtype=np.float32).reshape(-1, 1) / 20.0
    reference = _condition_reference(len(dataset))
    condition = np.stack(
        (
            margin,
            1.0 / (1.0 + np.exp(-margin)),
            margin - float(reference["tau50"]),
            np.abs(margin - float(reference["tau50"])),
            margin - float(reference["tau30"]),
            np.exp(-np.abs(margin - float(reference["tau50"])) / float(reference["near_tau50_scale"])),
        ),
        axis=1,
    ).astype(np.float32)
    metadata = {
        "contract": LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
        "label_names": ["QCD", "Hgg"],
        "positive_class_name": "Hgg",
        "positive_class_index": 1,
        "required_embedding_role": LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
        "embedding_dim": int(embedding_dim),
        "checkpoint_identity": _checkpoint_identity(),
        "checkpoint_identity_hash": "identity_hash_shared",
        "condition_reference": reference,
        "hlt_content_hash": f"hlt_hash_{split}",
        "jet_identity_hash": f"jet_hash_{split}",
        "split_manifest_hash": "manifest_hash_shared",
        "dataset": {
            "split": split,
            "hlt_content_hash": f"hlt_hash_{split}",
            "jet_identity_hash": f"jet_hash_{split}",
            "split_manifest_hash": "manifest_hash_shared",
            "n_jets": len(dataset),
        },
    }
    return LocalGraphResidualV2BaselineEmbeddingBlock(
        split=split,
        logits=logits,
        embedding=embedding,
        labels=labels,
        indices=np.arange(len(dataset), dtype=np.int64),
        metadata=metadata,
        condition_features_array=condition,
    )


def _fake_loader_factory(views: dict[str, SimpleNamespace]):
    def _load(_root, split, verify_hash=True):
        return views[str(split)]

    return _load


class LocalGraphResidualV2Step12ReportTest(unittest.TestCase):
    def test_report_evaluates_v2_checkpoints_on_final_test_and_writes_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = __import__("pathlib").Path(tmp)
            datasets = {
                "model_train": _dataset("model_train", n_jets=10, seed=1),
                "model_val": _dataset("model_val", n_jets=8, seed=2),
                "stack_val": _dataset("stack_val", n_jets=8, seed=3),
                "final_test": _dataset("final_test", n_jets=8, seed=4),
            }
            views = {
                split: _fake_view(split, n_jets=len(dataset), seed=10 + index)
                for index, (split, dataset) in enumerate(datasets.items())
            }
            blocks = {split: _block(split, dataset) for split, dataset in datasets.items()}
            cache_dir = root / "baseline_embeddings"
            for block in blocks.values():
                save_residual_v2_embedding_block(
                    block,
                    cache_dir,
                    condition_reference=_condition_reference(len(block.labels)),
                    checkpoint_identity=_checkpoint_identity(),
                    metric_splits=("model_train", "model_val", "stack_val"),
                    overwrite=True,
                )

            expert_dir = root / "experts" / "mode_a"
            train_config = LocalGraphResidualExpertV2TrainConfig(
                output_dir=str(expert_dir),
                hlt_cache_dir="unused_hlt",
                baseline_embedding_cache_dir=str(cache_dir),
                confirm_split_settings=True,
                seed=99,
                epochs=1,
                batch_size=5,
                eval_batch_size=4,
                num_workers=0,
                device="cpu",
                amp=False,
                early_stop_patience=-1,
                baseline_embedding_dim=8,
                max_constits=7,
                k=3,
                local_embed_dim=16,
                local_heads=4,
                local_context_dim=12,
                condition_embed_dim=5,
                residual_hidden_dim=18,
                dropout=0.0,
                attention_dropout=0.0,
                residual_dropout=0.0,
                gamma_learnable=False,
            )
            train_local_graph_residual_expert_v2(
                train_config,
                train_dataset=datasets["model_train"],
                val_dataset=datasets["model_val"],
                baseline_blocks={"model_train": blocks["model_train"], "model_val": blocks["model_val"]},
            )

            report_config = LocalGraphResidualExpertV2ReportConfig(
                output_dir=str(root / "report"),
                hlt_cache_dir="unused_hlt",
                baseline_embedding_cache_dir=str(cache_dir),
                residual_expert_root=str(root / "experts"),
                residual_variants=("mode_a",),
                comparison_split="final_test",
                confirm_final_test=True,
                batch_size=4,
                num_workers=0,
                device="cpu",
                amp=False,
                verify_hlt_params=False,
                max_model_val_jets=8,
                max_stack_val_jets=8,
                max_final_test_jets=8,
            )
            with patch(
                "teacher_logit_reco.local_graph_part.residual_v2_train.load_cached_hlt_view",
                side_effect=_fake_loader_factory(views),
            ):
                report = build_local_graph_residual_expert_v2_report(report_config)

            self.assertEqual(report["output_contract"], LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONTRACT)
            self.assertTrue(report["ok"])
            rows = [
                row
                for row in report["metric_table"]
                if row["split"] == "final_test" and row["source_type"].startswith("v2_residual_fused")
            ]
            self.assertTrue(rows)
            calibration_rows = [
                row
                for row in report["metric_table"]
                if row["split"] == "final_test" and row["source_type"] == "calibration_only_control"
            ]
            self.assertTrue(calibration_rows)
            self.assertIn("calibration_only_control", report)
            self.assertEqual(report["comparison_summary"]["comparison_split"], "final_test")
            self.assertEqual(report["comparison_summary"]["primary_metric"], "fpr_at_signal_eff_0p50")
            self.assertIn("best_v2_source_type", report["comparison_summary"])
            self.assertIn("best_v2_variant", report["comparison_summary"])
            self.assertIn("best_v2_learned_variant", report["comparison_summary"])
            self.assertIn("best_v2_val_shrunk_variant", report["comparison_summary"])
            self.assertIn(
                report["comparison_summary"]["best_v2_source_type"],
                {"v2_residual_fused_learned_gamma", "v2_residual_fused_val_shrunk"},
            )
            self.assertTrue(__import__("pathlib").Path(report["outputs"]["report_json"]).exists())
            self.assertTrue(__import__("pathlib").Path(report["outputs"]["metric_table_csv"]).exists())
            self.assertTrue(__import__("pathlib").Path(report["outputs"]["diagnostics_csv"]).exists())

    def test_report_requires_final_test_confirmation_and_fpr50_metric(self):
        with self.assertRaisesRegex(ValueError, "confirm_final_test"):
            LocalGraphResidualExpertV2ReportConfig(
                output_dir="out",
                hlt_cache_dir="hlt",
                baseline_embedding_cache_dir="emb",
                residual_expert_root="experts",
                comparison_split="final_test",
            )
        with self.assertRaisesRegex(ValueError, "FPR@50"):
            LocalGraphResidualExpertV2ReportConfig(
                output_dir="out",
                hlt_cache_dir="hlt",
                baseline_embedding_cache_dir="emb",
                residual_expert_root="experts",
                comparison_split="model_val",
                primary_metric="accuracy",
            )


if __name__ == "__main__":
    unittest.main()
