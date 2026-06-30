import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
    LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC,
    LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_TRAIN_STEP,
    LocalGraphResidualExpertV2TrainConfig,
    LocalGraphResidualV2BaselineEmbeddingBlock,
    build_local_graph_residual_expert_v2,
    load_local_graph_residual_expert_v2_checkpoint,
    run_local_graph_residual_expert_v2_epoch,
    train_local_graph_residual_expert_v2,
)
from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset, make_subtoken_hlt_loader


def _fake_view(split: str, *, n_jets: int, seed: int = 0):
    rng = np.random.default_rng(3100 + seed)
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
            "view": "fixed_hlt",
            "hlt_content_hash": f"hlt_hash_{split}",
            "jet_identity_hash": f"jet_hash_{split}",
            "source_manifest_hash": "manifest_hash_shared",
            "hlt_params": {"test_strength": 0.6},
            "seed": 4400 + seed,
        },
    )


def _dataset(split: str, *, n_jets: int, seed: int = 0) -> SubtokenHLTJetDataset:
    dataset = SubtokenHLTJetDataset(
        _fake_view(split, n_jets=n_jets, seed=seed),
        label_filter=(0, 1),
        label_names=("QCD", "Hgg"),
    )
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


class LocalGraphResidualV2Step8TrainingTest(unittest.TestCase):
    def test_epoch_collects_fused_baseline_residual_and_correction_predictions(self):
        train_dataset = _dataset("model_train", n_jets=8, seed=1)
        train_block = _block("model_train", train_dataset)
        config = LocalGraphResidualExpertV2TrainConfig(
            output_dir="unused",
            hlt_cache_dir="unused",
            baseline_embedding_cache_dir="unused",
            confirm_split_settings=True,
            epochs=1,
            batch_size=4,
            eval_batch_size=4,
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
        model = build_local_graph_residual_expert_v2(config.model_config(baseline_embedding_dim=8))
        loader = make_subtoken_hlt_loader(train_dataset, batch_size=4, shuffle=False, num_workers=0, seed=1)

        metrics = run_local_graph_residual_expert_v2_epoch(
            model,
            loader,
            baseline_block=train_block,
            device=__import__("torch").device("cpu"),
            loss_config=config.loss_config(train_block),
            amp=False,
            collect_predictions=True,
            collect_diagnostics=True,
        )

        self.assertEqual(metrics["n_jets"], len(train_dataset))
        self.assertIn("fpr_at_signal_eff_0p50", metrics)
        self.assertIn("baseline_metrics", metrics)
        self.assertIn("residual_metrics", metrics)
        self.assertIn("correction_metrics", metrics)
        arrays = metrics["_prediction_arrays"]
        self.assertEqual(arrays["fused_logits"].shape, (len(train_dataset), 2))
        self.assertEqual(arrays["baseline_logits"].shape, (len(train_dataset), 2))
        self.assertEqual(arrays["correction_logits"].shape, (len(train_dataset), 2))

    def test_train_writes_best_checkpoint_report_and_model_val_predictions(self):
        train_dataset = _dataset("model_train", n_jets=10, seed=2)
        val_dataset = _dataset("model_val", n_jets=8, seed=3)
        blocks = {
            "model_train": _block("model_train", train_dataset),
            "model_val": _block("model_val", val_dataset),
        }
        with tempfile.TemporaryDirectory() as tmp:
            config = LocalGraphResidualExpertV2TrainConfig(
                output_dir=tmp,
                hlt_cache_dir="unused",
                baseline_embedding_cache_dir="unused",
                confirm_split_settings=True,
                seed=88,
                epochs=2,
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

            report = train_local_graph_residual_expert_v2(
                config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                baseline_blocks=blocks,
            )
            model, payload = load_local_graph_residual_expert_v2_checkpoint(
                f"{tmp}/best_model_val.pt",
                device=__import__("torch").device("cpu"),
            )

            self.assertEqual(report["experiment_step"], LOCAL_GRAPH_RESIDUAL_V2_TRAIN_STEP)
            self.assertEqual(report["contract"], LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CONTRACT)
            self.assertEqual(report["selection_metric"], LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC)
            self.assertEqual(report["stack_or_final_loaded"], False)
            self.assertEqual(report["embedding_contract"], LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE)
            self.assertEqual(report["cache_alignment"]["family"]["checkpoint_identity_hash"], "identity_hash_shared")
            self.assertIn("fused_model_val_metrics", report)
            self.assertIn("fused_model_val_learned_gamma_metrics", report)
            self.assertIn("fused_model_val_val_shrunk_metrics", report)
            self.assertEqual(
                report["gamma_shrinkage_model_val"]["shrinkage_applies_to"],
                "learned_correction_delta",
            )
            self.assertEqual(
                report["alpha_shrinkage_model_val"]["selected_gamma"],
                report["gamma_shrinkage_model_val"]["selected_gamma"],
            )
            self.assertIn("baseline_model_val_metrics", report)
            self.assertEqual(payload["embedding_contract"], LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE)
            self.assertEqual(model.config.baseline_embedding_dim, 8)
            with np.load(report["model_val_learned_gamma_predictions"]) as predictions:
                self.assertEqual(predictions["fused_logits"].shape, (len(val_dataset), 2))
                self.assertEqual(predictions["correction_logits"].shape, (len(val_dataset), 2))

    def test_config_rejects_non_protocol_splits_and_accuracy_selection(self):
        with self.assertRaisesRegex(ValueError, "trains only on model_train"):
            LocalGraphResidualExpertV2TrainConfig(
                output_dir="unused",
                hlt_cache_dir="unused",
                baseline_embedding_cache_dir="unused",
                train_split="stack_train",
                confirm_split_settings=True,
            )
        with self.assertRaisesRegex(ValueError, "selects checkpoints"):
            LocalGraphResidualExpertV2TrainConfig(
                output_dir="unused",
                hlt_cache_dir="unused",
                baseline_embedding_cache_dir="unused",
                confirm_split_settings=True,
                selection_metric="accuracy",
            )


if __name__ == "__main__":
    unittest.main()
