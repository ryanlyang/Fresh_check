import tempfile
import unittest

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_RESIDUAL_EXPERT_TRAIN_STEP,
    LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE,
    LocalGraphBaselineLogitBlock,
    LocalGraphResidualExpertConfig,
    LocalGraphResidualExpertTrainConfig,
    baseline_condition_reference_from_block,
    build_local_graph_residual_expert,
    load_local_graph_residual_expert_checkpoint,
    run_local_graph_residual_expert_epoch,
    train_local_graph_residual_expert,
)
from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset, make_subtoken_hlt_loader


_TORCH = require_torch()


class TinyEmbeddingPartBackbone(_TORCH.nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        torch = require_torch()
        self.config = {"architecture": "tiny_step4_embedding_part_backbone", "output_dim": int(output_dim)}
        self.proj = torch.nn.Linear(17 + 4 + 1, int(output_dim))

    def __call__(self, points, features, lorentz_vectors, mask):
        torch = require_torch()
        del points
        particle_mask = mask.float()
        packed = torch.cat([features, lorentz_vectors, particle_mask], dim=1).transpose(1, 2)
        denom = torch.clamp(particle_mask.squeeze(1).sum(dim=1, keepdim=True), min=1.0)
        pooled = (packed * particle_mask.transpose(1, 2)).sum(dim=1) / denom
        return self.proj(pooled)

    def no_weight_decay(self):
        return set()


def make_toy_view(split: str, *, n_jets: int = 10) -> JetView:
    tokens = np.zeros((n_jets, 6, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, 6), dtype=bool)
    labels = np.asarray([index % 2 for index in range(n_jets)], dtype=np.int64)
    for jet in range(n_jets):
        valid = 5 if jet % 3 else 4
        mask[jet, :valid] = True
        for particle in range(valid):
            tokens[jet, particle, 0] = 15.0 + 0.3 * jet + 0.8 * particle + 1.5 * labels[jet]
            tokens[jet, particle, 1] = -0.4 + 0.11 * particle
            tokens[jet, particle, 2] = -0.2 + 0.19 * particle
            tokens[jet, particle, 3] = tokens[jet, particle, 0] * np.cosh(tokens[jet, particle, 1]) + 0.1
            tokens[jet, particle, 4] = -1.0 + (particle % 3)
            tokens[jet, particle, 5 + (particle % 5)] = 1.0
            tokens[jet, particle, 10] = 0.02 * particle
            tokens[jet, particle, 11] = 0.04
            tokens[jet, particle, 12] = -0.03 * particle
            tokens[jet, particle, 13] = 0.08
    jet_ids = [JetIdentity(file=f"{split}.root", entry=index, label=int(labels[index])) for index in range(n_jets)]
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata={
            "view": "fixed_hlt",
            "hlt_content_hash": f"toy-{split}",
            "jet_identity_hash": f"toy-jets-{split}",
            "hlt_params": {"strength": 0.6},
        },
    )


def make_dataset(split: str, *, n_jets: int = 10) -> SubtokenHLTJetDataset:
    return SubtokenHLTJetDataset(make_toy_view(split, n_jets=n_jets), label_filter=(0, 1), label_names=("QCD", "Hgg"))


def make_baseline_block(dataset: SubtokenHLTJetDataset, *, condition_reference=None) -> LocalGraphBaselineLogitBlock:
    labels = np.asarray(dataset.labels, dtype=np.int64)
    margins = np.where(labels == 1, 0.45, -0.45).astype(np.float32)
    if labels.shape[0] >= 4:
        margins[0] = 0.65
        margins[1] = 0.05
        margins[2] = 0.35
        margins[3] = -0.05
    logits = np.stack([-0.5 * margins, 0.5 * margins], axis=1).astype(np.float32)
    block = LocalGraphBaselineLogitBlock(
        split=dataset.split,
        logits=logits,
        labels=labels,
        indices=np.arange(labels.shape[0], dtype=np.int64),
        metadata={
            "source": "toy",
            "checkpoint": "toy_hlt_part_baseline.pt",
            "checkpoint_variant": "hlt_part_baseline",
            "checkpoint_epoch": 1,
            "run_report": "toy_hlt_part_baseline/run_report.json",
            "dataset": dict(dataset.metadata),
        },
    )
    block.metadata["condition_reference"] = dict(
        condition_reference or baseline_condition_reference_from_block(block, source_split="model_train")
    )
    return block


def make_model_config() -> LocalGraphResidualExpertConfig:
    return LocalGraphResidualExpertConfig(
        model_size="tiny",
        max_constits=6,
        local_adapter="point_attention",
        k=2,
        local_embed_dim=16,
        local_heads=2,
        dropout=0.0,
        attention_dropout=0.0,
        residual_gamma_init=0.01,
        backbone_output_dim=8,
        condition_embed_dim=4,
        residual_hidden_dim=12,
        residual_dropout=0.0,
        alpha_initial=0.2,
        alpha_learnable=False,
    )


class LocalGraphResidualExpertStep4Tests(unittest.TestCase):
    def test_epoch_computes_fused_baseline_and_residual_metrics(self):
        torch = require_torch()
        dataset = make_dataset("model_train", n_jets=8)
        block = make_baseline_block(dataset)
        loader = make_subtoken_hlt_loader(dataset, batch_size=4, shuffle=False, num_workers=0, seed=123)
        model_config = make_model_config()
        model = build_local_graph_residual_expert(
            model_config,
            part_model=TinyEmbeddingPartBackbone(output_dim=model_config.backbone_output_dim),
        )

        metrics = run_local_graph_residual_expert_epoch(
            model,
            loader,
            baseline_block=block,
            device=torch.device("cpu"),
            loss_config=LocalGraphResidualExpertTrainConfig(
                output_dir="unused",
                hlt_cache_dir="unused",
                baseline_logit_cache_dir="unused",
                confirm_split_settings=True,
                loss_mode=LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE,
            ).loss_config(block),
            amp=False,
            collect_predictions=True,
            collect_diagnostics=True,
            label_names=("QCD", "Hgg"),
        )

        self.assertIn("fused_metrics", metrics)
        self.assertIn("baseline_metrics", metrics)
        self.assertIn("residual_metrics", metrics)
        self.assertIn("binary_metrics", metrics)
        self.assertIn("diagnostics", metrics)

    def test_train_residual_expert_selects_model_val_checkpoint(self):
        train_dataset = make_dataset("model_train", n_jets=10)
        val_dataset = make_dataset("model_val", n_jets=10)
        train_block = make_baseline_block(train_dataset)
        reference = train_block.condition_reference(require=True)
        val_block = make_baseline_block(val_dataset, condition_reference=reference)
        model_config = make_model_config()
        model = build_local_graph_residual_expert(
            model_config,
            part_model=TinyEmbeddingPartBackbone(output_dim=model_config.backbone_output_dim),
        )

        with tempfile.TemporaryDirectory() as tmp:
            config = LocalGraphResidualExpertTrainConfig(
                output_dir=tmp,
                hlt_cache_dir="unused",
                baseline_logit_cache_dir="unused",
                confirm_split_settings=True,
                seed=123,
                batch_size=4,
                eval_batch_size=4,
                epochs=1,
                num_workers=0,
                device="cpu",
                amp=False,
                early_stop_patience=-1,
                max_train_batches=1,
                max_val_batches=1,
                model_size="tiny",
                max_constits=6,
                local_adapter="point_attention",
                k=2,
                local_embed_dim=16,
                local_heads=2,
                dropout=0.0,
                attention_dropout=0.0,
                residual_gamma_init=0.01,
                backbone_output_dim=8,
                condition_embed_dim=4,
                residual_hidden_dim=12,
                residual_dropout=0.0,
                alpha_initial=0.2,
                alpha_learnable=False,
                loss_mode=LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE,
            )
            report = train_local_graph_residual_expert(
                config,
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                baseline_blocks={"model_train": train_block, "model_val": val_block},
            )

            self.assertEqual(report["experiment_step"], LOCAL_GRAPH_RESIDUAL_EXPERT_TRAIN_STEP)
            self.assertEqual(report["selection_metric"], "fpr_at_signal_eff_0p50")
            self.assertFalse(report["stack_or_final_loaded"])
            self.assertIn("fused_model_val_metrics", report)
            self.assertIn("baseline_model_val_metrics", report)
            self.assertIn("residual_diagnostics_model_val", report)
            self.assertIsNotNone(report["residual_diagnostics_model_val"])
            self.assertIn("false_positive_overlap", report["residual_diagnostics_model_val"])
            self.assertIn("boundary_corrections", report["residual_diagnostics_model_val"])
            loaded_model, payload = load_local_graph_residual_expert_checkpoint(
                report["checkpoint"],
                device=require_torch().device("cpu"),
                part_model=TinyEmbeddingPartBackbone(output_dim=model_config.backbone_output_dim),
            )
            self.assertEqual(payload["experiment_step"], LOCAL_GRAPH_RESIDUAL_EXPERT_TRAIN_STEP)
            self.assertEqual(loaded_model.output_contract, model.output_contract)


if __name__ == "__main__":
    unittest.main()
