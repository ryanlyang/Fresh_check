from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, require_torch
from jetclass_fresh.jetclass_data import (
    SPLIT_ORDER,
    JetIdentity,
    JetView,
    RAW_TOKEN_DIM,
    SplitManifest,
    manifest_hash,
    save_split_manifest,
)

from teacher_logit_reco.architecture_view_part import (
    ARCHITECTURE_VIEW_PRIMARY_METRIC,
    ARCHITECTURE_VIEW_TRAIN_STEP,
    ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
    ArchitectureViewConfig,
    ArchitectureViewResidualParT,
    ArchitectureViewTaggerTrainConfig,
    architecture_view_label_filter_names_to_indices,
    train_architecture_view_tagger,
)
from teacher_logit_reco.local_compression_part import LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES
from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset


torch = require_torch()


class _SeqFirstEmbed(torch.nn.Module):
    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, embed_dim)

    def forward(self, rows):
        return self.linear(rows).transpose(0, 1).contiguous()


class _FakeParTMod(torch.nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embed = _SeqFirstEmbed(in_dim, embed_dim)
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.head = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, features, v=None, mask=None):
        del v
        rows = features.transpose(1, 2).contiguous()
        embedded = self.embed(rows).transpose(0, 1).contiguous()
        particle_mask = mask.squeeze(1).to(dtype=embedded.dtype)
        pooled = (embedded * particle_mask[:, :, None]).sum(dim=1)
        pooled = pooled / particle_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.head(self.norm(pooled))


class DummyEmbeddingPart(ParticleTransformerHLTClassifier):
    def __init__(self, embed_dim: int = 16, num_classes: int = 2) -> None:
        torch.nn.Module.__init__(self)
        self.config = {"dummy_embedding_part": True, "embed_dim": int(embed_dim), "num_classes": int(num_classes)}
        self.mod = _FakeParTMod(len(LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES), int(embed_dim), int(num_classes))

    def forward(self, points, features, lorentz_vectors, mask):
        del points
        return self.mod(features, v=lorentz_vectors, mask=mask)


def small_config() -> ArchitectureViewConfig:
    return ArchitectureViewConfig(
        view_dim=8,
        hidden_dim=16,
        pn_k=3,
        pn_layers=1,
        pfn_hidden_dim=16,
        pcnn_channels=16,
        pcnn_layers=1,
        fusion_hidden_dim=20,
        part_embed_dim=16,
        dropout=0.0,
        attention_dropout=0.0,
    )


def make_toy_view(split: str, *, n_jets: int = 10) -> JetView:
    tokens = np.zeros((n_jets, 6, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, 6), dtype=bool)
    labels = np.asarray([index % 2 for index in range(n_jets)], dtype=np.int64)
    for jet in range(n_jets):
        valid = 4 + (jet % 2)
        mask[jet, :valid] = True
        for particle in range(valid):
            pt = 12.0 + 0.5 * jet + 1.2 * particle + 1.5 * labels[jet]
            eta = -0.3 + 0.1 * particle
            phi = -math.pi + 0.2 * particle
            tokens[jet, particle, 0] = pt
            tokens[jet, particle, 1] = eta
            tokens[jet, particle, 2] = phi
            tokens[jet, particle, 3] = pt * math.cosh(eta) + 0.25
            tokens[jet, particle, 4] = -1.0 + (particle % 3)
            tokens[jet, particle, 5 + (particle % 5)] = 1.0
            tokens[jet, particle, 10] = 0.02 * particle
            tokens[jet, particle, 11] = 0.04
            tokens[jet, particle, 12] = -0.03 * particle
            tokens[jet, particle, 13] = 0.07
    jet_ids = [JetIdentity(file="toy.root", entry=index, label=int(labels[index])) for index in range(n_jets)]
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata={
            "view": "fixed_hlt",
            "hlt_content_hash": f"toy-{split}",
            "jet_identity_hash": f"toy-identity-{split}",
            "hlt_params": {"strength": 0.6},
        },
    )


def dataset(split: str) -> SubtokenHLTJetDataset:
    return SubtokenHLTJetDataset(
        make_toy_view(split),
        label_filter=(0, 1),
        label_names=("QCD", "Hgg"),
    )


def write_split_manifest(path: Path) -> str:
    manifest = SplitManifest(
        data_dir="toy",
        max_constits=6,
        class_names=["QCD", "Hgg"],
        file_prefix_to_label={"ZJetsToNuNu": 0, "HToGG": 1},
        split_sizes={split: 0 for split in SPLIT_ORDER},
        split_seeds={split: index + 1 for index, split in enumerate(SPLIT_ORDER)},
        file_records=[],
        splits={split: [] for split in SPLIT_ORDER},
        metadata={"test_manifest": True},
    )
    save_split_manifest(manifest, path)
    return manifest_hash(manifest)


def write_checkpoint(path: Path, model: torch.nn.Module, *, split_manifest_hash: str | None = None) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "selection_metric": ARCHITECTURE_VIEW_PRIMARY_METRIC,
        "hlt_degradation_strength": 0.6,
        "label_names": ["QCD", "Hgg"],
        "label_filter": [0, 1],
        "num_classes": 2,
        "model_config": dict(getattr(model, "config", {})),
    }
    if split_manifest_hash is not None:
        payload["split_manifest_hash"] = str(split_manifest_hash)
    torch.save(payload, path)


class ArchitectureViewStep3TrainTests(unittest.TestCase):
    def test_label_filter_names_resolve_in_binary_label_space_when_manifest_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "split_manifest.json"
            manifest.write_text("not-json\n", encoding="utf-8")

            labels = architecture_view_label_filter_names_to_indices(
                ("QCD", "Hgg"),
                manifest_path=manifest,
                label_names=("QCD", "Hgg"),
            )

            self.assertEqual(labels, (0, 1))

    def test_train_architecture_view_tagger_writes_step3_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest = tmp_path / "split_manifest.json"
            manifest_sha = write_split_manifest(manifest)
            checkpoint = tmp_path / "baseline.pt"
            write_checkpoint(checkpoint, DummyEmbeddingPart(), split_manifest_hash=manifest_sha)
            output_dir = tmp_path / "out"
            config = ArchitectureViewTaggerTrainConfig(
                output_dir=str(output_dir),
                manifest_path=str(manifest),
                hlt_cache_dir="unused",
                baseline_checkpoint=str(checkpoint),
                confirm_split_settings=True,
                confirm_final_test=True,
                seed=17,
                batch_size=4,
                eval_batch_size=4,
                epochs=1,
                adapter_lr=1.0e-3,
                part_lr=1.0e-4,
                num_workers=0,
                device="cpu",
                amp=False,
                early_stop_patience=-1,
                max_train_batches=1,
                max_val_batches=1,
                max_stack_val_batches=1,
                max_final_test_batches=1,
                label_names=("QCD", "Hgg"),
                label_filter=(0, 1),
                variant=ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
                view_dim=8,
                hidden_dim=16,
                pn_k=3,
                pn_layers=1,
                pfn_hidden_dim=16,
                pcnn_channels=16,
                pcnn_layers=1,
                fusion_hidden_dim=20,
                part_embed_dim=16,
                dropout=0.0,
                attention_dropout=0.0,
                freeze_part_epochs=0,
            )
            model = ArchitectureViewResidualParT(
                small_config(),
                variant=ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
                part_model=DummyEmbeddingPart(),
            )

            report = train_architecture_view_tagger(
                config,
                model=model,
                train_dataset=dataset("model_train"),
                val_dataset=dataset("model_val"),
                stack_val_dataset=dataset("stack_val"),
                final_test_dataset=dataset("final_test"),
            )

            self.assertEqual(report["experiment_step"], ARCHITECTURE_VIEW_TRAIN_STEP)
            self.assertEqual(report["selection_metric"], ARCHITECTURE_VIEW_PRIMARY_METRIC)
            self.assertTrue(report["final_test_evaluated"])
            self.assertIn("binary_metrics", report["final_test_metrics"])
            self.assertIn("prediction_arrays", report["best_model_val_metrics"])
            self.assertIn("prediction_arrays", report["final_test_metrics"])
            self.assertEqual(report["baseline_checkpoint_selection_metric"], ARCHITECTURE_VIEW_PRIMARY_METRIC)
            self.assertEqual(report["baseline_checkpoint_hlt_degradation_strength"], 0.6)
            self.assertTrue(report["init_logit_diff_vs_baseline"]["allclose_atol_1e_6"])
            self.assertTrue((output_dir / "run_report.json").exists())
            self.assertTrue((output_dir / "training_curves.json").exists())
            self.assertTrue((output_dir / "best_model_val.pt").exists())
            self.assertTrue((output_dir / "last.pt").exists())
            self.assertTrue((output_dir / "diagnostics" / "baseline_load_report.json").exists())
            self.assertTrue((output_dir / "diagnostics" / "init_logit_diff_vs_baseline.json").exists())
            saved = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["label_filter"], [0, 1])
            self.assertTrue(saved["config"]["require_baseline_split_manifest_hash"])
            self.assertIn("delta_l2_loss", saved["best_model_val_metrics"])
            curves = json.loads((output_dir / "training_curves.json").read_text(encoding="utf-8"))
            epoch = curves["epochs"][0]
            self.assertIn("ce_loss", epoch["train"])
            self.assertIn("delta_l2_loss", epoch["train"])
            self.assertIn("diagnostics", epoch["train"])
            self.assertIn("delta_h_norm_mean", epoch["train"]["diagnostics"])


if __name__ == "__main__":
    unittest.main()
