from __future__ import annotations

import math
from pathlib import Path
import tempfile

import numpy as np

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, require_torch
from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash
from jetclass_fresh.jetclass_data import (
    LABEL_NAMES,
    RAW_TOKEN_DIM,
    SPLIT_ORDER,
    JetIdentity,
    JetView,
    SplitManifest,
    manifest_hash,
    save_split_manifest,
)

from teacher_logit_reco.architecture_view_part import (
    ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
    ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_PCNN_CONTEXT,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_TRANSFER_VARIANTS,
    ArchitectureViewConfig,
    ArchitectureViewJetDataset,
    ArchitectureViewResidualParT,
    ArchitectureViewTaggerTrainConfig,
    architecture_view_variant_is_baseline_recheck,
    architecture_view_variant_spec,
    load_cached_offline_view,
    save_cached_offline_view,
    train_architecture_view_tagger,
)
from teacher_logit_reco.local_compression_part import LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES


torch = require_torch()


class _BatchFirstEmbed(torch.nn.Module):
    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, embed_dim)

    def forward(self, rows):
        return self.linear(rows)


class _FakeParTMod(torch.nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embed = _BatchFirstEmbed(in_dim, embed_dim)
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.head = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, features, v=None, mask=None):
        del v
        rows = features.transpose(1, 2).contiguous()
        embedded = self.embed(rows)
        particle_mask = mask.squeeze(1).to(dtype=embedded.dtype)
        pooled = (embedded * particle_mask[:, :, None]).sum(dim=1)
        pooled = pooled / particle_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.head(self.norm(pooled))


class DummyTenClassPart(ParticleTransformerHLTClassifier):
    def __init__(self, embed_dim: int = 16) -> None:
        torch.nn.Module.__init__(self)
        self.config = {"dummy_embedding_part": True, "embed_dim": int(embed_dim), "num_classes": 10}
        self.mod = _FakeParTMod(len(LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES), int(embed_dim), 10)

    def no_weight_decay(self):
        return set()

    def forward(self, points, features, lorentz_vectors, mask):
        del points
        return self.mod(features, v=lorentz_vectors, mask=mask)


def _config() -> ArchitectureViewConfig:
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
        num_classes=10,
        dropout=0.0,
        attention_dropout=0.0,
    )


def _toy_view(split: str, *, n_jets: int = 20) -> JetView:
    tokens = np.zeros((n_jets, 6, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, 6), dtype=bool)
    labels = np.asarray([index % 10 for index in range(n_jets)], dtype=np.int64)
    for jet in range(n_jets):
        valid = 4 + (jet % 2)
        mask[jet, :valid] = True
        for particle in range(valid):
            pt = 12.0 + 0.4 * jet + 0.7 * particle + 0.2 * labels[jet]
            eta = -0.25 + 0.08 * particle
            phi = -math.pi + 0.22 * particle
            tokens[jet, particle, 0] = pt
            tokens[jet, particle, 1] = eta
            tokens[jet, particle, 2] = phi
            tokens[jet, particle, 3] = pt * math.cosh(eta) + 0.25
            tokens[jet, particle, 5 + (particle % 5)] = 1.0
    jet_ids = [JetIdentity(file=f"toy_{split}.root", entry=index, label=int(labels[index])) for index in range(n_jets)]
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata={
            "view": "offline",
            "input_source": "offline",
            "source_manifest_hash": "toy-manifest",
        },
    )


def _dataset(split: str) -> ArchitectureViewJetDataset:
    return ArchitectureViewJetDataset(
        _toy_view(split),
        label_filter=ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
        label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
        expected_input_source="offline",
    )


def _write_manifest(path: Path) -> str:
    manifest = SplitManifest(
        data_dir="toy",
        max_constits=6,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label={str(name): index for index, name in enumerate(LABEL_NAMES)},
        split_sizes={split: 0 for split in SPLIT_ORDER},
        split_seeds={split: index + 1 for index, split in enumerate(SPLIT_ORDER)},
        file_records=[],
        splits={split: [] for split in SPLIT_ORDER},
        metadata={"test_manifest": True},
    )
    save_split_manifest(manifest, path)
    return manifest_hash(manifest)


def test_step4_offline_transfer_variants_are_explicitly_offline():
    assert ARCHITECTURE_VIEW_10CLASS_OFFLINE_TRANSFER_VARIANTS == (
        ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE,
        ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER,
        ARCHITECTURE_VIEW_10CLASS_OFFLINE_PCNN_CONTEXT,
    )
    for variant in ARCHITECTURE_VIEW_10CLASS_OFFLINE_TRANSFER_VARIANTS:
        spec = architecture_view_variant_spec(variant)
        assert spec.input_source == "offline"
        assert spec.suite == "av10_offline_transfer"
    assert not architecture_view_variant_is_baseline_recheck(ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE)


def test_step4_offline_cache_round_trip_preserves_hashes_and_source_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "offline_cache"
        view = _toy_view("model_train", n_jets=12)
        metadata = save_cached_offline_view(view, cache_dir)
        loaded = load_cached_offline_view(cache_dir, "model_train")

    assert loaded.metadata["view"] == "offline"
    assert loaded.metadata["input_source"] == "offline"
    assert loaded.metadata["offline_content_hash"] == metadata["offline_content_hash"]
    assert loaded.metadata["jet_identity_hash"] == jet_identity_hash(view.jet_ids)
    assert metadata["offline_content_hash"] == hash_arrays(
        {
            "tokens": view.tokens.astype(np.float32),
            "mask": view.mask.astype(bool),
            "labels": view.labels.astype(np.int64),
            "jet_file_indices": np.zeros((len(view.jet_ids),), dtype=np.int32),
            "jet_entries": np.asarray([identity.entry for identity in view.jet_ids], dtype=np.int64),
        }
    )
    assert np.allclose(loaded.tokens, view.tokens)
    assert np.array_equal(loaded.mask, view.mask)
    assert np.array_equal(loaded.labels, view.labels)


def test_step4_offline_baseline_trains_from_scratch_without_hlt_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest = tmp_path / "split_manifest.json"
        _write_manifest(manifest)
        output_dir = tmp_path / "out"
        config = ArchitectureViewTaggerTrainConfig(
            output_dir=str(output_dir),
            manifest_path=str(manifest),
            hlt_cache_dir="",
            offline_cache_dir=str(tmp_path / "offline_cache"),
            baseline_checkpoint="",
            input_source="offline",
            confirm_split_settings=True,
            confirm_final_test=True,
            seed=23,
            batch_size=5,
            eval_batch_size=5,
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
            label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
            label_filter=ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
            variant=ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE,
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
            freeze_part_epochs=2,
        )
        model = ArchitectureViewResidualParT(
            _config(),
            variant=ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE,
            part_model=DummyTenClassPart(),
        )

        report = train_architecture_view_tagger(
            config,
            model=model,
            train_dataset=_dataset("model_train"),
            val_dataset=_dataset("model_val"),
            stack_val_dataset=_dataset("stack_val"),
            final_test_dataset=_dataset("final_test"),
        )

    assert report["input_source"] == "offline"
    assert report["inference_consumes_hlt_only"] is False
    assert report["baseline_load_report"]["weight_warm_start_skipped"] is True
    assert report["baseline_load_report"]["weight_warm_start_skip_reason"] == "offline_part_baseline_trains_from_scratch"
    assert report["effective_freeze_part_epochs"] == 0
    assert report["epochs_completed"] == 1


def test_step4_offline_adapter_requires_offline_baseline_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "split_manifest.json"
        _write_manifest(manifest)
        try:
            ArchitectureViewTaggerTrainConfig(
                output_dir=str(Path(tmp) / "out"),
                manifest_path=str(manifest),
                hlt_cache_dir="",
                offline_cache_dir=str(Path(tmp) / "offline_cache"),
                baseline_checkpoint="",
                input_source="offline",
                confirm_split_settings=True,
                confirm_final_test=True,
                label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
                label_filter=ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
                variant=ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER,
            )
        except ValueError as exc:
            assert "baseline_checkpoint is required for offline adapter variants" in str(exc)
        else:
            raise AssertionError("offline adapter config should require an offline baseline checkpoint")
