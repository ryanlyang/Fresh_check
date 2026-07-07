from pathlib import Path

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch
from teacher_logit_reco.target_denoising_part import (
    TARGET_DENOISING_TAGGER_TRAINING_CONTRACT,
    TargetDenoisingTaggerTrainConfig,
    train_target_denoising_tagger,
)
from teacher_logit_reco.target_denoising_part.tagger_train import _validate_denoiser_checkpoint_compatibility


class TinyRawTokenDataset:
    def __init__(self, *, split: str, n_jets: int = 8, raw_dim: int = 19, num_classes: int = 2):
        rng = np.random.default_rng(1234 + len(split))
        labels = np.arange(n_jets, dtype=np.int64) % int(num_classes)
        tokens = rng.normal(size=(n_jets, 6, raw_dim)).astype(np.float32)
        tokens[:, :, 0] += labels[:, None].astype(np.float32) * 0.4
        mask = np.ones((n_jets, 6), dtype=bool)
        self.tokens = tokens
        self.mask = mask
        self.labels = labels
        self.split = split

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, index):
        return {
            "hlt_tokens": self.tokens[index],
            "hlt_constituent_mask": self.mask[index],
            "labels": int(self.labels[index]),
        }

    def to_metadata(self):
        return {
            "split": self.split,
            "n_jets": int(len(self)),
            "metadata": {
                "hlt_profile": "unit_test",
                "hlt_degradation_strength": 0.0,
                "hlt_content_hash": f"content-{self.split}",
                "source_manifest_hash": "manifest-step6",
                "jet_identity_hash": f"identity-{self.split}",
            },
        }


class TinyTagger:
    def __init__(self, raw_dim: int = 19, num_classes: int = 2):
        torch = require_torch()

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(raw_dim, num_classes)

            def forward(self, tokens, mask, **kwargs):
                del kwargs
                weights = mask.to(dtype=tokens.dtype)[:, :, None]
                pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                return self.linear(pooled)

            def to_config_dict(self):
                return {"model_class": "TinyTagger", "raw_dim": raw_dim, "num_classes": num_classes}

        self.model = _Model()

    def __getattr__(self, name):
        return getattr(self.model, name)


def test_step6_tagger_training_writes_complete_artifacts(tmp_path: Path):
    config = TargetDenoisingTaggerTrainConfig(
        output_dir=str(tmp_path / "tagger"),
        manifest_path=str(tmp_path / "manifest.json.gz"),
        hlt_cache_dir=str(tmp_path / "hlt_cache"),
        variant="hlt_part_baseline",
        num_classes=2,
        epochs=2,
        batch_size=4,
        eval_batch_size=4,
        num_workers=0,
        device="cpu",
        amp=False,
        evaluate_final_test=True,
        confirm_final_test=True,
        strict_hlt_metadata=False,
        require_denoiser_checkpoint=False,
    )
    train_dataset = TinyRawTokenDataset(split="model_train")
    val_dataset = TinyRawTokenDataset(split="model_val")
    final_dataset = TinyRawTokenDataset(split="final_test")
    report = train_target_denoising_tagger(
        config,
        model=TinyTagger().model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        final_test_dataset=final_dataset,
    )

    output_dir = tmp_path / "tagger"
    assert report["output_contract"] == TARGET_DENOISING_TAGGER_TRAINING_CONTRACT
    assert report["variant"] == "hlt_part_baseline"
    assert report["final_test_evaluated"] is True
    assert (output_dir / "best_model_val.pt").exists()
    assert (output_dir / "last.pt").exists()
    assert (output_dir / "run_report.json").exists()
    assert (output_dir / "model_val_report.json").exists()
    assert (output_dir / "final_test_report.json").exists()
    assert (output_dir / "training_curves.json").exists()
    assert (output_dir / "diagnostics" / "epoch_metrics.csv").exists()


def test_step6_rejects_denoiser_checkpoint_without_dataset_metadata(tmp_path: Path):
    config = TargetDenoisingTaggerTrainConfig(
        output_dir=str(tmp_path / "tagger"),
        manifest_path=str(tmp_path / "manifest.json.gz"),
        hlt_cache_dir=str(tmp_path / "hlt_cache"),
        variant="denoiser_features_frozen",
        num_classes=2,
        require_denoiser_checkpoint=True,
    )
    checkpoint_report = {
        "config": {"alignment_mode": "aligned_direct"},
        "model_config": {"use_pair_bias": True, "use_local_kernel": True},
    }
    train_dataset = TinyRawTokenDataset(split="model_train")
    val_dataset = TinyRawTokenDataset(split="model_val")

    try:
        _validate_denoiser_checkpoint_compatibility(
            checkpoint_report,
            config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
    except ValueError as exc:
        assert "dataset metadata is missing" in str(exc)
    else:
        raise AssertionError("expected missing denoiser dataset metadata to be rejected")


def _compatible_checkpoint_dataset(split: str, *, alignment_mode: str = "aligned_direct", content_hash: str | None = None):
    return {
        "split": split,
        "n_jets": 8,
        "alignment_mode": alignment_mode,
        "metadata": {
            "hlt_profile": "unit_test",
            "hlt_degradation_strength": 0.0,
            "hlt_content_hash": content_hash or f"content-{split}",
            "source_manifest_hash": "manifest-step6",
            "jet_identity_hash": f"identity-{split}",
        },
    }


def test_step6_rejects_denoiser_checkpoint_with_stale_hlt_content_hash(tmp_path: Path):
    config = TargetDenoisingTaggerTrainConfig(
        output_dir=str(tmp_path / "tagger"),
        manifest_path=str(tmp_path / "manifest.json.gz"),
        hlt_cache_dir=str(tmp_path / "hlt_cache"),
        variant="denoiser_features_frozen",
        num_classes=2,
    )
    checkpoint_report = {
        "config": {"alignment_mode": "aligned_direct"},
        "model_config": {"use_pair_bias": True, "use_local_kernel": True},
        "train_dataset": _compatible_checkpoint_dataset("model_train", content_hash="old-content"),
        "model_val_dataset": _compatible_checkpoint_dataset("model_val"),
    }

    try:
        _validate_denoiser_checkpoint_compatibility(
            checkpoint_report,
            config,
            train_dataset=TinyRawTokenDataset(split="model_train"),
            val_dataset=TinyRawTokenDataset(split="model_val"),
        )
    except ValueError as exc:
        assert "hlt_content_hash" in str(exc)
    else:
        raise AssertionError("expected stale denoiser HLT content hash to be rejected")


def test_step6_rejects_denoiser_checkpoint_with_wrong_alignment_mode(tmp_path: Path):
    config = TargetDenoisingTaggerTrainConfig(
        output_dir=str(tmp_path / "tagger"),
        manifest_path=str(tmp_path / "manifest.json.gz"),
        hlt_cache_dir=str(tmp_path / "hlt_cache"),
        variant="denoiser_features_frozen",
        num_classes=2,
        alignment_mode="aligned_direct",
    )
    checkpoint_report = {
        "config": {"alignment_mode": "rank_direct"},
        "model_config": {"use_pair_bias": True, "use_local_kernel": True},
        "train_dataset": _compatible_checkpoint_dataset("model_train", alignment_mode="rank_direct"),
        "model_val_dataset": _compatible_checkpoint_dataset("model_val", alignment_mode="rank_direct"),
    }

    try:
        _validate_denoiser_checkpoint_compatibility(
            checkpoint_report,
            config,
            train_dataset=TinyRawTokenDataset(split="model_train"),
            val_dataset=TinyRawTokenDataset(split="model_val"),
        )
    except ValueError as exc:
        assert "alignment_mode" in str(exc)
    else:
        raise AssertionError("expected denoiser alignment mismatch to be rejected")
