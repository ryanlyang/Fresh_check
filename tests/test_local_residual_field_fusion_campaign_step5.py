from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jetclass_fresh.fusion import PredictionBlock, load_prediction_block, save_prediction_block
from scripts.cache_local_residual_field_fusion_features import parse_args
from teacher_logit_reco.local_particle_residual_field import (
    FUSION_DEVELOPMENT_SPLITS,
    LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_CONTRACT,
    LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_SOURCES_CONTRACT,
    FusionFeatureCacheConfig,
    PreClassifierEmbeddingCapture,
    cache_local_residual_field_fusion_features,
    sha256_file,
    stable_fusion_json_hash,
)
from teacher_logit_reco.local_particle_residual_field import fusion_features as feature_module
from teacher_logit_reco.local_particle_residual_field import fusion_sources as source_module
from tests.test_local_residual_field_fusion_campaign_step3 import _OracleFreeModel, build_source_fixture


REPO_ROOT = Path(__file__).resolve().parents[1]


class _ExactPart(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(10, 10, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.eye(10))

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors, mask
        return self.classifier(features[:, :, 0])


class _ExactMember(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.part_model = _ExactPart()

    def forward(self, points, features, lorentz_vectors, mask, **kwargs):
        del kwargs
        return SimpleNamespace(logits=self.part_model(points, features, lorentz_vectors, mask))


def _prediction_registry(tmp_path: Path) -> tuple[dict, Path, Path]:
    audit_config, payload = build_source_fixture(tmp_path)
    audit = source_module.audit_fusion_source_artifacts(
        audit_config,
        model_loader=lambda path, device="cpu": (_OracleFreeModel(), payload),
    )
    seed_checkpoint = tmp_path / "A0_seed1" / "best_model_val.pt"
    seed_checkpoint.parent.mkdir(parents=True)
    seed_checkpoint.write_bytes(b"seed-checkpoint")
    seed_hash = sha256_file(seed_checkpoint)
    seed_root = tmp_path / "seed_predictions"
    seed_rows = {}
    for split in FUSION_DEVELOPMENT_SPLITS:
        source = load_prediction_block(audit_config.a0_prediction_dir, "A0", split)
        metadata = dict(source.metadata)
        metadata["checkpoint_hash"] = seed_hash
        save_prediction_block(
            PredictionBlock(
                model_name="A0_seed1",
                split=split,
                logits=source.logits,
                probs=source.probs,
                labels=source.labels,
                jet_ids=source.jet_ids,
                metadata=metadata,
            ),
            seed_root,
        )
        npz = seed_root / "A0_seed1" / f"{split}_predictions.npz"
        meta = seed_root / "A0_seed1" / f"{split}_predictions_metadata.json"
        loaded = load_prediction_block(seed_root, "A0_seed1", split)
        seed_rows[split] = {
            "prediction_path": str(npz.resolve()),
            "prediction_sha256": sha256_file(npz),
            "metadata_path": str(meta.resolve()),
            "metadata_sha256": sha256_file(meta),
            "prediction_content_hash": loaded.metadata["prediction_content_hash"],
            "jet_identity_hash": loaded.metadata["jet_identity_hash"],
            "n_jets": len(loaded.labels),
        }
    registry = {
        "ok": True,
        "contract": LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_SOURCES_CONTRACT,
        "source_artifact_audit": str(Path(audit_config.output_path).resolve()),
        "source_artifact_audit_hash": audit["audit_hash"],
        "development_splits": list(FUSION_DEVELOPMENT_SPLITS),
        "final_test_opened": False,
        "members": {
            "A0": audit["reusable_predictions"]["A0"],
            "A0_seed1": {"prediction_root": str(seed_root.resolve()), "splits": seed_rows},
            "P7b": audit["reusable_predictions"]["P7b"],
        },
    }
    registry["manifest_hash"] = stable_fusion_json_hash(registry)
    registry_path = tmp_path / "development_prediction_sources.json"
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit, registry_path, seed_checkpoint


def test_step5_capture_finds_unique_head_freezes_model_and_removes_hooks() -> None:
    model = _ExactMember()
    capture = PreClassifierEmbeddingCapture(model)
    features = torch.arange(20, dtype=torch.float32).reshape(2, 10, 1)

    report = capture.preflight(
        lambda: model(None, features, None, None)
    )

    assert report["final_head_name"] == "classifier"
    assert report["embedding_shape"] == [2, 10]
    assert report["normal_capture_logits_agree"] is True
    assert report["trainable_parameter_count"] == 0
    assert len(model.part_model.classifier._forward_hooks) == 0
    assert len(model.part_model.classifier._forward_pre_hooks) == 0


def test_step5_capture_fails_closed_for_ambiguous_or_missing_head() -> None:
    ambiguous = _ExactMember()
    ambiguous.part_model.auxiliary = torch.nn.Linear(10, 10)
    with pytest.raises(ValueError, match="ambiguous"):
        PreClassifierEmbeddingCapture(ambiguous)
    missing = _ExactMember()
    missing.part_model.classifier = torch.nn.Linear(10, 3)
    with pytest.raises(ValueError, match="could not locate"):
        PreClassifierEmbeddingCapture(missing)


def test_step5_cache_binds_representation_to_prediction_and_source_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, registry_path, _ = _prediction_registry(tmp_path)
    model = _ExactMember()
    checkpoint = Path(audit["config"]["a0_checkpoint"])

    def fake_dataset(config, split, *, model):
        del config, split, model
        return object()

    def fake_loader(dataset, *, batch_size, num_workers, seed, hlt_only):
        del dataset, batch_size, num_workers, seed, hlt_only
        block = load_prediction_block(audit["config"]["a0_prediction_dir"], "A0", "stack_train")
        logits = torch.from_numpy(block.logits)
        count = len(block.labels)
        return [{
            "points": torch.zeros((count, 2, 1)),
            "features": logits[:, :, None],
            "lorentz_vectors": torch.zeros((count, 4, 1)),
            "mask": torch.ones((count, 1, 1), dtype=torch.bool),
            "tokens": torch.zeros((count, 1, 14)),
            "raw_mask": torch.ones((count, 1), dtype=torch.bool),
            "indices": torch.arange(count),
            "labels": torch.from_numpy(block.labels),
        }]

    monkeypatch.setattr(feature_module, "load_local_residual_field_tagger_from_checkpoint", lambda path, device: (model, {}))
    monkeypatch.setattr(feature_module, "_prediction_dataset", fake_dataset)
    monkeypatch.setattr(feature_module, "_make_prediction_loader", fake_loader)
    output = tmp_path / "representations"
    report = cache_local_residual_field_fusion_features(
        FusionFeatureCacheConfig(
            checkpoint=str(checkpoint),
            member_id="A0",
            output_dir=str(output),
            prediction_sources=str(registry_path),
            source_artifact_audit=audit["config"]["output_path"],
            hlt_cache_dir="unused",
            manifest_path=audit["config"]["manifest_path"],
            amp=False,
        )
    )

    assert report["final_test_opened"] is False
    assert report["splits"]["stack_train"]["contract"] == LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_CONTRACT
    assert report["splits"]["stack_train"]["prediction_content_hash"]
    assert report["splits"]["stack_train"]["source_artifact_audit_hash"] == audit["audit_hash"]
    with np.load(output / "A0" / "stack_train_representations.npz", allow_pickle=False) as data:
        assert set(data.files) == {"jet_embedding", "labels", "jet_file_indices", "jet_entries"}
        assert data["jet_embedding"].shape == (6, 10)


def test_step5_cli_and_slurm_are_stack_only_and_tigris_safe() -> None:
    args = parse_args([
        "--checkpoint", "member.pt",
        "--member-id", "A0",
        "--output-dir", "features",
        "--prediction-sources", "sources.json",
        "--source-artifact-audit", "audit.json",
        "--hlt-cache-dir", "hlt",
        "--manifest-path", "manifest.json.gz",
    ])
    assert not hasattr(args, "splits")
    assert not hasattr(args, "confirm_final_test")
    shell = (REPO_ROOT / "sbatch" / "run_cache_local_residual_field_fusion_features.sh").read_text(encoding="utf-8")
    assert "#SBATCH --account=reu-aisocial" in shell
    assert "export PYTHONNOUSERSITE=1" in shell
    assert "A0|A0_seed1|P7b" in shell
    assert "stack_train stack_val" in shell
    assert "final_test" not in shell
