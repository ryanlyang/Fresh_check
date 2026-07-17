from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block
from jetclass_fresh.jetclass_data import JetIdentity
from teacher_logit_reco.constrained_coarse_to_fine.runtime_tagger_sanity import (
    PairedBootstrapConfig,
    paired_stratified_bootstrap,
    validate_tagger_sanity_report,
    write_tagger_sanity_report,
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tagger(path, *, variant: str, checkpoint_hash: str, source_state: dict[str, str]) -> None:
    path.mkdir(parents=True)
    config = {
        "variant": variant,
        "seed": 28031,
        "epochs": 12,
        "reconstructor_sources": ["checkpoint-specific-and-excluded-from-pair-config"],
        "batch_size": 32,
    }
    provenance = {"model_train": {"source_manifest_hash": "manifest"}, "model_val": {"source_manifest_hash": "manifest"}}
    source = {
        "config": config,
        "reconstructors": {"sources": {"source": {"checkpoint_sha256": checkpoint_hash}}},
        "trusted_hlt_warm_start": {"checkpoint_sha256": "shared-hlt"},
        "provenance": provenance,
        "selection_split": "model_val",
        "input_view": "fixed_hlt_v2_realistic",
        "runtime_compatibility": {"mode": "shared"},
        "source_state": source_state,
    }
    report = {
        "ok": True,
        "variant": variant,
        "selection_split": "model_val",
        "final_test_evaluated": False,
        "deployable_hlt_only": True,
        "phase_history": ["frozen_reconstructor"],
        "checkpoint_sha256": "tagger-checkpoint",
        "configuration_hash": "shared-config",
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "source_metadata.json").write_text(json.dumps(source), encoding="utf-8")
    (path / "run_report.json").write_text(json.dumps(report), encoding="utf-8")


def _write_predictions(root, model_name: str, logits: np.ndarray, identities, *, overwrite: bool = False) -> None:
    save_prediction_block(
        PredictionBlock(
            model_name=model_name,
            split="model_val",
            logits=np.asarray(logits, dtype=np.float32),
            probs=np.zeros_like(logits, dtype=np.float32),
            labels=np.asarray([row.label for row in identities], dtype=np.int64),
            jet_ids=identities,
            metadata={"source_manifest_hash": "manifest", "hlt_content_hash": "hlt"},
        ),
        root,
        overwrite=overwrite,
    )


def _write_reconstructor_checkpoint(path, *, variant: str, runtime_profile: str, runtime_profile_hash: str | None = None, epochs: int, source_state: dict[str, str]) -> None:
    torch.save(
        {
            "checkpoint_role": "best_model_val",
            "family": "C",
            "variant": variant,
            "runtime_profile": {"name": runtime_profile},
            "runtime_profile_hash": runtime_profile_hash,
            "config": {"epochs": epochs, "fixed_horizon": runtime_profile == "fp32_reference"},
            "provenance": {"model_train": {"source_manifest_hash": "manifest"}},
            "code_environment": source_state,
        },
        path,
    )


def test_paired_bootstrap_rejects_accuracy_and_ce_regression() -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    reference = np.asarray([[6.0, 0.0], [0.0, 6.0], [6.0, 0.0], [0.0, 6.0]])
    candidate = np.asarray([[0.0, 6.0], [6.0, 0.0], [0.0, 6.0], [6.0, 0.0]])
    result = paired_stratified_bootstrap(
        candidate_logits=candidate,
        reference_logits=reference,
        labels=labels,
        config=PairedBootstrapConfig(replicates=256, chunk_size=32, seed=9),
    )
    assert not result["ok"]
    assert result["upper_bound_negative_delta_accuracy"] > 0.005
    assert result["upper_bound_delta_ce"] > 0.010


def test_tagger_sanity_report_persists_matched_logits_and_cross_entropy(monkeypatch, tmp_path) -> None:
    candidate_checkpoint = tmp_path / "candidate.pt"
    reference_checkpoint = tmp_path / "reference.pt"
    state = {"source_commit": "abc", "source_status_hash": "clean"}
    _write_reconstructor_checkpoint(
        candidate_checkpoint,
        variant="C5-B3",
        runtime_profile="accelerated_candidate_v1",
        runtime_profile_hash="candidate-runtime-hash",
        epochs=10,
        source_state=state,
    )
    _write_reconstructor_checkpoint(
        reference_checkpoint,
        variant="C5-B3",
        runtime_profile="fp32_reference",
        epochs=30,
        source_state=state,
    )
    candidate_hash = _sha256(candidate_checkpoint)
    reference_hash = _sha256(reference_checkpoint)
    candidate_tagger = tmp_path / "taggers" / "candidate"
    reference_tagger = tmp_path / "taggers" / "reference"
    _write_tagger(candidate_tagger, variant="D5", checkpoint_hash=candidate_hash, source_state=state)
    _write_tagger(reference_tagger, variant="D5", checkpoint_hash=reference_hash, source_state=state)
    identities = [JetIdentity(file="a.root", entry=index, label=index % 2) for index in range(8)]
    logits = np.asarray([[5.0, 0.0] if row.label == 0 else [0.0, 5.0] for row in identities], dtype=np.float32)
    prediction_root = tmp_path / "predictions"
    _write_predictions(prediction_root, "candidate", logits, identities)
    _write_predictions(prediction_root, "reference", logits, identities)
    profile = tmp_path / "accelerated_candidate.json"
    profile.write_text(
        json.dumps(
            {
                "status": "accelerated_candidate_v1",
                "candidate_profile_hash": "candidate-profile",
                "code_environment": state,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "teacher_logit_reco.constrained_coarse_to_fine.runtime_tagger_sanity.validate_runtime_profile",
        lambda *_args, **_kwargs: {"profile": {"candidate_profile_hash": "candidate-profile", "code_environment": state, "code_environment_hash": "environment-hash"}, "file_sha256": "profile-file-hash"},
    )
    monkeypatch.setattr(
        "teacher_logit_reco.constrained_coarse_to_fine.runtime_tagger_sanity.resolve_execution",
        lambda *_args, **_kwargs: {"runtime_profile_hash": "candidate-runtime-hash"},
    )

    output = tmp_path / "report.json"
    report = write_tagger_sanity_report(
        path="C5-B3",
        candidate_tagger_dir=candidate_tagger,
        reference_tagger_dir=reference_tagger,
        candidate_prediction_dir=prediction_root,
        reference_prediction_dir=prediction_root,
        candidate_model_name="candidate",
        reference_model_name="reference",
        candidate_reconstructor_checkpoint=candidate_checkpoint,
        reference_reconstructor_checkpoint=reference_checkpoint,
        candidate_profile_path=profile,
        output_path=output,
        bootstrap=PairedBootstrapConfig(replicates=256, chunk_size=32),
    )
    assert report["ok"]
    paired = np.load(report["paired_arrays_path"])
    assert set(paired.files) == {"labels", "accelerated_logits", "fp32_logits", "accelerated_ce", "fp32_ce"}
    assert np.allclose(paired["accelerated_ce"], paired["fp32_ce"])
    assert report["prediction_provenance"]["paired_arrays_content_hash"]
    degraded_logits = np.asarray([[0.0, 5.0] if row.label == 0 else [5.0, 0.0] for row in identities], dtype=np.float32)
    _write_predictions(prediction_root, "candidate", degraded_logits, identities, overwrite=True)
    with pytest.raises(ValueError, match="paired-array provenance does not match live predictions"):
        validate_tagger_sanity_report(output)
