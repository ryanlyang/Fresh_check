from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block
from jetclass_fresh.jetclass_data import JetIdentity, JetView
from teacher_logit_reco.local_particle_residual_field import (
    LOCAL_RESIDUAL_FIELD_REPORT_CONTRACT,
    LocalParticleResidualFieldCache,
    LocalParticleResidualFieldDataset,
    LocalParticleResidualFieldDatasetConfig,
    LocalResidualFieldFusionConfig,
    LocalResidualFieldReconstructorConfig,
    LocalResidualFieldReportConfig,
    LocalResidualFieldTaggerTrainConfig,
    build_local_residual_field_reconstructor,
    build_local_residual_field_report,
    collate_local_particle_residual_field_batch,
    compute_local_particle_residual_fields,
    compute_local_residual_reconstruction_loss,
    load_local_residual_reconstructor_from_checkpoint,
    run_local_residual_field_fusion,
)
from tests.test_local_particle_residual_field_step5_tagger import (
    FIELD_GROUPS,
    FIELD_NAMES,
    FakePart,
    _dataset,
    fake_build_hlt_classifier,
)
import teacher_logit_reco.local_particle_residual_field.tagger as tagger_module
import teacher_logit_reco.local_particle_residual_field.tagger_train as tagger_train_module


def _one_jet_identity(label: int = 0) -> tuple[JetIdentity, ...]:
    return (JetIdentity(file="jet.root", entry=0, label=label),)


def test_step10_soft_assignment_target_values_and_mask_are_stable() -> None:
    hlt_tokens = np.zeros((1, 2, 14), dtype=np.float32)
    offline_tokens = np.zeros((1, 2, 14), dtype=np.float32)
    hlt_mask = np.array([[True, False]])
    offline_mask = np.array([[True, False]])
    hlt_tokens[0, 0, 0] = 10.0
    hlt_tokens[0, 0, 3] = 10.0
    hlt_tokens[0, 0, 5] = 1.0
    offline_tokens[0, 0, 0] = 20.0
    offline_tokens[0, 0, 3] = 20.0
    offline_tokens[0, 0, 5] = 1.0

    fields, mask, field_names, field_groups, diagnostics = compute_local_particle_residual_fields(
        hlt_tokens,
        hlt_mask,
        offline_tokens,
        offline_mask,
        radii=(0.10,),
    )

    assert fields.shape == (1, 2, len(field_names))
    assert np.array_equal(mask, hlt_mask)
    assert diagnostics["all_finite"] is True
    assert "pt_density" in field_groups
    assert fields[0, 1].tolist() == [0.0] * len(field_names)
    assert fields[0, 0, field_names.index("r0p1.delta_log_pt_sum")] == pytest.approx(np.log(2.0), rel=1.0e-5)
    assert fields[0, 0, field_names.index("r0p1.delta_pt_frac")] == pytest.approx(1.0, rel=1.0e-5)
    assert fields[0, 0, field_names.index("r0p1.missing_pt_frac")] == pytest.approx(1.0, rel=1.0e-5)
    assert fields[0, 0, field_names.index("r0p1.extra_pt_frac")] == pytest.approx(0.0, abs=1.0e-7)


def test_step10_phi_wrapping_keeps_local_centroid_near_boundary() -> None:
    hlt_tokens = np.zeros((1, 1, 14), dtype=np.float32)
    offline_tokens = np.zeros((1, 1, 14), dtype=np.float32)
    hlt_mask = np.array([[True]])
    offline_mask = np.array([[True]])
    hlt_tokens[0, 0, 0] = 10.0
    hlt_tokens[0, 0, 2] = np.float32(np.pi - 0.01)
    hlt_tokens[0, 0, 3] = 10.0
    offline_tokens[0, 0, 0] = 10.0
    offline_tokens[0, 0, 2] = np.float32(-np.pi + 0.01)
    offline_tokens[0, 0, 3] = 10.0

    fields, _, field_names, _, _ = compute_local_particle_residual_fields(
        hlt_tokens,
        hlt_mask,
        offline_tokens,
        offline_mask,
        radii=(10.0,),
    )

    assert fields[0, 0, field_names.index("r10.delta_phi_centroid")] == pytest.approx(0.02, abs=1.0e-4)
    assert fields[0, 0, field_names.index("r10.delta_r_centroid")] == pytest.approx(0.02, abs=1.0e-4)


def test_step10_dataset_rejects_stale_hlt_cache_hash() -> None:
    labels = np.asarray([0], dtype=np.int64)
    tokens = np.zeros((1, 2, 14), dtype=np.float32)
    mask = np.asarray([[True, False]])
    jet_ids = _one_jet_identity()
    view = JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split="model_train",
        metadata={
            "hlt_content_hash": "fresh-hlt",
            "jet_identity_hash": "identity",
            "source_manifest_hash": "manifest",
        },
    )
    cache = LocalParticleResidualFieldCache(
        target_fields=np.zeros((1, 2, 3), dtype=np.float32),
        target_mask=mask.copy(),
        labels=labels.copy(),
        jet_ids=jet_ids,
        field_names=("a", "b", "c"),
        field_groups={"all": [0, 1, 2]},
        radii=(0.1,),
        split="model_train",
        metadata={
            "hlt_content_hash": "stale-hlt",
            "target_content_hash": "target",
            "offline_content_hash": "offline",
            "jet_identity_hash": "identity",
            "source_manifest_hash": "manifest",
        },
    )

    with pytest.raises(ValueError, match="hlt_content_hash"):
        LocalParticleResidualFieldDataset(
            view,
            cache,
            config=LocalParticleResidualFieldDatasetConfig(
                hlt_cache_dir="unused",
                target_cache_dir="unused",
                split="model_train",
            ),
        )


def test_step10_frozen_reconstructor_checkpoint_roundtrip_and_loss_composition(tmp_path: Path) -> None:
    model = build_local_residual_field_reconstructor(
        LocalResidualFieldReconstructorConfig(
            variant="C6",
            field_dim=len(FIELD_NAMES),
            d_model=24,
            num_heads=3,
            num_layers=1,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        )
    )
    checkpoint = tmp_path / "reconstructor.pt"
    torch.save(
        {
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "metrics": {"mae": 0.1},
        },
        checkpoint,
    )

    loaded, payload = load_local_residual_reconstructor_from_checkpoint(checkpoint)
    dataset = _dataset("model_train", n_jets=3)
    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])
    output = loaded(batch["tokens"], batch["raw_mask"])
    base_loss, base_metrics = compute_local_residual_reconstruction_loss(
        output,
        batch,
        field_names=FIELD_NAMES,
        field_groups=FIELD_GROUPS,
        selected_indices=tuple(range(len(FIELD_NAMES))),
        consistency_loss_weight=0.0,
    )
    consistency_loss, consistency_metrics = compute_local_residual_reconstruction_loss(
        output,
        batch,
        field_names=FIELD_NAMES,
        field_groups=FIELD_GROUPS,
        selected_indices=tuple(range(len(FIELD_NAMES))),
        consistency_loss_weight=2.0,
    )

    assert payload["metrics"]["mae"] == 0.1
    assert output.predicted_fields.shape == batch["target_fields"].shape
    assert torch.isfinite(base_loss)
    assert torch.isfinite(consistency_loss)
    assert consistency_loss >= base_loss
    assert base_metrics["consistency_loss"] == 0.0
    assert consistency_metrics["consistency_loss"] >= 0.0


def test_step10_tagger_rejects_reconstructor_checkpoint_missing_provenance() -> None:
    current_metadata = {
        split: _dataset(split, n_jets=3).metadata
        for split in ("model_train", "model_val", "stack_val")
    }
    stale_payload = {
        "model_config": {
            "field_names": list(FIELD_NAMES),
        },
        "dataset_metadata": {
            split: {
                "alignment_report": {
                    "source_manifest_hash": "manifest_hash",
                    "jet_identity_hash": f"identity_{split}",
                }
            }
            for split in ("model_train", "model_val", "stack_val")
        },
    }

    with pytest.raises(ValueError, match="missing provenance"):
        tagger_train_module._validate_reconstructor_checkpoint_payload(
            stale_payload,
            current_dataset_metadata=current_metadata,
            full_field_names=FIELD_NAMES,
            selected_field_names=FIELD_NAMES,
        )


def test_step10_tagger_joint_training_reports_reconstruction_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = {
        "model_train": _dataset("model_train", n_jets=9),
        "model_val": _dataset("model_val", n_jets=6),
        "stack_val": _dataset("stack_val", n_jets=6),
    }

    def fake_load_dataset(config, split, *, max_jets):
        del config, max_jets
        return datasets[str(split)]

    monkeypatch.setattr(tagger_module, "build_hlt_classifier", fake_build_hlt_classifier)
    monkeypatch.setattr(tagger_train_module, "_load_tagger_dataset", fake_load_dataset)
    config = LocalResidualFieldTaggerTrainConfig(
        output_dir=str(tmp_path / "joint"),
        hlt_cache_dir="unused_hlt",
        target_cache_dir="unused_targets",
        field_source="joint_reconstructor",
        num_classes=3,
        label_names=("a", "b", "c"),
        model_size="tiny",
        batch_size=3,
        eval_batch_size=3,
        epochs=1,
        part_lr=1.0e-3,
        reconstructor_lr=1.0e-3,
        reconstructor_loss_weight=0.25,
        device="cpu",
        amp=False,
        early_stop_patience=-1,
    )

    report = tagger_train_module.train_local_residual_field_tagger(config)

    assert report["ok"] is True
    assert report["best_model_val"]["reconstructor_loss"] >= 0.0
    curves = json.loads((tmp_path / "joint" / "training_curves.json").read_text(encoding="utf-8"))
    assert curves["epochs"][0]["train"]["reconstructor_loss"] >= 0.0


def _write_prediction(prediction_dir: Path, *, model_name: str, split: str, labels: np.ndarray) -> None:
    logits = np.full((labels.shape[0], 3), -0.5, dtype=np.float32)
    logits[np.arange(labels.shape[0]), labels] = 2.0
    save_prediction_block(
        PredictionBlock(
            model_name=model_name,
            split=split,
            logits=logits,
            probs=np.zeros_like(logits),
            labels=labels,
            jet_ids=[JetIdentity(file=f"{split}.root", entry=int(index), label=int(label)) for index, label in enumerate(labels)],
            metadata={"contract": "local_particle_residual_field_predictions_v1"},
        ),
        prediction_dir,
        overwrite=True,
    )


def test_step10_fusion_fails_closed_on_label_mismatch(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    for split in ("stack_train", "stack_val", "final_test"):
        _write_prediction(prediction_dir, model_name="A0", split=split, labels=labels)
        bad_labels = labels.copy()
        if split == "final_test":
            bad_labels[0] = 2
        _write_prediction(prediction_dir, model_name="D5", split=split, labels=bad_labels)

    with pytest.raises(ValueError, match="Label mismatch|label mismatch"):
        run_local_residual_field_fusion(
            LocalResidualFieldFusionConfig(
                prediction_dir=str(prediction_dir),
                output_dir=str(tmp_path / "fusion"),
                groups={"G0": ("A0", "D5")},
                confirm_final_test=True,
            )
        )


def test_step10_report_cli_contract_and_final_test_gate(tmp_path: Path) -> None:
    from tests.test_local_particle_residual_field_step9_report import _write_campaign

    _write_campaign(tmp_path)
    report = build_local_residual_field_report(
        LocalResidualFieldReportConfig(
            output_dir=str(tmp_path / "final_report"),
            tagger_root=str(tmp_path / "taggers"),
            reconstructor_root=str(tmp_path / "reconstructors"),
            fusion_dir=str(tmp_path / "fusion"),
            prediction_dir=str(tmp_path / "predictions"),
            required_tagger_run_ids=("A0", "D5"),
            required_reconstructor_run_ids=("C0",),
            require_fusion=True,
            confirm_final_test=False,
        )
    )

    assert report["contract"] == LOCAL_RESIDUAL_FIELD_REPORT_CONTRACT
    assert report["ok"] is False
    assert any("final_test metrics found" in problem for problem in report["problems"])
