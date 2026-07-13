from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block
from jetclass_fresh.jetclass_data import JetIdentity, JetView
from teacher_logit_reco.local_particle_residual_field import (
    LOCAL_RESIDUAL_FIELD_FUSION_CONTRACT,
    LOCAL_RESIDUAL_FIELD_FUSION_MODE_SCALAR_WEIGHTED_LOGIT_MEAN,
    LOCAL_RESIDUAL_FIELD_FUSION_MODE_UNIFORM_LOGIT_MEAN,
    LocalResidualFieldAugmentedParT,
    LocalResidualFieldFusionConfig,
    LocalResidualFieldParticleViewFusion,
    LocalResidualFieldPredictionConfig,
    LocalResidualFieldTaggerConfig,
    run_local_residual_field_fusion,
)
import teacher_logit_reco.local_particle_residual_field.fusion as fusion_module


def _labels(n_jets: int = 24) -> np.ndarray:
    return (np.arange(n_jets, dtype=np.int64) % 3).astype(np.int64)


def _jet_ids(split: str, labels: np.ndarray) -> list[JetIdentity]:
    return [JetIdentity(file=f"{split}.root", entry=int(index), label=int(label)) for index, label in enumerate(labels)]


def _logits(labels: np.ndarray, *, good: bool = True, strength: float = 2.5) -> np.ndarray:
    logits = np.full((labels.shape[0], 3), -0.25, dtype=np.float32)
    target = labels if good else (labels + 1) % 3
    logits[np.arange(labels.shape[0]), target] = float(strength)
    return logits


def _write_block(prediction_dir: Path, *, model_name: str, split: str, labels: np.ndarray, logits: np.ndarray) -> None:
    block = PredictionBlock(
        model_name=model_name,
        split=split,
        logits=logits,
        probs=np.zeros_like(logits),
        labels=labels,
        jet_ids=_jet_ids(split, labels),
        metadata={
            "contract": "local_particle_residual_field_predictions_v1",
            "field_source": model_name,
            "checkpoint": f"/fake/{model_name}.pt",
        },
    )
    save_prediction_block(block, prediction_dir, overwrite=True)


def _write_prediction_suite(prediction_dir: Path, members: tuple[str, ...] = ("A0", "D5")) -> np.ndarray:
    labels = _labels()
    for split in ("stack_train", "stack_val", "final_test"):
        for member in members:
            if member == "A0":
                logits = _logits(labels, good=False, strength=1.6)
            elif member == "D5":
                logits = _logits(labels, good=True, strength=2.4)
            else:
                logits = _logits(labels, good=True, strength=1.8)
            _write_block(prediction_dir, model_name=member, split=split, labels=labels, logits=logits)
    return labels


def test_step7_uniform_logit_fusion_matches_manual_average(tmp_path: Path):
    prediction_dir = tmp_path / "predictions"
    labels = _write_prediction_suite(prediction_dir)

    report = run_local_residual_field_fusion(
        LocalResidualFieldFusionConfig(
            prediction_dir=str(prediction_dir),
            output_dir=str(tmp_path / "fusion"),
            groups={"G0": ("A0", "D5")},
            fusion_modes=(LOCAL_RESIDUAL_FIELD_FUSION_MODE_UNIFORM_LOGIT_MEAN,),
            confirm_final_test=True,
        )
    )

    metrics = report["groups"]["G0"]["fusion_modes"][LOCAL_RESIDUAL_FIELD_FUSION_MODE_UNIFORM_LOGIT_MEAN]["metrics"]["final_test"]
    manual_logits = 0.5 * (_logits(labels, good=False, strength=1.6) + _logits(labels, good=True, strength=2.4))
    manual_accuracy = float(np.mean(np.argmax(manual_logits, axis=1) == labels))
    assert report["contract"] == LOCAL_RESIDUAL_FIELD_FUSION_CONTRACT
    assert metrics["accuracy"] == manual_accuracy
    assert (tmp_path / "fusion" / "fusion_report.json").exists()
    assert (tmp_path / "fusion" / "fusion_metrics.csv").exists()


def test_step7_scalar_weighted_fusion_records_weights(tmp_path: Path):
    prediction_dir = tmp_path / "predictions"
    _write_prediction_suite(prediction_dir)

    report = run_local_residual_field_fusion(
        LocalResidualFieldFusionConfig(
            prediction_dir=str(prediction_dir),
            output_dir=str(tmp_path / "fusion"),
            groups={"G1_seed_ensemble": ("A0", "D5")},
            fusion_modes=(LOCAL_RESIDUAL_FIELD_FUSION_MODE_SCALAR_WEIGHTED_LOGIT_MEAN,),
            scalar_weight_trials=16,
            control_seed=11,
            confirm_final_test=True,
        )
    )

    fit = report["groups"]["G1_seed_ensemble"]["fusion_modes"][LOCAL_RESIDUAL_FIELD_FUSION_MODE_SCALAR_WEIGHTED_LOGIT_MEAN]["fit"]
    assert len(fit["weights"]) == 2
    assert abs(sum(fit["weights"]) - 1.0) < 1.0e-5


def test_step7_fusion_fails_closed_when_member_prediction_missing(tmp_path: Path):
    prediction_dir = tmp_path / "predictions"
    _write_prediction_suite(prediction_dir, members=("A0",))

    with pytest.raises(FileNotFoundError, match="Missing prediction blocks"):
        run_local_residual_field_fusion(
            LocalResidualFieldFusionConfig(
                prediction_dir=str(prediction_dir),
                output_dir=str(tmp_path / "fusion"),
                groups={"G0": ("A0", "D5")},
                fusion_modes=(LOCAL_RESIDUAL_FIELD_FUSION_MODE_UNIFORM_LOGIT_MEAN,),
                confirm_final_test=True,
            )
        )


def test_step7_particle_view_gated_fusion_masks_invalid_particles():
    module = LocalResidualFieldParticleViewFusion(field_dim=4, num_views=3, hidden_dim=12, dropout=0.0)
    view_fields = torch.randn(2, 3, 5, 4)
    mask = torch.tensor([[True, True, False, False, False], [True, False, False, False, False]])

    output = module(view_fields, mask)

    assert output.fused_fields.shape == (2, 5, 4)
    assert output.gates.shape == (2, 5, 3)
    assert torch.allclose(output.gates[mask].sum(dim=-1), torch.ones(int(mask.sum())))
    assert torch.all(output.fused_fields[~mask] == 0.0)
    assert output.diagnostics["num_views"] == 3


def test_step7_final_test_prediction_dataset_is_hlt_only_for_deployable_model(monkeypatch: pytest.MonkeyPatch):
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    jet_ids = tuple(JetIdentity(file="final.root", entry=int(index), label=int(label)) for index, label in enumerate(labels))
    hlt_view = JetView(
        tokens=np.zeros((3, 4, 14), dtype=np.float32),
        mask=np.ones((3, 4), dtype=bool),
        labels=labels,
        jet_ids=jet_ids,
        split="final_test",
        metadata={
            "view": "fixed_hlt",
            "hlt_content_hash": "hlt-final",
            "jet_identity_hash": "identity-final",
            "source_manifest_hash": "manifest",
            "hlt_profile": "fixed_hlt_v2_realistic",
            "hlt_profile_version": "2",
            "hlt_degradation_strength": 2.5,
        },
    )

    monkeypatch.setattr(fusion_module, "load_cached_hlt_view", lambda *args, **kwargs: hlt_view)
    config = LocalResidualFieldPredictionConfig(
        checkpoint="unused.pt",
        prediction_dir="unused_predictions",
        model_name="A0",
        hlt_cache_dir="unused_hlt",
        target_cache_dir="missing_target_cache",
        splits=("final_test",),
        confirm_final_test=True,
    )
    model = LocalResidualFieldAugmentedParT(
        LocalResidualFieldTaggerConfig(num_classes=3, field_dim=2, field_source="zero"),
        part_model=torch.nn.Identity(),
    )

    dataset = fusion_module._prediction_dataset(config, "final_test", model=model)

    assert dataset.metadata["target_fields_present"] is False
    assert dataset.metadata["allowed_inputs"] == "HLT_particles_only_deployable_final_test"
    assert len(dataset) == 3


def test_step7_final_test_prediction_refuses_target_dependent_control():
    config = LocalResidualFieldPredictionConfig(
        checkpoint="unused.pt",
        prediction_dir="unused_predictions",
        model_name="F0",
        hlt_cache_dir="unused_hlt",
        target_cache_dir="unused_target",
        splits=("final_test",),
        confirm_final_test=True,
    )
    model = LocalResidualFieldAugmentedParT(
        LocalResidualFieldTaggerConfig(num_classes=3, field_dim=2, field_source="random"),
        part_model=torch.nn.Identity(),
    )

    with pytest.raises(ValueError, match="target-dependent"):
        fusion_module._prediction_dataset(config, "final_test", model=model)
