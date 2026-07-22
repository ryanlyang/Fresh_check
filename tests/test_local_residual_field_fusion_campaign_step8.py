from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from jetclass_fresh.jetclass_data import LABEL_NAMES
from teacher_logit_reco.local_particle_residual_field import (
    FUSION_FIT_SPLIT,
    FUSION_SELECTION_SPLIT,
    FrozenRepresentationFusionHead,
    RepresentationFusionSplitData,
    RepresentationFusionTrainConfig,
    load_representation_fusion_head_from_checkpoint,
    representation_fusion_diagnostics,
    train_representation_fusion_campaign_candidate,
    train_representation_fusion_head,
)
from teacher_logit_reco.local_particle_residual_field import fusion_train as train_module


REPRESENTATION_IDS = (
    "R0_linear_embeddings",
    "R1_mlp_embeddings_logits",
    "R2_scalar_event_gate",
    "R3_classwise_event_gate",
    "R4_A0_anchored_residual",
)


def _data(rows_per_class: int = 3, dim_a: int = 8, dim_b: int = 8) -> RepresentationFusionSplitData:
    labels = np.repeat(np.arange(len(LABEL_NAMES), dtype=np.int64), rows_per_class)
    generator = np.random.default_rng(17 + dim_a + dim_b)
    logits_a = generator.normal(0.0, 0.2, (len(labels), len(LABEL_NAMES))).astype(np.float32)
    logits_b = generator.normal(0.0, 0.2, (len(labels), len(LABEL_NAMES))).astype(np.float32)
    logits_a[np.arange(len(labels)), labels] += 1.5
    logits_b[np.arange(len(labels)), labels] += 1.3
    return RepresentationFusionSplitData(
        embedding_a=generator.normal(size=(len(labels), dim_a)).astype(np.float32),
        embedding_b=generator.normal(size=(len(labels), dim_b)).astype(np.float32),
        logits_a=logits_a, logits_b=logits_b, labels=labels,
    )


@pytest.mark.parametrize("candidate_id", REPRESENTATION_IDS)
def test_step8_all_heads_are_finite_bounded_and_fusion_only(candidate_id: str) -> None:
    model = FrozenRepresentationFusionHead(candidate_id, 8, 6, hidden_width=64, dropout=0.0)
    data = _data(rows_per_class=1, dim_a=8, dim_b=6)
    output = model(
        torch.from_numpy(data.embedding_a), torch.from_numpy(data.embedding_b),
        torch.from_numpy(data.logits_a), torch.from_numpy(data.logits_b),
    )

    assert output.logits.shape == data.logits_a.shape
    assert torch.isfinite(output.logits).all()
    assert model.trainable_parameter_count <= 1_000_000
    assert model.backbone_parameter_count == 0
    assert not any("backbone" in name for name in model.state_dict())
    if output.gate is not None:
        assert output.gate.shape in {(len(data.labels), 1), (len(data.labels), len(LABEL_NAMES))}
        assert torch.all((output.gate >= 0.0) & (output.gate <= 1.0))
        assert "gate" in representation_fusion_diagnostics(output, logits_a=torch.from_numpy(data.logits_a))


def test_step8_zero_r4_correction_exactly_reproduces_a0() -> None:
    model = FrozenRepresentationFusionHead("R4_A0_anchored_residual", 8, 8)
    data = _data(rows_per_class=1)
    output = model(
        torch.from_numpy(data.embedding_a), torch.from_numpy(data.embedding_b),
        torch.from_numpy(data.logits_a), torch.from_numpy(data.logits_b),
    )
    assert torch.equal(output.logits, torch.from_numpy(data.logits_a))
    diagnostics = representation_fusion_diagnostics(output, logits_a=torch.from_numpy(data.logits_a))
    assert diagnostics["correction"]["mean_l2_norm"] == 0.0
    assert diagnostics["correction"]["fraction_a0_predictions_changed"] == 0.0


def test_step8_training_uses_locked_seed_and_never_creates_input_gradients() -> None:
    train = _data(rows_per_class=2)
    validation = _data(rows_per_class=1)
    model, audit, logits = train_representation_fusion_head(
        "R2_scalar_event_gate", train, validation, epochs=2, patience=2, batch_size=10, seed=5101,
    )

    assert audit["backbone_gradients_present"] is False
    assert audit["backbone_optimizer_state_present"] is False
    assert audit["optimizer_state_included_in_checkpoint"] is False
    assert "gate" in audit["diagnostics"][FUSION_SELECTION_SPLIT]
    assert logits[FUSION_FIT_SPLIT].shape == train.logits_a.shape
    with pytest.raises(ValueError, match="seed"):
        train_representation_fusion_head("R0_linear_embeddings", train, validation, seed=9999, epochs=1)


def test_step8_campaign_checkpoint_contains_provenance_and_no_backbone_or_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _data(rows_per_class=2)
    validation = _data(rows_per_class=1)
    hashes = {
        "member_ids": ["A0", "P7b"], "prediction_sources_hash": "a" * 64,
        "source_artifact_audit_hash": "b" * 64, "members": {"A0": {}, "P7b": {}},
    }
    monkeypatch.setattr(
        train_module, "load_representation_fusion_development_data",
        lambda **_kwargs: ({FUSION_FIT_SPLIT: train, FUSION_SELECTION_SPLIT: validation}, hashes),
    )
    checkpoint = tmp_path / "head.pt"
    report_path = tmp_path / "report.json"
    config = RepresentationFusionTrainConfig(
        campaign_id="toy", group_id="F_method", candidate_id="R0_linear_embeddings",
        feature_root="unused", prediction_sources="unused", source_artifact_audit="unused",
        checkpoint_path=str(checkpoint), report_path=str(report_path), epochs=2, patience=2, batch_size=10,
    )
    report = train_representation_fusion_campaign_candidate(config)
    loaded_model, payload = load_representation_fusion_head_from_checkpoint(checkpoint)

    assert report["final_test_opened"] is False
    assert payload["backbone_state_included"] is False
    assert payload["optimizer_state_included"] is False
    assert payload["source_hashes"] == hashes
    assert payload["normalization"]["embeddings"].startswith("per_jet_l2")
    assert loaded_model.backbone_parameter_count == 0
    assert "final" not in inspect.signature(train_representation_fusion_head).parameters

