from __future__ import annotations

import inspect

import numpy as np
import pytest

from jetclass_fresh.fusion import PredictionBlock
from jetclass_fresh.jetclass_data import JetIdentity, LABEL_NAMES
from teacher_logit_reco.local_particle_residual_field import (
    FUSION_FIT_SPLIT,
    FUSION_SELECTION_SPLIT,
    G0_DEVELOPMENT_REFERENCE,
    audit_g0_development_reproduction,
    fit_late_fusion_candidate,
)


def _block(name: str, split: str, logits: np.ndarray, labels: np.ndarray) -> PredictionBlock:
    probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    identities = [JetIdentity(file="toy.root", entry=index, label=int(label)) for index, label in enumerate(labels)]
    return PredictionBlock(
        model_name=name, split=split, logits=logits.astype(np.float32), probs=probabilities.astype(np.float32),
        labels=labels.astype(np.int64), jet_ids=identities,
    )


def _development_blocks(rows_per_class: int = 3) -> tuple[PredictionBlock, ...]:
    labels = np.repeat(np.arange(len(LABEL_NAMES), dtype=np.int64), rows_per_class)
    base = np.full((len(labels), len(LABEL_NAMES)), -0.4, dtype=np.float32)
    base[np.arange(len(labels)), labels] = 1.6
    alternate = base.copy()
    alternate[:, [1, 2]] = alternate[:, [2, 1]]
    alternate += np.linspace(-0.1, 0.1, len(labels), dtype=np.float32)[:, None]
    return (
        _block("A0", FUSION_FIT_SPLIT, base, labels),
        _block("P7b", FUSION_FIT_SPLIT, alternate, labels),
        _block("A0", FUSION_SELECTION_SPLIT, base * 0.9, labels),
        _block("P7b", FUSION_SELECTION_SPLIT, alternate * 1.1, labels),
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def test_step7_uniform_logit_and_probability_arithmetic_is_exact() -> None:
    train_a, train_b, val_a, val_b = _development_blocks()
    logits = fit_late_fusion_candidate("L0_mean_logits", train_a, train_b, val_a, val_b)
    probabilities = fit_late_fusion_candidate("L1_mean_probs", train_a, train_b, val_a, val_b)

    assert np.allclose(logits.train_logits, 0.5 * (train_a.logits + train_b.logits))
    expected_probability = 0.5 * (_softmax(val_a.logits) + _softmax(val_b.logits))
    assert np.allclose(_softmax(probabilities.validation_logits), expected_probability, atol=1.0e-6)


def test_step7_temperatures_and_scalar_weights_are_fit_on_stack_train_and_bounded() -> None:
    train_a, train_b, val_a, val_b = _development_blocks()
    first = fit_late_fusion_candidate("L2_temp_mean_logits", train_a, train_b, val_a, val_b)
    changed_val_a = _block("A0", FUSION_SELECTION_SPLIT, -val_a.logits, val_a.labels)
    second = fit_late_fusion_candidate("L2_temp_mean_logits", train_a, train_b, changed_val_a, val_b)
    scalar = fit_late_fusion_candidate("L3_scalar_simplex_logits", train_a, train_b, val_a, val_b)

    assert first.parameters["temperatures"] == second.parameters["temperatures"]
    assert all(0.25 <= value <= 5.0 for value in first.parameters["temperatures"])
    assert 0.0 <= scalar.parameters["weight"] <= 1.0
    assert scalar.parameters["weight"] in scalar.parameters["weight_grid"]


def test_step7_classwise_and_linear_stacker_use_locked_validation_search() -> None:
    train_a, train_b, val_a, val_b = _development_blocks(rows_per_class=2)
    classwise = fit_late_fusion_candidate(
        "L4_classwise_simplex_logits", train_a, train_b, val_a, val_b, classwise_steps=30,
    )
    stacker = fit_late_fusion_candidate(
        "L5_linear_stacker", train_a, train_b, val_a, val_b, stacker_max_steps=3,
    )

    assert len(classwise.parameters["weights"]) == len(LABEL_NAMES)
    assert all(0.0 <= value <= 1.0 for value in classwise.parameters["weights"])
    assert classwise.parameters["selected_l2"] in classwise.parameters["l2_grid"]
    assert stacker.parameters["feature_mode"] in stacker.parameters["feature_modes"]
    assert stacker.parameters["C"] in stacker.parameters["C_grid"]
    assert len(stacker.parameters["feature_mean"]) in {20, 40}


def test_step7_candidate_fitter_has_no_final_path_and_rejects_final_blocks() -> None:
    assert "final" not in inspect.signature(fit_late_fusion_candidate).parameters
    train_a, train_b, val_a, val_b = _development_blocks()
    final_a = _block("A0", "final_test", val_a.logits, val_a.labels)
    with pytest.raises(ValueError, match="accepts only"):
        fit_late_fusion_candidate("L0_mean_logits", train_a, train_b, final_a, val_b)


def test_step7_g0_reproduction_audit_is_bound_to_completed_pilot() -> None:
    metrics = {
        split: {"multiclass": dict(reference)}
        for split, reference in G0_DEVELOPMENT_REFERENCE.items()
    }
    assert audit_g0_development_reproduction(metrics)["ok"] is True
    metrics[FUSION_SELECTION_SPLIT]["multiclass"]["accuracy"] -= 0.001
    failed = audit_g0_development_reproduction(metrics)
    assert failed["ok"] is False
    assert "accuracy_absolute_difference" in failed["problems"][0]
