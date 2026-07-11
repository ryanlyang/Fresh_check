import json
from pathlib import Path

import numpy as np
import pytest
import torch

from teacher_logit_reco.canonical_state import (
    CANONICAL_STATE_EXPECTED_RUN_IDS,
    CANONICAL_STATE_VARIANT_REGISTRY_CONTRACT,
    FINAL_TEST_POLICY_MODEL_VAL_ONLY,
    FINAL_TEST_POLICY_PRIMARY_TEACHER_FREE,
    FINAL_TEST_POLICY_STACK_ONLY,
    MODEL_KIND_LOGIT_FUSION,
    MODEL_KIND_ORACLE_DIAGNOSTIC,
    MODEL_KIND_PART_BASELINE,
    MODEL_KIND_STATE_CONDITIONED_PART,
    PREDICTOR_VARIANT_GEOMETRY_BIASED,
    SCHEDULE_FROM_SCRATCH_PART_BASELINE,
    SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
    STATE_CONTEXT_ALL,
    STATE_CONTEXT_PARTICLES_ONLY,
    WARM_START_HLT_PART_BASELINE,
    CanonicalStateVariantRunConfig,
    CanonicalStateVariantSpec,
    canonical_state_diagnostic_run_ids,
    canonical_state_expected_run_ids,
    canonical_state_fusion_run_ids,
    canonical_state_oracle_run_ids,
    canonical_state_primary_run_ids,
    canonical_state_registry_manifest,
    canonical_state_required_dependencies,
    canonical_state_variant_registry,
    canonical_state_variant_spec,
    require_canonical_state_primary_final_test_allowed,
    run_canonical_state_variant,
)
from teacher_logit_reco.canonical_state.run_variants import _oracle_delta_for_batch, _state_context_mask_kwargs


def test_step8_registry_has_exact_a0_to_g3_run_ids() -> None:
    registry = canonical_state_variant_registry()

    assert tuple(registry) == CANONICAL_STATE_EXPECTED_RUN_IDS
    assert canonical_state_expected_run_ids() == CANONICAL_STATE_EXPECTED_RUN_IDS
    assert len(registry) == 39
    assert CANONICAL_STATE_EXPECTED_RUN_IDS[:4] == ("A0", "A1", "A2", "A3")
    assert CANONICAL_STATE_EXPECTED_RUN_IDS[-4:] == ("G0", "G1", "G2", "G3")


def test_step8_core_state_specs_match_plan() -> None:
    a0 = canonical_state_variant_spec("A0")
    assert a0.model_kind == MODEL_KIND_PART_BASELINE
    assert a0.tagger_mode == STATE_CONTEXT_PARTICLES_ONLY
    assert a0.warm_start_policy == "none"
    assert a0.loss_weights.active_terms() == ("ce",)

    a1 = canonical_state_variant_spec("A1")
    assert a1.warm_start_policy == WARM_START_HLT_PART_BASELINE

    d2 = canonical_state_variant_spec("D2")
    assert d2.model_kind == MODEL_KIND_STATE_CONDITIONED_PART
    assert d2.tagger_mode == STATE_CONTEXT_ALL
    assert d2.predictor_variant == PREDICTOR_VARIANT_GEOMETRY_BIASED
    assert d2.state_context_families == ("phi_hlt", "delta_phi_hat", "phi_pred")
    assert d2.loss_weights.state_huber > 0.0
    assert d2.loss_weights.logit_kd == 0.0

    d3 = canonical_state_variant_spec("D3")
    assert not d3.uses_teacher_logits
    assert not d3.teacher_for_training_only
    assert d3.final_test_policy == FINAL_TEST_POLICY_PRIMARY_TEACHER_FREE
    assert d3.loss_weights.logit_kd == 0.0
    assert "KD is intentionally disabled" in " ".join(d3.notes)

    d5 = canonical_state_variant_spec("D5")
    assert d5.loss_weights.active_terms() == ("ce",)


def test_step8_oracle_variants_are_diagnostic_and_blocked_from_primary_final_test() -> None:
    assert set(canonical_state_oracle_run_ids()) == {"G0", "G1"}

    for run_id in ("G0", "G1"):
        spec = canonical_state_variant_spec(run_id)
        assert spec.model_kind == MODEL_KIND_ORACLE_DIAGNOSTIC
        assert spec.is_oracle
        assert spec.uses_offline_phi
        assert spec.oracle_inputs_allowed
        assert not spec.primary
        assert spec.diagnostic_only
        assert spec.final_test_policy == FINAL_TEST_POLICY_MODEL_VAL_ONLY
        with pytest.raises(ValueError, match="not allowed as a primary final_test claim"):
            require_canonical_state_primary_final_test_allowed(run_id)


def test_step8_primary_final_test_allowlist_excludes_stack_and_report_controls() -> None:
    for run_id in ("A0", "A1", "D2", "D3", "E0", "F1"):
        assert require_canonical_state_primary_final_test_allowed(run_id).run_id == run_id

    for run_id in ("Fshuffle", "G2", "G3"):
        with pytest.raises(ValueError, match="primary final_test"):
            require_canonical_state_primary_final_test_allowed(run_id)

    assert canonical_state_variant_spec("Fshuffle").final_test_policy == FINAL_TEST_POLICY_STACK_ONLY


def test_step8_fusion_specs_declare_inputs_and_stack_dependencies() -> None:
    assert set(canonical_state_fusion_run_ids()) == {
        "A3",
        "F0",
        "F1",
        "F2",
        "F3",
        "F4",
        "Fseed",
        "Fshuffle",
    }

    f1 = canonical_state_variant_spec("F1")
    assert f1.model_kind == MODEL_KIND_LOGIT_FUSION
    assert f1.requires_stack_splits
    assert f1.fusion_inputs == ("A0", "A2", "D2", "D3")
    assert canonical_state_required_dependencies("F1") == ("A0", "A2", "D2", "D3")

    fseed = canonical_state_variant_spec("Fseed")
    assert fseed.seed_count == 3
    assert fseed.requires_stack_splits
    assert canonical_state_required_dependencies("Fseed") == ("A0",)

    fshuffle = canonical_state_variant_spec("Fshuffle")
    assert not fshuffle.primary
    assert fshuffle.diagnostic_only
    assert canonical_state_required_dependencies("Fshuffle") == ("F1", "A0", "D2")

    f2 = canonical_state_variant_spec("F2")
    assert not f2.primary
    assert f2.diagnostic_only
    assert f2.final_test_policy == FINAL_TEST_POLICY_STACK_ONLY
    assert "predictor-only" in " ".join(f2.notes)


def test_step8_primary_and_diagnostic_partitions_are_explicit() -> None:
    primary = set(canonical_state_primary_run_ids())
    diagnostic = set(canonical_state_diagnostic_run_ids())

    assert {"A0", "A2", "B0", "D2", "D3", "E0", "F1"} <= primary
    assert {"B2", "B3", "C0", "C6", "Fshuffle", "G0", "G1", "G2", "G3"} <= diagnostic
    assert primary.isdisjoint(diagnostic)


def test_step8_registry_manifest_is_serializable_and_lookup_is_strict() -> None:
    manifest = canonical_state_registry_manifest()

    assert manifest["contract"] == CANONICAL_STATE_VARIANT_REGISTRY_CONTRACT
    assert manifest["run_ids"] == list(CANONICAL_STATE_EXPECTED_RUN_IDS)
    assert manifest["variants"]["D2"]["loss_weights"]["active_terms"] == [
        "ce",
        "state_huber",
        "state_l1",
        "delta_norm",
        "smoothness",
    ]

    with pytest.raises(KeyError, match="unknown canonical-state run id"):
        canonical_state_variant_spec("NOPE")


def test_step8_variant_validation_rejects_unsafe_or_drifting_specs() -> None:
    with pytest.raises(ValueError, match="uses oracle inputs and cannot be primary"):
        CanonicalStateVariantSpec(
            run_id="G0",
            tier="G",
            title="bad primary oracle",
            model_kind=MODEL_KIND_ORACLE_DIAGNOSTIC,
            schedule=SCHEDULE_WARMSTART_FROZEN_TO_UPPER_UNFREEZE,
            oracle_inputs_allowed=True,
            uses_offline_phi=True,
            primary=True,
        )

    with pytest.raises(ValueError, match="unknown dependencies"):
        CanonicalStateVariantSpec(
            run_id="A0",
            tier="A",
            title="bad dependency",
            model_kind=MODEL_KIND_PART_BASELINE,
            schedule=SCHEDULE_FROM_SCRATCH_PART_BASELINE,
            dependencies=("NOPE",),
        )


def test_step8_fusion_refuses_partial_declared_inputs(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    d2_dir = run_root / "D2"
    d2_dir.mkdir(parents=True)
    (d2_dir / "run_report.json").write_text(
        json.dumps(
            {
                "run_id": "D2",
                "manifest": {"manifest_hash": "manifest-a"},
                "hlt_input_contract": {"hlt_profile": "fixed_hlt_v2_realistic"},
            }
        ),
        encoding="utf-8",
    )
    logits = np.zeros((4, 10), dtype=np.float32)
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    np.savez_compressed(d2_dir / "model_val_predictions.npz", logits=logits, labels=labels)
    np.savez_compressed(d2_dir / "stack_val_predictions.npz", logits=logits, labels=labels)

    report = run_canonical_state_variant(
        CanonicalStateVariantRunConfig(
            run_id="F2",
            output_dir=run_root / "F2",
            variant_root=run_root,
            manifest=tmp_path / "unused_manifest.json.gz",
            hlt_cache_dir=tmp_path / "unused_hlt_cache",
            phi_hlt_cache_dir=tmp_path / "unused_phi_hlt",
            phi_offline_cache_dir=tmp_path / "unused_phi_off",
            confirm_final_test=True,
        )
    )

    assert not report["ok"]
    assert report["model_val_metrics"]["available"] is False
    assert report["model_val_metrics"]["missing_inputs"] == ["C1", "C2", "C3"]
    assert "model_val" not in report["prediction_caches"]


def test_step8_g1_oracle_delta_uses_true_phi_off_minus_phi_hlt() -> None:
    spec = canonical_state_variant_spec("G1")
    phi_hlt = torch.randn(2, 48, 18)
    phi_off = phi_hlt + 0.125
    batch = {
        "phi_hlt": phi_hlt,
        "phi_off": phi_off,
        "has_phi_off": torch.ones(2, dtype=torch.bool),
    }

    delta = _oracle_delta_for_batch(spec, batch, split="model_val")

    assert torch.allclose(delta, phi_off - phi_hlt)
    assert _oracle_delta_for_batch(canonical_state_variant_spec("D0"), batch, split="model_val") is None
    with pytest.raises(ValueError, match="final_test"):
        _oracle_delta_for_batch(spec, batch, split="final_test")


def test_step8_oracle_mask_kwargs_expose_offline_valid_state_tokens() -> None:
    hlt_mask = torch.tensor([[True, False, True]])
    offline_mask = torch.tensor([[True, True, False]])
    batch = {
        "phi_hlt_state_mask": hlt_mask,
        "phi_off_state_mask": offline_mask,
        "has_phi_off": torch.ones(1, dtype=torch.bool),
    }

    g0_kwargs = _state_context_mask_kwargs(canonical_state_variant_spec("G0"), batch, split="model_val")
    g1_kwargs = _state_context_mask_kwargs(canonical_state_variant_spec("G1"), batch, split="model_val")
    d0_kwargs = _state_context_mask_kwargs(canonical_state_variant_spec("D0"), batch, split="model_val")

    assert torch.equal(g0_kwargs["state_mask"], offline_mask)
    assert torch.equal(g1_kwargs["state_mask"], hlt_mask)
    assert torch.equal(g1_kwargs["delta_state_mask"], offline_mask)
    assert torch.equal(d0_kwargs["state_mask"], hlt_mask)
    assert "delta_state_mask" not in d0_kwargs
