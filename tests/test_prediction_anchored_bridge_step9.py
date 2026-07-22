from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch

from teacher_logit_reco.local_particle_residual_field import (
    BASELINE_DEPLOYABLE_REQUIRED_IDS,
    B_STAGES,
    DeployableReplicaEvidence,
    PAIRED_SEED_IDS,
    PREDICTION_ANCHORED_ACCESS_RECEIPT_CONTRACT,
    PREDICTION_ANCHORED_CLEAN_RELOAD_AUDIT_CONTRACT,
    PredictionAnchoredDeployableBundle,
    advance_step9_campaign,
    aggregate_deployable_configuration,
    build_campaign_registry,
    build_campaign_reservations,
    build_deployable_bundle_manifest,
    build_deployable_confirmation,
    build_median_publication_manifest,
    build_stage_evidence,
    build_step9_report_row,
    build_step9_reports,
    clean_hlt_only_reload_audit,
    export_deployable_bundle,
    finalize_deployable_confirmation,
    initialize_step9_campaign,
    load_deployable_bundle,
    record_campaign_run_outcomes,
    record_registry_measurements,
    require_production_ready,
    select_deployable_preconfirmation,
    validate_final_test_request,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import with_content_hash


def _measured_registry(*, alternate=False):
    registry = build_campaign_registry(alternate_teacher_valid=alternate)
    measurements = {row["canonical_run_id"]: 1024 for row in registry["runs"]}
    return record_registry_measurements(registry, measurements)


def _execution_spec():
    return with_content_hash({
        "contract": "prediction_anchored_execution_spec_v1",
        "child_manifest": {"content_hash": "a" * 64},
        "parent_manifest": {"sha256": "b" * 64},
    })


def _reservations(registry, tmp_path: Path):
    readiness = require_production_ready(
        registry,
        fixed_persistent_bytes=4096,
        selected_budget_bytes=5 * 1024**3,
    )
    parents = {
        "r0": {"sha256": "1" * 64, "size_bytes": 512, "path": tmp_path / "r0.pt"},
        "consumer": {"sha256": "2" * 64, "size_bytes": 512, "path": tmp_path / "t10.pt"},
        "metadata": {"sha256": "3" * 64, "size_bytes": 512, "path": tmp_path / "metadata"},
    }
    return build_campaign_reservations(
        registry, execution_spec=_execution_spec(), production_readiness=readiness,
        fixed_parent_artifacts=parents, final_deployable_bundle_bytes=512,
    )


def _stage(stage: str, *, passed=True):
    parents = {
        "B0": {"step8_measurement": "0" * 64, "ram_audit": "1" * 64},
        "B1": {"r0_registration": "2" * 64},
        "B2": {"physical45_recipe": "3" * 64, "all50_recipe": "4" * 64},
        "B3": {"b0_b2_release": "5" * 64},
        "B4": {"selected_consumer": "6" * 64},
        "B5": {
            "selected_consumer": "6" * 64,
            "primary_binding": "7" * 64,
            "primary_cache": "8" * 64,
            "all50_cache": "9" * 64,
            "n3_cache": "a" * 64,
        },
        "B6": {"post_teacher_release": "b" * 64},
    }[stage]
    required = {
        "B0": ("data_audit_passed", "split_audit_passed", "mask_audit_passed", "unit_audit_passed", "ram_preflight_passed", "storage_preflight_passed", "r0_prerequisites_passed"),
        "B1": ("r0_registered", "live_f0_valid", "live_h0_valid"),
        "B2": ("physical45_recipe_valid", "all50_recipe_valid", "batch_audit_passed"),
        "B3": ("l0_launched", "upstream_paired3_completed"),
        "B4": ("consumer_aggregate_selected", "stack_val_consumer_confirmed"),
        "B5": ("teacher_bindings_valid", "target_cache_namespaces_valid"),
        "B6": ("post_teacher_matrix_released", "paired3_breadth_completed"),
    }[stage]
    flags = {name: True for name in required}
    if not passed:
        flags[required[-1]] = False
    return build_stage_evidence(
        stage,
        flags=flags,
        parent_artifact_sha256=parents,
        failure_reason=None if passed else "synthetic gate failure",
    )


def _replica(run_id: str, seed: int, accuracy: float, *, saturation=0.001, perturb=0.001):
    base = 0.70
    teacher = 0.72
    return DeployableReplicaEvidence(
        run_id=run_id,
        seed_id=seed,
        accuracy=accuracy,
        macro_per_class_accuracy=accuracy - 0.01,
        cross_entropy=1.0 - accuracy,
        baseline_accuracy=base,
        teacher_bridge_accuracy=teacher,
        recovery_fraction=(accuracy - base) / (teacher - base),
        epoch=4,
        checkpoint=f"{run_id}_{seed}.pt",
        checkpoint_sha256=hex(seed)[2:][-1] * 64,
        scaler_sha256="c" * 64,
        teacher_sha256="d" * 64,
        recipe_sha256="e" * 64,
        deployed_parameter_count=1000,
        persistent_bytes=1000,
        reserved_bytes=1200,
        saturation_fraction=saturation,
        perturbation_mean_accuracy_loss=perturb,
        perturbation_worst_accuracy_loss=perturb,
    ).to_artifact()


def _aggregate(registry, run_id="D10_L8_full_c0", scores=(0.705, 0.710, 0.715), **kwargs):
    rows = [_replica(run_id, seed, score, **kwargs) for seed, score in zip(PAIRED_SEED_IDS, scores)]
    return aggregate_deployable_configuration(registry, rows)


def _pre_and_locked(registry):
    pre = select_deployable_preconfirmation(registry, [_aggregate(registry)])
    receipt = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_ACCESS_RECEIPT_CONTRACT,
            "status": "AUTHORIZED",
            "split_name": "stack_val_deploy",
            "purpose": "deployable_confirmation",
            "parent_manifest_sha256": "1" * 64,
            "bound_split_sha256": "2" * 64,
            "seal_kind": "deployable_preconfirmation",
            "one_shot": True,
            "unlock_sha256": "3" * 64,
            "selection_sha256": pre["content_hash"],
        }
    )
    confirmation = build_deployable_confirmation(
        pre,
        access_receipt=receipt,
        deployable_gain=0.002,
        accuracy=0.705,
        cross_entropy=0.4,
        provenance_valid=True,
    )
    return pre, finalize_deployable_confirmation(pre, confirmation)


def test_reservations_reconcile_quota_and_reject_duplicate_parents(tmp_path):
    registry = _measured_registry()
    artifact = _reservations(registry, tmp_path)
    assert artifact["quota_preflight_passed"] is True
    assert len(artifact["run_reservations_bytes"]) == 54
    readiness = require_production_ready(
        registry, fixed_persistent_bytes=4096, selected_budget_bytes=5 * 1024**3
    )
    duplicate = {
        "r0": {"sha256": "1" * 64, "size_bytes": 10, "path": tmp_path / "a"},
        "consumer": {"sha256": "1" * 64, "size_bytes": 10, "path": tmp_path / "b"},
    }
    with pytest.raises(ValueError, match="duplicated"):
        build_campaign_reservations(
            registry, execution_spec=_execution_spec(), production_readiness=readiness,
            fixed_parent_artifacts=duplicate, final_deployable_bundle_bytes=512,
        )
    over_quota = dict(readiness)
    over_quota["projected_persistent_bytes"] = over_quota["selected_budget_bytes"] + 1
    with pytest.raises(PermissionError, match="exceeds"):
        build_campaign_reservations(
            registry,
            execution_spec=_execution_spec(),
            production_readiness=over_quota,
            fixed_parent_artifacts={
                "r0": {"sha256": "1" * 64, "size_bytes": 10, "path": tmp_path / "only"}
            },
            final_deployable_bundle_bytes=512,
        )


def test_b0_b6_success_state_machine_and_visible_conditional_skip(tmp_path):
    registry = _measured_registry()
    state = initialize_step9_campaign(registry, reservations=_reservations(registry, tmp_path))
    assert len(state["run_outcomes"]) == 54
    assert state["run_outcomes"]["D10_TALT_A3"] == "SKIPPED_INVALID_PARENT"
    for stage in B_STAGES:
        state = advance_step9_campaign(state, registry=registry, evidence=_stage(stage))
    assert state["status"] == "BREADTH_COMPLETE"
    assert all(row["status"] == "COMPLETED" for row in state["stages"].values())
    assert state["run_outcomes"]["D10_L0_bridge_only"] == "COMPLETED"
    assert state["run_outcomes"]["D10_A3_hlg_primary"] == "COMPLETED"


def test_stage_failure_stops_without_fallback_and_marks_children(tmp_path):
    registry = _measured_registry()
    state = initialize_step9_campaign(registry, reservations=_reservations(registry, tmp_path))
    state = advance_step9_campaign(state, registry=registry, evidence=_stage("B0", passed=False))
    assert state["status"] == "STOPPED"
    assert state["fallback_allowed"] is False
    assert state["stages"]["B1"]["status"] == "SKIPPED_FAILED_PARENT"
    with pytest.raises(PermissionError, match="cannot advance"):
        advance_step9_campaign(state, registry=registry, evidence=_stage("B1"))


def test_conditional_skip_cannot_be_promoted(tmp_path):
    registry = _measured_registry()
    state = initialize_step9_campaign(registry, reservations=_reservations(registry, tmp_path))
    with pytest.raises(PermissionError, match="cannot be made runnable"):
        record_campaign_run_outcomes(
            state, registry=registry, outcomes={"D10_TALT_A3": "COMPLETED"}
        )


def test_aggregate_enforces_three_seed_gate_and_ordered_median():
    registry = _measured_registry()
    aggregate = _aggregate(registry, scores=(0.715, 0.705, 0.710))
    assert aggregate["valid_for_primary_selection"] is True
    assert aggregate["ordered_seed_ids"] == [202, 303, 101]
    assert aggregate["median_seed_id"] == 303
    assert aggregate["best_seed_weights_rejected"] is True


def test_aggregate_rejects_saturation_and_nonselectable_control():
    registry = _measured_registry()
    saturated = _aggregate(registry, saturation=0.02)
    assert saturated["valid_for_primary_selection"] is False
    assert "trust_saturation_at_most_one_percent" in saturated["invalid_reasons"]
    control = _aggregate(registry, run_id="D10_N0_shuffled_logit_kd")
    assert control["valid_for_primary_selection"] is False
    assert control["validity_gates"]["registry_selectable"] is False


def test_final_selector_uses_fixed_tie_pool_and_exact_parameter_tie_break():
    registry = _measured_registry()
    first = _aggregate(registry, run_id="D10_L8_full_c0", scores=(0.7100, 0.7110, 0.7120))
    second_rows = [
        _replica("D10_A3_hlg_primary", seed, score)
        for seed, score in zip(PAIRED_SEED_IDS, (0.7104, 0.7114, 0.7124))
    ]
    # Equal secondary metrics except a smaller exact deployed parameter count.
    for row in second_rows:
        row.pop("content_hash")
        row["deployed_parameter_count"] = 999
        row.update(with_content_hash(row))
    second = aggregate_deployable_configuration(registry, second_rows)
    selected = select_deployable_preconfirmation(registry, [first, second])
    assert set(selected["tie_pool_run_ids"]) == {"D10_L8_full_c0", "D10_A3_hlg_primary"}
    assert selected["selected_run_id"] == "D10_A3_hlg_primary"
    assert selected["latency_used_for_selection"] is False


def test_confirmation_failure_is_final_and_never_promotes_runner_up():
    registry = _measured_registry()
    pre = select_deployable_preconfirmation(registry, [_aggregate(registry)])
    receipt = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_ACCESS_RECEIPT_CONTRACT,
            "status": "AUTHORIZED",
            "split_name": "stack_val_deploy",
            "purpose": "deployable_confirmation",
            "parent_manifest_sha256": "1" * 64,
            "bound_split_sha256": "2" * 64,
            "seal_kind": "deployable_preconfirmation",
            "one_shot": True,
            "unlock_sha256": "3" * 64,
            "selection_sha256": pre["content_hash"],
        }
    )
    failed = build_deployable_confirmation(
        pre, access_receipt=receipt, deployable_gain=-0.0001, accuracy=0.70,
        cross_entropy=0.5, provenance_valid=True,
    )
    stopped = finalize_deployable_confirmation(pre, failed)
    assert stopped["runner_up_promoted"] is False
    assert stopped["final_test_authorized"] is False
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_final_test_request(stopped, clean_reload_audit={})


def _report_row(row_id, section, *, deployable, teacher_gain=0.02, recovery=0.25):
    metadata = {}
    if row_id.startswith("A0"):
        metadata = {
            "training_manifest_sha256": "a" * 64,
            "unique_jet_count": 250000,
            "optimizer_step_budget": 100,
        }
    return build_step9_report_row(
        row_id=row_id,
        section=section,
        status="COMPLETED",
        metrics={"accuracy": 0.7},
        seed_metrics=[{"seed_id": seed, "accuracy": 0.7} for seed in PAIRED_SEED_IDS],
        aggregate={"mean_accuracy": 0.7, "sample_std": 0.001},
        median_seed_id=202,
        deployable=deployable,
        hlt_only_reload_passed=True if deployable else None,
        teacher_bridge_gain=teacher_gain,
        recovery_fraction=recovery if teacher_gain > 0 else None,
        metadata=metadata,
    )


def test_reports_are_disjoint_complete_and_keep_all_54_rows_visible():
    registry = _measured_registry()
    baseline_ids = list(BASELINE_DEPLOYABLE_REQUIRED_IDS) + ["selected_T10(fhat:D10_L8_full_c0)"]
    baseline = [_report_row(value, "baselines_and_deployable", deployable=True) for value in baseline_ids]
    privileged_ids = [
        "selected_T10(bridge_0.10)",
        "selected_T10(physical45_oracle_ceiling)",
        "selected_T10(full50_oracle_ceiling)",
        "selected_T10(zero_field)",
    ]
    privileged = [_report_row(value, "privileged_oracle", deployable=False) for value in privileged_ids]
    outcomes = {
        row["canonical_run_id"]: (
            "SKIPPED_INVALID_PARENT" if row["canonical_run_id"] == "D10_TALT_A3" else "PENDING"
        )
        for row in registry["runs"]
    }
    outcomes["D10_L0_bridge_only"] = "FAILED"
    report = build_step9_reports(
        registry,
        baseline_deployable_rows=baseline,
        privileged_rows=privileged,
        ablation_rows={},
        run_outcomes=outcomes,
        persistent_telemetry={"published_bytes": 1234},
        ram_telemetry={"peak_resident_bytes": 5678},
    )
    assert report["ablation_row_count"] == 54
    assert report["physical45_and_full50_ceilings_separate"] is True
    assert all(row["deployable"] for row in report["sections"]["baselines_and_deployable"])
    assert all(not row["deployable"] for row in report["sections"]["privileged_oracle"])
    assert next(row for row in report["sections"]["ablation_evidence"] if row["row_id"] == "D10_TALT_A3")["status"] == "SKIPPED_INVALID_PARENT"
    assert next(row for row in report["sections"]["ablation_evidence"] if row["row_id"] == "D10_L0_bridge_only")["status"] == "FAILED"


def test_report_null_recovery_and_missing_ceiling_fail_closed():
    row = _report_row(
        "diagnostic", "privileged_oracle", deployable=False, teacher_gain=-0.01, recovery=None
    )
    assert row["recovery_fraction"] is None
    with pytest.raises(ValueError, match="recovery must be null"):
        build_step9_report_row(
            row_id="bad", section="privileged_oracle", status="FAILED", deployable=False,
            hlt_only_reload_passed=None, teacher_bridge_gain=0.0, recovery_fraction=1.0,
        )


class _TinyR0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.field = torch.nn.Linear(14, 50)
        self.hidden = torch.nn.Linear(14, 160)

    def forward(self, tokens, mask):
        valid = mask.unsqueeze(-1).to(tokens.dtype)
        return SimpleNamespace(
            predicted_fields=self.field(tokens) * valid,
            hidden=self.hidden(tokens) * valid,
        )


class _TinyCorrection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(45))

    def forward(self, tokens, mask, f0, h0):
        del tokens, h0
        value = f0.clone()
        value[..., :45] = value[..., :45] + self.scale
        value = value * mask.unsqueeze(-1).to(value.dtype)
        return SimpleNamespace(f_hat=value)


class _TinyConsumer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(10))

    def forward(self, points, features, vectors, mask, *, tokens, raw_mask, residual_fields):
        del points, features, vectors, mask, tokens
        weights = raw_mask.unsqueeze(-1).to(residual_fields.dtype)
        pooled = (residual_fields[..., :10] * weights).sum(1) / weights.sum(1).clamp_min(1)
        return pooled + self.bias


def _bundle():
    return PredictionAnchoredDeployableBundle(_TinyR0(), _TinyCorrection(), _TinyConsumer()).eval()


def _hlt_batch():
    torch.manual_seed(9)
    tokens = torch.randn(3, 5, 14)
    raw_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    tokens = tokens * raw_mask.unsqueeze(-1)
    return {
        "tokens": tokens,
        "raw_mask": raw_mask,
        "points": torch.zeros(3, 2, 5),
        "features": torch.zeros(3, 17, 5),
        "lorentz_vectors": torch.zeros(3, 4, 5),
        "mask": raw_mask[:, None, :],
    }


def test_single_hlt_only_bundle_export_and_clean_reload(tmp_path):
    registry = _measured_registry()
    _, locked = _pre_and_locked(registry)
    component_hashes = {
        "r0": "f" * 64,
        "correction": locked["checkpoint_sha256"],
        "consumer": locked["teacher_sha256"],
    }
    manifest = build_deployable_bundle_manifest(
        locked,
        component_sha256=component_hashes,
        preprocessing={"source": "fixed_hlt"},
        residual_normalization={"space": "physical"},
        target_schema={"field_dim": 50},
        class_order=[f"class_{index}" for index in range(10)],
        architecture_manifest={"r0": "tiny", "correction": "tiny", "consumer": "tiny"},
        bundle_reservation_bytes=1024 * 1024,
    )
    source = _bundle()
    path = tmp_path / "deployable.pt"
    publication = export_deployable_bundle(source, manifest=manifest, output_path=path)
    assert publication["hlt_only"] is True
    loaded, loaded_manifest = load_deployable_bundle(path, bundle_factory=lambda _: _bundle())
    assert loaded_manifest["content_hash"] == manifest["content_hash"]
    assert torch.equal(source(_hlt_batch()), loaded(_hlt_batch()))
    offline = tmp_path / "offline.npz"
    target_logits = tmp_path / "target_logits.npz"
    offline.write_bytes(b"offline")
    target_logits.write_bytes(b"logits")
    audit = clean_hlt_only_reload_audit(
        source,
        checkpoint_path=path,
        locked_deployable=locked,
        bundle_factory=lambda _: _bundle(),
        fixed_hlt_batch=_hlt_batch(),
        privileged_source_paths=[offline, target_logits],
    )
    assert audit["passed"] is True
    assert audit["max_abs_logit_difference"] == 0.0
    assert audit["privileged_source_denial_probe_count"] == 2
    assert audit["privileged_access_attempt_count"] >= 2
    assert audit["literal_host_paths_removed"] is False
    final = validate_final_test_request(locked, clean_reload_audit=audit, evaluation_flags={})
    assert final["hlt_only"] is True
    with pytest.raises(PermissionError, match="forbidden evaluation flags"):
        validate_final_test_request(
            locked, clean_reload_audit=audit, evaluation_flags={"oracle_eval": True}
        )


def test_bundle_manifest_rejects_wrong_or_duplicated_parents():
    registry = _measured_registry()
    _, locked = _pre_and_locked(registry)
    kwargs = dict(
        preprocessing={}, residual_normalization={}, target_schema={},
        class_order=["a"], architecture_manifest={},
        bundle_reservation_bytes=1024 * 1024,
    )
    with pytest.raises(ValueError, match="duplicated"):
        build_deployable_bundle_manifest(
            locked,
            component_sha256={
                "r0": locked["checkpoint_sha256"],
                "correction": locked["checkpoint_sha256"],
                "consumer": locked["teacher_sha256"],
            },
            **kwargs,
        )
    with pytest.raises(ValueError, match="exact selected"):
        build_deployable_bundle_manifest(
            locked,
            component_sha256={"r0": "f" * 64, "correction": locked["checkpoint_sha256"], "consumer": "b" * 64},
            **kwargs,
        )


def test_bundle_export_refuses_quota_before_publication(tmp_path):
    registry = _measured_registry()
    _, locked = _pre_and_locked(registry)
    manifest = build_deployable_bundle_manifest(
        locked,
        component_sha256={
            "r0": "f" * 64,
            "correction": locked["checkpoint_sha256"],
            "consumer": locked["teacher_sha256"],
        },
        preprocessing={},
        residual_normalization={},
        target_schema={},
        class_order=["a"],
        architecture_manifest={},
        bundle_reservation_bytes=1,
    )
    destination = tmp_path / "must_not_exist.pt"
    with pytest.raises(PermissionError, match="exceed reservation"):
        export_deployable_bundle(_bundle(), manifest=manifest, output_path=destination)
    assert not destination.exists()


def test_median_publication_keeps_all_metrics_and_only_median_weights():
    registry = _measured_registry()
    manifest = build_median_publication_manifest([_aggregate(registry)])
    row = manifest["publications"][0]
    assert row["all_seed_metrics_retained"] == [101, 202, 303]
    assert row["nonmedian_weights_persisted"] is False
    assert row["frozen_parent_weights_duplicated"] is False


def test_step9_cli_help_is_importable():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_prediction_anchored_bridge_campaign.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "validate-final-test" in result.stdout
