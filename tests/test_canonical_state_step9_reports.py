import json
from pathlib import Path

import pytest

from teacher_logit_reco.canonical_state import (
    CANONICAL_STATE_HLT_DEGRADATION_STRENGTH,
    CANONICAL_STATE_HLT_PROFILE,
    CANONICAL_STATE_HLT_PROFILE_VERSION,
    CANONICAL_STATE_LABEL_NAMES,
    CANONICAL_STATE_REPORT_CONTRACT,
    CANONICAL_STATE_REPORT_TABLES,
    CanonicalStateReportConfig,
    build_canonical_state_report,
)


def _base_report(run_id: str, *, accuracy: float = 0.70, manifest_hash: str = "manifest-a") -> dict:
    return {
        "run_id": run_id,
        "manifest": {"manifest_hash": manifest_hash},
        "input_contract": {
            "hlt_profile": CANONICAL_STATE_HLT_PROFILE,
            "hlt_profile_version": CANONICAL_STATE_HLT_PROFILE_VERSION,
            "hlt_degradation_strength": CANONICAL_STATE_HLT_DEGRADATION_STRENGTH,
            "label_names": list(CANONICAL_STATE_LABEL_NAMES),
            "label_filter": list(range(10)),
        },
        "model_val_dataset": {
            "source_manifest_hash": manifest_hash,
            "hlt_profile": CANONICAL_STATE_HLT_PROFILE,
            "hlt_profile_version": CANONICAL_STATE_HLT_PROFILE_VERSION,
            "hlt_degradation_strength": CANONICAL_STATE_HLT_DEGRADATION_STRENGTH,
            "hlt_content_hash": "hlt-content-a",
            "phi_hlt_content_hash": "phi-hlt-a",
            "phi_hlt_source_cache_hash": "hlt-content-a",
            "jet_identity_hash": "jets-a",
        },
        "final_test_dataset": {
            "source_manifest_hash": manifest_hash,
            "hlt_profile": CANONICAL_STATE_HLT_PROFILE,
            "hlt_profile_version": CANONICAL_STATE_HLT_PROFILE_VERSION,
            "hlt_degradation_strength": CANONICAL_STATE_HLT_DEGRADATION_STRENGTH,
            "hlt_content_hash": "hlt-content-final",
            "phi_hlt_content_hash": "phi-hlt-final",
            "phi_hlt_source_cache_hash": "hlt-content-final",
            "jet_identity_hash": "jets-final",
        },
        "model_val_metrics": {
            "accuracy": accuracy - 0.01,
            "cross_entropy": 0.8,
            "n_jets": 1000,
            "per_class_accuracy": {"0": accuracy, "1": accuracy - 0.02},
        },
        "final_test_metrics": {
            "accuracy": accuracy,
            "cross_entropy": 0.82,
            "n_jets": 1000,
            "per_class_accuracy": {"0": accuracy + 0.01, "1": accuracy - 0.01},
        },
    }


def _predictor_report(run_id: str, *, loss: float = 0.4, manifest_hash: str = "manifest-a") -> dict:
    report = _base_report(run_id, accuracy=0.0, manifest_hash=manifest_hash)
    report.pop("final_test_metrics", None)
    report["state_prediction_metrics"] = {
        "model_val": {
            "state_huber": loss,
            "state_l1": loss * 0.5,
            "delta_l2": loss * 0.25,
            "smoothness": 0.01,
            "n_tokens": 64,
        }
    }
    report["per_token_family_residual_metrics"] = {
        "model_val": {
            "global": {"state_l1": loss * 0.2, "state_huber": loss * 0.3},
        }
    }
    report["per_field_residual_metrics"] = {
        "model_val": {
            "pt_sum": {"state_l1": loss * 0.4, "state_huber": loss * 0.6},
        }
    }
    return report


def _oracle_report(run_id: str, *, accuracy: float = 0.82, manifest_hash: str = "manifest-a") -> dict:
    report = _base_report(run_id, accuracy=accuracy, manifest_hash=manifest_hash)
    report.pop("final_test_metrics", None)
    return report


def _write_report(root: Path, run_id: str, payload: dict) -> None:
    path = root / run_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_report.json").write_text(json.dumps(payload), encoding="utf-8")


def test_step9_builds_report_tables_and_separates_fusion_and_oracle(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    output_dir = tmp_path / "report"
    for run_id, payload in {
        "A0": _base_report("A0", accuracy=0.70),
        "D2": _base_report("D2", accuracy=0.74),
        "D3": _base_report("D3", accuracy=0.745),
        "F1": _base_report("F1", accuracy=0.755),
        "Fseed": _base_report("Fseed", accuracy=0.735),
        "C0": _predictor_report("C0", loss=0.30),
        "C1": _predictor_report("C1", loss=0.37),
        "G0": _oracle_report("G0", accuracy=0.82),
        "G1": _oracle_report("G1", accuracy=0.80),
    }.items():
        _write_report(run_root, run_id, payload)

    report = build_canonical_state_report(
        CanonicalStateReportConfig(
            output_dir=output_dir,
            run_root=run_root,
            run_ids=("A0", "D2", "D3", "F1", "Fseed", "C0", "C1", "G0", "G1"),
            allow_missing_runs=True,
            confirm_final_test=True,
        )
    )

    assert report["report_contract"] == CANONICAL_STATE_REPORT_CONTRACT
    assert report["ok"]
    assert report["summary"]["best_final_test_run_id"] == "F1"
    assert (output_dir / "canonical_state_report.json").exists()
    for filename in CANONICAL_STATE_REPORT_TABLES.values():
        assert (output_dir / filename).exists()

    single_runs = {row["run_id"] for row in report["tables"]["single_model_tagging_metrics"]}
    fusion_runs = {row["run_id"] for row in report["tables"]["fusion_comparison"]}
    assert "D2" in single_runs
    assert "F1" not in single_runs
    assert "F1" in fusion_runs

    oracle_rows = report["tables"]["oracle_gaps"]
    assert {row["oracle_run_id"] for row in oracle_rows} == {"G0", "G1"}
    assert all(row["deployable"] is False for row in oracle_rows)
    assert all(row["metric"] == "model_val_accuracy" for row in oracle_rows)

    assert report["tables"]["state_prediction_metrics"][0]["run_id"] == "C0"
    assert report["tables"]["per_token_family_residual_metrics"][0]["token_family"] == "global"
    assert report["tables"]["per_field_residual_metrics"][0]["field"] == "pt_sum"
    assert any(row["comparison"] == "D2_vs_A0" and row["improvement"] > 0.0 for row in report["tables"]["control_gaps"])


def test_step9_missing_required_run_fails_unless_allowed(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_report(run_root, "A0", _base_report("A0"))

    report = build_canonical_state_report(
        CanonicalStateReportConfig(
            output_dir=tmp_path / "report",
            run_root=run_root,
            run_ids=("A0", "D2"),
            confirm_final_test=True,
        )
    )

    assert not report["ok"]
    assert any("missing required canonical-state run report for D2" in problem for problem in report["problems"])


def test_step9_provenance_consistency_is_enforced(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_report(run_root, "A0", _base_report("A0", manifest_hash="manifest-a"))
    _write_report(run_root, "D2", _base_report("D2", manifest_hash="manifest-b"))

    report = build_canonical_state_report(
        CanonicalStateReportConfig(
            output_dir=tmp_path / "report",
            run_root=run_root,
            run_ids=("A0", "D2"),
            allow_missing_runs=True,
            confirm_final_test=True,
        )
    )

    assert not report["ok"]
    assert any("disagree on manifest_hash" in problem for problem in report["problems"])


def test_step9_provenance_must_match_canonical_input_contract(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    wrong_profile = _base_report("A0")
    wrong_profile["input_contract"]["hlt_profile"] = "v1_legacy"
    wrong_profile["model_val_dataset"]["hlt_profile"] = "v1_legacy"
    wrong_profile["input_contract"]["label_filter"] = ["QCD", "Hbb"]
    _write_report(run_root, "A0", wrong_profile)

    report = build_canonical_state_report(
        CanonicalStateReportConfig(
            output_dir=tmp_path / "report",
            run_root=run_root,
            run_ids=("A0",),
            allow_missing_runs=True,
            confirm_final_test=True,
        )
    )

    assert not report["ok"]
    assert any("hlt_profile is 'v1_legacy'" in problem for problem in report["problems"])
    assert any("label_filter must be canonical integer class IDs" in problem for problem in report["problems"])


def test_step9_rejects_phi_hlt_bound_to_different_hlt_cache(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    stale_phi = _base_report("A0")
    stale_phi["model_val_dataset"]["phi_hlt_source_cache_hash"] = "old-hlt-content"
    _write_report(run_root, "A0", stale_phi)

    report = build_canonical_state_report(
        CanonicalStateReportConfig(
            output_dir=tmp_path / "report",
            run_root=run_root,
            run_ids=("A0",),
            allow_missing_runs=True,
            confirm_final_test=True,
        )
    )

    assert not report["ok"]
    assert any("phi_hlt_source_cache_hash does not match" in problem for problem in report["problems"])


def test_step9_rejects_failed_run_reports_and_unavailable_final_metrics(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    failed = _base_report("A0")
    failed["ok"] = False
    failed["final_test_metrics"] = {"available": False, "missing_inputs": ["D2"]}
    _write_report(run_root, "A0", failed)

    report = build_canonical_state_report(
        CanonicalStateReportConfig(
            output_dir=tmp_path / "report",
            run_root=run_root,
            run_ids=("A0",),
            allow_missing_runs=True,
            confirm_final_test=True,
        )
    )

    assert not report["ok"]
    assert any("run_report has ok=false" in problem for problem in report["problems"])
    assert any("final_test metrics are marked unavailable" in problem for problem in report["problems"])
    assert any("primary but missing final_test metrics" in problem for problem in report["problems"])


def test_step9_requires_final_test_phi_provenance_for_final_claims(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    missing_final_phi = _base_report("A0")
    missing_final_phi["final_test_dataset"].pop("phi_hlt_source_cache_hash")
    _write_report(run_root, "A0", missing_final_phi)

    report = build_canonical_state_report(
        CanonicalStateReportConfig(
            output_dir=tmp_path / "report",
            run_root=run_root,
            run_ids=("A0",),
            allow_missing_runs=True,
            confirm_final_test=True,
        )
    )

    assert not report["ok"]
    assert any("final_test metrics but is missing required provenance field final_test_phi_hlt_source_cache_hash" in problem for problem in report["problems"])


def test_step9_oracle_final_test_metrics_are_rejected(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_report(run_root, "A0", _base_report("A0"))
    oracle = _oracle_report("G0")
    oracle["final_test_metrics"] = {"accuracy": 0.99}
    _write_report(run_root, "G0", oracle)

    report = build_canonical_state_report(
        CanonicalStateReportConfig(
            output_dir=tmp_path / "report",
            run_root=run_root,
            run_ids=("A0", "G0"),
            allow_missing_runs=True,
            confirm_final_test=True,
        )
    )

    assert not report["ok"]
    assert any("policy model_val_only but includes final_test metrics" in problem for problem in report["problems"])


def test_step9_config_requires_explicit_final_test_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="confirm_final_test=True"):
        CanonicalStateReportConfig(output_dir=tmp_path / "report", run_root=tmp_path / "runs")
