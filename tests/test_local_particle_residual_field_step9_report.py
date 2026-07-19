from __future__ import annotations

import csv
import json
from pathlib import Path

from teacher_logit_reco.local_particle_residual_field import (
    LOCAL_RESIDUAL_FIELD_REPORT_CONTRACT,
    LocalResidualFieldReportConfig,
    build_local_residual_field_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dataset(split: str, *, hlt_hash: str = "hlt-shared", target_hash: str = "target-shared") -> dict:
    return {
        "contract": "local_particle_residual_field_dataset_v1",
        "split": split,
        "n_jets": 100,
        "target_field_dim": 6,
        "field_names": ["pt.r0", "pt.r1", "mult.r0", "comp.ch", "flag.merge", "flag.rel"],
        "alignment_report": {
            "source_manifest_hash": "manifest-shared",
            "hlt_content_hash": hlt_hash,
            "offline_content_hash": "offline-shared",
            "target_content_hash": target_hash,
            "jet_identity_hash": f"identity-{split}",
        },
        "hlt_metadata": {
            "source_manifest_hash": "manifest-shared",
            "hlt_content_hash": hlt_hash,
            "jet_identity_hash": f"identity-{split}",
            "hlt_profile": "fixed_hlt_v2_realistic",
            "hlt_profile_version": "2",
            "hlt_degradation_strength": 2.5,
        },
        "target_metadata": {
            "target_content_hash": target_hash,
            "offline_content_hash": "offline-shared",
        },
    }


def _metrics(acc: float = 0.72, ce: float = 0.7) -> dict:
    return {
        "accuracy": acc,
        "cross_entropy": ce,
        "loss": ce,
        "macro_per_class_accuracy": acc - 0.01,
        "n_jets": 100,
        "per_class_accuracy": [{"class_index": 0, "accuracy": acc}],
        "confusion_matrix": [[70, 30], [20, 80]],
    }


def _tagger_report(run_id: str, *, acc: float = 0.72, hlt_hash: str = "hlt-shared") -> dict:
    return {
        "ok": True,
        "contract": "local_particle_residual_field_augmented_part_train_v1",
        "model_contract": "local_particle_residual_field_augmented_part_v1",
        "field_source": "hlt_only" if run_id == "A0" else "frozen_reconstructor",
        "best_epoch": 3,
        "selection_metric": "accuracy",
        "best_model_selection_metric_value": acc,
        "best_model_val": _metrics(acc=acc - 0.01, ce=0.75),
        "stack_val": _metrics(acc=acc, ce=0.72),
        "checkpoint": f"/fake/{run_id}/best_model_val.pt",
        "selected_field_names": ["pt.r0", "mult.r0"],
        "dataset_metadata": {
            "model_train": _dataset("model_train", hlt_hash=hlt_hash),
            "model_val": _dataset("model_val", hlt_hash=hlt_hash),
            "stack_val": _dataset("stack_val", hlt_hash=hlt_hash),
        },
    }


def _reconstructor_report(run_id: str = "C0") -> dict:
    return {
        "ok": True,
        "contract": "local_particle_residual_field_reconstructor_train_v1",
        "model_contract": "local_particle_residual_field_reconstructor_v1",
        "variant": run_id,
        "best_epoch": 2,
        "selection_metric": "mae",
        "best_model_selection_metric_value": 0.2,
        "best_model_val": {
            "mae": 0.20,
            "mse": 0.08,
            "zero_baseline_mae": 0.40,
            "zero_baseline_mse": 0.16,
            "relative_mae_vs_zero": 0.5,
            "n_jets": 100,
            "per_field_mae": {"pt.r0": 0.1},
        },
        "stack_val": {
            "mae": 0.21,
            "mse": 0.09,
            "zero_baseline_mae": 0.42,
            "zero_baseline_mse": 0.18,
            "relative_mae_vs_zero": 0.5,
            "n_jets": 100,
        },
        "checkpoint": f"/fake/{run_id}/best_model_val.pt",
        "selected_field_names": ["pt.r0", "mult.r0"],
        "dataset_metadata": {
            "model_train": _dataset("model_train"),
            "model_val": _dataset("model_val"),
            "stack_val": _dataset("stack_val"),
        },
    }


def _prediction_metadata(run_id: str, *, acc: float) -> dict:
    return {
        "contract": "local_particle_residual_field_predictions_v1",
        "model_name": run_id,
        "split": "final_test",
        "metrics": _metrics(acc=acc, ce=0.68),
        "dataset_metadata": _dataset("final_test"),
    }


def _write_campaign(
    root: Path,
    *,
    d5_hlt_hash: str = "hlt-shared",
    include_d5: bool = True,
    fusion_groups: tuple[str, ...] = ("G0", "G1", "G2", "G3"),
) -> None:
    _write_json(root / "taggers" / "A0" / "run_report.json", _tagger_report("A0", acc=0.70))
    if include_d5:
        _write_json(root / "taggers" / "D5" / "run_report.json", _tagger_report("D5", acc=0.74, hlt_hash=d5_hlt_hash))
    _write_json(root / "taggers" / "B1" / "run_report.json", _tagger_report("B1", acc=0.78))
    _write_json(root / "taggers" / "F0" / "run_report.json", _tagger_report("F0", acc=0.69))
    _write_json(root / "taggers" / "E6" / "run_report.json", _tagger_report("E6", acc=0.735))
    _write_json(root / "reconstructors" / "C0" / "run_report.json", _reconstructor_report("C0"))
    _write_json(root / "predictions" / "A0" / "final_test_predictions_metadata.json", _prediction_metadata("A0", acc=0.71))
    if include_d5:
        _write_json(root / "predictions" / "D5" / "final_test_predictions_metadata.json", _prediction_metadata("D5", acc=0.75))
    _write_json(root / "fusion" / "fusion_report.json", {"ok": True, "contract": "local_particle_residual_field_fusion_v1"})
    (root / "fusion").mkdir(parents=True, exist_ok=True)
    with (root / "fusion" / "fusion_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "mode", "split", "accuracy", "cross_entropy", "members"])
        writer.writeheader()
        for group in fusion_groups:
            writer.writerow(
                {
                    "group": group,
                    "mode": "uniform_logit_mean",
                    "split": "final_test",
                    "accuracy": "0.755",
                    "cross_entropy": "0.67",
                    "members": "A0 D5",
                }
            )


def test_step9_report_writes_all_tables_and_summary(tmp_path: Path) -> None:
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
            confirm_final_test=True,
            require_final_test_provenance=True,
        )
    )

    assert report["ok"] is True
    assert report["contract"] == LOCAL_RESIDUAL_FIELD_REPORT_CONTRACT
    for name in (
        "summary.md",
        "tagger_metrics.csv",
        "reconstructor_metrics.csv",
        "oracle_gap.csv",
        "control_gap.csv",
        "field_importance.csv",
        "fusion_metrics.csv",
        "provenance_audit.json",
        "run_report.json",
    ):
        assert (tmp_path / "final_report" / name).exists()
    summary = (tmp_path / "final_report" / "summary.md").read_text(encoding="utf-8")
    assert "Best stack-val tagger" in summary


def test_step9_report_fails_when_required_run_missing(tmp_path: Path) -> None:
    _write_campaign(tmp_path, include_d5=False)
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
        )
    )

    assert report["ok"] is False
    assert any("missing required tagger run_report for D5" in problem for problem in report["problems"])


def test_step9_report_rejects_tiny_finite_tagger_validation_subset(tmp_path: Path) -> None:
    _write_campaign(tmp_path)
    path = tmp_path / "taggers" / "D5" / "run_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["best_model_val"]["n_jets"] = 1
    payload["best_model_val"]["nonfinite_batches"] = 99
    _write_json(path, payload)

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
        )
    )

    assert report["ok"] is False
    assert any("tagger D5 model_val finite metric coverage 1/100" in problem for problem in report["problems"])


def test_step9_report_fails_when_required_fusion_groups_are_missing(tmp_path: Path) -> None:
    _write_campaign(tmp_path, fusion_groups=("G0",))
    report = build_local_residual_field_report(
        LocalResidualFieldReportConfig(
            output_dir=str(tmp_path / "final_report"),
            tagger_root=str(tmp_path / "taggers"),
            reconstructor_root=str(tmp_path / "reconstructors"),
            fusion_dir=str(tmp_path / "fusion"),
            prediction_dir=str(tmp_path / "predictions"),
            required_tagger_run_ids=("A0", "D5"),
            required_reconstructor_run_ids=("C0",),
            required_fusion_groups=("G0", "G1", "G2", "G3"),
            require_fusion=True,
        )
    )

    assert report["ok"] is False
    assert any("missing required fusion groups: G1 G2 G3" in problem for problem in report["problems"])


def test_step9_report_fails_on_provenance_mismatch(tmp_path: Path) -> None:
    _write_campaign(tmp_path, d5_hlt_hash="different-hlt-hash")
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
        )
    )

    assert report["ok"] is False
    assert any("provenance mismatch" in problem and "hlt_content_hash" in problem for problem in report["problems"])


def test_step9_report_fails_on_missing_required_provenance(tmp_path: Path) -> None:
    _write_campaign(tmp_path)
    path = tmp_path / "taggers" / "D5" / "run_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset_metadata"]["model_val"]["alignment_report"].pop("hlt_content_hash", None)
    payload["dataset_metadata"]["model_val"]["hlt_metadata"].pop("hlt_content_hash", None)
    _write_json(path, payload)

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
        )
    )

    assert report["ok"] is False
    assert any("missing required provenance model_val.hlt_content_hash" in problem for problem in report["problems"])


def test_step9_final_test_hlt_only_provenance_does_not_require_target_fields(tmp_path: Path) -> None:
    _write_campaign(tmp_path)
    hlt_only_metadata = {
        "contract": "local_particle_residual_field_predictions_v1",
        "model_name": "D5",
        "split": "final_test",
        "metrics": _metrics(acc=0.75, ce=0.68),
        "dataset_metadata": {
            "contract": "local_particle_residual_field_hlt_only_prediction_dataset_v1",
            "allowed_inputs": "HLT_particles_only_deployable_final_test",
            "split": "final_test",
            "n_jets": 100,
            "target_fields_present": False,
            "alignment_report": {
                "source_manifest_hash": "manifest-shared",
                "hlt_content_hash": "hlt-shared",
                "jet_identity_hash": "identity-final_test",
            },
            "hlt_metadata": {
                "source_manifest_hash": "manifest-shared",
                "hlt_content_hash": "hlt-shared",
                "jet_identity_hash": "identity-final_test",
                "hlt_profile": "fixed_hlt_v2_realistic",
                "hlt_profile_version": "2",
                "hlt_degradation_strength": 2.5,
            },
        },
    }
    _write_json(tmp_path / "predictions" / "D5" / "final_test_predictions_metadata.json", hlt_only_metadata)

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
            confirm_final_test=True,
            require_final_test_provenance=True,
        )
    )

    assert report["ok"] is True
