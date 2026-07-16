from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from teacher_logit_reco.adaptive_binary_pseudooffline import (
    ABPH_CAMPAIGN_REPORT_CONTRACT,
    ABPH_EXPECTED_VARIANT_NAMES,
    ABPH_FINAL_CLAIM_CONTRACT,
    ABPH_FUSION_CANDIDATES,
    ABPH_POSTHOC_VARIANTS,
    ABPH_SLURM_ORCHESTRATION_CONTRACT,
    ABPH_STEP4_PREFLIGHT_CONTRACT,
    AdaptiveBinarySubmissionConfig,
    build_submission_graph,
    canonical_hash,
    freeze_final_claim_contract,
    load_final_claim_contract,
    require_actual_target_preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _executor_environment(tmp_path: Path) -> dict[str, str]:
    result = os.environ.copy()
    for name, filename in (
        ("ABPH_VARIANT_EXECUTOR", "variant_executor.py"),
        ("ABPH_PREDICTION_EXECUTOR", "prediction_executor.py"),
        ("ABPH_DIAGNOSTIC_EXECUTOR", "diagnostic_executor.py"),
    ):
        path = tmp_path / filename
        path.write_text("raise SystemExit(0)\n", encoding="utf-8")
        result[name] = str(path)
    return result


def _selection_report(
    path: Path,
    *,
    ok: bool = True,
    campaign_root: Path | None = None,
) -> Path:
    payload = {
        "contract": ABPH_CAMPAIGN_REPORT_CONTRACT,
        "ok": bool(ok),
        "problems": [] if ok else ["failed"],
        "campaign_root": str(campaign_root or path.parent),
        "required_variants": list(ABPH_EXPECTED_VARIANT_NAMES),
        "final_test_policy": {"confirmed": False, "claim_variants": []},
        "metrics": [],
        "provenance": [],
        "root_identity": [],
        "fusion_membership": [],
    }
    payload["report_content_hash"] = canonical_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _actual_target_preflight(path: Path) -> Path:
    reports = {}
    for split in ("model_train", "model_val"):
        for grouping in ("exclusive_kt", "cambridge_aachen"):
            reports[f"{split}/{grouping}"] = {
                "ok": True,
                "compiler_failure_count": 0,
                "class_counts": {str(index): 2 for index in range(10)},
            }
    payload = {
        "contract": ABPH_STEP4_PREFLIGHT_CONTRACT,
        "ok": True,
        "problems": [],
        "reports": reports,
        "synthetic_edge_cases": {"ok": True},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_full_graph_is_complete_and_hard_gated(tmp_path: Path) -> None:
    config = AdaptiveBinarySubmissionConfig(
        campaign_root=tmp_path / "campaign",
        data_dir=tmp_path / "data",
    )
    graph = build_submission_graph(config)
    keys = [job.key for job in graph]
    assert len(graph) == 81
    assert keys[:4] == ["input:splits", "input:hlt_cache", "input:offline_cache", "input:audit"]
    preflight_index = keys.index("preflight:actual_targets")
    for index, job in enumerate(graph):
        if job.stage in {"reconstructor", "renderer"}:
            assert index > preflight_index
            assert "preflight:actual_targets" in job.dependencies
    report = graph[-1]
    assert report.key == "report:model_selection"
    assert len(report.dependencies) == 73
    expected_reports = {
        f"baseline:{name}" if name.startswith("A") else f"variant:{name}"
        for name in ABPH_EXPECTED_VARIANT_NAMES
        if name not in ABPH_POSTHOC_VARIANTS
    }
    assert expected_reports.issubset(set(report.dependencies))
    assert "variant:F0_ce_reco_primary__seed2" in keys
    assert "variant:F0_ce_reco_primary__seed3" in keys
    assert "teacher_logits:A4_offline_part_ceiling" in keys
    f4 = next(job for job in graph if job.key == "variant:F4_ce_logit_kd")
    assert "teacher_logits:A4_offline_part_ceiling" in f4.dependencies


def test_actual_target_preflight_fails_closed(tmp_path: Path) -> None:
    path = _actual_target_preflight(tmp_path / "preflight.json")
    assert require_actual_target_preflight(path)["ok"] is True
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["reports"]["model_val/exclusive_kt"]["compiler_failure_count"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="compiler failures"):
        require_actual_target_preflight(path)


def test_highdata_approval_is_canonical(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="approval"):
        AdaptiveBinarySubmissionConfig(
            campaign_root=tmp_path / "highdata",
            data_dir=tmp_path / "data",
            campaign_mode="highdata",
        )
    report = _selection_report(tmp_path / "pilot" / "final_report.json")
    config = AdaptiveBinarySubmissionConfig(
        campaign_root=tmp_path / "highdata",
        data_dir=tmp_path / "data",
        campaign_mode="highdata",
        approve_highdata=True,
        pilot_report_path=report,
    )
    assert build_submission_graph(config)[0].key == "input:splits"


def test_final_claim_contract_expands_frozen_fusion_members(tmp_path: Path) -> None:
    highdata_root = tmp_path / "highdata"
    report = _selection_report(
        tmp_path / "selection" / "final_report.json",
        campaign_root=highdata_root,
    )
    contract_path = tmp_path / "selection" / "final_claim_contract.json"
    payload = freeze_final_claim_contract(
        report,
        contract_path,
        claim_variants=("A0_hlt_part", "G5_best_complementary_ensemble"),
        fusion_artifact_hashes={"G5_best_complementary_ensemble": "abc"},
        fusion_memberships={
            "G5_best_complementary_ensemble": (
                "E5_kt32_mh4_dualcross",
                "E6_ca32_mh4_dualcross",
                "E7_dual_hierarchy_dualcross",
            )
        },
    )
    assert payload["contract"] == ABPH_FINAL_CLAIM_CONTRACT
    assert load_final_claim_contract(contract_path, selection_report_path=report)
    config = AdaptiveBinarySubmissionConfig(
        campaign_root=highdata_root,
        data_dir=tmp_path / "data",
        campaign_mode="highdata",
        stage_mode="final_claims",
        approve_final_test=True,
        selection_report_path=report,
        final_claim_contract_path=contract_path,
        confirm_final_test=True,
        rebuild_inputs=False,
        rebuild_targets=False,
        rebuild_models=False,
    )
    # Bypass only the filesystem reuse check here; graph construction itself must
    # expand the frozen fusion to its exact member predictions.
    graph = build_submission_graph(config)
    keys = {job.key for job in graph}
    assert "final_prediction:G5_best_complementary_ensemble" not in keys
    assert "final_prediction:E5_kt32_mh4_dualcross" in keys
    fusion = next(job for job in graph if job.key == "final_fusion:G5_best_complementary_ensemble")
    assert fusion.dependencies == (
        "final_prediction:E5_kt32_mh4_dualcross",
        "final_prediction:E6_ca32_mh4_dualcross",
        "final_prediction:E7_dual_hierarchy_dualcross",
    )


@pytest.mark.parametrize("stage", ["predictions", "fusion", "diagnostics", "report"])
def test_partial_stage_rejects_upstream_rebuild(stage: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot rebuild"):
        AdaptiveBinarySubmissionConfig(
            campaign_root=tmp_path / stage,
            data_dir=tmp_path / "data",
            stage_mode=stage,
            rebuild_inputs=True,
            rebuild_targets=False,
            rebuild_models=False,
        )


def test_canonical_submitter_executes_full_tigris_dry_run(tmp_path: Path) -> None:
    manifest = tmp_path / "submission.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "submit_adaptive_binary_pseudooffline.py"),
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--data-dir",
            str(tmp_path / "data"),
            "--cluster",
            "tigris",
            "--project-dir",
            str(REPO_ROOT),
            "--dry-run",
            "--output-manifest",
            str(manifest),
        ],
        cwd=REPO_ROOT,
        env=_executor_environment(tmp_path),
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.returncode == 0
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["contract"] == ABPH_SLURM_ORCHESTRATION_CONTRACT
    assert len(payload["jobs"]) == 81
    assert payload["resource_profile"]["account"] == "reu-aisocial"
    assert all("--account=reu-aisocial" in row["command"] for row in payload["submission_commands"])
    assert all(row["environment"].get("ABPH_CONFIRM_FINAL_TEST") == "0" for row in payload["jobs"])
    assert set(payload["runtime_executors"]) == {
        "ABPH_VARIANT_EXECUTOR",
        "ABPH_PREDICTION_EXECUTOR",
        "ABPH_DIAGNOSTIC_EXECUTOR",
    }


def test_approved_highdata_cli_executes_with_quarter_independent_graph(tmp_path: Path) -> None:
    pilot_report = _selection_report(tmp_path / "pilot" / "final_report.json")
    output = tmp_path / "highdata_submission.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "submit_adaptive_binary_pseudooffline.py"),
            "--campaign-root",
            str(tmp_path / "highdata"),
            "--data-dir",
            str(tmp_path / "data"),
            "--campaign-mode",
            "highdata",
            "--approve-highdata",
            "--pilot-report",
            str(pilot_report),
            "--cluster",
            "tier3",
            "--project-dir",
            str(REPO_ROOT),
            "--dry-run",
            "--output-manifest",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=_executor_environment(tmp_path),
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["campaign_mode"] == "highdata"
    assert payload["split_sizes"]["model_train"] == 5_000_000
    assert payload["resource_profile"]["partition"] == "tier3"
    assert len(payload["jobs"]) == 81


def test_selection_report_hash_is_recomputed(tmp_path: Path) -> None:
    path = _selection_report(tmp_path / "report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["campaign_root"] = "changed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        AdaptiveBinarySubmissionConfig(
            campaign_root=tmp_path / "highdata",
            data_dir=tmp_path / "data",
            campaign_mode="highdata",
            approve_highdata=True,
            pilot_report_path=path,
        )


def _run_cli(root: Path, stage: str, output: Path, tmp_path: Path) -> dict[str, object]:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "submit_adaptive_binary_pseudooffline.py"),
            "--campaign-root",
            str(root),
            "--data-dir",
            str(root / "data"),
            "--stage-mode",
            stage,
            "--project-dir",
            str(REPO_ROOT),
            "--dry-run",
            "--output-manifest",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=_executor_environment(tmp_path),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_prediction_only_cli_executes_against_reused_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    audit = root / "audits" / "step1_input_audit.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({"ok": True}), encoding="utf-8")
    _actual_target_preflight(root / "audits" / "actual_target_feasibility.json")
    for member in ("D1_kt32_mh4_particles", "D2_ca32_mh4_particles"):
        run = root / "runs" / member
        run.mkdir(parents=True)
        (run / "run_report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        (run / "best_model_val.pt").write_bytes(b"checkpoint")
    payload = _run_cli(
        root,
        "predictions",
        tmp_path / "prediction_submission.json",
        tmp_path,
    )
    assert [row["stage"] for row in payload["jobs"]] == [
        "pseudo_prediction",
        "pseudo_prediction",
        "pseudo_prediction",
    ]
    assert payload["jobs"][-1]["arguments"][0] == "E7_shared_root_dual"
    assert payload["reuse_preflight"]["checked"]


def test_fusion_only_cli_executes_against_reused_predictions(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    members = tuple(
        dict.fromkeys(member for values in ABPH_FUSION_CANDIDATES.values() for member in values)
    )
    for member in members:
        directory = root / "logit_predictions" / member
        directory.mkdir(parents=True)
        for split in ("stack_train", "stack_val"):
            (directory / f"{split}.npz").write_bytes(b"prediction")
            (directory / f"{split}_metadata.json").write_text("{}", encoding="utf-8")
    payload = _run_cli(
        root,
        "fusion",
        tmp_path / "fusion_submission.json",
        tmp_path,
    )
    assert len(payload["jobs"]) == 4
    assert all(row["stage"] == "fusion" for row in payload["jobs"])
    assert payload["reuse_preflight"]["checked"]


def test_canonical_submitter_fails_before_sbatch_when_runtime_is_missing(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    for name in (
        "ABPH_VARIANT_EXECUTOR",
        "ABPH_PREDICTION_EXECUTOR",
        "ABPH_DIAGNOSTIC_EXECUTOR",
    ):
        environment[name] = str(tmp_path / f"missing_{name}.py")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "submit_adaptive_binary_pseudooffline.py"),
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--data-dir",
            str(tmp_path / "data"),
            "--project-dir",
            str(REPO_ROOT),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "requires ABPH_VARIANT_EXECUTOR" in result.stderr
    assert not (tmp_path / "campaign" / "submission_logs").exists()
