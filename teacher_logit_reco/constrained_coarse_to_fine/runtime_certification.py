"""Fail-closed ten/30-epoch reconstructor certification for runtime promotion."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .runtime_profiles import resolve_execution, validate_runtime_profile


RUNTIME_RECONSTRUCTOR_CERTIFICATION_CONTRACT = "constrained_c2f_runtime_reconstructor_certification_v1"


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metrics(report: Mapping[str, Any]) -> Mapping[str, Any]:
    value = report.get("best_model_val")
    if not isinstance(value, Mapping):
        raise ValueError("run report lacks best_model_val metrics")
    return value


def _skip_count(report: Mapping[str, Any], run_dir: Path) -> int:
    curves = _read(run_dir / "training_curves.json")
    epochs = curves.get("epochs")
    if not isinstance(epochs, list):
        raise ValueError("training curves lack epochs")
    skipped = 0
    for epoch in epochs:
        if not isinstance(epoch, Mapping):
            raise ValueError("training curves contain malformed epoch")
        train = epoch.get("train")
        if not isinstance(train, Mapping):
            raise ValueError("training curves lack train metrics")
        value = _finite(train.get("nonfinite_batches_skipped", 0))
        if value is None or value < 0:
            raise ValueError("training curves have invalid skipped-batch telemetry")
        skipped += int(value)
    return skipped


def _validate_run(
    *,
    run_dir: Path,
    expected_variant: str,
    expected_profile: str,
    expected_epochs: int,
    expected_runtime_hash: str | None,
    candidate_environment: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    report_path = run_dir / "run_report.json"
    report = _read(report_path)
    if not bool(report.get("ok")):
        raise ValueError(f"{run_dir} run_report is not ok")
    if report.get("variant") != expected_variant:
        raise ValueError(f"{run_dir} variant mismatch")
    runtime = report.get("runtime_profile")
    if not isinstance(runtime, Mapping) or runtime.get("name") != expected_profile:
        raise ValueError(f"{run_dir} runtime profile mismatch")
    if expected_runtime_hash is not None and report.get("runtime_profile_hash") != expected_runtime_hash:
        raise ValueError(f"{run_dir} runtime profile hash mismatch")
    config = report.get("training_config")
    if not isinstance(config, Mapping) or config.get("fixed_horizon") is not True:
        raise ValueError(f"{run_dir} is not a fixed-horizon certification run")
    if int(report.get("completed_epochs", -1)) != expected_epochs or int(config.get("epochs", -1)) != expected_epochs:
        raise ValueError(f"{run_dir} did not complete the required {expected_epochs} epochs")
    checkpoint = run_dir / "best_model_val.pt"
    if not checkpoint.is_file() or _sha256(checkpoint) != report.get("checkpoint_sha256"):
        raise ValueError(f"{run_dir} best checkpoint integrity mismatch")
    if report.get("code_environment") != candidate_environment:
        raise ValueError(f"{run_dir} code environment differs from candidate profile")
    if _skip_count(report, run_dir) != 0:
        raise ValueError(f"{run_dir} has skipped non-finite batches")
    return report, {
        "run_report_path": str(report_path.resolve()),
        "run_report_sha256": _sha256(report_path),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
    }


def _comparison(candidate: float, reference: float, *, higher_is_better: bool, relative_tolerance: float) -> dict[str, Any]:
    allowed = reference * (1.0 - relative_tolerance) if higher_is_better else reference * (1.0 + relative_tolerance)
    return {
        "candidate": candidate, "reference": reference, "allowed": allowed,
        "higher_is_better": higher_is_better,
        "ok": candidate >= allowed if higher_is_better else candidate <= allowed,
    }


def validate_reconstructor_certification(
    *,
    path: str,
    mode: str,
    candidate_profile_path: str | Path,
    candidate_run_dir: str | Path,
    fp32_reference_run_dir: str | Path,
) -> dict[str, Any]:
    """Recompute every 10/30-epoch promotion condition from source artifacts."""

    mode = str(mode)
    if mode not in {"ten_epoch_certification", "fp32_reference_promotion"}:
        raise ValueError("mode must be ten_epoch_certification or fp32_reference_promotion")
    candidate = validate_runtime_profile(candidate_profile_path, expected_status="accelerated_candidate_v1")
    profile = candidate["profile"]
    execution = resolve_execution(profile, path)
    candidate_epochs = 10
    reference_epochs = 10 if mode == "ten_epoch_certification" else 30
    candidate_report, candidate_files = _validate_run(
        run_dir=Path(candidate_run_dir), expected_variant=path,
        expected_profile="accelerated_candidate_v1", expected_epochs=candidate_epochs,
        expected_runtime_hash=str(execution["runtime_profile_hash"]), candidate_environment=profile["code_environment"],
    )
    reference_report, reference_files = _validate_run(
        run_dir=Path(fp32_reference_run_dir), expected_variant=path,
        expected_profile="fp32_reference", expected_epochs=reference_epochs,
        expected_runtime_hash=None, candidate_environment=profile["code_environment"],
    )
    if candidate_report.get("provenance") != reference_report.get("provenance"):
        raise ValueError("candidate and FP32 reference provenance differs")
    candidate_metrics, reference_metrics = _metrics(candidate_report), _metrics(reference_report)
    comparisons: dict[str, Any] = {}
    required = ["selection.reconstruction_score"]
    diagnostics = sorted(
        key for key in candidate_metrics
        if key.startswith(("hierarchy.metric.", "slot.metric.")) and _finite(candidate_metrics[key]) is not None
    )
    required.extend(key for key in diagnostics if _finite(reference_metrics.get(key)) is not None)
    for key in required:
        left, right = _finite(candidate_metrics.get(key)), _finite(reference_metrics.get(key))
        if left is None or right is None:
            raise ValueError(f"missing or non-finite promotion metric {key}")
        comparisons[key] = _comparison(
            left, right,
            higher_is_better=key.endswith("matched_pid_accuracy"),
            relative_tolerance=0.01 if key == "selection.reconstruction_score" else 0.02,
        )
    report = {
        "contract": RUNTIME_RECONSTRUCTOR_CERTIFICATION_CONTRACT,
        "ok": all(row["ok"] for row in comparisons.values()),
        "promotion_evidence_kind": mode,
        "promotion_gate": {"ok": all(row["ok"] for row in comparisons.values()), "comparisons": comparisons},
        "path": path,
        "candidate_profile_path": candidate["path"],
        "candidate_profile_file_sha256": candidate["file_sha256"],
        "candidate_profile_hash": profile["candidate_profile_hash"],
        "code_environment_hash": profile["code_environment_hash"],
        "candidate": candidate_files,
        "fp32_reference": reference_files,
        "candidate_epochs": candidate_epochs,
        "fp32_reference_epochs": reference_epochs,
        "final_test_loaded": False,
    }
    return report


def write_reconstructor_certification(
    *,
    path: str,
    mode: str,
    candidate_profile_path: str | Path,
    candidate_run_dir: str | Path,
    fp32_reference_run_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate actual checkpoints/curves and persist promotion evidence."""

    report = validate_reconstructor_certification(
        path=path,
        mode=mode,
        candidate_profile_path=candidate_profile_path,
        candidate_run_dir=candidate_run_dir,
        fp32_reference_run_dir=fp32_reference_run_dir,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = [
    "RUNTIME_RECONSTRUCTOR_CERTIFICATION_CONTRACT",
    "validate_reconstructor_certification",
    "write_reconstructor_certification",
]
