"""Reproduction gate for the already-opened A0/P7b exploratory final split."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.fusion import load_prediction_block, prediction_paths, validate_prediction_alignment
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .fusion_atomic import publish_temporary_file
from .fusion_campaign import FUSION_MEMBER_A0, FUSION_MEMBER_P7B, stable_fusion_json_hash
from .fusion_metrics import (
    local_residual_field_binary_projection_metrics,
    local_residual_field_complementarity_metrics,
    local_residual_field_multiclass_metrics,
)
from .fusion_seed_control import sha256_file
from .fusion_sources import require_fusion_source_artifact_audit


LOCAL_RESIDUAL_FIELD_FUSION_METRIC_REPRODUCTION_CONTRACT = (
    "local_residual_field_fusion_metric_reproduction_v1"
)
LOCAL_RESIDUAL_FIELD_FUSION_TEST_STATUS = "exploratory_previously_partially_opened"
LOCAL_RESIDUAL_FIELD_PILOT_FINAL_EXPECTATIONS = {
    FUSION_MEMBER_A0: {
        "accuracy": 0.74946,
        "hgg_qcd_50_false_positives": 271,
        "qcd_support": 15000,
        "hgg_qcd_50_rejection": 55.35055350553505,
    },
    FUSION_MEMBER_P7B: {
        "accuracy": 0.75328,
        "hgg_qcd_50_false_positives": 259,
        "qcd_support": 15000,
        "hgg_qcd_50_rejection": 57.915057915057915,
    },
}


def _atomic_json(path: str | Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    output = Path(path)
    if output.exists() and not bool(overwrite):
        raise FileExistsError(f"refusing to overwrite immutable artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        publish_temporary_file(temporary, output, overwrite=overwrite)
    finally:
        if temporary.exists():
            temporary.unlink()


def audit_local_residual_field_pilot_metric_reproduction(
    source_artifact_audit: str | Path,
    *,
    output_path: str | Path,
    confirm_exploratory_final_test: bool,
    expectations: Mapping[str, Mapping[str, Any]] = LOCAL_RESIDUAL_FIELD_PILOT_FINAL_EXPECTATIONS,
    accuracy_atol: float = 1.0e-12,
    rejection_atol: float = 1.0e-9,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Recompute the exact published pilot checks before fusion results are trusted."""

    if not bool(confirm_exploratory_final_test):
        raise ValueError("metric reproduction requires explicit confirmation of the already-opened final_test")
    audit = require_fusion_source_artifact_audit(source_artifact_audit)
    audit_config = audit.get("config")
    if not isinstance(audit_config, Mapping):
        raise ValueError("source-artifact audit does not contain its resolved config")
    roots = {
        FUSION_MEMBER_A0: audit_config.get("a0_prediction_dir"),
        FUSION_MEMBER_P7B: audit_config.get("p7b_prediction_dir"),
    }
    problems: list[str] = []
    member_reports: dict[str, Any] = {}
    blocks = []
    for member in (FUSION_MEMBER_A0, FUSION_MEMBER_P7B):
        root = roots[member]
        try:
            block = load_prediction_block(root, member, "final_test", verify_hash=True)
            blocks.append(block)
            if block.metadata.get("runtime_inputs") != "HLT_only" or any(
                bool(block.metadata.get(key))
                for key in ("uses_true_fields", "uses_offline_particles", "uses_teacher_logits_at_runtime")
            ) or block.metadata.get("deployable") is not True:
                raise ValueError(f"{member} final_test is not deployable HLT-only")
            multiclass = local_residual_field_multiclass_metrics(
                block.logits,
                block.labels,
                label_names=LABEL_NAMES,
            )
            binary = local_residual_field_binary_projection_metrics(
                block.logits,
                block.labels,
                label_names=LABEL_NAMES,
            )
            expected = dict(expectations[member])
            hgg50 = binary["projections"]["QCD_vs_Hgg"]["operating_points"]["signal_efficiency_0.50"]
            checks = {
                "accuracy": bool(
                    np.isclose(multiclass["accuracy"], float(expected["accuracy"]), atol=accuracy_atol, rtol=0.0)
                ),
                "hgg_qcd_50_false_positives": int(hgg50["qcd_false_positive_count"])
                == int(expected["hgg_qcd_50_false_positives"]),
                "qcd_support": int(hgg50["qcd_support"]) == int(expected["qcd_support"]),
                "hgg_qcd_50_rejection": bool(
                    np.isclose(
                        hgg50["background_rejection"],
                        float(expected["hgg_qcd_50_rejection"]),
                        atol=rejection_atol,
                        rtol=0.0,
                    )
                ),
                "all_nine_qcd_projections": len(binary["projections"]) == 9,
            }
            stored_metrics = block.metadata.get("metrics")
            if isinstance(stored_metrics, Mapping) and stored_metrics.get("accuracy") is not None:
                checks["stored_prediction_accuracy"] = bool(
                    np.isclose(
                        multiclass["accuracy"],
                        float(stored_metrics["accuracy"]),
                        atol=accuracy_atol,
                        rtol=0.0,
                    )
                )
            failed = [name for name, ok in checks.items() if not ok]
            if failed:
                problems.append(f"{member} reproduction checks failed: {failed}")
            npz_path, metadata_path = prediction_paths(root, member, "final_test")
            member_reports[member] = {
                "ok": not failed,
                "expectations": expected,
                "checks": checks,
                "prediction_path": str(npz_path.resolve()),
                "prediction_sha256": sha256_file(npz_path),
                "metadata_path": str(metadata_path.resolve()),
                "metadata_sha256": sha256_file(metadata_path),
                "checkpoint_hash": block.metadata.get("checkpoint_hash"),
                "multiclass": multiclass,
                "binary_projection": binary,
            }
        except Exception as exc:
            problems.append(f"{member} reproduction failed: {exc}")
    complementarity = None
    if len(blocks) == 2:
        try:
            validate_prediction_alignment(blocks)
            complementarity = local_residual_field_complementarity_metrics(
                blocks[0].logits,
                blocks[1].logits,
                blocks[0].labels,
                label_names=LABEL_NAMES,
                member_a=FUSION_MEMBER_A0,
                member_b=FUSION_MEMBER_P7B,
            )
        except Exception as exc:
            problems.append(f"A0/P7b final alignment or complementarity failed: {exc}")
    report = {
        "ok": not problems,
        "contract": LOCAL_RESIDUAL_FIELD_FUSION_METRIC_REPRODUCTION_CONTRACT,
        "test_status": LOCAL_RESIDUAL_FIELD_FUSION_TEST_STATUS,
        "selection_allowed": False,
        "source_artifact_audit": str(Path(source_artifact_audit).resolve()),
        "source_artifact_audit_path": str(Path(source_artifact_audit).resolve()),
        "source_artifact_audit_hash": audit["audit_hash"],
        "problems": problems,
        "members": member_reports,
        "complementarity": complementarity,
    }
    report["audit_hash"] = stable_fusion_json_hash(report)
    _atomic_json(output_path, report, overwrite=overwrite)
    return report


__all__ = [
    "LOCAL_RESIDUAL_FIELD_FUSION_METRIC_REPRODUCTION_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_FUSION_TEST_STATUS",
    "LOCAL_RESIDUAL_FIELD_PILOT_FINAL_EXPECTATIONS",
    "audit_local_residual_field_pilot_metric_reproduction",
]
