"""Authenticated execution plans for genuine Stage-L confirmation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .confirmation import SEED_COMPONENT_KEYS
from .contracts import require_sha256, validate_content_hash
from .scale_execution import validate_execution_steps


CONFIRMATION_EXECUTION_PLAN_CONTRACT = (
    "retb_500k_confirmation_execution_plan_v1"
)


def _inside(path: str | Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "500k confirmation execution path escapes campaign"
        ) from error
    return resolved


def validate_confirmation_execution_plan(
    payload: Mapping[str, Any],
    *,
    campaign_source: Mapping[str, Any],
    campaign_root: str | Path,
    repo_root: str | Path,
) -> str:
    """Validate one complete, matched-pipeline-seed training plan."""

    digest = validate_content_hash(
        payload, expected_contract=CONFIRMATION_EXECUTION_PLAN_CONTRACT
    )
    required = {
        "contract",
        "schema_version",
        "graph_id",
        "pipeline_seed",
        "stage_l_graph_registry_sha256",
        "steps",
        "component_artifacts",
        "training_summary",
        "val_design_label_manifest_sha256",
        "source",
        "content_hash",
    }
    if (
        set(payload) != required
        or int(payload["schema_version"]) != 1
        or payload["source"] != campaign_source
        or int(payload["pipeline_seed"]) not in {101, 202, 303}
        or set(payload["component_artifacts"]) != set(SEED_COMPONENT_KEYS)
    ):
        raise ValueError("500k confirmation execution-plan semantics differ")
    require_sha256(
        payload["stage_l_graph_registry_sha256"],
        name="stage_l_graph_registry_sha256",
    )
    require_sha256(
        payload["val_design_label_manifest_sha256"],
        name="val_design_label_manifest_sha256",
    )
    root = Path(campaign_root).resolve()
    source = Path(repo_root).resolve()
    steps = validate_execution_steps(
        payload["steps"], campaign_root=root, repo_root=source
    )
    forbidden = {
        "execute_retb_500k_seed_confirmation.py",
        "register_retb_500k_seed_confirmation.py",
    }
    if any(Path(step["argv"][1]).name in forbidden for step in steps):
        raise ValueError(
            "500k confirmation plan uses a registration-only worker"
        )
    for path in payload["component_artifacts"].values():
        if _inside(path, root).suffix.lower() != ".json":
            raise ValueError(
                "500k component evidence must use JSON manifests"
            )
    _inside(payload["training_summary"], root)
    return digest


__all__ = [
    "CONFIRMATION_EXECUTION_PLAN_CONTRACT",
    "validate_confirmation_execution_plan",
]
