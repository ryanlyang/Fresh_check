"""Authenticated execution plans for genuine Stage-M refits and training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .contracts import require_sha256, validate_content_hash
from .scale_up import SCALE_COMPONENT_KEYS, SCALE_REFIT_KEYS


SCALE_REFIT_EXECUTION_PLAN_CONTRACT = (
    "retb_scale_refit_execution_plan_v1"
)
SCALE_GRAPH_EXECUTION_PLAN_CONTRACT = (
    "retb_scale_graph_execution_plan_v1"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_digest(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"Stage-M artifact is absent: {resolved}")
    if resolved.suffix.lower() == ".json":
        return validate_content_hash(
            json.loads(resolved.read_text(encoding="utf-8"))
        )
    return file_sha256(resolved)


def source_bound_artifact_digest(
    path: str | Path, *, campaign_source: Mapping[str, Any]
) -> str:
    """Hash a JSON manifest only after exact campaign-source validation."""

    resolved = Path(path)
    if resolved.suffix.lower() != ".json":
        raise ValueError(
            "component evidence must be a source-bound JSON manifest"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    digest = validate_content_hash(payload)
    if payload.get("source") != campaign_source:
        raise ValueError("component artifact source differs")
    return digest


def _inside(path: str | Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("Stage-M execution path escapes campaign") from error
    return resolved


def validate_execution_steps(
    steps: Any,
    *,
    campaign_root: Path,
    repo_root: Path,
    forbidden_terms: tuple[str, ...] = ("stack_val", "final_test"),
    forbidden_entrypoints: frozenset[str] = frozenset(
        {
            "execute_retb_scale_refits.py",
            "execute_retb_scale_graph_training.py",
            "train_retb_scale_shortlist.py",
            "register_retb_scale_refits.py",
        }
    ),
) -> list[dict[str, Any]]:
    if not isinstance(steps, list) or not steps:
        raise ValueError("Stage-M execution steps are empty")
    checked = []
    for index, raw in enumerate(steps):
        if set(raw) != {"step_id", "argv", "expected_outputs"}:
            raise ValueError("Stage-M execution-step fields differ")
        argv = [str(value) for value in raw["argv"]]
        outputs = [
            str(_inside(value, campaign_root))
            for value in raw["expected_outputs"]
        ]
        if (
            str(raw["step_id"]) != f"step_{index:03d}"
            or len(argv) < 2
            or Path(argv[0]).name.lower().startswith("python") is False
            or Path(argv[1]).is_absolute()
            or Path(argv[1]).suffix != ".py"
            or "--dry-run" in argv
            or not outputs
        ):
            raise ValueError("Stage-M execution-step semantics differ")
        entrypoint = (repo_root / argv[1]).resolve()
        try:
            entrypoint.relative_to(repo_root)
        except ValueError as error:
            raise ValueError("Stage-M entry point escapes repository") from error
        lowered = " ".join(argv).lower()
        if (
            not entrypoint.is_file()
            or any(term in lowered for term in forbidden_terms)
            or entrypoint.name in forbidden_entrypoints
        ):
            raise ValueError("Stage-M execution step is not a training/refit worker")
        checked.append(
            {
                "step_id": str(raw["step_id"]),
                "argv": argv,
                "expected_outputs": outputs,
            }
        )
    return checked


def validate_scale_refit_execution_plan(
    payload: Mapping[str, Any],
    *,
    campaign_source: Mapping[str, Any],
    campaign_root: str | Path,
    repo_root: str | Path,
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=SCALE_REFIT_EXECUTION_PLAN_CONTRACT
    )
    required = {
        "contract",
        "schema_version",
        "graph_id",
        "pipeline_seed",
        "locked_scale_shortlist_sha256",
        "scale_train_manifest_sha256",
        "val_design_identity_manifest_sha256",
        "five_hundred_k_artifact_hashes",
        "operations",
        "source",
        "content_hash",
    }
    operations = payload.get("operations", {})
    if (
        set(payload) != required
        or int(payload["schema_version"]) != 1
        or payload["source"] != campaign_source
        or int(payload["pipeline_seed"]) not in {101, 202, 303}
        or set(operations) != set(SCALE_REFIT_KEYS)
    ):
        raise ValueError("scale-refit execution plan semantics differ")
    root = Path(campaign_root).resolve()
    source = Path(repo_root).resolve()
    for name in SCALE_REFIT_KEYS:
        row = operations[name]
        if set(row) != {
            "population",
            "identity_manifest_sha256",
            "recipe_sha256",
            "replica_ids",
            "steps",
            "output_artifact",
        }:
            raise ValueError("scale-refit operation fields differ")
        validate_execution_steps(
            row["steps"], campaign_root=root, repo_root=source
        )
        _inside(row["output_artifact"], root)
        require_sha256(
            row["identity_manifest_sha256"],
            name=f"{name}.identity_manifest_sha256",
        )
        require_sha256(
            row["recipe_sha256"], name=f"{name}.recipe_sha256"
        )
    return digest


def validate_scale_graph_execution_plan(
    payload: Mapping[str, Any],
    *,
    campaign_source: Mapping[str, Any],
    campaign_root: str | Path,
    repo_root: str | Path,
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=SCALE_GRAPH_EXECUTION_PLAN_CONTRACT
    )
    required = {
        "contract",
        "schema_version",
        "graph_id",
        "pipeline_seed",
        "locked_scale_shortlist_sha256",
        "scale_refit_bundle_sha256",
        "steps",
        "component_artifacts",
        "training_summary",
        "pre_stack_metrics",
        "source",
        "content_hash",
    }
    if (
        set(payload) != required
        or int(payload["schema_version"]) != 1
        or payload["source"] != campaign_source
        or int(payload["pipeline_seed"]) not in {101, 202, 303}
        or set(payload["component_artifacts"])
        != set(SCALE_COMPONENT_KEYS)
    ):
        raise ValueError("scale-graph execution plan semantics differ")
    root = Path(campaign_root).resolve()
    validate_execution_steps(
        payload["steps"],
        campaign_root=root,
        repo_root=Path(repo_root).resolve(),
    )
    for path in payload["component_artifacts"].values():
        if _inside(path, root).suffix.lower() != ".json":
            raise ValueError(
                "scale component evidence must use JSON manifests"
            )
    _inside(payload["training_summary"], root)
    _inside(payload["pre_stack_metrics"], root)
    return digest


def execute_plan_steps(
    steps: list[Mapping[str, Any]],
    *,
    campaign_root: str | Path,
    repo_root: str | Path,
    forbidden_terms: tuple[str, ...] = ("stack_val", "final_test"),
    forbidden_entrypoints: frozenset[str] = frozenset(
        {
            "execute_retb_scale_refits.py",
            "execute_retb_scale_graph_training.py",
            "train_retb_scale_shortlist.py",
            "register_retb_scale_refits.py",
        }
    ),
) -> list[dict[str, Any]]:
    checked = validate_execution_steps(
        steps,
        campaign_root=Path(campaign_root).resolve(),
        repo_root=Path(repo_root).resolve(),
        forbidden_terms=forbidden_terms,
        forbidden_entrypoints=forbidden_entrypoints,
    )
    receipts = []
    for step in checked:
        existing = all(
            Path(path).is_file() and not Path(path).is_symlink()
            for path in step["expected_outputs"]
        )
        if not existing:
            completed = subprocess.run(
                step["argv"], cwd=repo_root, check=False
            )
            if completed.returncode:
                raise RuntimeError(
                    f"Stage-M step failed: {step['step_id']} "
                    f"({completed.returncode})"
                )
        hashes = {
            path: artifact_digest(path)
            for path in step["expected_outputs"]
        }
        receipts.append(
            {
                "step_id": step["step_id"],
                "reused": existing,
                "output_hashes": hashes,
            }
        )
    return receipts


__all__ = [
    "SCALE_GRAPH_EXECUTION_PLAN_CONTRACT",
    "SCALE_REFIT_EXECUTION_PLAN_CONTRACT",
    "artifact_digest",
    "execute_plan_steps",
    "file_sha256",
    "source_bound_artifact_digest",
    "validate_execution_steps",
    "validate_scale_graph_execution_plan",
    "validate_scale_refit_execution_plan",
]
