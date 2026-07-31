#!/usr/bin/env python3
"""Audit HOSD source capabilities and compile the immutable target registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    PARENT_STATUS_CONTRACT,
    authorize_access,
    build_structure_target_registry,
    build_target_capability_audit,
    load_and_validate_campaign,
    load_hashed_json,
    require_parents_ready,
    validate_structure_target_registry,
    validate_target_capability_audit,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    RAW_INPUT_SCHEMA_CONTRACT,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--raw-input-schema", type=Path)
    parser.add_argument("--resolved-parent-lock", type=Path)
    parser.add_argument("--capability-output", type=Path)
    parser.add_argument("--registry-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    schema_path = args.raw_input_schema or root / "inputs" / "raw_input_schema.json"
    parent_path = args.resolved_parent_lock or (
        root / "inputs" / "resolved_inherited_parent_lock.json"
    )
    authorize_access(
        worker_role="target_builder",
        requested_resource="authenticated_raw_schema",
    )
    raw_schema = load_hashed_json(
        schema_path, expected_contract=RAW_INPUT_SCHEMA_CONTRACT
    )
    authorize_access(
        worker_role="target_builder",
        requested_resource="resolved_inherited_parent_lock",
    )
    parents = load_hashed_json(
        parent_path, expected_contract=PARENT_STATUS_CONTRACT
    )
    if parents.get("source") != campaign.get("source"):
        raise ValueError("resolved parent lock source differs from campaign")
    require_parents_ready(parents, before_stage="B")

    audit = build_target_capability_audit(
        raw_input_schema=raw_schema,
        source=campaign["source"],
    )
    registry = build_structure_target_registry(
        capability_audit=audit,
        raw_input_schema=raw_schema,
        source=campaign["source"],
    )
    validate_target_capability_audit(
        audit, raw_input_schema=raw_schema, source=campaign["source"]
    )
    validate_structure_target_registry(
        registry,
        capability_audit=audit,
        raw_input_schema=raw_schema,
        source=campaign["source"],
    )

    capability_output = args.capability_output or (
        root / "capability" / "target_capability_audit.json"
    )
    registry_output = args.registry_output or (
        root / "registry" / "structure_target_registry.json"
    )
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "campaign_spec_sha256": campaign["content_hash"],
        "resolved_parent_lock_sha256": parents["content_hash"],
        "raw_input_schema_sha256": raw_schema["content_hash"],
        "target_capability_audit_sha256": audit["content_hash"],
        "structure_target_registry_sha256": registry["content_hash"],
        "current_executable_target_count": len(
            registry["current_executable_target_ids"]
        ),
        "future_unavailable_target_ids": registry[
            "future_unavailable_target_ids"
        ],
        "capability_output": str(capability_output.resolve()),
        "registry_output": str(registry_output.resolve()),
        "scientific_outputs_created": False,
    }
    if not args.dry_run:
        authorize_access(
            worker_role="target_builder",
            requested_resource="capability_audit",
            operation="write",
        )
        result["capability_publication"] = write_immutable_json(
            capability_output, audit
        )
        authorize_access(
            worker_role="target_builder",
            requested_resource="target_registry",
            operation="write",
        )
        result["registry_publication"] = write_immutable_json(
            registry_output, registry
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
