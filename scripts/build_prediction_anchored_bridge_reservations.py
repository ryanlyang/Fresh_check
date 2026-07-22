#!/usr/bin/env python3
"""Build the immutable, filesystem-measured reservation ledger for production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import build_campaign_reservations  # noqa: E402
from teacher_logit_reco.local_particle_residual_field.bridge_campaign import (  # noqa: E402
    validate_campaign_registry,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    canonical_sha256,
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_execution import (  # noqa: E402
    PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
    validate_prediction_anchored_execution_spec,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--step8-measurement", required=True)
    parser.add_argument("--execution-spec", required=True)
    parser.add_argument(
        "--fixed-parent",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="measured immutable parent; repeat for R0, consumer, metadata, and other parents",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _measure(path_text: str) -> dict[str, object]:
    root = Path(path_text).resolve()
    if root.is_symlink() or not root.exists():
        raise FileNotFoundError(f"missing/unsafe fixed parent: {root}")
    files = [root] if root.is_file() else sorted(root.rglob("*"))
    if root.is_dir():
        if any(path.is_symlink() for path in files):
            raise ValueError(f"fixed parent contains a symlink: {root}")
        files = [path for path in files if path.is_file()]
    if not files:
        raise ValueError(f"fixed parent contains no files: {root}")
    rows = []
    size = 0
    for path in files:
        if path.is_symlink():
            raise ValueError(f"fixed parent contains a symlink: {path}")
        measured = int(path.stat().st_size)
        size += measured
        rows.append(
            {
                "path": path.name if root.is_file() else path.relative_to(root).as_posix(),
                "size_bytes": measured,
                "sha256": sha256_file(path),
            }
        )
    digest = rows[0]["sha256"] if root.is_file() else canonical_sha256(rows)
    return {"path": str(root), "size_bytes": size, "sha256": digest}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry = load_hashed_json(args.registry)
    validate_campaign_registry(registry)
    measurement = load_hashed_json(args.step8_measurement)
    validate_content_hash(
        measurement, expected_contract="prediction_anchored_step8_registry_measurement_v1"
    )
    execution_spec = load_hashed_json(
        args.execution_spec, expected_contract=PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT
    )
    validate_prediction_anchored_execution_spec(execution_spec, verify_file_hashes=True)
    if measurement.get("updated_registry_sha256") != registry["content_hash"]:
        raise ValueError("Step 8 measurement belongs to another measured registry")
    if measurement.get("source_manifest_sha256") != execution_spec["parent_manifest"]["sha256"]:
        raise ValueError("Step 8 measurement belongs to another execution-spec source manifest")
    fixed = measurement.get("fixed_storage", {})
    validate_content_hash(fixed, expected_contract="prediction_anchored_step8_fixed_storage_v1")
    parents = {}
    for binding in args.fixed_parent:
        if "=" not in binding:
            raise ValueError("--fixed-parent must use ROLE=PATH")
        role, path = binding.split("=", 1)
        if not role.strip() or not path.strip() or role in parents:
            raise ValueError("fixed-parent roles must be nonempty and unique")
        parents[role] = _measure(path)
    if not parents:
        raise ValueError("at least one --fixed-parent ROLE=PATH is required")
    artifact = build_campaign_reservations(
        registry,
        execution_spec=execution_spec,
        production_readiness=measurement["production_readiness"],
        fixed_parent_artifacts=parents,
        final_deployable_bundle_bytes=int(fixed["final_deployable_bundle_bytes"]),
    )
    publication = None
    if not args.dry_run:
        if not args.output:
            raise ValueError("--output is required unless --dry-run is used")
        publication = write_immutable_json(args.output, artifact)
    print(json.dumps({"dry_run": bool(args.dry_run), "reservations": artifact,
                      "publication": publication}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
