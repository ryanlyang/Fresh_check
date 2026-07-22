#!/usr/bin/env python3
"""Aggregate, confirm, report, and validate the Step 9 bridge campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    DeployableReplicaEvidence,
    aggregate_deployable_configuration,
    build_deployable_confirmation,
    build_step9_reports,
    finalize_deployable_confirmation,
    select_deployable_preconfirmation,
    validate_final_test_request,
    write_step9_decision_artifact,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import load_hashed_json  # noqa: E402


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="validate/print the action without publishing")
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select", help="aggregate paired replicas and lock pre-confirmation")
    select.add_argument("--registry", required=True)
    select.add_argument("--evidence", required=True, help="JSON object with a replicas list")
    select.add_argument("--output", required=True)

    confirm = sub.add_parser("confirm", help="finalize the one-shot stack_val_deploy result")
    confirm.add_argument("--preconfirmation", required=True)
    confirm.add_argument("--access-receipt", required=True)
    confirm.add_argument("--metrics", required=True)
    confirm.add_argument("--output", required=True)

    reports = sub.add_parser("reports", help="build the three disjoint Step 9 report tables")
    reports.add_argument("--registry", required=True)
    reports.add_argument("--evidence", required=True)
    reports.add_argument("--output", required=True)

    final = sub.add_parser("validate-final-test", help="fail closed unless final-test is HLT-only and unlocked")
    final.add_argument("--locked-deployable", required=True)
    final.add_argument("--clean-reload-audit", required=True)
    final.add_argument("--flags", help="optional JSON object of evaluation flags")
    return parser


def _replica(row: Mapping[str, Any]) -> dict[str, Any]:
    # The dataclass is the canonical validation/construction boundary.  Unknown
    # evidence keys are rejected instead of silently affecting selection.
    return DeployableReplicaEvidence(**dict(row)).to_artifact()


def main() -> int:
    args = _parser().parse_args()
    if args.command == "select":
        registry = load_hashed_json(args.registry)
        raw = _json(args.evidence)
        replicas = [_replica(row) for row in raw.get("replicas", [])]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in replicas:
            grouped.setdefault(str(row["run_id"]), []).append(row)
        aggregates = [
            aggregate_deployable_configuration(registry, rows)
            for _, rows in sorted(grouped.items())
        ]
        result = select_deployable_preconfirmation(registry, aggregates)
        action = "select_deployable_preconfirmation"
        output = args.output
    elif args.command == "confirm":
        pre = load_hashed_json(args.preconfirmation)
        receipt = load_hashed_json(args.access_receipt)
        metrics = _json(args.metrics)
        confirmation = build_deployable_confirmation(
            pre,
            access_receipt=receipt,
            deployable_gain=metrics["deployable_gain"],
            accuracy=metrics["accuracy"],
            cross_entropy=metrics["cross_entropy"],
            provenance_valid=bool(metrics["provenance_valid"]),
        )
        result = finalize_deployable_confirmation(pre, confirmation)
        action = "finalize_deployable_confirmation"
        output = args.output
    elif args.command == "reports":
        registry = load_hashed_json(args.registry)
        evidence = _json(args.evidence)
        result = build_step9_reports(
            registry,
            baseline_deployable_rows=evidence["baseline_deployable_rows"],
            privileged_rows=evidence["privileged_rows"],
            ablation_rows=evidence.get("ablation_rows", {}),
            run_outcomes=evidence["run_outcomes"],
            persistent_telemetry=evidence["persistent_telemetry"],
            ram_telemetry=evidence["ram_telemetry"],
        )
        action = "build_step9_reports"
        output = args.output
    else:
        locked = load_hashed_json(args.locked_deployable)
        audit = load_hashed_json(args.clean_reload_audit)
        flags = {} if args.flags is None else _json(args.flags)
        result = validate_final_test_request(
            locked, evaluation_flags=flags, clean_reload_audit=audit
        )
        print(json.dumps({"dry_run": bool(args.dry_run), "action": "validate_final_test", "result": result}, indent=2, sort_keys=True))
        return 0

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "action": action,
                    "output_would_be": str(Path(output).resolve()),
                    "result_content_hash": result["content_hash"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        publication = write_step9_decision_artifact(output, result)
        print(json.dumps({"dry_run": False, "action": action, "publication": publication}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
