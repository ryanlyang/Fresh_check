#!/usr/bin/env python3
"""Run or attest the ordered RETB preproduction validation gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge import (  # noqa: E402
    JOB_LEDGER_CONTRACT,
    PRODUCTION_GRAPH_CONTRACT,
    build_job_ledger,
    build_production_graph,
    build_full_submission_authorization,
    build_production_dry_run_evidence,
    build_tigris_smoke_evidence,
    load_hashed_json,
    run_local_synthetic_dag,
    source_snapshot,
    validate_full_submission_authorization,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    CAMPAIGN_SPEC_CONTRACT,
    bind_source,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.operational_validation import (  # noqa: E402
    LOCAL_OPERATIONAL_REPORT_CONTRACT,
    PRODUCTION_DRY_RUN_EVIDENCE_CONTRACT,
    TIGRIS_SMOKE_EVIDENCE_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.storage import (  # noqa: E402
    STORAGE_MEASUREMENTS_CONTRACT,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    local = sub.add_parser("local")
    local.add_argument("--campaign-root", required=True, type=Path)
    local.add_argument("--output", type=Path)
    smoke = sub.add_parser("attest-tigris-smoke")
    smoke.add_argument("--campaign-root", required=True, type=Path)
    smoke.add_argument("--completed-ledger", required=True, type=Path)
    smoke.add_argument("--output", required=True, type=Path)
    dry = sub.add_parser("attest-production-dry-run")
    dry.add_argument("--production-graph", required=True, type=Path)
    dry.add_argument("--dry-run-ledger", required=True, type=Path)
    dry.add_argument("--storage-measurements", required=True, type=Path)
    dry.add_argument("--output", required=True, type=Path)
    prepare = sub.add_parser("production-dry-run")
    prepare.add_argument("--storage-measurements", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    authorize = sub.add_parser("authorize-full-submission")
    authorize.add_argument("--local-report", required=True, type=Path)
    authorize.add_argument("--tigris-smoke-evidence", required=True, type=Path)
    authorize.add_argument(
        "--production-dry-run-evidence", required=True, type=Path
    )
    authorize.add_argument("--output", required=True, type=Path)
    verify = sub.add_parser("verify-authorization")
    verify.add_argument("--authorization", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    snapshot = source_snapshot(REPO_ROOT)
    result: dict[str, object]
    if args.command == "local":
        report = run_local_synthetic_dag(
            campaign_root=args.campaign_root, repo_root=REPO_ROOT
        )
        output = args.output or (
            args.campaign_root
            / "operational_validation"
            / "local_report.json"
        )
        result = {
            "local_operational_report_sha256": report["content_hash"],
            "publication": write_immutable_json(output, report),
        }
    elif args.command == "attest-tigris-smoke":
        campaign = load_hashed_json(
            args.campaign_root / "campaign_spec.json",
            expected_contract=CAMPAIGN_SPEC_CONTRACT,
        )
        graph = load_hashed_json(
            args.campaign_root / "job_ledgers" / "production_graph.json",
            expected_contract=PRODUCTION_GRAPH_CONTRACT,
        )
        ledger = load_hashed_json(
            args.completed_ledger, expected_contract=JOB_LEDGER_CONTRACT
        )
        evidence = build_tigris_smoke_evidence(
            campaign_spec=campaign,
            production_graph=graph,
            completed_ledger=ledger,
            source_snapshot=snapshot,
        )
        result = {
            "tigris_smoke_evidence_sha256": evidence["content_hash"],
            "publication": write_immutable_json(args.output, evidence),
        }
    elif args.command in {
        "attest-production-dry-run",
        "production-dry-run",
    }:
        if args.command == "production-dry-run":
            storage = load_hashed_json(
                args.storage_measurements,
                expected_contract=STORAGE_MEASUREMENTS_CONTRACT,
            )
            bound_storage_sha = bind_source(
                storage, source_snapshot=snapshot
            )["content_hash"]
            campaign_id = (
                "retb_production_dry_run_"
                f"{str(snapshot['source_commit'])[:10]}_"
                f"{str(snapshot['source_status_sha256'])[:10]}"
            )
            graph = build_production_graph(
                campaign_root=args.output_dir / campaign_id,
                campaign_id=campaign_id,
                source_commit=str(snapshot["source_commit"]),
                source_status_sha256=str(
                    snapshot["source_status_sha256"]
                ),
                storage_measurements_sha256=bound_storage_sha,
                miniature=False,
            )
            ledger = build_job_ledger(
                production_graph=graph,
                jobs={
                    node["node_id"]: None for node in graph["nodes"]
                },
                submission_mode="dry_run",
            )
        else:
            graph = load_hashed_json(
                args.production_graph,
                expected_contract=PRODUCTION_GRAPH_CONTRACT,
            )
            ledger = load_hashed_json(
                args.dry_run_ledger, expected_contract=JOB_LEDGER_CONTRACT
            )
            storage = load_hashed_json(
                args.storage_measurements,
                expected_contract=STORAGE_MEASUREMENTS_CONTRACT,
            )
        evidence = build_production_dry_run_evidence(
            production_graph=graph,
            dry_run_ledger=ledger,
            storage_measurements=storage,
            source_snapshot=snapshot,
        )
        output = (
            args.output
            if args.command == "attest-production-dry-run"
            else args.output_dir / "production_dry_run_evidence.json"
        )
        publications = {
            "evidence": write_immutable_json(output, evidence)
        }
        if args.command == "production-dry-run":
            publications.update(
                {
                    "production_graph": write_immutable_json(
                        args.output_dir / "production_graph.json", graph
                    ),
                    "dry_run_ledger": write_immutable_json(
                        args.output_dir / "dry_run_ledger.json", ledger
                    ),
                }
            )
        result = {
            "production_dry_run_evidence_sha256": evidence[
                "content_hash"
            ],
            "production_submission_performed": False,
            "publications": publications,
        }
    elif args.command == "authorize-full-submission":
        local = load_hashed_json(
            args.local_report,
            expected_contract=LOCAL_OPERATIONAL_REPORT_CONTRACT,
        )
        smoke = load_hashed_json(
            args.tigris_smoke_evidence,
            expected_contract=TIGRIS_SMOKE_EVIDENCE_CONTRACT,
        )
        dry = load_hashed_json(
            args.production_dry_run_evidence,
            expected_contract=PRODUCTION_DRY_RUN_EVIDENCE_CONTRACT,
        )
        authorization = build_full_submission_authorization(
            local_report=local,
            tigris_smoke_evidence=smoke,
            production_dry_run_evidence=dry,
            source_snapshot=snapshot,
        )
        result = {
            "full_submission_authorization_sha256": authorization[
                "content_hash"
            ],
            "publication": write_immutable_json(
                args.output, authorization
            ),
        }
    else:
        authorization = load_hashed_json(args.authorization)
        result = {
            "full_submission_authorization_sha256": (
                validate_full_submission_authorization(
                    authorization, current_source_snapshot=snapshot
                )
            ),
            "valid_for_current_source": True,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
