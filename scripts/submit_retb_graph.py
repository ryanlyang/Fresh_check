#!/usr/bin/env python3
"""Resolve, validate, and optionally publish the complete RETB Tigris DAG."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge import (  # noqa: E402
    DEFAULT_CONCURRENCY,
    build_job_ledger,
    build_production_graph,
    build_step15_contract_bundle,
    load_hashed_json,
    publish_step15_contract_bundle,
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.production import (  # noqa: E402
    HOSD_MINIATURE_SPLIT_PROFILE,
    RETB_MINIATURE_SPLIT_PROFILE,
)
from teacher_logit_reco.relation_expert_token_bridge.storage import (  # noqa: E402
    STORAGE_MEASUREMENTS_CONTRACT,
    miniature_storage_measurements,
    validate_storage_measurements,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
)


def _campaign_id(source: dict[str, object]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"retb_relation_expert_bridge_{stamp}_"
        f"{str(source['source_commit'])[:10]}_"
        f"{str(source['source_status_sha256'])[:10]}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/ryreu/atlas/Fresh_check/checkpoints"),
    )
    parser.add_argument("--campaign-id")
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--storage-measurements", type=Path)
    parser.add_argument("--storage-measurements-sha256")
    parser.add_argument("--miniature", action="store_true")
    parser.add_argument(
        "--miniature-split-profile",
        choices=(
            RETB_MINIATURE_SPLIT_PROFILE,
            HOSD_MINIATURE_SPLIT_PROFILE,
        ),
        default=RETB_MINIATURE_SPLIT_PROFILE,
    )
    parser.add_argument("--split-profile-parent-sha256")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-simulate", action="store_true")
    parser.add_argument("--write-artifacts", action="store_true")
    for name, default in DEFAULT_CONCURRENCY.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}-concurrency",
            type=int,
            default=default,
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_run and args.write_artifacts:
        raise ValueError("--dry-run may not write campaign artifacts")
    if args.smoke_simulate and not args.miniature:
        raise ValueError("--smoke-simulate requires --miniature")
    if (
        args.miniature_split_profile == HOSD_MINIATURE_SPLIT_PROFILE
        and not args.miniature
    ):
        raise ValueError("the HOSD split profile requires --miniature")
    if (
        args.storage_measurements is not None
        and args.storage_measurements_sha256 is not None
    ):
        raise ValueError("storage measurement path/hash options are exclusive")
    if args.storage_measurements_sha256 is not None:
        if args.write_artifacts or not (args.dry_run or args.smoke_simulate):
            raise ValueError(
                "a storage hash without its artifact is allowed only in "
                "non-mutating graph resolution"
            )
        measurements_sha = str(args.storage_measurements_sha256)
        if (
            len(measurements_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in measurements_sha
            )
        ):
            raise ValueError("storage measurement hash differs")
        measurements = None
    elif args.storage_measurements is None:
        if not args.miniature:
            raise ValueError("production graph requires --storage-measurements")
        measurements = miniature_storage_measurements()
    else:
        measurements = load_hashed_json(
            args.storage_measurements,
            expected_contract=STORAGE_MEASUREMENTS_CONTRACT,
        )
    if measurements is not None:
        measurements_sha = validate_storage_measurements(measurements)
    source = source_snapshot(REPO_ROOT)
    if measurements is not None:
        measurements_sha = bind_source(
            measurements, source_snapshot=source
        )["content_hash"]
    campaign_id = args.campaign_id or _campaign_id(source)
    campaign_root = args.campaign_root or (
        args.output_root / "relation_expert_token_bridge" / campaign_id
    )
    concurrency = {
        name: int(getattr(args, f"{name}_concurrency"))
        for name in DEFAULT_CONCURRENCY
    }
    graph = build_production_graph(
        campaign_root=campaign_root,
        campaign_id=campaign_id,
        source_commit=str(source["source_commit"]),
        source_status_sha256=str(source["source_status_sha256"]),
        storage_measurements_sha256=measurements_sha,
        miniature=bool(args.miniature),
        miniature_split_profile=str(args.miniature_split_profile),
        split_profile_parent_sha256=args.split_profile_parent_sha256,
        concurrency=concurrency,
    )
    mode = "smoke_simulation" if args.smoke_simulate else "dry_run"
    jobs = {
        node["node_id"]: (
            str(50_000 + index) if args.smoke_simulate else None
        )
        for index, node in enumerate(graph["nodes"])
    }
    # Virtual selector/lock nodes are one immutable selector invocation.
    if args.smoke_simulate:
        selector_id = jobs["accuracy_finalist_selector"]
        jobs["rejection_finalist_selector"] = selector_id
        jobs["locked_scale_finalists"] = selector_id
    ledger = build_job_ledger(
        production_graph=graph, jobs=jobs, submission_mode=mode
    )
    bundle = build_step15_contract_bundle(
        production_graph=graph,
        dry_run_ledger=ledger,
        source_snapshot=source,
    )
    result: dict[str, object] = {
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
        "source": source,
        "storage_measurements_sha256": measurements_sha,
        "storage_measurements_artifact_resolved": measurements is not None,
        "degradation_profile": graph["degradation_profile"],
        "bounded_concurrency": graph["bounded_concurrency"],
        "dry_run": bool(args.dry_run),
        "smoke_simulate": bool(args.smoke_simulate),
        "production_submission_performed": False,
        "nodes": [
            {
                "node_id": node["node_id"],
                "stage": node["stage"],
                "dependencies": node["dependencies"],
                "resource": node["resource"],
                "array": node["array"],
                "worker": node["worker"],
                "dynamic_continuation": node["dynamic_continuation"],
            }
            for node in graph["nodes"]
        ],
        "job_ledger": ledger,
        "monitoring": graph["monitoring"],
        "report_download": (
            f"rsync -av tigris:{campaign_root}/reports/ "
            "./retb_reports/"
        ),
        "next_command": "bash sbatch/submit_retb_tigris_full.sh",
    }
    if args.write_artifacts:
        if campaign_root.exists() and not campaign_root.is_dir():
            raise ValueError("campaign root exists and is not a directory")
        campaign_root.mkdir(parents=True, exist_ok=True)
        result["publication"] = publish_step15_contract_bundle(
            campaign_root=campaign_root, bundle=bundle
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
