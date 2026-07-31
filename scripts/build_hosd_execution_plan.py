#!/usr/bin/env python3
"""Compile the exhaustive source-bound HOSD production command DAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_production_execution_plan,
    build_node_factory_registry,
    build_registered_command_matrix,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    RESOURCE_MEASUREMENTS_CONTRACT,
    RUNTIME_MANIFEST_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--profile", choices=("miniature_test", "production_500k_scale3m"), required=True)
    parser.add_argument("--resource-measurements", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    registry = load_hashed_json(args.campaign_root / "registry" / "stage_job_registry.json", expected_contract="hosd_registry_v1")
    runtime_path = args.runtime_manifest or (
        args.campaign_root / "registry" / "runtime_manifest.json"
    )
    runtime = load_hashed_json(
        runtime_path, expected_contract=RUNTIME_MANIFEST_CONTRACT
    )
    factory_registry = build_node_factory_registry(
        stage_job_registry=registry, source=campaign["source"]
    )
    write_immutable_json(
        args.campaign_root / "registry" / "node_factory_registry.json",
        factory_registry,
    )
    commands = build_registered_command_matrix(
        stage_job_registry=registry,
        factory_registry=factory_registry,
        runtime_manifest=runtime,
        campaign_root=args.campaign_root,
    )
    resource_measurements = (
        None
        if args.resource_measurements is None
        else load_hashed_json(
            args.resource_measurements,
            expected_contract=RESOURCE_MEASUREMENTS_CONTRACT,
        )
    )
    artifact = build_production_execution_plan(stage_job_registry=registry, commands_by_node=commands, campaign_spec_sha256=campaign["content_hash"], source=campaign["source"], profile=args.profile, resource_measurements=resource_measurements, node_factory_registry=factory_registry, runtime_manifest=runtime)
    output = args.output or args.campaign_root / "job_ledgers" / "production_execution_plan.json"
    publication = write_immutable_json(output, artifact)
    print(json.dumps({"content_hash": artifact["content_hash"], "publication": publication["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
