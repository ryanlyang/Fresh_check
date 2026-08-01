#!/usr/bin/env python3
"""Compile runtime, node factories, and an executable HOSD campaign plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_node_factory_registry,
    build_production_execution_plan,
    build_registered_command_matrix,
    build_runtime_manifest,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    RESOURCE_MEASUREMENTS_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.runtime_support import (  # noqa: E402
    publish_runtime_support,
)
from scripts.validate_hosd_weaver_parity import main as parity_main  # noqa: E402


CONFIG_KEYS = {
    "files",
    "directories",
    "infrastructure_arguments_by_node",
}


def _load_config(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != CONFIG_KEYS:
        raise ValueError("runtime config has unexpected or missing keys")
    if any(not isinstance(raw[key], dict) for key in CONFIG_KEYS):
        raise ValueError("runtime config sections must be JSON objects")
    return raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        help="Legacy explicit external-scalar config; normally omit this.",
    )
    parser.add_argument("--available-storage-bytes", type=int)
    parser.add_argument("--production-batch-size", type=int, default=32)
    parser.add_argument(
        "--clock-power-mode", default="miniature_control_plane_unmeasured"
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=("miniature_test", "production_500k_scale3m"),
    )
    parser.add_argument("--resource-measurements", type=Path)
    args = parser.parse_args(argv)
    if (
        args.profile == "production_500k_scale3m"
        and args.resource_measurements is None
    ):
        raise ValueError(
            "production preparation requires miniature-derived resources"
        )
    if (
        args.profile == "production_500k_scale3m"
        and args.clock_power_mode == "miniature_control_plane_unmeasured"
    ):
        raise ValueError(
            "production preparation requires an explicit measured "
            "--clock-power-mode"
        )
    if args.profile == "miniature_test" and args.resource_measurements is not None:
        raise ValueError("miniature preparation cannot consume production resources")

    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    parity_path = (
        root
        / "registry"
        / "runtime_support"
        / "hosd_weaver_parity.json"
    )
    parity_main(
        [
            "--campaign-root",
            str(root),
            "--output",
            str(parity_path),
        ]
    )
    support = publish_runtime_support(
        campaign_root=root, campaign=campaign
    )
    if args.runtime_config is None:
        available = args.available_storage_bytes
        if available is None:
            if args.profile != "miniature_test":
                raise ValueError(
                    "production preparation requires an explicit measured "
                    "--available-storage-bytes"
                )
            available = int(shutil.disk_usage(root).free)
        if available <= 0 or args.production_batch_size <= 0:
            raise ValueError("runtime scalar controls must be positive")
        config = {
            "files": {},
            "directories": {},
            "infrastructure_arguments_by_node": {
                "storage_measurement": [
                    "--available-storage-bytes",
                    str(available),
                ],
                "scale_efficiency": [
                    "--production-batch-size",
                    str(args.production_batch_size),
                    "--clock-power-mode",
                    str(args.clock_power_mode),
                ],
            },
        }
    else:
        config = _load_config(args.runtime_config)
    runtime = build_runtime_manifest(
        campaign_spec_sha256=campaign["content_hash"],
        files=config["files"],
        directories=config["directories"],
        infrastructure_arguments_by_node=config[
            "infrastructure_arguments_by_node"
        ],
        source=campaign["source"],
        runtime_support_sha256=support["content_hash"],
    )
    if not runtime["execution_ready"]:
        raise ValueError(
            "runtime configuration remains incomplete: "
            f"{runtime['missing_required_options_by_node']}"
        )
    registry = load_hashed_json(
        root / "registry" / "stage_job_registry.json",
        expected_contract="hosd_registry_v1",
    )
    factories = build_node_factory_registry(
        stage_job_registry=registry,
        source=campaign["source"],
    )
    commands = build_registered_command_matrix(
        stage_job_registry=registry,
        factory_registry=factories,
        runtime_manifest=runtime,
        campaign_root=root,
    )
    resources = (
        None
        if args.resource_measurements is None
        else load_hashed_json(
            args.resource_measurements,
            expected_contract=RESOURCE_MEASUREMENTS_CONTRACT,
        )
    )
    plan = build_production_execution_plan(
        stage_job_registry=registry,
        commands_by_node=commands,
        campaign_spec_sha256=campaign["content_hash"],
        source=campaign["source"],
        profile=args.profile,
        resource_measurements=resources,
        node_factory_registry=factories,
        runtime_manifest=runtime,
    )
    publications = {
        "runtime_manifest": write_immutable_json(
            root / "registry" / "runtime_manifest.json", runtime
        )["status"],
        "node_factory_registry": write_immutable_json(
            root / "registry" / "node_factory_registry.json", factories
        )["status"],
        "production_execution_plan": write_immutable_json(
            root / "job_ledgers" / "production_execution_plan.json", plan
        )["status"],
    }
    print(
        json.dumps(
            {
                "campaign_root": str(root),
                "profile": args.profile,
                "runtime_manifest_sha256": runtime["content_hash"],
                "node_factory_registry_sha256": factories["content_hash"],
                "execution_plan_sha256": plan["content_hash"],
                "node_count": plan["node_count"],
                "runtime_support_sha256": support["content_hash"],
                "publications": publications,
                "next_command": (
                    "bash sbatch/submit_hosd_tigris_full.sh --smoke-submit"
                    if args.profile == "miniature_test"
                    else (
                        "bash sbatch/submit_hosd_tigris_full.sh "
                        "--authorization "
                        f"{root / 'job_ledgers' / 'full_authorization.json'}"
                    )
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
