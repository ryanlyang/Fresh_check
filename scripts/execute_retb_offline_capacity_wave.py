#!/usr/bin/env python3
"""Authenticate and aggregate the complete Stage-C offline capacity wave."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.capacity import (  # noqa: E402
    build_capacity_control_registry,
    validate_offline_capacity_control_registration,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


CAPACITY_WAVE_CONTRACT = "retb_offline_capacity_wave_v2"


def _registration_path(root: Path, control_id: str) -> Path:
    return (
        root
        / "runs"
        / "stage_c"
        / "capacity_controls"
        / control_id
        / "control_registration.json"
    )


def _validate_registration(
    artifact: Mapping[str, Any],
    *,
    control_id: str,
    campaign: Mapping[str, Any],
) -> dict[str, Any]:
    digest = validate_offline_capacity_control_registration(artifact)
    capacity = artifact.get("complete_graph_capacity", {})
    exposure = artifact.get("label_exposure", {})
    if (
        artifact.get("control_id") != control_id
        or artifact.get("source") != campaign.get("source")
        or artifact.get("fixed_budget_completed") is not True
        or artifact.get("performance_based_termination") is not False
        or set(capacity)
        != {
            "parameter_count",
            "inference_flops_batch1",
            "inference_flops_batch128",
            "profile_sha256",
        }
        or min(
            int(capacity["parameter_count"]),
            float(capacity["inference_flops_batch1"]),
            float(capacity["inference_flops_batch128"]),
        )
        <= 0
        or int(exposure.get("labeled_example_presentations", 0)) <= 0
    ):
        raise ValueError(f"capacity registration {control_id} is incomplete")
    require_sha256(capacity["profile_sha256"], name=f"{control_id}.profile")
    return {
        "control_id": control_id,
        "registration_sha256": digest,
        "checkpoint_sha256": (
            None
            if artifact.get("checkpoint_sha256") is None
            else require_sha256(
                artifact["checkpoint_sha256"],
                name=f"{control_id}.checkpoint_sha256",
            )
        ),
        "complete_graph_capacity": dict(capacity),
        "label_exposure": dict(exposure),
        "fixed_budget_completed": True,
        "performance_based_termination": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_capacity_controls.json"
    )
    expected = build_capacity_control_registry()
    actual = dict(registry)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash", None)
    if actual != expected:
        raise ValueError("offline capacity-control registry differs")
    execution = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_offline_capacity_execution.json"
    )
    missing = [
        control_id
        for control_id in registry["offline_controls"]
        if not _registration_path(args.campaign_root, control_id).is_file()
    ]
    if missing:
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    REPO_ROOT
                    / "scripts"
                    / "train_retb_offline_capacity_controls.py"
                ),
                "--campaign-root",
                str(args.campaign_root),
            ],
            cwd=REPO_ROOT,
            check=False,
        )
        if completed.returncode != 0:
            return int(completed.returncode)
    controls = [
        _validate_registration(
            load_hashed_json(_registration_path(args.campaign_root, control_id)),
            control_id=control_id,
            campaign=campaign,
        )
        for control_id in registry["offline_controls"]
    ]
    artifact = bind_source(
        with_content_hash(
            {
                "contract": CAPACITY_WAVE_CONTRACT,
                "schema_version": 2,
                "capacity_control_registry_sha256": registry["content_hash"],
                "capacity_execution_registry_sha256": execution[
                    "content_hash"
                ],
                "control_order": list(registry["offline_controls"]),
                "control_count": len(controls),
                "controls": controls,
                "all_complete_deployable_graphs_profiled": True,
                "all_label_exposure_ledgers_complete": True,
                "all_declared_controls_executed": True,
                "latency_used_for_selection": False,
                "scientific_underperformance_blocks_continuation": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(
        args.campaign_root
        / "reports"
        / "stage_c"
        / "offline_capacity_controls.json",
        artifact,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
