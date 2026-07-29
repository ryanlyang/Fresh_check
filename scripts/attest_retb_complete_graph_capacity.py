#!/usr/bin/env python3
"""Attest complete parameters and analytical FLOPs for one RETB export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
    require_sha256,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.deployment import (  # noqa: E402
    CAPACITY_COMPONENTS,
    DEPLOYABLE_EXPORT_CONTRACT,
    DeployableRetbGraph,
    build_complete_graph_capacity,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--deployment-manifest", required=True, type=Path)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--prepared-components", required=True, type=Path)
    parser.add_argument("--prepared-components-sha256", required=True)
    parser.add_argument("--analytical-flops", required=True, type=Path)
    parser.add_argument("--measured-diagnostics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    deployment = load_hashed_json(
        args.deployment_manifest,
        expected_contract=DEPLOYABLE_EXPORT_CONTRACT,
    )
    expected_hash = require_sha256(
        args.prepared_components_sha256,
        name="prepared_components_sha256",
    )
    if (
        deployment.get("source") != campaign.get("source")
        or not args.prepared_components.is_file()
        or args.prepared_components.is_symlink()
        or _sha256(args.prepared_components) != expected_hash
    ):
        raise ValueError("complete-graph capacity lineage differs")
    graph_path = (
        args.deployment_manifest.parent / deployment["graph_filename"]
    )
    if (
        not graph_path.is_file()
        or graph_path.is_symlink()
        or _sha256(graph_path) != deployment["graph_sha256"]
    ):
        raise ValueError("deployed graph bytes differ")
    graph_payload = torch.load(
        graph_path, map_location="cpu", weights_only=False
    )
    exported_graph = graph_payload.get("graph")
    if (
        graph_payload.get("contract") != DEPLOYABLE_EXPORT_CONTRACT
        or not isinstance(exported_graph, DeployableRetbGraph)
    ):
        raise ValueError("deployed graph payload differs")
    payload = torch.load(
        args.prepared_components, map_location="cpu", weights_only=False
    )
    modules = payload.get("component_modules") if isinstance(
        payload, dict
    ) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != {"component_modules"}
        or set(modules or {}) != set(CAPACITY_COMPONENTS)
    ):
        raise ValueError("complete-graph component payload differs")
    flops = json.loads(args.analytical_flops.read_text("utf-8"))
    diagnostics = json.loads(
        args.measured_diagnostics.read_text("utf-8")
    )
    artifact = build_complete_graph_capacity(
        graph_id=args.graph_id,
        deployment_export_sha256=deployment["content_hash"],
        component_modules=modules,
        exported_graph=exported_graph,
        analytical_component_flops=flops,
        measured_diagnostics=diagnostics,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "capacity_sha256": artifact["content_hash"],
                "totals": artifact["totals"],
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
