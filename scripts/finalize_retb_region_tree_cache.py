#!/usr/bin/env python3
"""Finalize all Stage-A RETB REGION views and publish their lineage index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    HLT_V3_ARRAY_FILENAME,
    HLT_V3_CACHE_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_a import (  # noqa: E402
    STAGE_A_HLT_TREE_VIEWS,
    STAGE_A_OFFLINE_TREE_ROLES,
    build_stage_a_tree_index,
    identity_newline_sha256,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relation_expert_token_bridge.streamed_abc import (  # noqa: E402
    STREAMED_HLT_VIEWS,
    STREAMED_OFFLINE_ROLES,
    build_streamed_tree_index,
    validate_streamed_abc_execution_profile,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    ANGULAR_TREE_RESOURCE_CONTRACT,
    finalize_tree_split,
)


def _offline_view(root: Path, role: str) -> dict:
    cache = root / "inputs" / "offline" / role
    metadata = load_hashed_json(
        cache / "offline_input_manifest.json",
        expected_contract="retb_offline_input_cache_v1",
    )
    with np.load(cache / metadata["npz_filename"], allow_pickle=False) as payload:
        identities = [str(value) for value in payload["identities"].tolist()]
    return {
        "view_id": f"offline:{role}",
        "view_kind": "offline",
        "logical_role": role,
        "replica_id": None,
        "realization_policy": "OFFLINE",
        "identities": identities,
        "view_content_sha256": metadata["npz_sha256"],
        "identity_manifest_sha256": metadata["identity_manifest_sha256"],
        "cache_metadata_sha256": metadata["content_hash"],
        "tree_dir": (
            root / "inputs" / "region_tree" / "offline"
            / f"{role}_exclusive_ca_v1"
        ),
    }


def _hlt_view(
    root: Path,
    role: str,
    replica: int,
    policy: str,
    *,
    streamed_abc: bool = False,
) -> dict:
    cache_namespace = (
        "hlt_v3_streamed_normalizer_sample" if streamed_abc else "hlt_v3"
    )
    tree_namespace = (
        "hlt_streamed_normalizer_sample" if streamed_abc else "hlt"
    )
    cache = (
        root
        / "inputs"
        / cache_namespace
        / role
        / f"replica_{replica}"
        / policy
        / "D_NOMINAL"
    )
    metadata = load_hashed_json(
        cache / "hlt_v3_metadata.json",
        expected_contract=HLT_V3_CACHE_CONTRACT,
    )
    with np.load(cache / HLT_V3_ARRAY_FILENAME, allow_pickle=False) as payload:
        identities = [str(value) for value in payload["identities"].tolist()]
    if (
        metadata["logical_role"] != role
        or int(metadata["replica_id"]) != replica
        or metadata["realization_policy"] != policy
    ):
        raise ValueError("Stage-A HLT REGION view metadata differs")
    return {
        "view_id": f"hlt:{role}:r{replica}:{policy}",
        "view_kind": "hlt",
        "logical_role": role,
        "replica_id": replica,
        "realization_policy": policy,
        "identities": identities,
        "view_content_sha256": metadata["array_content_sha256"],
        "identity_manifest_sha256": metadata["identity_manifest_sha256"],
        "cache_metadata_sha256": metadata["content_hash"],
        "tree_dir": (
            root / "inputs" / "region_tree" / tree_namespace
            / f"{role}_r{replica}_exclusive_ca_v1"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--streamed-abc", action="store_true")
    args = parser.parse_args()
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    backend = load_hashed_json(
        args.campaign_root / "backend" / "backend_manifest.json",
        expected_contract=ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    )
    resource = load_hashed_json(
        args.campaign_root / "inputs" / "inherited_angular_tree_resource.json",
        expected_contract=ANGULAR_TREE_RESOURCE_CONTRACT,
    )
    if resource.get("source") != campaign.get("source"):
        raise ValueError("Stage-A REGION resource source lineage differs")
    offline_roles = (
        STREAMED_OFFLINE_ROLES
        if args.streamed_abc
        else STAGE_A_OFFLINE_TREE_ROLES
    )
    hlt_views = (
        STREAMED_HLT_VIEWS
        if args.streamed_abc
        else STAGE_A_HLT_TREE_VIEWS
    )
    views = [
        *(
            _offline_view(args.campaign_root, role)
            for role in offline_roles
        ),
        *(
            _hlt_view(
                args.campaign_root,
                role,
                replica,
                policy,
                streamed_abc=bool(args.streamed_abc),
            )
            for role, replica, policy in hlt_views
        ),
    ]
    rows = []
    for view in views:
        identities = view.pop("identities")
        tree_dir = Path(view.pop("tree_dir"))
        shard_paths = []
        for index, start in enumerate(range(0, len(identities), 10_000)):
            stop = min(start + 10_000, len(identities))
            metadata_path = (
                tree_dir
                / "shards"
                / f"shard_{index:05d}.metadata.json"
            )
            shard = load_hashed_json(metadata_path)
            if (
                int(shard["jet_count"]) != stop - start
                or shard["identity_sha256"]
                != identity_newline_sha256(identities[start:stop])
            ):
                raise ValueError("REGION shard identity coverage differs")
            shard_paths.append(metadata_path)
        if args.dry_run:
            manifest_sha = "0" * 64
        else:
            manifest = finalize_tree_split(
                tree_dir / "manifest.json",
                shard_paths,
                split=str(view["view_id"]),
                expected_jet_count=len(identities),
                hlt_content_sha256=view["view_content_sha256"],
                tree_resource_sha256=resource["content_hash"],
                backend_manifest_sha256=backend["content_hash"],
            )
            manifest_sha = manifest["content_hash"]
        rows.append(
            {
                **view,
                "jet_count": len(identities),
                "identity_order_sha256": identity_newline_sha256(identities),
                "tree_manifest_sha256": manifest_sha,
            }
        )
    result = {
        "dry_run": bool(args.dry_run),
        "view_count": len(rows),
        "total_tree_count": sum(int(row["jet_count"]) for row in rows),
    }
    if not args.dry_run:
        if args.streamed_abc:
            profile = load_hashed_json(
                args.campaign_root
                / "registry"
                / "retb_streamed_abc_execution_profile.json"
            )
            validate_streamed_abc_execution_profile(profile)
            artifact = build_streamed_tree_index(
                campaign_spec_sha256=campaign["content_hash"],
                backend_manifest_sha256=backend["content_hash"],
                angular_tree_resource_sha256=resource["content_hash"],
                execution_profile_sha256=profile["content_hash"],
                views=rows,
                source=campaign["source"],
            )
        else:
            artifact = build_stage_a_tree_index(
                campaign_spec_sha256=campaign["content_hash"],
                backend_manifest_sha256=backend["content_hash"],
                angular_tree_resource_sha256=resource["content_hash"],
                views=rows,
                source_snapshot=source_snapshot(REPO_ROOT),
            )
        output = args.output or (
            args.campaign_root / "inputs" / "region_tree" / "tree_cache_index.json"
        )
        result["tree_index_sha256"] = artifact["content_hash"]
        result["publication"] = write_immutable_json(output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
