#!/usr/bin/env python3
"""Generate genuine scale-teacher targets after the finalist lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_selection import (  # noqa: E402
    LOCKED_SCALE_FINALISTS_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _run(arguments: list[str], expected: list[Path]) -> None:
    if all(path.is_file() and not path.is_symlink() for path in expected):
        return
    result = subprocess.run(
        [sys.executable, *arguments], cwd=REPO_ROOT, check=False
    )
    if result.returncode or any(not path.is_file() for path in expected):
        raise RuntimeError("postlock target generation failed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--locked-scale-finalists", required=True, type=Path)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument(
        "--split", required=True, choices=("stack_val", "final_test")
    )
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    finalists = load_hashed_json(
        args.locked_scale_finalists,
        expected_contract=LOCKED_SCALE_FINALISTS_CONTRACT,
    )
    run_map = {
        (row["graph_id"], int(row["pipeline_seed"])): row
        for row in finalists["all_shortlisted_scale_runs"]
    }
    key = (args.graph_id, args.pipeline_seed)
    if (
        finalists.get("source") != campaign.get("source")
        or args.graph_id not in finalists["finalist_graph_ids"]
        or key not in run_map
    ):
        raise ValueError("postlock target graph lineage differs")
    definition = finalists["locked_graph_definitions"][args.graph_id]
    role = definition["configuration"]["source_carried_shape_role"]
    role_root = (
        root
        / "runs"
        / "scale"
        / "refits"
        / f"seed_{args.pipeline_seed}"
        / "roles"
        / role
    )
    configuration = role_root / "target_configuration.json"
    coordinates = args.output_dir / "coordinates"
    official = args.output_dir / "target_cache"
    _run(
        [
            "scripts/prepare_retb_scale_target_coordinates.py",
            "--campaign-root",
            str(root),
            "--configuration",
            str(configuration),
            "--split",
            args.split,
            "--output-dir",
            str(coordinates),
            "--device",
            args.device,
        ],
        [
            coordinates / "scale_target_coordinate_index.json",
            coordinates / f"{args.split}_frozen_tokens.json",
        ],
    )
    _run(
        [
            "scripts/finalize_retb_scale_target_cache.py",
            "--campaign-root",
            str(root),
            "--configuration",
            str(configuration),
            "--coordinate-index",
            str(coordinates / "scale_target_coordinate_index.json"),
            "--coordinate-cache",
            str(coordinates / f"{args.split}_frozen_tokens.json"),
            "--fusion-registration",
            str(role_root / "offline_fusion" / "fusion_registration.json"),
            "--fusion-checkpoint",
            str(role_root / "offline_fusion" / "best_model_val.pt"),
            "--output-dir",
            str(official),
        ],
        [
            official / "target_cache_manifest.json",
            official / "identity_manifest.json",
        ],
    )
    target = load_hashed_json(official / "target_cache_manifest.json")
    identity = load_hashed_json(official / "identity_manifest.json")
    # Float32 is the deterministic fallback. The optional float16 path is
    # deliberately not asserted until a real numerical round-trip passes.
    audit = bind_source(
        with_content_hash(
            {
                "contract": "retb_postlock_float16_audit_v1",
                "schema_version": 1,
                "graph_id": args.graph_id,
                "pipeline_seed": args.pipeline_seed,
                "split": args.split,
                "target_cache_manifest_sha256": target["content_hash"],
                "passed": False,
                "published_dtype": "float32_fallback",
                "fallback_is_nonblocking": True,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    audit_path = args.output_dir / "float16_audit.json"
    write_immutable_json(audit_path, audit)
    run = run_map[key]
    evidence = bind_source(
        with_content_hash(
            {
                "contract": "retb_postlock_target_generation_evidence_v1",
                "schema_version": 1,
                "graph_id": args.graph_id,
                "pipeline_seed": args.pipeline_seed,
                "split": args.split,
                "parent_hashes": {
                    "locked_scale_finalists": finalists["content_hash"],
                    "scale_graph_run": run["scale_graph_run_sha256"],
                    "scale_offline_experts": run["component_hashes"][
                        "offline_experts"
                    ],
                    "scale_offline_fusion": run["component_hashes"][
                        "offline_fusion"
                    ],
                    "scale_target_normalizer": run["component_hashes"][
                        "scale_target_token_normalizer"
                    ],
                    "split_identity_manifest": identity["content_hash"],
                    "input_manifest": args.input_manifest_sha256,
                },
                "target_cache_manifest_sha256": target["content_hash"],
                "target_identity_order_sha256": identity[
                    "identity_order_sha256"
                ],
                "target_dtype": "float32_fallback",
                "float16_audit_sha256": audit["content_hash"],
                "float16_audit_passed": False,
                "actual_target_generation_executed": True,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    publication = write_immutable_json(args.output, evidence)
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
