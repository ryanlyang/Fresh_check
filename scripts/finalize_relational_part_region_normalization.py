#!/usr/bin/env python3
"""Reduce authenticated REGION sample shards into the canonical scaler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT,
    REGION_NORMALIZATION_CONTRACT,
    RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    bind_source_provenance,
    build_region_normalization_from_samples,
    load_hashed_json,
    sha256_file,
    validate_campaign_source,
    validate_region_normalization,
    write_immutable_json,
)
from teacher_logit_reco.relational_part.ca_tree import (  # noqa: E402
    VIEW_TREE_SPLIT_MANIFEST_CONTRACT,
)
from teacher_logit_reco.relational_part.normalization import (  # noqa: E402
    RELATION_NORMALIZATION_ARTIFACT_CONTRACT_V3,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    REGION_NORMALIZATION_PARTIAL_CONTRACT,
    REGION_NORMALIZATION_PLAN_CONTRACT,
    assemble_region_normalization_partials,
    validate_region_normalization_partial,
    validate_region_normalization_plan,
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--partials-dir", type=Path, required=True)
    parser.add_argument("--tree-dir", type=Path, required=True)
    parser.add_argument("--relation-normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    campaign = load_hashed_json(args.campaign_spec)
    current_source = validate_campaign_source(campaign, repo_root=REPO_ROOT)
    plan = load_hashed_json(args.plan)
    validate_region_normalization_plan(plan)
    if plan.get("source") != campaign.get("source"):
        raise ValueError("REGION map plan source differs from campaign")
    offline = plan["parents"].get("input_view") == "offline"
    relation = load_hashed_json(
        args.relation_normalization,
        expected_contract=(
            RELATION_NORMALIZATION_ARTIFACT_CONTRACT_V3
            if offline
            else RELATION_NORMALIZATION_ARTIFACT_CONTRACT
        ),
    )
    if relation["content_hash"] != plan["parents"][
        "relation_normalization_sha256"
    ]:
        raise ValueError("REGION reducer base normalizer differs from plan")
    manifest = load_hashed_json(
        args.tree_dir / "manifest.json",
        expected_contract=(
            VIEW_TREE_SPLIT_MANIFEST_CONTRACT
            if offline
            else ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT
        ),
    )
    if (
        manifest["content_hash"] != plan["parents"]["tree_manifest_sha256"]
        or manifest["parents"]["tree_resource_sha256"]
        != plan["parents"]["tree_resource_sha256"]
    ):
        raise ValueError("REGION reducer tree manifest differs from plan")
    if args.output.is_file():
        artifact = load_hashed_json(
            args.output, expected_contract=REGION_NORMALIZATION_CONTRACT
        )
        validate_region_normalization(
            artifact,
            relation_normalization_sha256=relation["content_hash"],
            angular_tree_resource_sha256=manifest["parents"][
                "tree_resource_sha256"
            ],
        )
        if (
            artifact.get("source") != campaign.get("source")
            or int(artifact["selected_jet_count"])
            != int(plan["selected_jet_count"])
            or artifact["selected_jet_identity_sha256"]
            != plan["selected_jet_identity_sha256"]
        ):
            raise ValueError("reusable REGION normalization differs")
        print(
            json.dumps(
                {
                    "reused": True,
                    "output": str(args.output),
                    "content_hash": artifact["content_hash"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.output.exists():
        raise FileExistsError("REGION normalization destination is unsafe")

    partials = []
    for shard_index in range(int(plan["shard_count"])):
        metadata_path = (
            args.partials_dir / f"shard_{shard_index:05d}.metadata.json"
        )
        sample_path = (
            args.partials_dir / f"shard_{shard_index:05d}.samples.npz"
        )
        metadata = load_hashed_json(metadata_path)
        validate_region_normalization_partial(
            metadata, plan=plan, shard_index=shard_index
        )
        if (
            metadata.get("source") != campaign.get("source")
            or sha256_file(sample_path)
            != metadata["parents"]["sample_npz_sha256"]
        ):
            raise ValueError("REGION reducer partial bytes/source differ")
        partials.append((metadata, _load_npz(sample_path)))
        if (shard_index + 1) % 10 == 0 or shard_index + 1 == int(
            plan["shard_count"]
        ):
            print(
                json.dumps(
                    {
                        "progress": {
                            "stage": "load_region_partials",
                            "processed_shards": shard_index + 1,
                            "total_shards": plan["shard_count"],
                        }
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    samples, hashes, identities = assemble_region_normalization_partials(
        plan, partials
    )
    artifact = build_region_normalization_from_samples(
        samples,
        hashes,
        identities,
        relation_normalization_artifact=relation,
        angular_tree_resource_sha256=manifest["parents"][
            "tree_resource_sha256"
        ],
    )
    artifact = bind_source_provenance(
        artifact, source_snapshot=current_source
    )
    publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "fit_split": "model_train",
                "final_test_accessed": False,
                "artifact": artifact,
                "publication": publication,
                "parallel_reduction": {
                    "plan_sha256": plan["content_hash"],
                    "partial_count": len(partials),
                    "canonical_order": "salted_selection_rank",
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
