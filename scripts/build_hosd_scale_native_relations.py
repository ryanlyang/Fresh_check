#!/usr/bin/env python3
"""Build one scale native-relation target and close the four-replica wave."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    NATIVE_RELATION_TARGET_CONTRACT,
    SCALE_EXECUTION_PLAN_CONTRACT,
    SCALE_NATIVE_RELATION_WAVE_CONTRACT,
    load_and_validate_campaign,
    materialize_native_relation_target_from_family_caches,
    native_relation_target_ids,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    SCALE_TARGET_WAVE_COMPLETION_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _requires_native_relations(plan: dict) -> bool:
    return any(
        (
            row["graph_definition"].get("baseline_id")
            == "H_NATIVE_REL_AUX"
            or row["graph_definition"].get("graph", {}).get(
                "native_relation_auxiliary"
            )
            is not None
        )
        for row in plan["graph_rows"]
    )


def _try_finalize(root: Path, campaign: dict, plan: dict, target_wave: dict):
    required = _requires_native_relations(plan)
    if not required:
        completion = with_content_hash(
            {
                "contract": SCALE_NATIVE_RELATION_WAVE_CONTRACT,
                "schema_version": 2,
                "source": campaign["source"],
                "campaign_spec_sha256": campaign["content_hash"],
                "scale_execution_plan_sha256": plan["content_hash"],
                "scale_target_wave_sha256": target_wave["content_hash"],
                "required_by_shortlist": False,
                "artifact_hashes_by_replica": {},
                "replicas": [],
                "artifact_count": 0,
                "same_view_replica_binding": True,
                "offline_information_consumed": False,
                "complete_exact_coverage": True,
            }
        )
        write_immutable_json(
            root
            / "scale_up"
            / "targets"
            / "native_relations"
            / "completion.json",
            completion,
        )
        return completion
    hashes = {}
    for replica in range(4):
        npz_path = (
            root
            / "scale_up"
            / "targets"
            / "native_relations"
            / f"replica_{replica}.npz"
        )
        manifest_path = npz_path.with_suffix(".manifest.json")
        if not npz_path.is_file() or not manifest_path.is_file():
            return None
        artifact = load_hashed_json(
            manifest_path,
            expected_contract=NATIVE_RELATION_TARGET_CONTRACT,
        )
        if (
            artifact.get("source") != campaign["source"]
            or artifact.get("campaign_spec_sha256")
            != campaign["content_hash"]
            or artifact.get("split") != "scale_train"
            or int(artifact.get("hlt_replica_id", -1)) != replica
            or artifact.get("target_ids")
            != list(native_relation_target_ids())
            or artifact.get("storage_layout")
            != "compressed_npz_plus_authenticated_npy_mmap_v1"
            or artifact.get("mmap_store", {}).get("contract")
            != "hosd_native_relation_npy_store_v1"
            or _sha256_file(npz_path) != artifact.get("npz_sha256")
        ):
            raise ValueError("scale native-relation artifact lineage differs")
        hashes[str(replica)] = artifact["content_hash"]
    completion = with_content_hash(
        {
            "contract": SCALE_NATIVE_RELATION_WAVE_CONTRACT,
            "schema_version": 2,
            "source": campaign["source"],
            "campaign_spec_sha256": campaign["content_hash"],
            "scale_execution_plan_sha256": plan["content_hash"],
            "scale_target_wave_sha256": target_wave["content_hash"],
            "required_by_shortlist": True,
            "artifact_hashes_by_replica": hashes,
            "replicas": [0, 1, 2, 3],
            "artifact_count": len(hashes),
            "same_view_replica_binding": True,
            "offline_information_consumed": False,
            "complete_exact_coverage": True,
        }
    )
    write_immutable_json(
        root
        / "scale_up"
        / "targets"
        / "native_relations"
        / "completion.json",
        completion,
    )
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--replica", required=True, choices=("none", "0", "1", "2", "3")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        root / "scale_up" / "execution_plan.json",
        expected_contract=SCALE_EXECUTION_PLAN_CONTRACT,
    )
    target_wave = load_hashed_json(
        root / "scale_up" / "target_completion.json",
        expected_contract=SCALE_TARGET_WAVE_COMPLETION_CONTRACT,
    )
    if (
        plan.get("source") != campaign["source"]
        or target_wave.get("source") != campaign["source"]
        or target_wave.get("scale_execution_plan_sha256")
        != plan["content_hash"]
    ):
        raise ValueError("scale native-relation parent lineage differs")
    required = _requires_native_relations(plan)
    if (args.replica == "none") == required:
        raise ValueError("scale native-relation coordinate/shortlist differs")
    replica = None if args.replica == "none" else int(args.replica)
    output = (
        root
        / "scale_up"
        / "targets"
        / "native_relations"
        / (
            "not_required"
            if replica is None
            else f"replica_{replica}.npz"
        )
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "replica": args.replica,
                    "required_by_shortlist": required,
                    "output": str(output),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    artifact = None
    if replica is not None:
        artifact = materialize_native_relation_target_from_family_caches(
            target_cache_roots={
                target_id: (
                    root
                    / "scale_up"
                    / "targets"
                    / target_id
                    / "hlt"
                    / f"replica_{replica}"
                )
                for target_id in native_relation_target_ids()
            },
            output_path=output,
            campaign_spec_sha256=campaign["content_hash"],
            source=campaign["source"],
        )
    completion = _try_finalize(root, campaign, plan, target_wave)
    print(
        json.dumps(
            {
                "replica": args.replica,
                "artifact_sha256": (
                    None if artifact is None else artifact["content_hash"]
                ),
                "wave_complete": completion is not None,
                "wave_completion_sha256": (
                    None
                    if completion is None
                    else completion["content_hash"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
