#!/usr/bin/env python3
"""Materialize the five authenticated, label-blind Stage-J scale views."""

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
    INPUT_VIEW_MANIFEST_CONTRACT,
    SCALE_EXECUTION_PLAN_CONTRACT,
    SCALE_INPUT_COMPLETION_CONTRACT,
    load_and_validate_campaign,
    materialize_hlt_input_view,
    materialize_retb_offline_input_view,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    load_hashed_json,
    require_sha256,
    with_content_hash,
    write_immutable_json,
)
def _mapping(values: list[str], *, name: str) -> dict[int, Path]:
    rows: dict[int, Path] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or not key.isdigit() or int(key) in rows:
            raise ValueError(f"{name} must contain unique REPLICA=PATH values")
        rows[int(key)] = Path(path).resolve()
    if set(rows) != {0, 1, 2, 3}:
        raise ValueError(f"{name} must cover exact replicas 0,1,2,3")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scale_manifest_sha(campaign: dict) -> str:
    digest = (
        campaign.get("shared_parent_hashes", {}).get("scale_train_manifest")
        or campaign.get("parent_artifact_hashes", {}).get(
            "scale_train_manifest"
        )
    )
    return require_sha256(digest, name="scale_train_manifest_sha256")


COORDINATES = ("offline", "0", "1", "2", "3")


def _output_path(output_root: Path, coordinate: str) -> Path:
    if coordinate == "offline":
        return output_root / "offline" / "scale_train.npz"
    return output_root / "hlt" / f"replica_{int(coordinate)}.npz"


def _try_finalize(
    *,
    output_root: Path,
    campaign: dict,
    plan: dict,
    parent_lock: dict,
    expected_identity: str,
) -> dict | None:
    rows = []
    for coordinate in COORDINATES:
        output = _output_path(output_root, coordinate)
        manifest_path = output.with_suffix(output.suffix + ".json")
        if not output.is_file() or not manifest_path.is_file():
            return None
        manifest = load_hashed_json(
            manifest_path, expected_contract=INPUT_VIEW_MANIFEST_CONTRACT
        )
        replica = None if coordinate == "offline" else int(coordinate)
        expected_kind = (
            "canonical_offline"
            if coordinate == "offline"
            else "hlt_analogue"
        )
        if (
            manifest.get("source") != campaign["source"]
            or manifest.get("split") != "scale_train"
            or manifest.get("view_kind") != expected_kind
            or manifest.get("replica_id") != replica
            or manifest.get("parent_hashes", {}).get(
                "scale_train_manifest"
            )
            != expected_identity
            or manifest.get("contains_measurement_states")
            != (coordinate != "offline")
            or _sha256(output) != manifest.get("npz_sha256")
        ):
            raise ValueError("Stage-J input-view wave lineage differs")
        rows.append(
            {
                "view_id": (
                    "offline:scale_train"
                    if coordinate == "offline"
                    else f"hlt:scale_train:r{replica}"
                ),
                "view_kind": expected_kind,
                "replica_id": replica,
                "identity_count": int(manifest["identity_count"]),
                "npz_path": str(output.resolve()),
                "npz_sha256": manifest["npz_sha256"],
                "view_manifest_sha256": manifest["content_hash"],
            }
        )
    if len({int(row["identity_count"]) for row in rows}) != 1:
        raise ValueError("Stage-J input-view identity counts differ")
    completion = with_content_hash(
        {
            "contract": SCALE_INPUT_COMPLETION_CONTRACT,
            "schema_version": 2,
            "source": campaign["source"],
            "campaign_spec_sha256": campaign["content_hash"],
            "scale_execution_plan_sha256": plan["content_hash"],
            "resolved_parent_lock_sha256": parent_lock["content_hash"],
            "scale_train_manifest_sha256": expected_identity,
            "rows": rows,
            "coordinate_count": len(rows),
            "coordinate_order": list(COORDINATES),
            "offline_coordinate_count": 1,
            "hlt_coordinate_count": 4,
            "hlt_realization_policy": "R_MULTI",
            "degradation_profile_id": "D_NOMINAL",
            "label_access": False,
            "constituent_matching_used": False,
            "complete_exact_coverage": True,
        }
    )
    write_immutable_json(output_root / "completion.json", completion)
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--coordinate", required=True, choices=COORDINATES
    )
    parser.add_argument("--scale-offline-cache", required=True, type=Path)
    parser.add_argument(
        "--scale-hlt-cache",
        action="append",
        default=[],
        metavar="REPLICA=DIR",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        root / "scale_up" / "execution_plan.json",
        expected_contract=SCALE_EXECUTION_PLAN_CONTRACT,
    )
    parent_lock = load_hashed_json(
        root / "inputs" / "resolved_inherited_parent_lock.json",
        expected_contract="hosd_parent_status_v1",
    )
    if (
        plan.get("source") != campaign["source"]
        or parent_lock.get("source") != campaign["source"]
        or not parent_lock.get("all_stage_b_parents_reusable")
    ):
        raise ValueError("Stage-J input parent lineage differs")
    expected_identity = _scale_manifest_sha(campaign)
    hlt = _mapping(args.scale_hlt_cache, name="--scale-hlt-cache")
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else root / "scale_up" / "inputs"
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "coordinate": args.coordinate,
                    "output_root": str(output_root),
                    "scale_execution_plan_sha256": plan["content_hash"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    parent_hashes = {
        "campaign_spec": campaign["content_hash"],
        "resolved_parent_lock": parent_lock["content_hash"],
        "scale_execution_plan": plan["content_hash"],
        "scale_train_manifest": expected_identity,
    }
    output = _output_path(output_root, args.coordinate)
    if args.coordinate == "offline":
        manifest = materialize_retb_offline_input_view(
            offline_cache_dir=args.scale_offline_cache,
            split="scale_train",
            output=output,
            parent_hashes=parent_hashes,
            source=campaign["source"],
            expected_identity_manifest_sha256=expected_identity,
        )
    else:
        replica = int(args.coordinate)
        manifest = materialize_hlt_input_view(
            hlt_cache_path=hlt[replica],
            split="scale_train",
            replica_id=replica,
            output=output,
            parent_hashes=parent_hashes,
            source=campaign["source"],
            expected_identity_manifest_sha256=expected_identity,
        )
    completion = _try_finalize(
        output_root=output_root,
        campaign=campaign,
        plan=plan,
        parent_lock=parent_lock,
        expected_identity=expected_identity,
    )
    print(
        json.dumps(
            {
                "coordinate": args.coordinate,
                "view_manifest_sha256": manifest["content_hash"],
                "wave_complete": completion is not None,
                "completion_sha256": (
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
