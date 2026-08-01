#!/usr/bin/env python3
"""Materialize all label-blind HOSD target-builder views before DAG planning."""

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
    load_and_validate_campaign,
    materialize_hlt_input_view,
    materialize_retb_offline_input_view,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    PARENT_STATUS_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)


SPLITS = ("model_train", "val_stop", "val_design")
HLT_COORDINATES = (
    *(("model_train", replica, "R_MULTI") for replica in range(4)),
    ("val_stop", 0, "R_FIXED"),
    ("val_design", 0, "R_FIXED"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--hlt-cache-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)

    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    parent_lock = load_hashed_json(
        root / "inputs" / "resolved_inherited_parent_lock.json",
        expected_contract=PARENT_STATUS_CONTRACT,
    )
    if (
        parent_lock.get("source") != campaign["source"]
        or not parent_lock.get("all_stage_b_parents_reusable")
    ):
        raise ValueError(
            "runtime input views require the complete same-source parent lock"
        )
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else root / "inputs" / "hosd_views"
    )
    hlt_root = (
        args.hlt_cache_root.resolve()
        if args.hlt_cache_root is not None
        else root
        / "inputs"
        / "shared_retb_parent_campaign"
        / "inputs"
        / "hlt_v3"
    )
    rows = []
    for split in SPLITS:
        output = output_root / "offline" / f"{split}.npz"
        manifest = materialize_retb_offline_input_view(
            offline_cache_dir=(
                root
                / "inputs"
                / "shared_retb_parent_campaign"
                / "inputs"
                / "offline"
                / split
            ),
            split=split,
            output=output,
            parent_hashes={
                "campaign_spec": campaign["content_hash"],
                "resolved_parent_lock": parent_lock["content_hash"],
            },
            source=campaign["source"],
        )
        rows.append(
            {
                "view_kind": "canonical_offline",
                "split": split,
                "replica": None,
                "npz_path": str(output.resolve()),
                "npz_sha256": _sha256(output),
                "manifest_sha256": manifest["content_hash"],
            }
        )
    for split, replica, policy in HLT_COORDINATES:
        source_cache = (
            hlt_root
            / split
            / f"replica_{replica}"
            / policy
            / "D_NOMINAL"
        )
        output = (
            output_root
            / "hlt"
            / split
            / f"replica_{replica}.npz"
        )
        manifest = materialize_hlt_input_view(
            hlt_cache_path=source_cache,
            split=split,
            replica_id=replica,
            output=output,
            parent_hashes={
                "campaign_spec": campaign["content_hash"],
                "resolved_parent_lock": parent_lock["content_hash"],
            },
            source=campaign["source"],
        )
        rows.append(
            {
                "view_kind": "hlt_analogue",
                "split": split,
                "replica": replica,
                "npz_path": str(output.resolve()),
                "npz_sha256": _sha256(output),
                "manifest_sha256": manifest["content_hash"],
            }
        )
    completion = with_content_hash(
        {
            "contract": "hosd_runtime_input_view_completion_v1",
            "schema_version": 1,
            "source": campaign["source"],
            "campaign_spec_sha256": campaign["content_hash"],
            "resolved_parent_lock_sha256": parent_lock["content_hash"],
            "rows": rows,
            "coordinate_count": len(rows),
            "offline_coordinate_count": len(SPLITS),
            "hlt_coordinate_count": len(HLT_COORDINATES),
            "hlt_replica_policy": {
                "model_train": "R_MULTI_0_1_2_3",
                "val_stop": "R_FIXED_0",
                "val_design": "R_FIXED_0",
            },
            "label_access": False,
            "constituent_matching_used": False,
        }
    )
    output = output_root / "completion.json"
    publication = write_immutable_json(output, completion)
    print(
        json.dumps(
            {
                "completion_sha256": completion["content_hash"],
                "coordinate_count": len(rows),
                "output": str(output.resolve()),
                "publication": publication["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
