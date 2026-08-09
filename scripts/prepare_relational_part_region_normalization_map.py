#!/usr/bin/env python3
"""Prepare authenticated selected-HLT inputs for parallel REGION fitting."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_cache import load_cached_hlt_view  # noqa: E402
from teacher_logit_reco.architecture_view_part.train import (  # noqa: E402
    load_cached_offline_view,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT,
    RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    bind_source_provenance,
    load_hashed_json,
    select_normalization_jet_indices,
    sha256_file,
    validate_campaign_source,
    write_immutable_bytes,
    write_immutable_json,
)
from teacher_logit_reco.relational_part.ca_tree import (  # noqa: E402
    VIEW_TREE_SPLIT_MANIFEST_CONTRACT,
)
from teacher_logit_reco.relational_part.normalization import (  # noqa: E402
    RELATION_NORMALIZATION_ARTIFACT_CONTRACT_V3,
    _identity_key,
    _identity_sequence_hash,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    REGION_NORMALIZATION_PLAN_CONTRACT,
    build_region_normalization_plan,
    validate_region_normalization_plan,
)


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tree-dir", type=Path, required=True)
    parser.add_argument("--relation-normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-view", choices=("fixed_hlt", "offline"), default="fixed_hlt"
    )
    args = parser.parse_args()

    campaign = load_hashed_json(args.campaign_spec)
    current_source = validate_campaign_source(campaign, repo_root=REPO_ROOT)
    relation = load_hashed_json(
        args.relation_normalization,
        expected_contract=(
            RELATION_NORMALIZATION_ARTIFACT_CONTRACT
            if args.input_view == "fixed_hlt"
            else RELATION_NORMALIZATION_ARTIFACT_CONTRACT_V3
        ),
    )
    manifest = load_hashed_json(
        args.tree_dir / "manifest.json",
        expected_contract=(
            ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT
            if args.input_view == "fixed_hlt"
            else VIEW_TREE_SPLIT_MANIFEST_CONTRACT
        ),
    )
    if manifest.get("split") != "model_train":
        raise ValueError("REGION normalization map may only access model_train")
    view = (
        load_cached_hlt_view(args.cache_dir, "model_train", verify_hash=True)
        if args.input_view == "fixed_hlt"
        else load_cached_offline_view(
            args.cache_dir, "model_train", verify_hash=True
        )
    )
    hlt_sha = str(
        view.metadata[
            "hlt_content_hash"
            if args.input_view == "fixed_hlt"
            else "offline_content_hash"
        ]
    )
    manifest_content_key = (
        "hlt_content_sha256"
        if args.input_view == "fixed_hlt"
        else "input_content_sha256"
    )
    if manifest["parents"][manifest_content_key] != hlt_sha:
        raise ValueError("REGION tree manifest belongs to another input cache")

    plan_path = args.output_dir / "plan.json"
    input_dir = args.output_dir / "selected_inputs"
    if plan_path.is_file():
        plan = load_hashed_json(plan_path)
        validate_region_normalization_plan(plan)
        if plan.get("source") != campaign.get("source"):
            raise ValueError("reusable REGION map plan source differs")
        expected_parents = {
            "tree_manifest_sha256": manifest["content_hash"],
            "tree_resource_sha256": manifest["parents"][
                "tree_resource_sha256"
            ],
            "relation_normalization_sha256": relation["content_hash"],
            **(
                {"hlt_content_sha256": hlt_sha}
                if args.input_view == "fixed_hlt"
                else {
                    "input_view": "offline",
                    "input_content_sha256": hlt_sha,
                }
            ),
        }
        if plan["parents"] != expected_parents:
            raise ValueError("reusable REGION map plan parents differ")
        for row in plan["shards"]:
            path = input_dir / str(row["selected_input_filename"])
            if sha256_file(path) != row["selected_input_npz_sha256"]:
                raise ValueError("reusable REGION selected input differs")
        print(
            json.dumps(
                {
                    "reused": True,
                    "plan": str(plan_path),
                    "selected_jet_count": plan["selected_jet_count"],
                    "shard_count": plan["shard_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    if plan_path.exists():
        raise FileExistsError("REGION map plan destination is unsafe")

    selected = select_normalization_jet_indices(view.jet_ids)
    selection_rank = {
        int(global_index): rank
        for rank, global_index in enumerate(selected.tolist())
    }
    selected_identities = [view.jet_ids[int(index)] for index in selected]
    rows = []
    global_start = 0
    input_dir.mkdir(parents=True, exist_ok=True)
    for shard_index, manifest_row in enumerate(manifest["shards"]):
        shard_jet_count = int(manifest_row["jet_count"])
        global_stop = global_start + shard_jet_count
        selected_global = sorted(
            index
            for index in selection_rank
            if global_start <= index < global_stop
        )
        selected_local = [
            index - global_start for index in selected_global
        ]
        ranks = [selection_rank[index] for index in selected_global]
        identities = [
            _identity_key(view.jet_ids[index]) for index in selected_global
        ]
        filename = f"shard_{shard_index:05d}.npz"
        destination = input_dir / filename
        if destination.exists():
            # Without plan.json this is an unregistered interrupted prepare.
            if destination.is_symlink() or not destination.is_file():
                raise FileExistsError("unregistered REGION input is unsafe")
            destination.unlink()
        publication = write_immutable_bytes(
            destination,
            _npz_bytes(
                {
                    "tokens": view.tokens[selected_global],
                    "mask": view.mask[selected_global],
                    "identity": np.asarray(identities),
                    "local_index": np.asarray(
                        selected_local, dtype=np.int64
                    ),
                    "global_index": np.asarray(
                        selected_global, dtype=np.int64
                    ),
                    "selection_rank": np.asarray(ranks, dtype=np.int64),
                }
            ),
        )
        rows.append(
            {
                "shard_index": shard_index,
                "shard_jet_count": shard_jet_count,
                "global_start": global_start,
                "global_stop": global_stop,
                "selected_count": len(selected_local),
                "selected_local_indices": selected_local,
                "selection_ranks": ranks,
                "selected_identity_sha256": _identity_sequence_hash(
                    identities
                ),
                "selected_input_filename": filename,
                "selected_input_npz_sha256": publication["sha256"],
                "tree_shard_metadata_sha256": manifest_row[
                    "metadata_sha256"
                ],
            }
        )
        global_start = global_stop
        if (shard_index + 1) % 10 == 0 or shard_index + 1 == len(
            manifest["shards"]
        ):
            print(
                json.dumps(
                    {
                        "progress": {
                            "stage": "prepare_region_map",
                            "processed_shards": shard_index + 1,
                            "total_shards": len(manifest["shards"]),
                        }
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if global_start != len(view.jet_ids):
        raise ValueError("REGION plan tree shard coverage differs from HLT")
    plan = build_region_normalization_plan(
        tree_manifest_sha256=manifest["content_hash"],
        tree_resource_sha256=manifest["parents"]["tree_resource_sha256"],
        relation_normalization_sha256=relation["content_hash"],
        hlt_content_sha256=hlt_sha,
        input_view=args.input_view,
        selected_identities=selected_identities,
        shard_rows=rows,
    )
    plan = bind_source_provenance(plan, source_snapshot=current_source)
    publication = write_immutable_json(plan_path, plan)
    print(
        json.dumps(
            {
                "reused": False,
                "plan": plan,
                "publication": publication,
                "final_test_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
