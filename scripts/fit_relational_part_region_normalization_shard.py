#!/usr/bin/env python3
"""Extract one authenticated shard of REGION normalization samples."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    ANGULAR_TREE_SHARD_CONTRACT,
    ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT,
    bind_source_provenance,
    load_hashed_json,
    sha256_file,
    unpack_tree_shard,
    validate_campaign_source,
    write_immutable_bytes,
    write_immutable_json,
)
from teacher_logit_reco.relational_part.ca_tree import (  # noqa: E402
    VIEW_TREE_SHARD_CONTRACT,
    VIEW_TREE_SPLIT_MANIFEST_CONTRACT,
)
from teacher_logit_reco.relational_part.normalization import (  # noqa: E402
    _identity_sequence_hash,
)
from teacher_logit_reco.relational_part.region_normalization import (  # noqa: E402
    _collect_region_domain_samples,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    REGION_NORMALIZATION_PARTIAL_CONTRACT,
    REGION_NORMALIZATION_PLAN_CONTRACT,
    build_region_normalization_partial,
    validate_region_normalization_partial,
    validate_region_normalization_partial_arrays,
    validate_region_normalization_plan,
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--tree-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--shard-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")),
    )
    args = parser.parse_args()

    campaign = load_hashed_json(args.campaign_spec)
    current_source = validate_campaign_source(campaign, repo_root=REPO_ROOT)
    plan = load_hashed_json(args.plan)
    validate_region_normalization_plan(plan)
    if plan.get("source") != campaign.get("source"):
        raise ValueError("REGION map plan source differs from campaign")
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= int(plan["shard_count"]):
        raise ValueError("REGION shard index is out of range")
    plan_row = plan["shards"][shard_index]

    sample_path = args.output_dir / f"shard_{shard_index:05d}.samples.npz"
    metadata_path = args.output_dir / f"shard_{shard_index:05d}.metadata.json"
    if sample_path.is_file() and metadata_path.is_file():
        metadata = load_hashed_json(metadata_path)
        validate_region_normalization_partial(
            metadata, plan=plan, shard_index=shard_index
        )
        if (
            metadata.get("source") != campaign.get("source")
            or sha256_file(sample_path)
            != metadata["parents"]["sample_npz_sha256"]
        ):
            raise ValueError("reusable REGION partial differs")
        validate_region_normalization_partial_arrays(
            _load_npz(sample_path), metadata
        )
        print(
            json.dumps(
                {
                    "reused": True,
                    "shard_index": shard_index,
                    "metadata": str(metadata_path),
                },
                sort_keys=True,
            )
        )
        return 0
    if sample_path.exists() or metadata_path.exists():
        for path in (sample_path, metadata_path):
            if path.exists():
                if path.is_symlink() or not path.is_file():
                    raise FileExistsError(
                        "unregistered REGION partial is unsafe"
                    )
                path.unlink()

    input_path = (
        args.plan.parent
        / "selected_inputs"
        / str(plan_row["selected_input_filename"])
    )
    if sha256_file(input_path) != plan_row["selected_input_npz_sha256"]:
        raise ValueError("REGION selected input differs from plan")
    selected_input = _load_npz(input_path)
    local_indices = np.asarray(selected_input["local_index"])
    selection_ranks = np.asarray(selected_input["selection_rank"])
    identities = [str(value) for value in selected_input["identity"]]
    selected_count = int(plan_row["selected_count"])
    if (
        local_indices.dtype != np.int64
        or selection_ranks.dtype != np.int64
        or local_indices.tolist() != plan_row["selected_local_indices"]
        or selection_ranks.tolist() != plan_row["selection_ranks"]
        or len(identities) != selected_count
        or _identity_sequence_hash(identities)
        != plan_row["selected_identity_sha256"]
    ):
        raise ValueError("REGION selected input layout differs from plan")

    offline = plan["parents"].get("input_view") == "offline"
    manifest = load_hashed_json(
        args.tree_dir / "manifest.json",
        expected_contract=(
            VIEW_TREE_SPLIT_MANIFEST_CONTRACT
            if offline
            else ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT
        ),
    )
    if manifest["content_hash"] != plan["parents"]["tree_manifest_sha256"]:
        raise ValueError("REGION worker tree manifest differs from plan")
    manifest_row = manifest["shards"][shard_index]
    metadata = load_hashed_json(
        args.tree_dir
        / "shards"
        / f"shard_{shard_index:05d}.metadata.json",
        expected_contract=(
            VIEW_TREE_SHARD_CONTRACT if offline else ANGULAR_TREE_SHARD_CONTRACT
        ),
    )
    if (
        metadata["content_hash"] != plan_row["tree_shard_metadata_sha256"]
        or metadata["content_hash"] != manifest_row["metadata_sha256"]
    ):
        raise ValueError("REGION worker tree shard metadata differs")
    tree_path = (
        args.tree_dir / "shards" / f"shard_{shard_index:05d}.npz"
    )
    if sha256_file(tree_path) != metadata["npz_sha256"]:
        raise ValueError("REGION worker tree shard bytes differ")
    shard_identities, trees = unpack_tree_shard(
        tree_path, rows=local_indices.tolist()
    )
    if len(shard_identities) != int(plan_row["shard_jet_count"]):
        raise ValueError("REGION worker tree shard width differs")
    if [
        shard_identities[int(index)] for index in local_indices
    ] != identities:
        raise ValueError("REGION selected tree identities differ")

    tokens = np.asarray(selected_input["tokens"])
    mask = np.asarray(selected_input["mask"])
    if (
        tokens.dtype != np.float32
        or mask.dtype != np.bool_
        or tokens.shape[0] != selected_count
        or mask.shape != tokens.shape[:2]
        or tokens.shape[2] != 14
    ):
        raise ValueError("REGION selected input arrays differ")
    samples, hashes, layout = _collect_region_domain_samples(
        tokens,
        mask,
        identities,
        trees,
        np.arange(selected_count, dtype=np.int64),
        include_layout=True,
        allow_empty_domains=True,
    )
    arrays = {
        "identity": np.asarray(identities),
        "selection_rank": selection_ranks,
        **{
            f"{domain}_samples": values
            for domain, values in samples.items()
        },
        **layout,
    }
    publication = write_immutable_bytes(sample_path, _npz_bytes(arrays))
    partial = build_region_normalization_partial(
        plan=plan,
        shard_index=shard_index,
        tree_shard_metadata_sha256=metadata["content_hash"],
        sample_npz_sha256=publication["sha256"],
        selected_count=selected_count,
        sample_counts={
            domain: int(values.shape[0])
            for domain, values in samples.items()
        },
        sample_identity_sha256=hashes,
    )
    partial = bind_source_provenance(
        partial, source_snapshot=current_source
    )
    validate_region_normalization_partial_arrays(arrays, partial)
    metadata_publication = write_immutable_json(metadata_path, partial)
    print(
        json.dumps(
            {
                "reused": False,
                "shard_index": shard_index,
                "selected_count": selected_count,
                "sample_counts": partial["sample_counts"],
                "sample_publication": publication,
                "metadata_publication": metadata_publication,
                "final_test_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
