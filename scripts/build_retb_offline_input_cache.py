#!/usr/bin/env python3
"""Build one checkpoint-free RETB offline raw-particle input cache."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import (  # noqa: E402
    JetIdentity,
    load_offline_view,
    load_split_manifest,
)
from teacher_logit_reco.relation_expert_token_bridge import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


CONTRACT = "retb_offline_input_cache_v1"
ROLES = (
    "model_train",
    "val_stop",
    "val_design",
    "stack_val",
    "final_test",
    "scale_train",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identities(
    role: str, *, campaign_root: Path, manifest
) -> tuple[list[JetIdentity], str]:
    if role in {"model_train", "stack_val", "final_test"}:
        campaign = load_hashed_json(campaign_root / "campaign_spec.json")
        return (
            list(manifest.splits[role]),
            campaign["parent_artifact_hashes"]["split_manifest"],
        )
    if role in {"val_stop", "val_design"}:
        artifact = load_hashed_json(
            campaign_root / "inputs" / "validation_partition_manifest.json.gz"
        )
        return (
            [JetIdentity.from_dict(row) for row in artifact["roles"][role]],
            artifact["content_hash"],
        )
    artifact = load_hashed_json(
        campaign_root / "inputs" / "scale_train_manifest.json.gz"
    )
    return (
        [JetIdentity.from_dict(row) for row in artifact["identities"]],
        artifact["content_hash"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--logical-role", required=True, choices=ROLES)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    manifest = load_split_manifest(
        args.campaign_root / "bootstrap" / "split_manifest.json.gz"
    )
    identities, identity_parent = _identities(
        args.logical_role,
        campaign_root=args.campaign_root,
        manifest=manifest,
    )
    output_dir = args.output_dir
    npz_path = output_dir / "offline_inputs.npz"
    metadata_path = output_dir / "offline_input_manifest.json"
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "logical_role": args.logical_role,
        "event_count": len(identities),
        "output": str(npz_path),
    }
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if metadata_path.is_file():
        metadata = load_hashed_json(metadata_path, expected_contract=CONTRACT)
        if (
            metadata["campaign_spec_sha256"] != campaign["content_hash"]
            or metadata["logical_role"] != args.logical_role
            or metadata["identity_manifest_sha256"] != identity_parent
            or not npz_path.is_file()
            or _file_sha256(npz_path) != metadata["npz_sha256"]
        ):
            raise ValueError("reusable offline input cache differs")
        result["reused"] = True
        result["offline_input_manifest_sha256"] = metadata["content_hash"]
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if npz_path.exists():
        raise FileExistsError("offline input NPZ exists without its manifest")
    split_map = {name: [] for name in manifest.splits}
    split_map["model_train"] = identities
    selected = replace(
        manifest,
        split_sizes={
            **manifest.split_sizes,
            "model_train": len(identities),
        },
        splits=split_map,
    )
    view = load_offline_view(
        selected,
        "model_train",
        data_dir=args.data_dir,
        tree_name="tree",
        max_constits=128,
        verify_label_branches=True,
    )
    canonical_ids = np.asarray(
        [identity.key() for identity in identities], dtype=np.str_
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".offline_inputs.", suffix=".npz", dir=output_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            tokens=np.asarray(view.tokens, dtype=np.float32),
            mask=np.asarray(view.mask, dtype=bool),
            labels=np.asarray(view.labels, dtype=np.int64),
            identities=canonical_ids,
        )
        temporary.replace(npz_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata = bind_source(
        with_content_hash(
            {
                "contract": CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": campaign["content_hash"],
                "logical_role": args.logical_role,
                "identity_manifest_sha256": identity_parent,
                "event_count": len(identities),
                "npz_filename": npz_path.name,
                "npz_sha256": _file_sha256(npz_path),
                "raw_input_schema_sha256": campaign[
                    "parent_artifact_hashes"
                ]["raw_input_schema"],
                "contains_model_output": False,
                "checkpoint_loaded": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    publication = write_immutable_json(metadata_path, metadata)
    result.update(
        {
            "reused": False,
            "offline_input_manifest_sha256": metadata["content_hash"],
            "publication": publication,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
