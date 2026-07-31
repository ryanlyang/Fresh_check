#!/usr/bin/env python3
"""Build one deterministic val-design degradation view for Stage K."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fixed_hlt import (  # noqa: E402
    build_fixed_hlt_v2_realistic_view,
    build_fixed_hlt_view,
)
from jetclass_fresh.part_inputs import (  # noqa: E402
    build_particle_transformer_inputs_from_tokens,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    canonical_sha256,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    build_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    build_compiled_tree,
    load_tree_backend,
)


CONTRACT = "retb_stage_k_robustness_view_v1"
PROFILES = (
    "D_OFFLINE_IDENTITY",
    "D_KIN_ONLY",
    "D_TRACK_ONLY",
    "D_MISSING_ONLY",
    "D_MILD",
    "D_NOMINAL",
    "D_SEVERE",
    "D_LEGACY_V1",
    "D_LEGACY_V2",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _legacy(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    profile: str,
    replica: int,
) -> tuple[np.ndarray, np.ndarray]:
    seed = 730_001 + 100_003 * int(replica)
    if profile == "D_LEGACY_V1":
        values, valid, _ = build_fixed_hlt_view(tokens, mask, seed=seed)
    else:
        values, valid, _ = build_fixed_hlt_v2_realistic_view(
            tokens, mask, seed=seed
        )
    return (
        np.asarray(values, dtype=np.float32),
        np.asarray(valid, dtype=bool),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--replica", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.replica not in range(4):
        raise ValueError("robustness replica lies outside 0..3")
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    offline_manifest = __import__(
        "teacher_logit_reco.relation_expert_token_bridge.contracts",
        fromlist=["load_hashed_json"],
    ).load_hashed_json(
        root / "inputs" / "offline" / "val_design" / "offline_input_manifest.json"
    )
    offline_npz = (
        root
        / "inputs"
        / "offline"
        / "val_design"
        / offline_manifest["npz_filename"]
    )
    if _sha256(offline_npz) != offline_manifest["npz_sha256"]:
        raise ValueError("robustness offline source bytes differ")
    with np.load(offline_npz, allow_pickle=False) as payload:
        tokens = np.asarray(payload["tokens"], dtype=np.float32)
        mask = np.asarray(payload["mask"], dtype=bool)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        identities = [str(value) for value in payload["identities"].tolist()]
    if args.profile.startswith("D_LEGACY"):
        degraded, degraded_mask = _legacy(
            tokens, mask, profile=args.profile, replica=args.replica
        )
    else:
        arrays, _ = build_hlt_v3_cache(
            tokens,
            mask,
            canonical_identities=identities,
            logical_role="val_design",
            replica_id=args.replica,
            realization_policy="R_FIXED",
            profile_id=args.profile,
        )
        degraded = np.asarray(arrays["tokens"], dtype=np.float32)
        degraded_mask = np.asarray(arrays["mask"], dtype=bool)
    inputs = build_particle_transformer_inputs_from_tokens(
        degraded,
        degraded_mask,
        labels=np.zeros(len(identities), dtype=np.int64),
        source_view="hlt",
    )
    backend_manifest = root / "backend" / "backend_manifest.json"
    backend_payload = __import__(
        "teacher_logit_reco.relation_expert_token_bridge.contracts",
        fromlist=["load_hashed_json"],
    ).load_hashed_json(backend_manifest)
    backend = load_tree_backend(
        backend_manifest.parent / backend_payload["binary_filename"],
        backend_manifest,
        source_path=(
            REPO_ROOT
            / "teacher_logit_reco"
            / "relational_part"
            / "csrc"
            / "relational_ca_tree_v1.cpp"
        ),
    )
    vectors = inputs.pf_vectors.transpose(0, 2, 1)
    trees = [
        build_compiled_tree(backend, vectors[index], degraded[index], degraded_mask[index])
        for index in range(len(identities))
    ]
    identity_hash = canonical_sha256(identities)
    degraded_hashes = []
    for index, identity in enumerate(identities):
        digest = hashlib.sha256()
        digest.update(b"retb_stage_k_view_v1\0")
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(args.profile.encode("ascii"))
        digest.update(bytes((args.replica,)))
        digest.update(degraded[index].tobytes(order="C"))
        digest.update(degraded_mask[index].tobytes(order="C"))
        degraded_hashes.append(digest.hexdigest())
    payload: dict[str, Any] = {
        "identities": identities,
        "labels": labels,
        "replica_ids": torch.full(
            (len(identities),), args.replica, dtype=torch.int64
        ),
        "degraded_view_hashes": degraded_hashes,
        "features": torch.from_numpy(inputs.pf_features),
        "vectors": torch.from_numpy(inputs.pf_vectors),
        "mask": torch.from_numpy(inputs.pf_mask),
        "raw_tokens": torch.from_numpy(degraded),
        "region_trees_by_expert": {
            expert: trees for expert in EXPERT_ORDER
        },
    }
    stream = io.BytesIO()
    torch.save({"contract": CONTRACT, "view": payload}, stream)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    binary = write_immutable_bytes(
        args.output_dir / "robustness_view.pt", stream.getvalue()
    )
    manifest = bind_source(
        with_content_hash(
            {
                "contract": CONTRACT,
                "schema_version": 1,
                "profile": args.profile,
                "replica": args.replica,
                "split": "val_design",
                "event_count": len(identities),
                "identity_order_sha256": identity_hash,
                "offline_input_manifest_sha256": offline_manifest[
                    "content_hash"
                ],
                "view_filename": "robustness_view.pt",
                "view_sha256": binary["file_sha256"],
                "backend_manifest_sha256": backend_payload["content_hash"],
                "labels_persisted_only_for_evaluation": True,
                "stack_val_consumed": False,
                "final_test_consumed": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(args.output_dir / "robustness_view.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
