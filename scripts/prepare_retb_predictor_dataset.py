#!/usr/bin/env python3
"""Join one HLT evidence view to one authenticated offline target cache."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

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
from teacher_logit_reco.relation_expert_token_bridge.fusion import (  # noqa: E402
    build_fusion_model,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_training import (  # noqa: E402
    evaluate_fusion,
    make_fusion_loader,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.target_cache import (  # noqa: E402
    identity_order_sha256,
    load_offline_target_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.target_coordinates import (  # noqa: E402
    target_slot_queries,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)

import torch  # noqa: E402


PREDICTOR_DATASET_PREPARATION_CONTRACT = (
    "retb_predictor_dataset_preparation_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _queries(
    checkpoint: Path,
    *,
    expected_sha256: str,
    expected_query_sha256: str,
    target_mode: str,
) -> np.ndarray:
    if checkpoint.is_symlink() or _sha256(checkpoint) != expected_sha256:
        raise ValueError("predictor target checkpoint bytes differ")
    queries = target_slot_queries(
        checkpoint, target_mode=target_mode
    )
    if (
        hashlib.sha256(queries.tobytes(order="C")).hexdigest()
        != expected_query_sha256
    ):
        raise ValueError("predictor offline slot-query hash differs")
    return queries


def _publish_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    data = stream.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise ValueError("predictor dataset output already differs")
    else:
        path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--target-cache-manifest", required=True, type=Path)
    parser.add_argument("--evidence-npz", required=True, type=Path)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--fusion-checkpoint", required=True, type=Path)
    parser.add_argument("--expert-id", required=True, choices=EXPERT_ORDER)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    target_manifest = load_hashed_json(args.target_cache_manifest)
    specification = load_hashed_json(
        args.target_cache_manifest.parent
        / "target_cache_specification.json"
    )
    manifest, targets = load_offline_target_cache(
        args.target_cache_manifest,
        expected_pipeline_seed=args.pipeline_seed,
        expected_specification_sha256=specification["content_hash"],
    )
    evidence = _npz(args.evidence_npz)
    evidence_ids = np.asarray(evidence["identities"]).astype(str)
    identities = evidence_ids
    if (
        not np.array_equal(targets["labels"], evidence["labels"])
        or len(identities) != int(manifest["event_count"])
        or identity_order_sha256(
            identities.tolist(), targets["labels"]
        )
        != manifest["identity_order_sha256"]
    ):
        raise ValueError("predictor target/evidence identity join differs")
    expert = args.expert_id
    descriptor = manifest["target_descriptors"][expert]
    queries = _queries(
        args.target_checkpoint,
        expected_sha256=descriptor["checkpoint_sha256"],
        expected_query_sha256=descriptor["slot_query_sha256"],
        target_mode=descriptor["target_mode"],
    )
    required_evidence = {
        "identities",
        "labels",
        "unbiased_particle_states",
        "particle_mask",
        *{f"hlt_tokens_{name}" for name in EXPERT_ORDER},
    }
    if not required_evidence <= set(evidence):
        raise ValueError("predictor evidence NPZ fields are incomplete")
    if (
        _sha256(args.fusion_checkpoint)
        != specification["offline_fusion_checkpoint_sha256"]
    ):
        raise ValueError("predictor target fusion checkpoint bytes differ")
    fusion_payload = torch.load(
        args.fusion_checkpoint, map_location="cpu", weights_only=False
    )
    fusion = build_fusion_model(
        "F_TOKEN_TRANSFORMER",
        bank_dimensions={
            name: int(manifest["allocation"][name][1])
            for name in EXPERT_ORDER
        },
    )
    fusion.load_state_dict(
        fusion_payload["model_state_dict"], strict=True
    )
    _, fusion_prediction = evaluate_fusion(
        fusion,
        make_fusion_loader(
            {
                "identities": identities,
                "labels": targets["labels"],
                "token_banks": targets["tokens"],
                "expert_logits": targets["expert_logits"],
            },
            batch_size=512,
            seed=0,
            training=False,
        ),
        device="cpu",
        split=str(manifest["split"]),
    )
    arrays = {
        name: value
        for name, value in evidence.items()
        if name in required_evidence
        or name.startswith("relation_particle_")
    }
    arrays.update(
        {
            "target_tokens": targets["tokens"][expert],
            "target_expert_logits": targets["expert_logits"][expert],
            "target_hybrid_logits": np.asarray(
                fusion_prediction["logits"], dtype=np.float32
            ),
            "offline_slot_queries": queries,
            **{
                f"oracle_tokens_{name}": targets["tokens"][name]
                for name in EXPERT_ORDER
            },
        }
    )
    npz_sha = _publish_npz(args.output, arrays)
    metadata = bind_source(
        with_content_hash(
            {
                "contract": PREDICTOR_DATASET_PREPARATION_CONTRACT,
                "schema_version": 1,
                "split": manifest["split"],
                "pipeline_seed": int(args.pipeline_seed),
                "expert_id": expert,
                "event_count": len(identities),
                "target_cache_manifest_sha256": manifest["content_hash"],
                "target_cache_specification_sha256": specification[
                    "content_hash"
                ],
                "evidence_npz_sha256": _sha256(args.evidence_npz),
                "target_checkpoint_sha256": descriptor[
                    "checkpoint_sha256"
                ],
                "offline_fusion_checkpoint_sha256": specification[
                    "offline_fusion_checkpoint_sha256"
                ],
                "slot_query_sha256": descriptor["slot_query_sha256"],
                "prepared_npz_sha256": npz_sha,
                "identity_join_exact": True,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    publication = write_immutable_json(args.metadata_output, metadata)
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
