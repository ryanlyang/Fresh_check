#!/usr/bin/env python3
"""Certify one Stage-E bridge candidate on authenticated val_design arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.bridge_certification import (  # noqa: E402
    certify_bridge_content,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.dynamic_continuation import (  # noqa: E402
    add_dynamic_continuation_arguments,
    resolve_selector_continuation,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--candidate-registration", required=True, type=Path)
    parser.add_argument("--t0-registration", required=True, type=Path)
    parser.add_argument("--t0-normalizer", required=True, type=Path)
    parser.add_argument("--bridge-normalizer", type=Path)
    parser.add_argument("--identity-manifest", required=True, type=Path)
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    add_dynamic_continuation_arguments(parser)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    candidate = load_hashed_json(args.candidate_registration)
    t0 = load_hashed_json(args.t0_registration)
    t0_normalizer = load_hashed_json(args.t0_normalizer)
    bridge_normalizer = (
        load_hashed_json(args.bridge_normalizer)
        if args.bridge_normalizer is not None
        else t0_normalizer
    )
    identities_manifest = load_hashed_json(args.identity_manifest)
    if (
        t0_normalizer.get("source") != campaign.get("source")
        or bridge_normalizer.get("source") != campaign.get("source")
        or identities_manifest.get("source") != campaign.get("source")
        or any(
            parent.get("source") is not None
            and parent.get("source") != campaign.get("source")
            for parent in (candidate, t0)
        )
    ):
        raise ValueError("bridge certification source lineage differs")
    if (
        candidate.get("contract") != "retb_bridge_candidate_registration_v1"
        or t0_normalizer.get("contract") != "retb_bridge_token_normalizer_v1"
        or bridge_normalizer.get("contract")
        != "retb_bridge_token_normalizer_v1"
        or t0_normalizer.get("expert_id") != candidate.get("expert_id")
        or bridge_normalizer.get("expert_id") != candidate.get("expert_id")
        or t0_normalizer.get("target_checkpoint_sha256")
        != t0.get("checkpoint_sha256")
        or t0.get("expert_id") != candidate.get("expert_id")
        or candidate.get("parent_checkpoint_hashes", {}).get("T0_checkpoint")
        != t0.get("checkpoint_sha256")
    ):
        raise ValueError("bridge certification parent identity differs")
    if candidate["target_mode"] == "T2_PROJECT":
        if (
            args.bridge_normalizer is None
            or bridge_normalizer.get("target_checkpoint_sha256")
            != candidate.get("checkpoint_sha256")
        ):
            raise ValueError("T2 certification requires its bridge normalizer")
    elif args.bridge_normalizer is not None:
        raise ValueError("T1 certification uses the T0 normalizer on both sides")
    authorize_dataset_access(
        worker_role="design_worker", requested_resource="val_design"
    )
    with np.load(args.arrays, allow_pickle=False) as payload:
        required = {
            "identities",
            "labels",
            "moving_tokens",
            "t0_tokens",
            "predicted_hlt_tokens",
            "moving_expert_logits",
            "moving_fusion_logits",
            "t0_expert_logits",
            "t0_fusion_logits",
        }
        fields = set(payload.files)
        if fields != required and fields != required | {"decoded_tokens"}:
            raise ValueError("bridge certification NPZ fields differ")
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    def normalize(values, artifact):
        data = np.asarray(values, dtype=np.float32)
        mean = np.asarray(artifact["mean"], dtype=np.float32)
        scale = np.maximum(
            np.asarray(artifact["standard_deviation"], dtype=np.float32),
            np.float32(artifact["standard_deviation_floor"]),
        )
        if data.shape[1:] != mean.shape or mean.shape != scale.shape:
            raise ValueError("bridge certification normalizer shape differs")
        normalized = (data - mean[None]) / scale[None]
        if not np.isfinite(normalized).all():
            raise FloatingPointError("bridge certification normalization is nonfinite")
        return normalized
    moving = normalize(arrays["moving_tokens"], bridge_normalizer)
    predicted = normalize(arrays["predicted_hlt_tokens"], bridge_normalizer)
    pure = normalize(arrays["t0_tokens"], t0_normalizer)
    decoded = (
        None
        if "decoded_tokens" not in arrays
        else normalize(arrays["decoded_tokens"], t0_normalizer)
    )
    certification = bind_source(
        certify_bridge_content(
            target_mode=candidate["target_mode"],
            expert_id=candidate["expert_id"],
            shape_id=candidate["shape_id"],
            pipeline_seed=int(candidate["pipeline_seed"]),
            moving_tokens=moving,
            t0_tokens=pure,
            predicted_hlt_tokens=predicted,
            frozen_moving_logits={
                "expert": arrays["moving_expert_logits"],
                "fusion": arrays["moving_fusion_logits"],
            },
            frozen_t0_logits={
                "expert": arrays["t0_expert_logits"],
                "fusion": arrays["t0_fusion_logits"],
            },
            identities=[str(value) for value in arrays["identities"].tolist()],
            labels=arrays["labels"],
            candidate_checkpoint_sha256=candidate["checkpoint_sha256"],
            t0_checkpoint_sha256=t0["checkpoint_sha256"],
            identity_manifest_sha256=identities_manifest["content_hash"],
            coordinate_normalizer_sha256=bridge_normalizer["content_hash"],
            t0_normalizer_sha256=t0_normalizer["content_hash"],
            decoded_tokens=decoded,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    certification["input_npz_sha256"] = _sha256(args.arrays)
    # Rehash after binding the exact array bytes.
    certification.pop("content_hash")
    certification = with_content_hash(certification)
    result = {
        "dry_run": bool(args.dry_run),
        "certification": certification,
        "output": str(args.output.resolve()),
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, certification)
    continuation = resolve_selector_continuation(
        args=args,
        campaign=campaign,
        campaign_root=args.campaign_root,
        selector_output=certification,
        selector_output_path=args.output,
        load_hashed_json=load_hashed_json,
        dry_run=bool(args.dry_run),
    )
    if continuation is not None:
        result["continuation"] = continuation
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
