#!/usr/bin/env python3
"""Seal seven native-HLT expert output replicas into a fusion cache."""

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

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.native_fusion import (  # noqa: E402
    publish_native_fusion_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def _expert_outputs(rows: Sequence[str]) -> dict[tuple[str, int], Path]:
    output = {}
    for row in rows:
        if "=" not in row or ":" not in row.split("=", 1)[0]:
            raise ValueError("--expert-output requires EXPERT:REPLICA=NPZ")
        key, value = row.split("=", 1)
        expert, replica_text = key.split(":", 1)
        pair = (expert, int(replica_text))
        if expert not in EXPERT_ORDER or pair in output:
            raise ValueError("expert-output key is unknown or duplicated")
        output[pair] = Path(value)
    return output


def _registrations(rows: Sequence[str]) -> dict[str, Path]:
    output = {}
    for row in rows:
        if "=" not in row:
            raise ValueError("--expert-registration requires EXPERT=JSON")
        expert, value = row.split("=", 1)
        if expert not in EXPERT_ORDER or expert in output:
            raise ValueError("expert registration is unknown or duplicated")
        output[expert] = Path(value)
    if set(output) != set(EXPERT_ORDER):
        raise ValueError("expert registration coverage differs")
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replica_manifests(rows: Sequence[str]) -> dict[int, Path]:
    output = {}
    for row in rows:
        if "=" not in row:
            raise ValueError("--hlt-cache-manifest requires REPLICA=JSON")
        replica, value = row.split("=", 1)
        key = int(replica)
        if key not in range(4) or key in output:
            raise ValueError("HLT cache replica is invalid or duplicated")
        output[key] = Path(value)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--shape-id", required=True)
    parser.add_argument("--realization-policy", required=True)
    parser.add_argument("--expert-output", action="append", default=[])
    parser.add_argument("--expert-registration", action="append", default=[])
    parser.add_argument("--expert-output-manifest", action="append", default=[])
    parser.add_argument("--hlt-cache-manifest", action="append", default=[])
    parser.add_argument("--identity-manifest", required=True, type=Path)
    parser.add_argument("--label-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    requested = _expert_outputs(args.expert_output)
    registration_paths = _registrations(args.expert_registration)
    output_manifest_paths = _registrations(args.expert_output_manifest)
    hlt_paths = _replica_manifests(args.hlt_cache_manifest)
    expected_replicas = (
        {0}
        if args.split not in {"model_train", "scale_train"}
        or args.realization_policy == "R_FIXED"
        else {0, 1, 2, 3}
    )
    if set(hlt_paths) != expected_replicas or set(requested) != {
        (expert, replica)
        for expert in EXPERT_ORDER
        for replica in expected_replicas
    }:
        raise ValueError("native fusion input replica coverage differs")
    registrations = {
        expert: load_hashed_json(path)
        for expert, path in registration_paths.items()
    }
    output_manifests = {
        expert: load_hashed_json(path)
        for expert, path in output_manifest_paths.items()
    }
    hlt_manifests = {
        replica: load_hashed_json(path) for replica, path in hlt_paths.items()
    }
    identity_manifest = load_hashed_json(args.identity_manifest)
    label_manifest = load_hashed_json(args.label_manifest)
    for artifact in (
        *registrations.values(),
        *output_manifests.values(),
        *hlt_manifests.values(),
        identity_manifest,
        label_manifest,
    ):
        source = artifact.get("source")
        if source is not None and source != campaign.get("source"):
            raise ValueError("native fusion cache parent belongs to another source")
    for expert in EXPERT_ORDER:
        manifest = output_manifests[expert]
        if (
            manifest.get("contract")
            not in {
                "retb_native_hlt_expert_outputs_v3",
                "retb_native_hlt_expert_outputs_v4",
            }
            or manifest.get("expert_id") != expert
            or manifest.get("expert_registration_sha256")
            != registrations[expert]["content_hash"]
            or int(manifest.get("pipeline_seed", -1)) != args.pipeline_seed
            or manifest.get("realization_policy") != args.realization_policy
        ):
            raise ValueError("native expert output-manifest lineage differs")
    identities = None
    labels = None
    token_banks = {replica: {} for replica in expected_replicas}
    expert_logits = {replica: {} for replica in expected_replicas}
    unbiased_states = {}
    particle_masks = {}
    for (expert, replica), path in sorted(requested.items()):
        output_manifest_path = output_manifest_paths[expert]
        output_row = output_manifests[expert]["files"].get(
            f"{args.split}_replica_{replica}"
        )
        expected_path = (
            output_manifest_path.parent / output_row["relative_path"]
            if output_row is not None
            else None
        )
        if (
            output_row is None
            or path.resolve() != expected_path.resolve()
            or _sha256(path) != output_row["file_sha256"]
        ):
            raise ValueError("native expert output bytes/manifest differ")
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {
                "identities",
                "labels",
                "tokens",
                "logits",
                "particle_states",
                "particle_mask",
            }:
                raise ValueError("native expert output NPZ fields differ")
            current_ids = np.asarray(payload["identities"])
            current_labels = np.asarray(payload["labels"], dtype=np.int64)
            current_tokens = np.asarray(payload["tokens"], dtype=np.float32)
            current_logits = np.asarray(payload["logits"], dtype=np.float32)
            if expert == "BASE4":
                unbiased_states[replica] = np.asarray(
                    payload["particle_states"], dtype=np.float32
                )
                particle_masks[replica] = np.asarray(
                    payload["particle_mask"], dtype=bool
                )
        if identities is None:
            identities, labels = current_ids, current_labels
        elif not np.array_equal(identities, current_ids) or not np.array_equal(
            labels, current_labels
        ):
            raise ValueError("native expert output populations differ")
        token_banks[replica][expert] = current_tokens
        expert_logits[replica][expert] = current_logits
    result = {
        "dry_run": bool(args.dry_run),
        "split": args.split,
        "replica_ids": sorted(expected_replicas),
        "event_count": len(identities),
        "output_dir": str(args.output_dir.resolve()),
    }
    if not args.dry_run:
        authorize_dataset_access(
            worker_role=(
                "scale_training_worker"
                if args.split == "scale_train"
                else "training_worker"
            ),
            requested_resource=args.split,
        )
        result["manifest"] = publish_native_fusion_cache(
            output_dir=args.output_dir,
            split=args.split,
            pipeline_seed=args.pipeline_seed,
            shape_id=args.shape_id,
            realization_policy=args.realization_policy,
            identities=[str(value) for value in identities.tolist()],
            labels=labels,
            token_banks_by_replica=token_banks,
            expert_logits_by_replica=expert_logits,
            unbiased_particle_states_by_replica=unbiased_states,
            particle_masks_by_replica=particle_masks,
            expert_registration_hashes={
                expert: registration["content_hash"]
                for expert, registration in registrations.items()
            },
            hlt_cache_hashes_by_replica={
                replica: manifest["content_hash"]
                for replica, manifest in hlt_manifests.items()
            },
            identity_manifest_sha256=identity_manifest["content_hash"],
            label_manifest_sha256=label_manifest["content_hash"],
            source_snapshot=source_snapshot(REPO_ROOT),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
