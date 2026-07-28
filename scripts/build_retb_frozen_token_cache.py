#!/usr/bin/env python3
"""Publish one identity-bound seven-expert frozen-token fusion cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    load_hashed_json,
    require_sha256,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_cache import (
    publish_frozen_token_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import EXPERT_ORDER
from teacher_logit_reco.relation_expert_token_bridge.provenance import source_snapshot
from teacher_logit_reco.relation_expert_token_bridge.workflow import (
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def _named_paths(rows: Sequence[str]) -> dict[str, Path]:
    output = {}
    for row in rows:
        if "=" not in row:
            raise ValueError("--expert-registration requires EXPERT=PATH")
        name, path = row.split("=", 1)
        if name in output:
            raise ValueError("expert registration is duplicated")
        output[name] = Path(path)
    if set(output) != set(EXPERT_ORDER):
        raise ValueError("exactly seven expert registrations are required")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("model_train", "val_stop", "val_design"))
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--shape-id", required=True)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--expert-registration", action="append", default=[])
    parser.add_argument("--identity-manifest-sha256", required=True)
    parser.add_argument("--label-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    load_and_validate_campaign_source(args.campaign_root, repo_root=REPO_ROOT)
    role = "training_worker" if args.split != "val_design" else "design_worker"
    authorize_dataset_access(worker_role=role, requested_resource=args.split)
    paths = _named_paths(args.expert_registration)
    registrations = {
        name: load_hashed_json(path) for name, path in paths.items()
    }
    for name, registration in registrations.items():
        if (
            registration.get("contract") != "retb_offline_expert_registration_v1"
            or registration.get("expert_id") != name
            or registration.get("shape_id") != args.shape_id
            or registration.get("seed") != args.pipeline_seed
            or registration.get("fixed_epoch_budget_completed") is not True
        ):
            raise ValueError(f"expert registration {name} is ineligible")
    resolved = {
        "dry_run": bool(args.dry_run),
        "split": args.split,
        "pipeline_seed": args.pipeline_seed,
        "shape_id": args.shape_id,
        "input_npz": str(args.input_npz.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "expert_registration_hashes": {
            name: value["content_hash"] for name, value in registrations.items()
        },
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return 0
    with np.load(args.input_npz, allow_pickle=False) as payload:
        required = {"identities", "labels"} | {
            f"{kind}_{name}"
            for name in EXPERT_ORDER
            for kind in ("tokens", "logits")
        }
        if set(payload.files) != required:
            raise ValueError("frozen-token input NPZ fields differ")
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    manifest = publish_frozen_token_cache(
        output_dir=args.output_dir,
        split=args.split,
        pipeline_seed=args.pipeline_seed,
        shape_id=args.shape_id,
        identities=arrays["identities"],
        labels=arrays["labels"],
        token_banks={
            name: arrays[f"tokens_{name}"] for name in EXPERT_ORDER
        },
        expert_logits={
            name: arrays[f"logits_{name}"] for name in EXPERT_ORDER
        },
        expert_checkpoint_hashes={
            name: require_sha256(
                registrations[name]["checkpoint_sha256"],
                name=f"{name}.checkpoint_sha256",
            )
            for name in EXPERT_ORDER
        },
        expert_registration_hashes={
            name: registrations[name]["content_hash"] for name in EXPERT_ORDER
        },
        identity_manifest_sha256=args.identity_manifest_sha256,
        label_manifest_sha256=args.label_manifest_sha256,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    print(json.dumps({**resolved, "manifest": manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
