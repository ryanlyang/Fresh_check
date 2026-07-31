#!/usr/bin/env python3
"""Combine scale-trained native HLT outputs into predictor evidence."""

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
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


EVIDENCE_MANIFEST_CONTRACT = "retb_scale_predictor_evidence_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    data = stream.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise FileExistsError("scale predictor evidence differs")
    else:
        path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--split",
        required=True,
        choices=("scale_train", "val_stop", "val_design"),
    )
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--shape-role", required=True)
    parser.add_argument(
        "--native-output",
        action="append",
        required=True,
        metavar="EXPERT=MANIFEST",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    authorize_dataset_access(
        worker_role=(
            "scale_training_worker"
            if args.split == "scale_train"
            else "training_worker"
            if args.split == "val_stop"
            else "design_worker"
        ),
        requested_resource=args.split,
    )
    paths = {}
    for raw in args.native_output:
        if "=" not in raw:
            raise ValueError("--native-output requires EXPERT=MANIFEST")
        expert, value = raw.split("=", 1)
        if expert in paths:
            raise ValueError("native output expert is duplicated")
        paths[expert] = Path(value)
    if set(paths) != set(EXPERT_ORDER):
        raise ValueError("native output expert coverage differs")
    replicas = (0, 1, 2, 3) if args.split == "scale_train" else (0,)
    manifests, values = {}, {}
    identities = labels = None
    for expert in EXPERT_ORDER:
        manifest = load_hashed_json(paths[expert])
        if (
            manifest.get("contract") != "retb_native_hlt_expert_outputs_v4"
            or manifest.get("source") != campaign.get("source")
            or manifest.get("expert_id") != expert
            or int(manifest.get("pipeline_seed", -1))
            != args.pipeline_seed
            or manifest.get("training_population") != "scale_train"
        ):
            raise ValueError("scale native-output lineage differs")
        manifests[expert] = manifest
        rows = []
        for replica in replicas:
            record = manifest["files"][f"{args.split}_replica_{replica}"]
            path = paths[expert].parent / record["relative_path"]
            if _sha256(path) != record["file_sha256"]:
                raise ValueError("scale native-output bytes differ")
            with np.load(path, allow_pickle=False) as payload:
                row = {
                    name: np.asarray(payload[name])
                    for name in payload.files
                }
            if identities is None:
                identities = row["identities"]
                labels = row["labels"].astype(np.int64, copy=False)
            elif not np.array_equal(identities, row["identities"]) or not np.array_equal(
                labels, row["labels"]
            ):
                raise ValueError("scale predictor evidence identities differ")
            rows.append(row)
        values[expert] = rows
    arrays = {
        "identities": identities,
        "labels": labels,
        **{
            f"hlt_tokens_{expert}": (
                values[expert][0]["tokens"]
                if len(replicas) == 1
                else np.stack(
                    [row["tokens"] for row in values[expert]], axis=0
                )
            )
            for expert in EXPERT_ORDER
        },
        "unbiased_particle_states": (
            values["BASE4"][0]["particle_states"]
            if len(replicas) == 1
            else np.stack(
                [
                    row["particle_states"]
                    for row in values["BASE4"]
                ],
                axis=0,
            )
        ),
        "particle_mask": (
            values["BASE4"][0]["particle_mask"]
            if len(replicas) == 1
            else np.stack(
                [row["particle_mask"] for row in values["BASE4"]],
                axis=0,
            )
        ),
        **{
            f"relation_particle_states_{expert}": (
                values[expert][0]["particle_states"]
                if len(replicas) == 1
                else np.stack(
                    [
                        row["particle_states"]
                        for row in values[expert]
                    ],
                    axis=0,
                )
            )
            for expert in ("PT", "TRACK", "REGION")
        },
        **{
            f"relation_particle_mask_{expert}": (
                values[expert][0]["particle_mask"]
                if len(replicas) == 1
                else np.stack(
                    [
                        row["particle_mask"]
                        for row in values[expert]
                    ],
                    axis=0,
                )
            )
            for expert in ("PT", "TRACK", "REGION")
        },
    }
    file_sha = _write_npz(args.output, arrays)
    artifact = bind_source(
        with_content_hash(
            {
                "contract": EVIDENCE_MANIFEST_CONTRACT,
                "schema_version": 1,
                "split": args.split,
                "shape_role": args.shape_role,
                "pipeline_seed": args.pipeline_seed,
                "replica_ids": list(replicas),
                "event_count": len(labels),
                "native_output_manifest_hashes": {
                    expert: manifests[expert]["content_hash"]
                    for expert in EXPERT_ORDER
                },
                "npz_sha256": file_sha,
                "npz_path": str(args.output.resolve()),
                "labels_included_for_training_only": True,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    publication = write_immutable_json(args.metadata_output, artifact)
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
