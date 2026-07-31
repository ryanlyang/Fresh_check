#!/usr/bin/env python3
"""Stream exact same-view HLT pair targets into one frozen-probe input bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    ExtractorResources,
    authorize_access,
    extract_registered_target,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    STAGE_C_PLAN_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relational_part.ca_tree import unpack_tree_shard  # noqa: E402
from teacher_logit_reco.hlt_offline_structure_distillation.wave_completion import (  # noqa: E402
    try_finalize_row_wave,
)


ROLES = ("model_train", "val_stop", "design_select")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _role_mapping(rows, name, *, replicated):
    output = {}
    for row in rows:
        key, separator, value = row.partition("=")
        if not separator:
            raise ValueError(f"{name} requires ROLE[:REPLICA]=PATH")
        pieces = key.split(":")
        role = pieces[0]
        replica = int(pieces[1]) if len(pieces) == 2 else 0
        if role not in ROLES or replica not in range(4):
            raise ValueError(f"{name} role/replica differs")
        output[(role, replica)] = Path(value)
    expected = {
        *((("model_train", replica) for replica in range(4)) if replicated else (("model_train", 0),)),
        ("val_stop", 0),
        ("design_select", 0),
    }
    if set(output) != expected:
        raise ValueError(f"{name} does not cover exact required roles/replicas")
    return output


def _simple_mapping(rows, name):
    output = {}
    for row in rows:
        role, separator, value = row.partition("=")
        if not separator or role not in ROLES or role in output:
            raise ValueError(f"{name} requires unique ROLE=PATH")
        output[role] = Path(value)
    if set(output) != set(ROLES):
        raise ValueError(f"{name} requires all roles")
    return output


def _npz(path):
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _trees(path, identities):
    manifest = load_hashed_json(path / "manifest.json")
    by_identity = {}
    for shard in sorted((path / "shards").glob("shard_*.npz")):
        shard_ids, rows = unpack_tree_shard(shard)
        for identity, tree in zip(shard_ids, rows):
            if identity in by_identity:
                raise ValueError("pair tree cache duplicates an identity")
            by_identity[identity] = tree
    if set(by_identity) != set(identities):
        raise ValueError("pair tree cache identity coverage differs")
    return tuple(by_identity[value] for value in identities), manifest


def _publish(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _npz(path)
        if set(existing) != set(arrays) or any(
            not np.array_equal(existing[name], value)
            for name, value in arrays.items()
        ):
            raise FileExistsError("pair probe input differs on reuse")
        return _sha(path)
    np.savez_compressed(path, **arrays)
    return _sha(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--hlt-input", action="append", default=[])
    parser.add_argument("--labels", action="append", default=[])
    parser.add_argument("--tap-cache", action="append", default=[])
    parser.add_argument("--tree-cache", action="append", default=[])
    parser.add_argument("--relation-normalizer", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    for resource in (
        "model_train_hlt", "model_train_labels", "model_train_targets",
        "val_stop_hlt", "val_stop_labels", "val_stop_targets",
        "design_select_hlt", "design_select_labels", "design_select_targets",
    ):
        authorize_access(worker_role="probe_worker", requested_resource=resource)
    plan = load_hashed_json(
        args.campaign_root / "job_ledgers" / "stage_c_execution_plan.json",
        expected_contract=STAGE_C_PLAN_CONTRACT,
    )
    row = next((item for item in plan["probe_rows"] if item["row_id"] == args.row_id), None)
    if row is None or row["target_id"] not in {
        "T_HLT_TRACK_PAIR_13", "T_HLT_REGION_PAIR_8"
    }:
        raise ValueError("row is not a current HLT pair probe")
    inputs = _role_mapping(args.hlt_input, "--hlt-input", replicated=True)
    labels = _simple_mapping(args.labels, "--labels")
    taps = _simple_mapping(args.tap_cache, "--tap-cache")
    requires_tree = row["target_id"] == "T_HLT_REGION_PAIR_8"
    trees = (
        _role_mapping(args.tree_cache, "--tree-cache", replicated=True)
        if requires_tree
        else {}
    )
    normalizer = load_hashed_json(args.relation_normalizer)
    if normalizer.get("source") != campaign["source"]:
        raise ValueError("pair normalizer source differs")
    floors = normalizer.get("track_uncertainty_floors", {})
    resources = ExtractorResources(
        d0_uncertainty_floor=float(floors.get("d0", {}).get("floor", 0.0)),
        dz_uncertainty_floor=float(floors.get("dz", {}).get("floor", 0.0)),
        sentinel_policy=normalizer.get("track_sentinel_policy"),
    )
    parents = {
        "relation_normalizer": normalizer["content_hash"],
        **{f"hlt_{role}_{replica}": _sha(path) for (role, replica), path in inputs.items()},
        **{f"labels_{role}": _sha(path) for role, path in labels.items()},
        **{f"tap_{role}": _sha(path) for role, path in taps.items()},
    }
    extracted = {}
    canonical_ids = {}
    label_values = {}
    for role in ROLES:
        label_npz = _npz(labels[role])
        ids = tuple(str(value) for value in label_npz["identities"].tolist())
        canonical_ids[role] = ids
        label_values[role] = np.asarray(label_npz["labels"], dtype=np.int64)
        replicas = range(4) if role == "model_train" else (0,)
        role_batches = []
        for replica in replicas:
            view = _npz(inputs[(role, replica)])
            view_ids = tuple(
                str(value) for value in view[
                    "identity" if "identity" in view else "identities"
                ].tolist()
            )
            if view_ids != ids:
                raise ValueError(f"{role} replica {replica} identities differ")
            tree_rows = None
            if requires_tree:
                tree_rows, tree_manifest = _trees(trees[(role, replica)], ids)
                parents[f"tree_{role}_{replica}"] = tree_manifest["content_hash"]
            batch = extract_registered_target(
                row["target_id"],
                np.asarray(view["raw_tokens"], dtype=np.float32),
                np.asarray(view["mask"], dtype=bool),
                resources=resources,
                vectors=(
                    None if "vectors" not in view
                    else np.asarray(view["vectors"], dtype=np.float32)
                ),
                trees=tree_rows,
            )
            role_batches.append(
                (
                    np.moveaxis(
                        batch.values.detach().cpu().numpy(), 1, -1
                    ).astype(np.float32),
                    np.moveaxis(
                        batch.loss_mask.detach().cpu().numpy(), 1, -1
                    ).astype(bool),
                )
            )
        extracted[role] = role_batches
    lineage = with_content_hash({
        "contract": "hosd_probe_input_lineage_v1",
        "schema_version": 1,
        "source": campaign["source"],
        "campaign_spec_sha256": campaign["content_hash"],
        "stage_c_plan_sha256": plan["content_hash"],
        "row_id": args.row_id,
        "target_id": row["target_id"],
        "streaming": "same_view_pair_target_recomputed_from_bound_HLT_replica",
        "parents": parents,
    })
    output = args.output_dir or args.campaign_root / "probes" / "inputs" / args.row_id
    write_immutable_json(output / "input_lineage.json", lineage)
    files = {}
    for role in ROLES:
        tap = _npz(taps[role])
        tap_ids = tuple(str(value) for value in tap["identities"].tolist())
        if tap_ids != canonical_ids[role] or str(tap["tap"].item()) != row["tap"]:
            raise ValueError(f"{role} pair tap lineage differs")
        values = (
            np.stack([item[0] for item in extracted[role]])
            if role == "model_train" else extracted[role][0][0]
        )
        masks = (
            np.stack([item[1] for item in extracted[role]])
            if role == "model_train" else extracted[role][0][1]
        )
        arrays = {
            "identities": np.asarray(canonical_ids[role]),
            "labels": label_values[role],
            "target": values,
            "target_mask": masks,
            "states": tap["states"].astype(np.float32),
            "particle_mask": tap["particle_mask"].astype(bool),
            "probe_encoder_lock_sha256": tap["probe_encoder_lock_sha256"],
            "tap": tap["tap"],
            "target_cache_manifest_sha256": np.asarray(lineage["content_hash"]),
        }
        path = output / f"{role}.npz"
        files[role] = {"path": str(path.resolve()), "sha256": _publish(path, arrays)}
    completion = with_content_hash({
        "contract": "hosd_probe_input_completion_v1",
        "schema_version": 1,
        "source": campaign["source"],
        "row_id": args.row_id,
        "input_lineage_sha256": lineage["content_hash"],
        "files": files,
    })
    write_immutable_json(output / "completion.json", completion)
    wave = try_finalize_row_wave(
        wave_id="stage_c_probe_inputs",
        expected_paths={
            item["row_id"]: args.campaign_root
            / "probes"
            / "inputs"
            / item["row_id"]
            / "completion.json"
            for item in plan["probe_rows"]
        },
        expected_rows={
            item["row_id"]: {
                "row_id": item["row_id"],
                "target_id": item["target_id"],
                "probe_kind": item["probe_kind"],
                "tap": item["tap"],
            }
            for item in plan["probe_rows"]
        },
        expected_contract="hosd_probe_input_completion_v1",
        parent_hashes={"stage_c_plan": plan["content_hash"]},
        source=campaign["source"],
        output=args.campaign_root
        / "probes"
        / "inputs"
        / "input_completion.json",
    )
    print(json.dumps({
        "row_id": args.row_id,
        "target_cache_manifest_sha256": lineage["content_hash"],
        "completion_sha256": completion["content_hash"],
        "wave_completion_sha256": None if wave is None else wave["content_hash"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
