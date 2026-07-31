#!/usr/bin/env python3
"""Join frozen taps, labels, and persisted targets for one HOSD probe row."""

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
    authorize_access,
    load_and_validate_campaign,
    load_target_cache,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    STAGE_C_PLAN_CONTRACT,
    TARGET_CACHE_SPEC_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.wave_completion import (  # noqa: E402
    try_finalize_row_wave,
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(rows, name):
    output = {}
    for row in rows:
        key, separator, value = row.partition("=")
        if not separator or key not in {"model_train", "val_stop", "design_select"}:
            raise ValueError(f"{name} requires ROLE=PATH")
        if key in output:
            raise ValueError(f"{name} duplicates {key}")
        output[key] = Path(value)
    if set(output) != {"model_train", "val_stop", "design_select"}:
        raise ValueError(f"{name} requires all three probe roles")
    return output


def _npz(path):
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _labels(path):
    arrays = _npz(path)
    identities = tuple(str(value) for value in arrays["identities"].tolist())
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    if labels.shape != (len(identities),) or len(identities) != len(set(identities)):
        raise ValueError("probe labels differ")
    return identities, labels


def _availability(mask, target_row):
    components = target_row["components"]
    if len(components) != mask.shape[-1]:
        raise ValueError("target registry component count differs")
    order = []
    indices = {}
    for index, component in enumerate(components):
        group = str(component["availability_group"])
        if group not in indices:
            order.append(group)
            indices[group] = []
        indices[group].append(index)
    values = []
    for group in order:
        group_mask = mask[..., indices[group]]
        reference = group_mask[..., :1]
        if not np.array_equal(group_mask, np.broadcast_to(reference, group_mask.shape)):
            raise ValueError(f"availability group {group} component bits disagree")
        values.append(reference[..., 0])
    return np.stack(values, axis=-1), order


def _publish(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _npz(path)
        if set(existing) != set(arrays) or any(
            not np.array_equal(existing[name], value)
            for name, value in arrays.items()
        ):
            raise FileExistsError("probe input differs on reuse")
        return _sha(path)
    np.savez_compressed(path, **arrays)
    return _sha(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--target-cache", action="append", default=[])
    parser.add_argument("--labels", action="append", default=[])
    parser.add_argument("--tap-cache", action="append", default=[])
    parser.add_argument("--raw-input", action="append", default=[])
    parser.add_argument("--hlt-input", action="append", default=[])
    parser.add_argument("--tree-cache", action="append", default=[])
    parser.add_argument("--relation-normalizer", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    for resource in (
        "model_train_targets",
        "model_train_labels",
        "val_stop_targets",
        "val_stop_labels",
        "design_select_labels",
        "design_select_targets",
    ):
        authorize_access(worker_role="probe_worker", requested_resource=resource)
    plan = load_hashed_json(
        args.campaign_root / "job_ledgers" / "stage_c_execution_plan.json",
        expected_contract=STAGE_C_PLAN_CONTRACT,
    )
    row = next((item for item in plan["probe_rows"] if item["row_id"] == args.row_id), None)
    if row is None:
        raise ValueError("probe row is absent from Stage-C plan")
    if row["target_id"] in {"T_HLT_TRACK_PAIR_13", "T_HLT_REGION_PAIR_8"}:
        if args.relation_normalizer is None:
            raise ValueError("pair probe input requires relation normalizer")
        from scripts.materialize_hosd_pair_probe_inputs import (
            main as pair_main,
        )

        command = [
            "--campaign-root",
            str(args.campaign_root),
            "--row-id",
            args.row_id,
            "--relation-normalizer",
            str(args.relation_normalizer),
        ]
        if args.output_dir is not None:
            command.extend(["--output-dir", str(args.output_dir)])
        for option, values in (
            ("--hlt-input", args.hlt_input),
            ("--labels", args.labels),
            ("--tap-cache", args.tap_cache),
            ("--tree-cache", args.tree_cache),
        ):
            for value in values:
                command.extend([option, value])
        return pair_main(command)
    target_paths = _mapping(args.target_cache, "--target-cache")
    label_paths = _mapping(args.labels, "--labels")
    tap_paths = (
        _mapping(args.tap_cache, "--tap-cache")
        if row["probe_kind"] in {"P_LINEAR", "P_SHALLOW"}
        else {}
    )
    raw_paths = (
        _mapping(args.raw_input, "--raw-input")
        if row["probe_kind"] == "P_RAW_MLP"
        else {}
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "structure_target_registry.json",
        expected_contract="hosd_structure_target_registry_v1",
    )
    target_row = next(
        item for item in registry["targets"]
        if item["target_id"] == row["target_id"]
    )
    parents = {}
    loaded = {}
    for role, path in target_paths.items():
        spec = load_hashed_json(
            path / "cache_spec.json", expected_contract=TARGET_CACHE_SPEC_CONTRACT
        )
        cache = load_target_cache(path, cache_spec=spec)
        if (
            cache.manifest.get("source") != campaign["source"]
            or row["target_id"] not in cache.values
        ):
            raise ValueError(f"{role} target cache lineage/coverage differs")
        loaded[role] = cache
        parents[f"{role}_target_manifest"] = cache.manifest["content_hash"]
        parents[f"{role}_labels_file"] = _sha(label_paths[role])
        if tap_paths:
            parents[f"{role}_tap_file"] = _sha(tap_paths[role])
        if raw_paths:
            parents[f"{role}_raw_file"] = _sha(raw_paths[role])
    lineage = with_content_hash({
        "contract": "hosd_probe_input_lineage_v1",
        "schema_version": 1,
        "source": campaign["source"],
        "campaign_spec_sha256": campaign["content_hash"],
        "stage_c_plan_sha256": plan["content_hash"],
        "target_registry_sha256": registry["content_hash"],
        "row_id": args.row_id,
        "target_id": row["target_id"],
        "parents": parents,
    })
    output = args.output_dir or args.campaign_root / "probes" / "inputs" / args.row_id
    write_immutable_json(output / "input_lineage.json", lineage)
    files = {}
    for role in ("model_train", "val_stop", "design_select"):
        identities, labels = _labels(label_paths[role])
        cache = loaded[role]
        positions = {value: index for index, value in enumerate(cache.identities)}
        if set(positions) != set(identities):
            raise ValueError(f"{role} target and labels identities differ")
        order = np.asarray([positions[value] for value in identities], dtype=np.int64)
        target = cache.values[row["target_id"]][order].astype(np.float32)
        target_mask = cache.masks[row["target_id"]][order].astype(bool)
        availability, group_order = _availability(target_mask, target_row)
        arrays = {
            "identities": np.asarray(identities),
            "labels": labels,
            "target": target,
            "target_mask": target_mask,
            "availability": availability.astype(np.float32),
            "availability_group_order": np.asarray(group_order),
            "target_cache_manifest_sha256": np.asarray(lineage["content_hash"]),
        }
        if tap_paths:
            tap = _npz(tap_paths[role])
            tap_ids = tuple(str(value) for value in tap["identities"].tolist())
            if tap_ids != identities or str(tap["tap"].item()) != row["tap"]:
                raise ValueError(f"{role} frozen tap identities/tap differ")
            arrays.update({
                "states": tap["states"].astype(np.float32),
                "particle_mask": tap["particle_mask"].astype(bool),
                "probe_encoder_lock_sha256": tap["probe_encoder_lock_sha256"],
                "tap": tap["tap"],
            })
        if raw_paths:
            raw = _npz(raw_paths[role])
            raw_ids = tuple(str(value) for value in raw["identities"].tolist())
            if raw_ids != identities or not {"raw_summary", "jet_context"}.issubset(raw):
                raise ValueError(f"{role} raw-summary probe input differs")
            if raw["jet_context"].shape != (len(identities), 5):
                raise ValueError("raw jet context must have five locked fields")
            arrays.update({
                "raw_summary": raw["raw_summary"].astype(np.float32),
                "jet_context": raw["jet_context"].astype(np.float32),
            })
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
