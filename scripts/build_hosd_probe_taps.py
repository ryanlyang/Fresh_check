#!/usr/bin/env python3
"""Capture authenticated seed-101 H_BASE block-2/4/8 probe states."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    HBaseParticleTransformer,
    TAP_BLOCKS,
    build_probe_encoder_lock,
    load_and_validate_campaign,
    authorize_access,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    BASELINE_COMPLETION_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    identity_order_hash,
    load_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_experts import (  # noqa: E402
    NativeHLTExpertDataset,
    collate_native_hlt_expert_batch,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(rows, name, *, required_replicas):
    output = {}
    for row in rows:
        key, separator, value = row.partition("=")
        if not separator:
            raise ValueError(f"{name} requires REPLICA=PATH")
        replica = int(key)
        if replica not in range(4) or replica in output:
            raise ValueError(f"{name} replica differs")
        output[replica] = Path(value)
    if set(output) != set(required_replicas):
        raise ValueError(
            f"{name} requires replicas {sorted(required_replicas)}"
        )
    return output


def _labels(path):
    with np.load(path, allow_pickle=False) as payload:
        identities = tuple(str(value) for value in payload["identities"].tolist())
        labels = np.asarray(payload["labels"], dtype=np.int64)
    if labels.shape != (len(identities),) or len(identities) != len(set(identities)):
        raise ValueError("probe tap labels differ")
    return labels, identities


def _dataset(paths, labels_path, role, *, realization_policy):
    arrays, metadata = {}, {}
    for replica, path in paths.items():
        arrays[replica], metadata[replica] = load_hlt_v3_cache(path)
    labels, identities = _labels(labels_path)
    source_role = "val_design" if role == "design_select" else role
    if {
        str(value.get("logical_role")) for value in metadata.values()
    } != {source_role}:
        raise ValueError("probe tap HLT cache logical role differs")
    source_indices = {}
    for replica, replica_arrays in arrays.items():
        raw_source_ids = replica_arrays["identities"]
        if (
            len(raw_source_ids) == len(identities)
            and identity_order_hash(raw_source_ids)
            == identity_order_hash(identities)
        ):
            source_indices[replica] = range(len(identities))
            continue
        source_ids = tuple(str(value) for value in raw_source_ids)
        positions = {value: index for index, value in enumerate(source_ids)}
        requested = set(identities)
        if (
            len(positions) != len(source_ids)
            or len(requested) != len(identities)
            or not requested.issubset(positions)
            or (role != "design_select" and requested != set(positions))
        ):
            raise ValueError("probe tap HLT cache lacks label identities")
        source_indices[replica] = np.asarray(
            [positions[value] for value in identities], dtype=np.int64
        )
    return NativeHLTExpertDataset(
        replica_arrays=arrays,
        replica_metadata=metadata,
        labels=labels,
        identities=identities,
        logical_role=role,
        source_logical_role=source_role,
        source_indices_by_replica=source_indices,
        realization_policy=realization_policy,
    )


def _capture(model, dataset, replica, *, device):
    rows = {tap: [] for tap in TAP_BLOCKS}
    masks = {tap: [] for tap in TAP_BLOCKS}
    full_length = int(dataset.replicas[replica]["tokens"].shape[1])
    with torch.no_grad():
        for start in range(0, len(dataset), 64):
            samples = [
                dataset.item_for_replica(index, replica)
                for index in range(start, min(start + 64, len(dataset)))
            ]
            batch = collate_native_hlt_expert_batch(samples)
            result = model.forward_with_taps(
                batch["features"][:, 15:17].to(device),
                batch["features"].to(device),
                batch["vectors"].to(device),
                batch["mask"].to(device),
            )
            for tap in TAP_BLOCKS:
                state = result.states[tap].float().cpu()
                mask = result.masks[tap].cpu()
                if state.shape[1] > full_length:
                    raise RuntimeError("Weaver tap exceeds source particle length")
                padded = torch.zeros(
                    state.shape[0], full_length, state.shape[2], dtype=state.dtype
                )
                padded_mask = torch.zeros(
                    state.shape[0], full_length, dtype=torch.bool
                )
                padded[:, : state.shape[1]] = state
                padded_mask[:, : mask.shape[1]] = mask
                rows[tap].append(padded.numpy())
                masks[tap].append(padded_mask.numpy())
    return (
        {tap: np.concatenate(value) for tap, value in rows.items()},
        {tap: np.concatenate(value) for tap, value in masks.items()},
    )


def _publish(path: Path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(
                not np.array_equal(existing[name], value)
                for name, value in arrays.items()
            ):
                raise FileExistsError("probe tap cache differs on reuse")
        return _sha256(path)
    np.savez_compressed(path, **arrays)
    return _sha256(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--baseline-completion", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-cache", action="append", default=[])
    parser.add_argument("--val-stop-cache", action="append", default=[])
    parser.add_argument("--design-select-cache", action="append", default=[])
    parser.add_argument("--train-labels", required=True, type=Path)
    parser.add_argument("--val-stop-labels", required=True, type=Path)
    parser.add_argument("--design-select-labels", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    for resource in (
        "model_train_hlt",
        "val_stop_hlt",
        "design_select_hlt",
    ):
        authorize_access(worker_role="probe_worker", requested_resource=resource)
    completion = load_hashed_json(
        args.baseline_completion, expected_contract=BASELINE_COMPLETION_CONTRACT
    )
    if (
        completion.get("source") != campaign["source"]
        or completion["baseline_id"] != "H_BASE"
        or completion["checkpoint_sha256"] != _sha256(args.checkpoint)
    ):
        raise ValueError("probe baseline checkpoint lineage differs")
    module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    model = HBaseParticleTransformer(weaver_module=module)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    model.to(device).eval()
    lock = build_probe_encoder_lock(
        encoder=model,
        checkpoint_path=args.checkpoint,
        checkpoint_registration_sha256=completion["content_hash"],
        campaign_spec_sha256=campaign["content_hash"],
        source=campaign["source"],
    )
    output = args.output_dir or args.campaign_root / "probes" / "frozen_taps"
    write_immutable_json(output / "probe_encoder_lock.json", lock)
    datasets = {
        "model_train": _dataset(
            _mapping(
                args.train_cache,
                "--train-cache",
                required_replicas={0, 1, 2, 3},
            ),
            args.train_labels,
            "model_train",
            realization_policy="R_MULTI",
        ),
        "val_stop": _dataset(
            _mapping(
                args.val_stop_cache,
                "--val-stop-cache",
                required_replicas={0},
            ),
            args.val_stop_labels,
            "val_stop",
            realization_policy="R_FIXED",
        ),
        "design_select": _dataset(
            _mapping(
                args.design_select_cache,
                "--design-select-cache",
                required_replicas={0},
            ),
            args.design_select_labels,
            "design_select",
            realization_policy="R_FIXED",
        ),
    }
    files = {}
    for split, dataset in datasets.items():
        replicas = range(4) if split == "model_train" else (0,)
        captured = {}
        for replica in replicas:
            captured[replica] = _capture(model, dataset, replica, device=device)
        for tap in TAP_BLOCKS:
            states = (
                np.stack([captured[r][0][tap] for r in replicas])
                if split == "model_train"
                else captured[0][0][tap]
            )
            mask = (
                np.stack([captured[r][1][tap] for r in replicas])
                if split == "model_train"
                else captured[0][1][tap]
            )
            path = output / f"{split}__{tap}.npz"
            files[f"{split}/{tap}"] = {
                "path": str(path.resolve()),
                "sha256": _publish(path, {
                    "identities": np.asarray(dataset.identities),
                    "labels": dataset.labels,
                    "states": states.astype(np.float32),
                    "particle_mask": mask.astype(bool),
                    "probe_encoder_lock_sha256": np.asarray(lock["content_hash"]),
                    "tap": np.asarray(tap),
                }),
            }
    manifest = {
        "contract": "hosd_frozen_probe_tap_cache_manifest_v1",
        "schema_version": 1,
        "source": campaign["source"],
        "campaign_spec_sha256": campaign["content_hash"],
        "probe_encoder_lock_sha256": lock["content_hash"],
        "model_train_replicas": [0, 1, 2, 3],
        "evaluation_replica": 0,
        "files": files,
    }
    from teacher_logit_reco.hlt_offline_structure_distillation.contracts import with_content_hash
    manifest = with_content_hash(manifest)
    write_immutable_json(output / "tap_cache_manifest.json", manifest)
    print(json.dumps({
        "probe_encoder_lock_sha256": lock["content_hash"],
        "tap_cache_manifest_sha256": manifest["content_hash"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
