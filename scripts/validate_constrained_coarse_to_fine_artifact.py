#!/usr/bin/env python3
"""Fail-closed completion checks for constrained coarse-to-fine artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import zipfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HIERARCHY_TARGET_CACHE_CONTRACT = "constrained_coarse_to_fine_hierarchy_target_cache_v1"
HIERARCHY_TARGET_CACHE_SET_CONTRACT = "constrained_coarse_to_fine_hierarchy_target_cache_set_v1"
HIERARCHY_TARGET_EXPECTED_HLT_PROFILE = "fixed_hlt_v2_realistic"
HIERARCHY_TARGET_EXPECTED_HLT_PROFILE_VERSION = "v1"
HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH = 2.5

_VIEW_ARRAY_KEYS = ("tokens", "mask", "labels", "jet_file_indices", "jet_entries")
_PREDICTION_ARRAY_KEYS = ("logits", "probs", "labels", "jet_file_indices", "jet_entries")
_TARGET_SHARD_ARRAY_KEYS = (
    "global_accounting",
    "level1_accounting",
    "level2_accounting",
    "level3_accounting",
    "final_cell_indices",
    "reference_eta",
    "reference_phi",
    "valid_hlt_counts",
    "valid_offline_counts",
    "unknown_pid_counts",
    "clipped_particle_counts",
    "labels",
    "jet_file_indices",
    "jet_entries",
)

_RECONSTRUCTOR_REPORT_VARIANTS = {
    "B0": "B0_global_only",
    "B1": "B1_global_8",
    "B2": "B2_global_8_32",
    "B3": "B3_global_8_32_128",
    "B4": "B4_no_moments",
    "B5": "B5_no_composition",
    "B6": "B6_no_counts",
    "B7": "B7_direct_child_totals",
    "C0": "C0_deterministic_k8",
    "C1": "C1_deterministic_k16",
    "C2": "C2_no_dust",
    "C3": "C3_sinkhorn",
    "C4": "C4_hungarian",
    "C5": "C5_uncertainty",
    "C6": "C6_multiview",
    "C5-B1": "C5-B1",
    "C5-B2": "C5-B2",
    "C5-B3": "C5-B3",
    "C5-no-slot": "C5_uncertainty",
    "C5-unconstrained": "C5_uncertainty",
    "Cdirect-unconstrained": "C5_uncertainty",
}


def _expected_training_variant(run_id: str, *, tagger: bool) -> str:
    if not tagger:
        return _RECONSTRUCTOR_REPORT_VARIANTS.get(run_id, run_id)
    if "-seed" in run_id:
        base, suffix = run_id.rsplit("-seed", 1)
        if base.startswith("D") and suffix.isdigit():
            return base
    return run_id


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing required JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _manifest_hash(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else path.open
    if path.suffix == ".gz":
        with opener(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        with opener("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty required file: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _npz_logical_hash(path: Path, names: tuple[str, ...]) -> str:
    """Recompute hash_arrays() while streaming decompressed NPY payloads."""

    digest = hashlib.sha256()
    with zipfile.ZipFile(path, "r") as archive:
        members = set(archive.namelist())
        for name in sorted(names):
            member = f"{name}.npy"
            if member not in members:
                raise ValueError(f"{path} lacks required NPZ member {member}")
            with archive.open(member, "r") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version in {(2, 0), (3, 0)}:
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError(f"unsupported NPY version {version} in {path}/{member}")
                if fortran_order:
                    raise ValueError(f"{path}/{member} is Fortran-ordered; C-order hash is unavailable")
                dtype = np.dtype(dtype)
                if dtype.hasobject:
                    raise ValueError(f"{path}/{member} contains object data")
                digest.update(name.encode("utf-8"))
                digest.update(str(dtype).encode("utf-8"))
                digest.update(json.dumps(tuple(int(value) for value in shape)).encode("utf-8"))
                remaining = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
                while remaining:
                    block = handle.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise ValueError(f"truncated NPY payload in {path}/{member}")
                    digest.update(block)
                    remaining -= len(block)
                if handle.read(1):
                    raise ValueError(f"unexpected trailing bytes in {path}/{member}")
    return digest.hexdigest()


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _target_content_hash_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cache_contract": metadata["cache_contract"],
        "builder_version": metadata["builder_version"],
        "split": metadata["split"],
        "n_jets": metadata["n_jets"],
        "target_dtype": metadata["target_dtype"],
        "source_manifest_hash": metadata["source_manifest_hash"],
        "hlt_content_hash": metadata["hlt_content_hash"],
        "offline_content_hash": metadata["offline_content_hash"],
        "jet_identity_hash": metadata["jet_identity_hash"],
        "layout": metadata["layout"],
        "shards": [
            {
                "filename": shard["filename"],
                "shard_index": shard["shard_index"],
                "start": shard["start"],
                "stop": shard["stop"],
                "n_jets": shard["n_jets"],
                "content_hash": shard["content_hash"],
            }
            for shard in metadata["shards"]
        ],
    }


def _active_split_provenance(
    split: str,
    *,
    hlt_cache_dir: Path,
    offline_cache_dir: Path,
    target_cache_dir: Path,
    manifest_sha: str,
) -> dict[str, Any]:
    hlt = _read_json(hlt_cache_dir / f"{split}_fixed_hlt_metadata.json")
    offline = _read_json(offline_cache_dir / f"{split}_offline_metadata.json")
    target = _read_json(target_cache_dir / f"{split}_hierarchy_targets_metadata.json")
    _require_hlt_contract(hlt, manifest_sha, f"active HLT cache/{split}")
    _require_equal(offline.get("source_manifest_hash"), manifest_sha, f"active offline cache/{split} manifest")
    _require_hlt_contract(target, manifest_sha, f"active target cache/{split}")
    offline_hash = offline.get("offline_content_hash") or offline.get("content_hash")
    _require_equal(target.get("hlt_content_hash"), hlt.get("hlt_content_hash"), f"active target/{split} HLT")
    _require_equal(target.get("offline_content_hash"), offline_hash, f"active target/{split} offline")
    _require_equal(target.get("jet_identity_hash"), hlt.get("jet_identity_hash"), f"active target/{split} HLT identity")
    _require_equal(target.get("jet_identity_hash"), offline.get("jet_identity_hash"), f"active target/{split} offline identity")
    return {
        "source_manifest_hash": manifest_sha,
        "hlt_content_hash": hlt.get("hlt_content_hash"),
        "offline_content_hash": offline_hash,
        "target_content_hash": target.get("target_content_hash"),
        "jet_identity_hash": target.get("jet_identity_hash"),
    }


def _require_active_training_provenance(
    report: Mapping[str, Any],
    *,
    hlt_cache_dir: Path,
    offline_cache_dir: Path,
    target_cache_dir: Path,
    manifest_sha: str,
    label: str,
) -> None:
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{label} lacks provenance")
    for split in ("model_train", "model_val"):
        row = provenance.get(split)
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} lacks {split} provenance")
        active = _active_split_provenance(
            split,
            hlt_cache_dir=hlt_cache_dir,
            offline_cache_dir=offline_cache_dir,
            target_cache_dir=target_cache_dir,
            manifest_sha=manifest_sha,
        )
        for field, expected in active.items():
            _require_equal(row.get(field), expected, f"{label}/{split} active {field}")


def _require_hlt_contract(row: Mapping[str, Any], manifest_sha: str, label: str) -> None:
    _require_equal(row.get("source_manifest_hash"), manifest_sha, f"{label} source_manifest_hash")
    _require_equal(row.get("hlt_profile"), HIERARCHY_TARGET_EXPECTED_HLT_PROFILE, f"{label} hlt_profile")
    _require_equal(
        str(row.get("hlt_profile_version") or ""),
        HIERARCHY_TARGET_EXPECTED_HLT_PROFILE_VERSION,
        f"{label} hlt_profile_version",
    )
    try:
        strength = float(row.get("hlt_degradation_strength"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} lacks numeric hlt_degradation_strength") from exc
    if abs(strength - HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH) > 1.0e-12:
        raise ValueError(
            f"{label} HLT strength mismatch: {strength} != {HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH}"
        )


def _require_provenance(report: Mapping[str, Any], manifest_sha: str, label: str) -> None:
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{label} lacks provenance")
    for split in ("model_train", "model_val"):
        row = provenance.get(split)
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} lacks {split} provenance")
        _require_hlt_contract(row, manifest_sha, f"{label}/{split}")
        for field in ("hlt_content_hash", "jet_identity_hash"):
            if row.get(field) in (None, ""):
                raise ValueError(f"{label}/{split} lacks {field}")


def _validate_hlt_cache(root: Path, splits: tuple[str, ...], manifest_sha: str) -> None:
    for split in splits:
        _require_file(root / f"{split}_fixed_hlt.npz")
        metadata = _read_json(root / f"{split}_fixed_hlt_metadata.json")
        _require_hlt_contract(metadata, manifest_sha, f"HLT cache/{split}")
        for field in ("hlt_content_hash", "jet_identity_hash"):
            if metadata.get(field) in (None, ""):
                raise ValueError(f"HLT cache/{split} lacks {field}")
        observed = _npz_logical_hash(root / f"{split}_fixed_hlt.npz", _VIEW_ARRAY_KEYS)
        _require_equal(observed, metadata.get("hlt_content_hash"), f"HLT cache/{split} content")


def _validate_offline_cache(root: Path, splits: tuple[str, ...], manifest_sha: str) -> None:
    for split in splits:
        _require_file(root / f"{split}_offline.npz")
        metadata = _read_json(root / f"{split}_offline_metadata.json")
        _require_equal(metadata.get("source_manifest_hash"), manifest_sha, f"offline cache/{split} manifest")
        for field in ("offline_content_hash", "jet_identity_hash"):
            if metadata.get(field) in (None, ""):
                raise ValueError(f"offline cache/{split} lacks {field}")
        observed = _npz_logical_hash(root / f"{split}_offline.npz", _VIEW_ARRAY_KEYS)
        _require_equal(observed, metadata.get("offline_content_hash") or metadata.get("content_hash"), f"offline cache/{split} content")


def _validate_target_cache(
    root: Path,
    splits: tuple[str, ...],
    manifest_sha: str,
    *,
    hlt_cache_dir: Path | None = None,
    offline_cache_dir: Path | None = None,
) -> None:
    cache_set = _read_json(root / "hierarchy_target_cache_manifest.json")
    _require_equal(cache_set.get("cache_set_contract"), HIERARCHY_TARGET_CACHE_SET_CONTRACT, "target cache set contract")
    _require_equal(cache_set.get("source_manifest_hash"), manifest_sha, "target cache set manifest")
    _require_hlt_contract(cache_set, manifest_sha, "target cache set")
    declared = set(str(value) for value in cache_set.get("splits", ()))
    missing = sorted(set(splits) - declared)
    if missing:
        raise ValueError(f"target cache set is missing splits: {missing}")
    observed_split_hashes: dict[str, str] = {}
    for split in splits:
        metadata = _read_json(root / f"{split}_hierarchy_targets_metadata.json")
        _require_equal(metadata.get("cache_contract"), HIERARCHY_TARGET_CACHE_CONTRACT, f"target/{split} contract")
        _require_hlt_contract(metadata, manifest_sha, f"target/{split}")
        for field in ("target_content_hash", "hlt_content_hash", "offline_content_hash", "jet_identity_hash"):
            if metadata.get(field) in (None, ""):
                raise ValueError(f"target/{split} lacks {field}")
        shards = metadata.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"target/{split} has no declared shards")
        shard_root = root / f"{split}_hierarchy_targets"
        for row in shards:
            if not isinstance(row, Mapping) or not row.get("filename"):
                raise ValueError(f"target/{split} contains malformed shard metadata")
            _require_file(shard_root / str(row["filename"]))
            observed_hash = _npz_logical_hash(
                shard_root / str(row["filename"]),
                _TARGET_SHARD_ARRAY_KEYS,
            )
            _require_equal(observed_hash, row.get("content_hash"), f"target/{split}/{row['filename']} content")
        observed_target_hash = _json_hash(_target_content_hash_payload(metadata))
        _require_equal(observed_target_hash, metadata.get("target_content_hash"), f"target/{split} aggregate content")
        observed_split_hashes[split] = observed_target_hash
        if hlt_cache_dir is not None:
            hlt = _read_json(hlt_cache_dir / f"{split}_fixed_hlt_metadata.json")
            _require_equal(metadata.get("hlt_content_hash"), hlt.get("hlt_content_hash"), f"target/{split} active HLT")
            _require_equal(metadata.get("jet_identity_hash"), hlt.get("jet_identity_hash"), f"target/{split} active HLT identity")
        if offline_cache_dir is not None:
            offline = _read_json(offline_cache_dir / f"{split}_offline_metadata.json")
            offline_hash = offline.get("offline_content_hash") or offline.get("content_hash")
            _require_equal(metadata.get("offline_content_hash"), offline_hash, f"target/{split} active offline")
            _require_equal(metadata.get("jet_identity_hash"), offline.get("jet_identity_hash"), f"target/{split} active offline identity")
    declared_hashes = cache_set.get("split_target_content_hashes")
    if not isinstance(declared_hashes, Mapping):
        raise ValueError("target cache set lacks split_target_content_hashes")
    for split, observed_hash in observed_split_hashes.items():
        _require_equal(declared_hashes.get(split), observed_hash, f"target cache set/{split} content")
    aggregate_payload = dict(cache_set)
    declared_aggregate_hash = aggregate_payload.pop("cache_set_content_hash", None)
    _require_equal(_json_hash(aggregate_payload), declared_aggregate_hash, "target cache set aggregate content")


def _validate_training(
    root: Path,
    run_id: str,
    manifest_sha: str,
    *,
    tagger: bool,
    hlt_cache_dir: Path,
    offline_cache_dir: Path,
    target_cache_dir: Path,
) -> None:
    report = _read_json(root / "run_report.json")
    if report.get("ok") is not True:
        raise ValueError(f"{run_id} run_report is not successful")
    observed = report.get("variant") if tagger else (report.get("variant") or report.get("family"))
    expected = _expected_training_variant(run_id, tagger=tagger)
    if observed not in (expected, None):
        raise ValueError(f"{run_id} report identifies a different run: {observed}")
    declared_run_id = report.get("run_id")
    if declared_run_id not in (None, run_id):
        raise ValueError(f"{run_id} report declares a different campaign run_id: {declared_run_id}")
    if not tagger and run_id in {"C5-unconstrained", "Cdirect-unconstrained"}:
        model = report.get("model")
        slot = model.get("slot_config") if isinstance(model, Mapping) else None
        if not isinstance(slot, Mapping) or slot.get("direct_particle_decoding") is not True:
            raise ValueError(f"{run_id} is not attested as a direct particle decoder")
        if slot.get("constrain_accounting") is not False:
            raise ValueError(f"{run_id} unexpectedly retains hierarchical slot accounting constraints")
    if not tagger and run_id == "C5-no-slot":
        config = report.get("training_config")
        if not isinstance(config, Mapping) or float(config.get("slot_loss_weight", -1.0)) != 0.0:
            raise ValueError("C5-no-slot is not attested with slot_loss_weight == 0")
    checkpoint = root / "best_model_val.pt"
    _require_file(checkpoint)
    checkpoint_sha = report.get("checkpoint_sha256")
    if checkpoint_sha in (None, ""):
        raise ValueError(f"{run_id} run_report lacks checkpoint_sha256")
    _require_equal(_file_sha256(checkpoint), checkpoint_sha, f"{run_id} checkpoint_sha256")
    if tagger and not isinstance(report.get("provenance"), Mapping):
        source_metadata = _read_json(root / "source_metadata.json")
        report = {**report, "provenance": source_metadata.get("provenance")}
    _require_provenance(report, manifest_sha, run_id)
    _require_active_training_provenance(
        report,
        hlt_cache_dir=hlt_cache_dir,
        offline_cache_dir=offline_cache_dir,
        target_cache_dir=target_cache_dir,
        manifest_sha=manifest_sha,
        label=run_id,
    )


def _validate_predictions(
    root: Path,
    run_id: str,
    splits: tuple[str, ...],
    manifest_sha: str,
    *,
    hlt_cache_dir: Path,
    offline_cache_dir: Path,
    tagger_root: Path,
) -> None:
    report = _read_json(root / "prediction_run_report.json")
    if report.get("ok") is not True:
        raise ValueError(f"{run_id} prediction_run_report is not successful")
    rows = report.get("splits")
    if not isinstance(rows, Mapping):
        raise ValueError(f"{run_id} prediction report lacks split rows")
    for split in splits:
        _require_file(root / f"{split}_predictions.npz")
        _require_file(root / f"{split}_representations.npz")
        metadata = _read_json(root / f"{split}_predictions_metadata.json")
        _require_equal(metadata.get("run_id"), run_id, f"prediction/{run_id}/{split} run_id")
        offline_reference = run_id == "A2"
        if offline_reference:
            _require_equal(metadata.get("source_manifest_hash"), manifest_sha, f"prediction/{run_id}/{split} source_manifest_hash")
            if metadata.get("input_view") != "offline" or metadata.get("offline_inputs_loaded") is not True:
                raise ValueError(f"prediction/{run_id}/{split} is not an attested offline reference")
            if metadata.get("offline_content_hash") in (None, ""):
                raise ValueError(f"prediction/{run_id}/{split} lacks offline_content_hash")
            active = _read_json(offline_cache_dir / f"{split}_offline_metadata.json")
            active_content_hash = active.get("offline_content_hash") or active.get("content_hash")
        else:
            _require_hlt_contract(metadata, manifest_sha, f"prediction/{run_id}/{split}")
            if metadata.get("deployable_hlt_only") is not True:
                raise ValueError(f"prediction/{run_id}/{split} is not deployable_hlt_only")
            active = _read_json(hlt_cache_dir / f"{split}_fixed_hlt_metadata.json")
            active_content_hash = active.get("hlt_content_hash")
        _require_equal(
            metadata.get("offline_content_hash" if offline_reference else "hlt_content_hash"),
            active_content_hash,
            f"prediction/{run_id}/{split} active input content",
        )
        _require_equal(
            metadata.get("jet_identity_hash"),
            active.get("jet_identity_hash"),
            f"prediction/{run_id}/{split} active input identity",
        )
        required_fields = [
            "prediction_content_hash",
            "checkpoint_sha256",
            "jet_identity_hash",
            "fusion_representation_sha256",
            "offline_content_hash" if offline_reference else "hlt_content_hash",
        ]
        for field in required_fields:
            if metadata.get(field) in (None, ""):
                raise ValueError(f"prediction/{run_id}/{split} lacks {field}")
        prediction_hash = _npz_logical_hash(
            root / f"{split}_predictions.npz",
            _PREDICTION_ARRAY_KEYS,
        )
        _require_equal(
            prediction_hash,
            metadata.get("prediction_content_hash"),
            f"prediction/{run_id}/{split} content",
        )
        representation = root / f"{split}_representations.npz"
        _require_equal(
            _file_sha256(representation),
            metadata.get("fusion_representation_sha256"),
            f"prediction/{run_id}/{split} representation sha256",
        )
        checkpoint = Path(str(metadata.get("checkpoint_path") or ""))
        _require_file(checkpoint)
        _require_equal(
            _file_sha256(checkpoint),
            metadata.get("checkpoint_sha256"),
            f"prediction/{run_id}/{split} checkpoint sha256",
        )
        active_checkpoint = tagger_root / run_id / "best_model_val.pt"
        _require_file(active_checkpoint)
        _require_equal(
            _file_sha256(active_checkpoint),
            metadata.get("checkpoint_sha256"),
            f"prediction/{run_id}/{split} active tagger checkpoint",
        )
        if split == "final_test":
            if metadata.get("final_test_confirmed") is not True:
                raise ValueError(f"prediction/{run_id}/final_test is not confirmed")
            receipt = _read_json(root / "final_test_claim_receipt.json")
            if receipt.get("immutable_final_claim") is not True:
                raise ValueError(f"prediction/{run_id}/final_test lacks immutable claim receipt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("hlt-cache", "offline-cache", "target-cache", "reconstructor", "tagger", "prediction"))
    parser.add_argument("--path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--splits", nargs="*", default=[])
    parser.add_argument("--hlt-cache-dir")
    parser.add_argument("--offline-cache-dir")
    parser.add_argument("--target-cache-dir")
    parser.add_argument("--tagger-root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.path)
    manifest_sha = _manifest_hash(Path(args.manifest))
    splits = tuple(str(value) for value in args.splits)
    if args.kind.endswith("cache") and not splits:
        raise ValueError(f"{args.kind} validation requires --splits")
    if args.kind in ("reconstructor", "tagger", "prediction") and not args.run_id:
        raise ValueError(f"{args.kind} validation requires --run-id")
    if args.kind == "hlt-cache":
        _validate_hlt_cache(root, splits, manifest_sha)
    elif args.kind == "offline-cache":
        _validate_offline_cache(root, splits, manifest_sha)
    elif args.kind == "target-cache":
        _validate_target_cache(
            root,
            splits,
            manifest_sha,
            hlt_cache_dir=None if not args.hlt_cache_dir else Path(args.hlt_cache_dir),
            offline_cache_dir=None if not args.offline_cache_dir else Path(args.offline_cache_dir),
        )
    elif args.kind == "reconstructor":
        if not args.hlt_cache_dir or not args.offline_cache_dir or not args.target_cache_dir:
            raise ValueError("reconstructor validation requires active --hlt-cache-dir, --offline-cache-dir, and --target-cache-dir")
        _validate_training(
            root,
            str(args.run_id),
            manifest_sha,
            tagger=False,
            hlt_cache_dir=Path(args.hlt_cache_dir),
            offline_cache_dir=Path(args.offline_cache_dir),
            target_cache_dir=Path(args.target_cache_dir),
        )
    elif args.kind == "tagger":
        if not args.hlt_cache_dir or not args.offline_cache_dir or not args.target_cache_dir:
            raise ValueError("tagger validation requires active --hlt-cache-dir, --offline-cache-dir, and --target-cache-dir")
        _validate_training(
            root,
            str(args.run_id),
            manifest_sha,
            tagger=True,
            hlt_cache_dir=Path(args.hlt_cache_dir),
            offline_cache_dir=Path(args.offline_cache_dir),
            target_cache_dir=Path(args.target_cache_dir),
        )
    else:
        if not splits:
            raise ValueError("prediction validation requires --splits")
        if not args.hlt_cache_dir or not args.offline_cache_dir or not args.tagger_root:
            raise ValueError("prediction validation requires active --hlt-cache-dir, --offline-cache-dir, and --tagger-root")
        _validate_predictions(
            root,
            str(args.run_id),
            splits,
            manifest_sha,
            hlt_cache_dir=Path(args.hlt_cache_dir),
            offline_cache_dir=Path(args.offline_cache_dir),
            tagger_root=Path(args.tagger_root),
        )
    print(f"valid {args.kind}: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
